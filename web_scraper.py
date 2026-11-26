import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import unquote, unquote_plus, urljoin, urlsplit, urlunsplit

import requests
from mysql.connector import DataError, Error
from requests.adapters import HTTPAdapter
from requests.utils import requote_uri
from urllib3.exceptions import HTTPError as TransportError
from urllib3.util import Timeout


LOGGER = logging.getLogger(__name__)
DISCOVERY_STATE_KEY = "web_articles_last_tweetid"
DISCOVERY_DAYS = 7
RECOVERY_STATE_KEY = "web_articles_recover"
RECOVERY_MAX_ATTEMPTS = 3
TCO_URL_PREFIXES = ("https://t.co/%", "http://t.co/%")
USER_AGENT = "NetVoraBot/1.0"
MAX_URL_LENGTH = 4096
ROBOTS_MAX_BYTES = 512_000
TRACKING_KEYS = {"fbclid", "gclid", "dclid", "msclkid", "igshid", "mc_cid", "mc_eid", "twclid"}
SOCIAL_DOMAINS = {
    "x.com", "twitter.com", "twimg.com", "facebook.com", "fb.com", "fb.me", "fb.watch",
    "fbcdn.net", "instagram.com", "threads.net", "threads.com", "tiktok.com", "youtube.com",
    "youtu.be", "linkedin.com", "reddit.com", "redd.it", "twitch.tv", "pinterest.com", "pin.it",
    "bsky.app", "snapchat.com", "whatsapp.com", "t.me",
}
FILE_EXTENSIONS = frozenset(
    "7z aac apk avi avif bmp csv doc docx exe gif gz heic ico jpeg jpg json m4a m4v mov mp3 mp4 "
    "mpeg msi ogg ogv pdf png ppt pptx rar rss svg tar tif tiff txt wav webm webp xls xlsx xml zip"
    .split()
)
URL_TEXT = r"[^\s<>\x00-\x1f\"“”‘’]+"
BARE_URL_PATTERN = re.compile(r"(?:(?:[^\W_]|-){1,63}\.)+[a-z]{2,63}/" + URL_TEXT, re.I)
URL_PATTERN = re.compile(
    rf"(?<![\w@/])(?:https?://|www\.){URL_TEXT}|(?<![\w@./:%=?&+#\\-]){BARE_URL_PATTERN.pattern}",
    re.I,
)
DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
NUMERIC_HOST = re.compile(r"(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*")
NON_PUBLIC_NETWORKS = tuple(map(ipaddress.ip_network, (
    "192.0.0.0/24", "192.88.99.0/24", "168.63.129.16/32", "2001::/23", "2002::/16",
)))
GLOBAL_IPV6 = ipaddress.ip_network("2000::/3")
DNS_WORKERS = ThreadPoolExecutor(max_workers=2, thread_name_prefix="article-dns")
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
ALLOWED_HTML = {"text/html", "application/xhtml+xml"}
BLOCKED_TITLES = re.compile(
    r"^(?:403\b|404\b|access denied\b|acceso denegado\b|forbidden\b|just a moment\b|"
    r"attention required\b|verify (?:that )?you are human\b|consent required\b|cookie consent\b|"
    r"before you continue\b|antes de continuar\b|"
    r"page not found\b|p[áa]gina no encontrada\b|enable javascript and cookies\b)", re.I
)
RESTRICTED_ROBOTS = re.compile(r"(?:^|[\s,:])(?:none|noarchive|nosnippet)(?=$|[\s,])", re.I)


def _setting(name, default, low, high):
    key = f"WEB_ARTICLES_{name}"
    try:
        value = int(os.getenv(key, str(default)))
    except ValueError:
        raise ValueError(f"{key} debe ser un entero") from None
    if not low <= value <= high:
        raise ValueError(f"{key} debe estar entre {low} y {high}")
    return value


@dataclass(frozen=True)
class ScrapeSettings:
    limit: int = 10
    max_seconds: int = 45
    max_attempts: int = 3
    retry_seconds: int = 60
    processing_ttl_seconds: int = 900
    connect_timeout: int = 4
    read_timeout: int = 8
    max_bytes: int = 2_097_152
    max_redirects: int = 5
    domain_delay: int = 1
    min_words: int = 80

    @classmethod
    def from_env(cls):
        return cls(
            limit=_setting("SCRAPE_LIMIT", 10, 1, 100),
            max_seconds=_setting("SCRAPE_SECONDS", 45, 1, 240),
            max_attempts=_setting("MAX_ATTEMPTS", 3, 1, 20),
            retry_seconds=_setting("RETRY_SECONDS", 60, 1, 86400),
            processing_ttl_seconds=_setting("PROCESSING_TTL_SECONDS", 900, 60, 86400),
            connect_timeout=_setting("CONNECT_TIMEOUT", 4, 1, 60),
            read_timeout=_setting("READ_TIMEOUT", 8, 1, 60),
            max_bytes=_setting("MAX_BYTES", 2_097_152, 1024, 10_000_000),
            max_redirects=_setting("MAX_REDIRECTS", 5, 0, 10),
            domain_delay=_setting("DOMAIN_DELAY", 1, 1, 60),
            min_words=_setting("MIN_WORDS", 80, 10, 1000),
        )


def _ip_address(host):
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _public_ip(address):
    if address.version == 6 and (address.scope_id is not None or address not in GLOBAL_IPV6):
        return False
    return (
        address.is_global and not address.is_multicast and not address.is_reserved
        and not any(address in network for network in NON_PUBLIC_NETWORKS)
    )


def normalize_url(url, *, strip_tracking=True):
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url or len(url) > MAX_URL_LENGTH or re.search(r"[\s\x00-\x1f\x7f\\]", url):
        return None
    if url.lower().startswith("www.") or BARE_URL_PATTERN.fullmatch(url):
        url = "https://" + url
    try:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or "@" in parts.netloc:
            return None
        default_port = 443 if parts.scheme == "https" else 80
        if parts.netloc.endswith(":") or parts.port not in (None, default_port):
            return None
        host = parts.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        if "%" in host:
            return None
        address = _ip_address(host)
        if address is None:
            if (len(host) > 253 or "[" in parts.netloc or "." not in host or NUMERIC_HOST.fullmatch(host)
                    or not all(DNS_LABEL.fullmatch(label) for label in host.split("."))):
                return None
        elif address.version == 6:
            if not re.fullmatch(r"\[[^\]]+\](?::[0-9]+)?", parts.netloc):
                return None
            host = f"[{address.compressed}]"

        query = parts.query
        if strip_tracking:
            kept = []
            for parameter in query.split("&"):
                key = unquote_plus(parameter.partition("=")[0]).lower()
                if not key.startswith("utm_") and key not in TRACKING_KEYS:
                    kept.append(parameter)
            query = "&".join(kept)
        normalized = requote_uri(urlunsplit((parts.scheme, host, parts.path or "/", query, "")))
        return normalized if len(normalized) <= MAX_URL_LENGTH else None
    except (UnicodeError, ValueError):
        return None


def _public_target(host):
    address = _ip_address(host)
    if address is not None:
        return _public_ip(address)
    return host != "metadata.goog" and not host.endswith((".localhost", ".local", ".internal", ".home.arpa"))


def _social_host(host):
    return any(host == domain or host.endswith("." + domain) for domain in SOCIAL_DOMAINS)


def _article_target(url):
    parts = urlsplit(url)
    host = parts.hostname
    if not _public_target(host) or _social_host(host):
        return False
    path = unquote(parts.path).lower().rstrip("/")
    extension = path.rsplit("/", 1)[-1].partition(";")[0].rpartition(".")[2]
    return extension not in FILE_EXTENSIONS


def is_valid_article_url(url):
    normalized = normalize_url(url)
    return bool(normalized and _article_target(normalized))


def extract_urls_from_text(text):
    if not isinstance(text, str):
        return []
    urls = []
    for match in URL_PATTERN.finditer(text):
        url = match.group().rstrip(".,;:!?…'»")
        for left, right in (("(", ")"), ("[", "]"), ("{", "}")):
            while url.endswith(right) and url.count(right) > url.count(left):
                url = url[:-1]
        if is_valid_article_url(url):
            urls.append(url)
    return urls


def _url_entities(part):
    entities = part.get("entities")
    urls = entities.get("urls") if isinstance(entities, dict) else None
    return [item for item in urls if isinstance(item, dict)] if isinstance(urls, list) else []


def _tco_key(normalized_url):
    parts = urlsplit(normalized_url or "")
    if parts.hostname == "t.co":
        return parts._replace(scheme="https").geturl()
    return None


@dataclass(frozen=True)
class TweetLink:
    source_url: str
    target_url: str
    provisional_url: str | None = None


def extract_external_urls(tweet, *, include_text=True, metrics=None):
    note = tweet.get("note_tweet")
    parts = (note, tweet) if isinstance(note, dict) else (tweet,)
    choices = {}
    covered = set()
    expanded = set()
    has_entities = False
    for part in parts:
        for entity in _url_entities(part):
            has_entities = True
            candidates = []
            for priority, field in enumerate(("unwound_url", "expanded_url", "url")):
                target = entity.get(field)
                normalized = normalize_url(target)
                if normalized:
                    short = _tco_key(normalized)
                    # A known blocked destination also suppresses fallback to the short link.
                    rank = 2 if short else int(not _article_target(normalized))
                    candidates.append(((rank, priority), normalized, target))
                    covered.add(short or normalized)
            if candidates:
                priority, normalized, target = min(candidates)
                original = entity.get("url")
                source = normalize_url(original)
                if priority[0] == 2 and _tco_key(source):
                    normalized, target = source, original
                key = _tco_key(source) or source or normalized
                if any(rank < 2 and field_index < 2 for (rank, field_index), _, _ in candidates):
                    expanded.add(key)
                if key not in choices or priority < choices[key][0]:
                    choices[key] = (priority, normalized, TweetLink(original if source else target, target))

    selected = [link for _, normalized, link in choices.values() if _article_target(normalized)]
    if include_text:
        for part in parts:
            for source in extract_urls_from_text(part.get("text")):
                normalized = normalize_url(source)
                key = _tco_key(normalized) or normalized
                if key not in covered:
                    selected.append(TweetLink(source, source))
                    covered.add(key)
    if metrics is not None:
        metrics["tweets_with_url_entities"] += int(has_entities)
        metrics["expanded_urls_received"] += len(expanded)
        metrics["tweets_only_tco"] += int(bool(selected) and all(
            _tco_key(normalize_url(link.target_url)) for link in selected
        ))
    return selected


def _sha256(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _first_column(row, name):
    return row[name] if isinstance(row, dict) else row[0]


def _registration_target(cursor, url_hash):
    cursor.execute(
        """
        SELECT article_id FROM web_articles
        WHERE url_hash = %s OR canonical_hash = %s
        ORDER BY (canonical_hash = %s) DESC, article_id
        LIMIT 1
        """,
        (url_hash, url_hash, url_hash),
    )
    row = cursor.fetchone()
    if row:
        cursor.execute(
            "SELECT article_id, canonical_hash, canonical_url FROM web_articles WHERE article_id = %s FOR SHARE",
            (_first_column(row, "article_id"),),
        )
        row = cursor.fetchone()
        if row:
            article_id, canonical_hash, canonical_url = (
                (row["article_id"], row["canonical_hash"], row["canonical_url"])
                if isinstance(row, dict) else row
            )
            if canonical_hash is None and canonical_url:
                cursor.execute("SELECT article_id FROM web_articles WHERE canonical_hash = %s FOR SHARE", (
                    _sha256(canonical_url),
                ))
                owner = cursor.fetchone()
                if owner:
                    return _first_column(owner, "article_id")
            return article_id

    # A merged URL is still recorded in tweet_articles after its article row is removed.
    cursor.execute(
        "SELECT id FROM tweet_articles WHERE source_url_hash = %s ORDER BY id LIMIT 1",
        (url_hash,),
    )
    alias = cursor.fetchone()
    if alias:
        cursor.execute(
            "SELECT article_id FROM tweet_articles WHERE id = %s FOR SHARE",
            (_first_column(alias, "id"),),
        )
        row = cursor.fetchone()
        if row:
            return _first_column(row, "article_id")
    return None


def _delete_empty_article(cursor, article_id):
    cursor.execute(
        """
        DELETE FROM web_articles WHERE article_id = %s
          AND COALESCE(content, '') = '' AND summary IS NULL AND key_points IS NULL AND summarized_at IS NULL
          AND NOT EXISTS (SELECT 1 FROM tweet_articles WHERE article_id = %s)
        """,
        (article_id, article_id),
    )
    if not cursor.rowcount:
        raise RuntimeError("No se pudo retirar el artículo duplicado")


def _merge_provisional_source(cursor, source_hash, article_id):
    cursor.execute(
        """
        SELECT article_id FROM web_articles
        WHERE url_hash = %s AND article_id <> %s
          AND scraping_status IN ('PENDING', 'FAILED', 'SKIPPED')
        """,
        (source_hash, article_id),
    )
    row = cursor.fetchone()
    if not row:
        return
    provisional_id = _first_column(row, "article_id")
    cursor.execute(
        """
        SELECT article_id FROM web_articles WHERE article_id = %s
          AND scraping_status IN ('PENDING', 'FAILED', 'SKIPPED')
          AND canonical_hash IS NULL AND canonical_url IS NULL AND COALESCE(content, '') = ''
          AND summary IS NULL AND key_points IS NULL AND summarized_at IS NULL
        FOR UPDATE
        """,
        (provisional_id,),
    )
    if cursor.fetchone():
        cursor.execute("UPDATE tweet_articles SET article_id = %s WHERE article_id = %s", (article_id, provisional_id))
        _delete_empty_article(cursor, provisional_id)


def register_tweet_urls(cursor, tweetid, urls):
    normalized_urls = {}
    for item in urls:
        link = item if isinstance(item, TweetLink) else TweetLink(item, item)
        source = normalize_url(link.source_url)
        target = normalize_url(link.target_url)
        if source and target and _article_target(target):
            normalized_urls.setdefault(source, (target, link.source_url.strip(), link.provisional_url))

    registered = 0
    for normalized_source, (url, source, provisional) in sorted(normalized_urls.items()):
        source_hash = _sha256(normalized_source)
        url_hash = _sha256(url)
        has_expansion = source_hash != url_hash
        try:
            if not has_expansion:
                cursor.execute(
                    "SELECT id FROM tweet_articles WHERE tweetid = %s AND source_url_hash = %s",
                    (tweetid, source_hash),
                )
                if cursor.fetchone():
                    continue

            cursor.execute("SAVEPOINT web_article_url")
            try:
                article_id = _registration_target(cursor, url_hash)
                if article_id is None:
                    cursor.execute(
                        """
                        INSERT INTO web_articles
                            (url, url_hash, scraping_status, summary_status, created_at, updated_at)
                        VALUES (%s, %s, 'PENDING', 'PENDING', UTC_TIMESTAMP(), UTC_TIMESTAMP())
                        ON DUPLICATE KEY UPDATE article_id = LAST_INSERT_ID(article_id)
                        """,
                        (url, url_hash),
                    )
                    article_id = cursor.lastrowid
                update = "article_id = %s" if has_expansion else "id = id"
                params = (tweetid, article_id, source, source_hash)
                cursor.execute(
                    """
                    INSERT INTO tweet_articles (tweetid, article_id, source_url, source_url_hash, created_at)
                    VALUES (%s, %s, %s, %s, UTC_TIMESTAMP())
                    ON DUPLICATE KEY UPDATE
                    """ + update,
                    (*params, article_id) if has_expansion else params,
                )
                inserted = cursor.rowcount == 1
                if has_expansion:
                    _merge_provisional_source(cursor, source_hash, article_id)
                    if provisional and provisional != normalized_source:
                        short_key = _tco_key(normalized_source)
                        if short_key and _tco_key(provisional) == short_key:
                            _merge_provisional_source(cursor, _sha256(provisional), article_id)
                elif not inserted:
                    # Another transaction registered this link. Discard any newly inserted article.
                    cursor.execute("ROLLBACK TO SAVEPOINT web_article_url")
                registered += int(inserted or has_expansion)
            except Error as error:
                if not isinstance(error, DataError) and error.errno not in (1366, 1406):
                    raise
                cursor.execute("ROLLBACK TO SAVEPOINT web_article_url")
                LOGGER.warning("Enlace descartado para tweet %s (MySQL %s)", tweetid, error.errno)
            cursor.execute("RELEASE SAVEPOINT web_article_url")
        except Error as error:
            raise RuntimeError(f"Fallo registrando enlaces (MySQL {error.errno})") from None
    return registered


def _discovery_rows(cursor, cutoff, now, after, through, limit):
    cursor.execute(
        """
        SELECT tweetid, text FROM Tweets
        WHERE created >= %s AND created <= %s AND tweetid > %s
        """ + ("AND tweetid <= %s " if through is not None else "") + "ORDER BY tweetid LIMIT %s",
        (cutoff, now, after, through, limit) if through is not None else (cutoff, now, after, limit),
    )
    return cursor.fetchall()


def discover_historical_urls(conn, cursor, now, started, get_state, set_state):
    limit = _setting("DISCOVER_LIMIT", 200, 2, 1000)
    deadline = started + _setting("DISCOVER_SECONDS", 20, 1, 240)
    cutoff = now - timedelta(days=DISCOVERY_DAYS)
    state = json.loads(get_state(cursor, DISCOVERY_STATE_KEY, "0") or "0")
    if isinstance(state, dict):
        high, review = int(state["last_tweetid"]), int(state["review_tweetid"])
    else:
        high, review = int(state), 0

    metrics = {"tweets_scanned": 0, "tweets_reviewed": 0, "urls_found": 0, "tco_urls_found": 0, "links_registered": 0}

    def register_text(tweetid, tweet_text):
        urls = extract_urls_from_text(tweet_text)
        metrics["urls_found"] += len(urls)
        metrics["tco_urls_found"] += sum(urlsplit(normalize_url(url)).hostname == "t.co" for url in urls)
        metrics["links_registered"] += register_tweet_urls(cursor, tweetid, urls)

    if high and time.monotonic() < deadline:
        # Reserve part of each batch for rows that can arrive below the high-water mark.
        quota = max(1, limit // 4)
        rows = _discovery_rows(cursor, cutoff, now, review, high, quota)
        for tweetid, tweet_text in rows:
            if time.monotonic() >= deadline:
                break
            register_text(tweetid, tweet_text)
            metrics["tweets_reviewed"] += 1
            review = int(tweetid)
        if metrics["tweets_reviewed"] == len(rows) < quota:
            review = 0

    if time.monotonic() < deadline:
        rows = _discovery_rows(cursor, cutoff, now, high, None, limit - metrics["tweets_reviewed"])
        for tweetid, tweet_text in rows:
            if time.monotonic() >= deadline:
                break
            register_text(tweetid, tweet_text)
            metrics["tweets_scanned"] += 1
            high = int(tweetid)

    set_state(cursor, DISCOVERY_STATE_KEY, json.dumps({"last_tweetid": high, "review_tweetid": review}))
    conn.commit()
    metrics["time_limit_reached"] = time.monotonic() >= deadline
    return metrics


RECOVERABLE_ARTICLE = """(a.url LIKE %s OR a.url LIKE %s)
    AND a.scraping_status IN ('PENDING', 'FAILED', 'SKIPPED')"""


def _recovery_candidates(cursor, limit):
    cursor.execute(
        f"""
        SELECT DISTINCT ta.tweetid, s.v
        FROM tweet_articles ta JOIN web_articles a ON a.article_id = ta.article_id
        LEFT JOIN ingest_state s ON s.k = CONCAT(%s, ta.tweetid)
        WHERE {RECOVERABLE_ARTICLE} AND ta.tweetid > 0
          AND (s.k IS NULL OR (s.v LIKE 'retry:%'
               AND CAST(SUBSTRING_INDEX(s.v, ':', -1) AS UNSIGNED) <= UNIX_TIMESTAMP()))
        ORDER BY ta.tweetid LIMIT %s
        """,
        (f"{RECOVERY_STATE_KEY}:", *TCO_URL_PREFIXES, limit),
    )
    return {str(tweetid): state for tweetid, state in cursor.fetchall()}


def _recovery_sources(cursor, post_ids):
    if not post_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(post_ids))
    cursor.execute(
        f"""
        SELECT ta.tweetid, ta.source_url, a.url
        FROM tweet_articles ta JOIN web_articles a ON a.article_id = ta.article_id
        WHERE ta.tweetid IN ({placeholders}) AND {RECOVERABLE_ARTICLE}
        """,
        (*post_ids, *TCO_URL_PREFIXES),
    )
    sources = {}
    for tweetid, source, provisional in cursor.fetchall():
        normalized_source = normalize_url(source)
        short_key = _tco_key(normalized_source)
        normalized_provisional = normalize_url(provisional)
        if not short_key:
            source = provisional
            normalized_source = normalized_provisional
            short_key = _tco_key(normalized_source)
        if short_key:
            alias = normalized_provisional if normalized_provisional != normalized_source else None
            sources.setdefault(str(tweetid), {})[source] = (short_key, alias)
    return sources


def _matching_expansions(post, sources):
    expanded = {}
    for link in extract_external_urls(post, include_text=False):
        short_key = _tco_key(normalize_url(link.source_url))
        if short_key and not _tco_key(normalize_url(link.target_url)):
            expanded[short_key] = link.target_url
    return [TweetLink(source, expanded[key], alias) for source, (key, alias) in sources.items() if key in expanded]


def _problem_type(problem):
    kind = problem.get("type") if isinstance(problem, dict) else None
    return kind.rsplit("/", 1)[-1] if isinstance(kind, str) else None


def _lookup_reset_at(headers):
    now = time.time()
    reset = headers.get("x-rate-limit-reset") or ""
    reset = int(reset) if len(reset) <= 12 and reset.isascii() and reset.isdecimal() else 0
    wait = max(reset - now, _retry_after(headers.get("Retry-After")) or 0)
    return math.ceil(now + (wait if wait > 0 else 60))


def recover_historical_urls(conn, cursor, lookup, get_state, set_state):
    limit = _setting("RECOVER_LIMIT", 10, 1, 100)
    metrics = {
        "batch_limit": limit, "posts_requested": 0, "posts_returned": 0, "posts_recovered": 0,
        "posts_unavailable": 0, "posts_without_expansion": 0, "posts_retryable": 0,
        "posts_retry_exhausted": 0, "expanded_urls_recovered": 0,
    }
    wait = math.ceil(int(get_state(cursor, RECOVERY_STATE_KEY, "0") or 0) - time.time())
    if wait > 0:
        return {**metrics, "status_code": 429, "error": "x_rate_limited", "retry_after_seconds": wait}

    candidates = _recovery_candidates(cursor, limit)
    conn.commit()
    if not candidates:
        return metrics

    metrics["posts_requested"] = len(candidates)
    try:
        response = lookup(list(candidates))
    except requests.RequestException:
        return {**metrics, "status_code": 502, "error": "x_network_error"}

    try:
        payload = response.json()
    except ValueError:
        payload = None

    status = response.status_code
    metrics["x_status"] = status
    if status == 402 or _problem_type(payload) == "usage-capped":
        return {**metrics, "status_code": 402, "error": "x_usage_limit"}
    if status == 429 or response.headers.get("x-rate-limit-remaining") == "0":
        reset_at = _lookup_reset_at(response.headers)
        set_state(cursor, RECOVERY_STATE_KEY, str(reset_at))
        conn.commit()
        metrics["retry_after_seconds"] = max(0, math.ceil(reset_at - time.time()))
        if status == 429:
            return {**metrics, "status_code": 429, "error": "x_rate_limited"}
    if status != 200:
        return {**metrics, "status_code": 502, "error": "x_lookup_failed"}
    if (not isinstance(payload, dict) or not {"data", "errors"}.intersection(payload)
            or not isinstance(payload.get("data", []), list) or not isinstance(payload.get("errors", []), list)):
        return {**metrics, "status_code": 502, "error": "x_invalid_response"}

    posts = {str(post.get("id")): post for post in payload.get("data", [])
             if isinstance(post, dict) and str(post.get("id")) in candidates}
    unavailable = {str(problem.get("resource_id") or problem.get("value")) for problem in payload.get("errors", [])
                   if _problem_type(problem) in {"resource-not-found", "not-authorized-for-resource"}}
    metrics["posts_returned"] = len(posts)
    sources = _recovery_sources(cursor, list(posts))
    for tweetid, previous in candidates.items():
        state = None
        if tweetid in posts:
            links = _matching_expansions(posts[tweetid], sources.get(tweetid, {}))
            if links:
                registered = register_tweet_urls(cursor, tweetid, links)
                metrics["expanded_urls_recovered"] += registered
                if registered == len(links):
                    state = "recovered"
            else:
                state = "without_expansion"
        elif tweetid in unavailable:
            state = "unavailable"

        if state is None:
            attempt = (int(previous.split(":")[1]) if previous else 0) + 1
            if attempt >= RECOVERY_MAX_ATTEMPTS:
                state = "retry_exhausted"
            else:
                retry_at = math.ceil(time.time() + 60 * 2 ** (attempt - 1))
                set_state(cursor, f"{RECOVERY_STATE_KEY}:{tweetid}", f"retry:{attempt}:{retry_at}")
                metrics["posts_retryable"] += 1
                continue
        set_state(cursor, f"{RECOVERY_STATE_KEY}:{tweetid}", state)
        metrics[f"posts_{state}"] += 1

    conn.commit()
    return metrics


class ScrapeError(Exception):
    def __init__(self, code, *, status=None, retryable=False, retry_after=None, skipped=False, domain=None):
        super().__init__(code)
        self.code = code
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after
        self.skipped = skipped
        self.domain = domain


def _remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise ScrapeError("runtime_limit", retryable=True)
    return seconds


def _resolve_public_ip(host, timeout):
    literal = _ip_address(host)
    if literal is not None:
        if not _public_ip(literal):
            raise ScrapeError("unsafe_address", skipped=True)
        return str(literal)

    future = DNS_WORKERS.submit(socket.getaddrinfo, host + ".", None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    try:
        records = future.result(timeout=timeout)
    except (OSError, FutureTimeout):
        future.cancel()
        raise ScrapeError("dns_error", retryable=True) from None

    addresses = [_ip_address(record[4][0]) for record in records]
    if not addresses:
        raise ScrapeError("dns_error", retryable=True)
    if any(address is None or not _public_ip(address) for address in addresses):
        raise ScrapeError("unsafe_address", skipped=True)
    # Prefer IPv4 when both families are offered. The connection uses this exact IP.
    return str(min(addresses, key=lambda address: address.version))


class _PinnedHTTPSAdapter(HTTPAdapter):
    def __init__(self, hostname):
        self.hostname = hostname
        super().__init__(pool_connections=1, pool_maxsize=1)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        super().init_poolmanager(
            connections, maxsize, block,
            server_hostname=self.hostname, assert_hostname=self.hostname, **pool_kwargs,
        )


def _retry_after(value):
    if not value:
        return None
    value = value.strip()
    if re.fullmatch(r"[0-9]+", value):
        # Avoid integer overflow and keep the timestamp within MySQL's DATETIME range.
        return min(int(value.lstrip("0")[:12] or "0"), 100_000_000_000)
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return min(100_000_000_000, max(0, (date - datetime.now(timezone.utc)).total_seconds()))
    except (ValueError, TypeError, OverflowError):
        return None


def _retryable_http(status):
    return status in {408, 425, 429} or (500 <= status < 600 and status not in {501, 505})


def _http_error(status, headers):
    return ScrapeError(
        f"http_{status}", status=status, retryable=_retryable_http(status),
        retry_after=_retry_after(headers.get("Retry-After")),
        skipped=status in {401, 403, 404, 405, 407, 410, 451},
    )


def _read_body(response, max_bytes, deadline):
    length = response.headers.get("Content-Length", "")
    if length.isascii() and length.isdecimal() and (len(length) > 12 or int(length) > max_bytes):
        raise ScrapeError("response_too_large", status=response.status_code, skipped=True)
    if response.headers.get("Content-Encoding", "identity").strip().lower() not in {"", "identity"}:
        raise ScrapeError("unsupported_encoding", status=response.status_code, skipped=True)

    body = bytearray()
    while True:
        _remaining(deadline)
        chunk = response.raw.read1(min(65536, max_bytes + 1 - len(body)), decode_content=False)
        if not chunk:
            return bytes(body)
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ScrapeError("response_too_large", status=response.status_code, skipped=True)


class ArticleScraper:
    def __init__(self, settings, deadline):
        self.settings = settings
        self.deadline = deadline
        self.robots = {}
        self.last_request = {}
        self.domain_delays = {}
        self.cooldowns = {}

    def _wait_for_domain(self, domain):
        now = time.monotonic()
        cooldown = self.cooldowns.get(domain, 0) - now
        delay = self.domain_delays.get(domain, self.settings.domain_delay)
        wait = max(cooldown, self.last_request.get(domain, -delay) + delay - now)
        if wait >= _remaining(self.deadline):
            raise ScrapeError("domain_delay", retryable=True, retry_after=max(0, wait))
        if wait > 0:
            time.sleep(wait)

    def _download(self, url, *, robots=False):
        parts = urlsplit(url)
        host = parts.hostname
        domain = host.removeprefix("www.")
        self._wait_for_domain(domain)
        address = _resolve_public_ip(host, min(self.settings.connect_timeout, _remaining(self.deadline)))
        ip_authority = f"[{address}]" if ":" in address else address
        pinned_url = urlunsplit((parts.scheme, ip_authority, parts.path, parts.query, ""))
        prepared = requests.Request("GET", pinned_url, headers={
            "Host": parts.netloc,
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "identity",
            "Accept": "text/plain, */*;q=0.1" if robots else "text/html, application/xhtml+xml",
        }).prepare()
        adapter = _PinnedHTTPSAdapter(host) if parts.scheme == "https" else HTTPAdapter()
        remaining = _remaining(self.deadline)
        timeout = Timeout(total=remaining, connect=self.settings.connect_timeout, read=self.settings.read_timeout)
        self.last_request[domain] = time.monotonic()
        try:
            # Send directly so Requests cannot use proxies, netrc, cookies or read redirect bodies.
            with closing(adapter), adapter.send(prepared, stream=True, verify=True, proxies={}, timeout=timeout) as response:
                status = response.status_code
                headers = response.headers
                if status != 200:
                    retry_after = _retry_after(headers.get("Retry-After"))
                    if retry_after and _retryable_http(status):
                        self.cooldowns[domain] = time.monotonic() + retry_after
                    return status, headers, b""

                content_type = headers.get("Content-Type", "").partition(";")[0].strip().lower()
                if robots:
                    if content_type not in {"", "text/plain"}:
                        raise ScrapeError("invalid_robots", status=status, retryable=True)
                    max_bytes = ROBOTS_MAX_BYTES
                else:
                    if content_type not in ALLOWED_HTML:
                        raise ScrapeError("non_html", status=status, skipped=True)
                    if RESTRICTED_ROBOTS.search(headers.get("X-Robots-Tag", "")):
                        raise ScrapeError("restricted_content", status=status, skipped=True)
                    max_bytes = self.settings.max_bytes
                return status, headers, _read_body(response, max_bytes, self.deadline)
        except requests.exceptions.SSLError:
            raise ScrapeError("tls_error") from None
        except (requests.RequestException, TransportError, OSError):
            raise ScrapeError("network_error", retryable=True) from None

    def _fetch(self, url, *, robots=False):
        visited = set()
        for hop in range(self.settings.max_redirects + 1):
            host = urlsplit(url).hostname
            allowed = _public_target(host) and not _social_host(host) if robots else _article_target(url)
            if not allowed:
                raise ScrapeError("unsafe_or_unsupported_url", skipped=True)
            if url in visited:
                raise ScrapeError("redirect_cycle", skipped=True)
            visited.add(url)
            if not robots:
                self._check_robots(url)
            status, headers, body = self._download(url, robots=robots)
            if status not in REDIRECT_STATUSES:
                return url, status, headers, body
            if hop == self.settings.max_redirects:
                raise ScrapeError("redirect_limit", status=status, skipped=True)
            location = headers.get("Location", "")
            try:
                url = normalize_url(urljoin(url, location), strip_tracking=False) if location else None
            except ValueError:
                url = None
            if not url:
                raise ScrapeError("invalid_redirect", status=status, skipped=True)
            retry_after = _retry_after(headers.get("Retry-After"))
            if retry_after:
                if retry_after >= _remaining(self.deadline):
                    raise ScrapeError("redirect_delay", status=status, retryable=True, retry_after=retry_after)
                time.sleep(retry_after)

    def _check_robots(self, url):
        from protego import Protego

        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self.robots:
            _, status, headers, body = self._fetch(origin + "/robots.txt", robots=True)
            if status not in {200, 204, 404, 410}:
                raise _http_error(status, headers)
            self.robots[origin] = Protego.parse(body.decode("utf-8-sig", errors="replace"))
        rules = self.robots[origin]
        if not rules.can_fetch(url, USER_AGENT):
            raise ScrapeError("robots_disallowed", skipped=True, domain=parts.hostname)
        delay = rules.crawl_delay(USER_AGENT) or 0
        rate = rules.request_rate(USER_AGENT)
        rate_delay = rate.seconds / rate.requests if rate and rate.requests else 0
        domain = parts.hostname.removeprefix("www.")
        self.domain_delays[domain] = max(
            self.domain_delays.get(domain, self.settings.domain_delay), delay, rate_delay,
        )

    def scrape_article(self, url):
        normalized = normalize_url(url, strip_tracking=False)
        if not normalized:
            raise ScrapeError("invalid_url", skipped=True)
        final_url, status, headers, body = self._fetch(normalized)
        if status != 200:
            raise _http_error(status, headers)
        if urlsplit(final_url).hostname == "t.co":
            raise ScrapeError("unresolved_short_url", status=200, skipped=True)
        _remaining(self.deadline)
        return extract_article(body, final_url, self.settings.min_words)


def _meta_values(tree, *keys):
    return [
        meta.get("content") for meta in tree.iter("meta")
        if (meta.get("property") or meta.get("name") or meta.get("itemprop") or "").lower() in keys
        and meta.get("content")
    ]


def _structured_article(tree):
    article = {}
    restricted = False
    for script in tree.xpath('//script[@type="application/ld+json"]'):
        try:
            values = [json.loads(script.text or "")]
        except (ValueError, RecursionError):
            continue
        while values:
            value = values.pop()
            if isinstance(value, list):
                values.extend(value)
            elif isinstance(value, dict):
                types = value.get("@type", [])
                if isinstance(types, str):
                    types = [types]
                if isinstance(types, list) and any(
                    isinstance(kind, str) and kind.rsplit("/", 1)[-1].lower().endswith("article")
                    for kind in types
                ):
                    article = article or value
                free = str(value.get("isAccessibleForFree", "")).strip().lower()
                if free in {"false", "0", "https://schema.org/false", "http://schema.org/false"}:
                    restricted = True
                values.extend(item for item in value.values() if isinstance(item, (dict, list)))
    return article, restricted


def _metadata_text(value, max_length=16000):
    if not isinstance(value, str):
        return None
    return " ".join(value.split())[:max_length] or None


def _published_at(values):
    for value in values:
        if not isinstance(value, str):
            continue
        try:
            date = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if date.tzinfo:
                date = date.astimezone(timezone.utc).replace(tzinfo=None)
            if date.year >= 1000:
                return date
        except (ValueError, OverflowError):
            continue
    return None


def _metadata_url(url, base):
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        normalized = normalize_url(urljoin(base, url.strip()))
    except ValueError:
        return None
    return normalized if normalized and _public_target(urlsplit(normalized).hostname) else None


def extract_article(body, final_url, min_words):
    from trafilatura import bare_extraction
    from trafilatura.utils import load_html

    if body.startswith((b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"PK\x03\x04")):
        raise ScrapeError("invalid_html", status=200, skipped=True)
    tree = load_html(body, max_size=len(body))
    if tree is None:
        raise ScrapeError("invalid_html", status=200, skipped=True)
    structured, restricted = _structured_article(tree)
    paywall = tree.xpath('//*[@id="paywall" or @data-paywall="true" or @id="challenge-form" '
                         'or contains(concat(" ", normalize-space(@class), " "), " paywall ")]')
    if (restricted or paywall
            or any(value.strip().lower() in {"false", "0"} for value in _meta_values(tree, "isaccessibleforfree"))
            or any(RESTRICTED_ROBOTS.search(value) for value in _meta_values(tree, "robots", "netvorabot"))
            or "premium" in _meta_values(tree, "article:content_tier")):
        raise ScrapeError("restricted_content", status=200, skipped=True)
    headings = tree.xpath("//head/title/text() | //h1//text()")
    if any(BLOCKED_TITLES.search(heading.strip()) for heading in headings):
        raise ScrapeError("error_or_consent_page", status=200, skipped=True)

    canonical_tags = tree.xpath('//head/link[contains(concat(" ", normalize-space(@rel), " "), " canonical ")]/@href')
    dates = _meta_values(tree, "article:published_time", "datepublished", "pubdate")
    language = tree.get("lang") or next(iter(_meta_values(tree, "og:locale")), None)
    article_markup = structured or tree.xpath('//article | //*[@itemprop="articleBody"]') or any(
        value.strip().lower() == "article" for value in _meta_values(tree, "og:type")
    )
    document = bare_extraction(
        tree, with_metadata=True, include_comments=False, include_tables=False, favor_precision=True,
        date_extraction_params={"original_date": True, "extensive_search": False},
    )
    title = _metadata_text(document.title) if document else None
    content = document.text.strip() if document and document.text else ""
    words = re.findall(r"\b[^\W_]+\b", content)
    if not title or len(words) < min_words or not (article_markup or dates):
        raise ScrapeError("insufficient_article_content", status=200, skipped=True)
    if len({word.lower() for word in words}) < min(20, min_words // 3):
        raise ScrapeError("insufficient_article_content", status=200, skipped=True)

    host = urlsplit(final_url).hostname
    canonical = _metadata_url(next(iter(canonical_tags), None) or document.url, final_url)
    if (not canonical or not _article_target(canonical)
            or urlsplit(canonical).hostname.removeprefix("www.") != host.removeprefix("www.")):
        canonical = normalize_url(final_url)
    language = _metadata_text(language or document.language, 32)
    if language:
        language = language.replace("_", "-")
        if len(language) > 10:
            language = language.split("-", 1)[0]
        if not re.fullmatch(r"[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]+)*", language):
            language = None
    return {
        "canonical_url": canonical,
        "canonical_hash": _sha256(canonical),
        "content_hash": _sha256(" ".join(content.split())),
        "domain": host,
        "title": title,
        "description": _metadata_text(document.description),
        "content": content,
        "author": _metadata_text(document.author or next(iter(_meta_values(tree, "author")), None), 255),
        "image_url": _metadata_url(document.image, final_url),
        "published_at": _published_at([*dates, structured.get("datePublished"), document.date]),
        "language": language,
        "word_count": len(words),
    }


def _recover_stale_articles(cursor, settings):
    cursor.execute(
        """
        SELECT article_id FROM web_articles
        WHERE scraping_status = 'PROCESSING'
          AND (scraping_started_at IS NULL OR scraping_started_at <= TIMESTAMPADD(SECOND, %s, UTC_TIMESTAMP()))
        ORDER BY article_id LIMIT %s
        """,
        (-settings.processing_ttl_seconds, settings.limit),
    )
    ids = [row[0] for row in cursor.fetchall()]
    if not ids:
        return 0
    placeholders = ", ".join(["%s"] * len(ids))
    cursor.execute(
        f"""
        UPDATE web_articles
        SET scraping_status = 'FAILED', scraping_error = 'processing_expired',
            scraping_retry_at = IF(COALESCE(scraping_attempts, 0) < %s, UTC_TIMESTAMP(), NULL),
            updated_at = UTC_TIMESTAMP()
        WHERE article_id IN ({placeholders}) AND scraping_status = 'PROCESSING'
        """,
        (settings.max_attempts, *ids),
    )
    return cursor.rowcount


# Previous versions could mistake redirects that changed only the query for a loop.
ELIGIBLE_FOR_SCRAPING = """(
    scraping_status = 'PENDING'
    OR (scraping_status = 'FAILED' AND scraping_retry_at <= UTC_TIMESTAMP())
    OR (scraping_status IN ('FAILED', 'SKIPPED') AND scraping_error = 'redirect_loop' AND scraping_retry_at IS NULL)
)"""


def fetch_pending_articles(cursor, settings):
    cursor.execute(
        f"""
        SELECT article_id, url, COALESCE(scraping_attempts, 0) FROM web_articles
        WHERE COALESCE(scraping_attempts, 0) < %s AND {ELIGIBLE_FOR_SCRAPING}
        ORDER BY (url LIKE %s OR url LIKE %s), article_id LIMIT %s
        """,
        (settings.max_attempts, *TCO_URL_PREFIXES, settings.limit),
    )
    return cursor.fetchall()


def _claim_article(cursor, article_id, previous_attempts):
    cursor.execute(
        f"""
        UPDATE web_articles
        SET scraping_status = 'PROCESSING', scraping_attempts = COALESCE(scraping_attempts, 0) + 1,
            scraping_started_at = UTC_TIMESTAMP(), scraping_retry_at = NULL, scraping_error = NULL,
            http_status = NULL, updated_at = UTC_TIMESTAMP()
        WHERE article_id = %s AND COALESCE(scraping_attempts, 0) = %s AND {ELIGIBLE_FOR_SCRAPING}
        """,
        (article_id, previous_attempts),
    )
    return cursor.rowcount == 1


def _save_failure(cursor, article_id, attempt, error, settings):
    status = "SKIPPED" if error.skipped else "FAILED"
    retry_at = None
    if error.retryable and attempt < settings.max_attempts:
        delay = max(min(86400, settings.retry_seconds * 2 ** (attempt - 1)), error.retry_after or 0)
        now = datetime.now(timezone.utc).timestamp()
        retry_at = datetime.fromtimestamp(math.ceil(now + delay), timezone.utc).replace(tzinfo=None)
    cursor.execute(
        """
        UPDATE web_articles
        SET scraping_status = %s, scraping_error = %s, http_status = %s, scraping_retry_at = %s,
            domain = COALESCE(%s, domain), updated_at = UTC_TIMESTAMP()
        WHERE article_id = %s AND scraping_status = 'PROCESSING' AND scraping_attempts = %s
        """,
        (status, error.code, error.status, retry_at, error.domain, article_id, attempt),
    )
    if not cursor.rowcount:
        return "superseded"
    log = LOGGER.info if error.skipped else LOGGER.warning
    log("Artículo %s, intento %s, %s (%s)", article_id, attempt, status, error.code)
    return status.lower()


def _save_content(cursor, article_id, result, canonical_hash):
    cursor.execute(
        """
        UPDATE web_articles
        SET canonical_url = %(canonical_url)s, canonical_hash = %(canonical_hash)s, content_hash = %(content_hash)s,
            domain = %(domain)s, title = %(title)s, description = %(description)s, content = %(content)s,
            author = %(author)s, image_url = %(image_url)s, published_at = %(published_at)s,
            language = %(language)s, word_count = %(word_count)s, scraping_status = 'COMPLETED',
            scraping_method = 'trafilatura', scraping_error = NULL, scraping_retry_at = NULL,
            http_status = 200, scraped_at = UTC_TIMESTAMP(), updated_at = UTC_TIMESTAMP(),
            summary_status = IF(summary IS NULL AND key_points IS NULL AND summarized_at IS NULL, 'PENDING', summary_status)
        WHERE article_id = %(article_id)s
        """,
        {**result, "article_id": article_id, "canonical_hash": canonical_hash},
    )


def _canonical_owner(cursor, article_id, canonical_hash):
    fields = "article_id, content_hash, COALESCE(content, '') <> ''"
    cursor.execute(
        f"SELECT {fields} FROM web_articles WHERE canonical_hash = %s AND article_id <> %s FOR UPDATE",
        (canonical_hash, article_id),
    )
    row = cursor.fetchone()
    if row:
        return row
    cursor.execute(
        f"""
        SELECT {fields} FROM web_articles
        WHERE url_hash = %s AND article_id <> %s AND canonical_hash IS NULL AND canonical_url IS NULL
        FOR UPDATE
        """,
        (canonical_hash, article_id),
    )
    return cursor.fetchone()


def save_article_result(cursor, article_id, attempt, result):
    cursor.execute(
        """
        SELECT canonical_hash, COALESCE(content, '') <> '',
               summary IS NOT NULL OR key_points IS NOT NULL OR summarized_at IS NOT NULL
        FROM web_articles
        WHERE article_id = %s AND scraping_status = 'PROCESSING' AND scraping_attempts = %s
        FOR UPDATE
        """,
        (article_id, attempt),
    )
    source = cursor.fetchone()
    if not source:
        return "superseded"

    canonical_hash = result["canonical_hash"]
    source_canonical_hash, stored_content, stored_summary = source
    owner = None if source_canonical_hash == canonical_hash else _canonical_owner(cursor, article_id, canonical_hash)
    if not owner:
        _save_content(cursor, article_id, result, canonical_hash)
        return "completed"

    owner_id, owner_content_hash, owner_has_content = owner
    if not owner_has_content:
        _save_content(cursor, owner_id, result, canonical_hash)
    else:
        cursor.execute(
            """
            UPDATE web_articles
            SET canonical_url = %s, canonical_hash = %s, domain = %s, scraping_status = 'COMPLETED', scraping_error = NULL,
                scraping_retry_at = NULL, http_status = 200, updated_at = UTC_TIMESTAMP()
            WHERE article_id = %s
            """,
            (result["canonical_url"], canonical_hash, result["domain"], owner_id),
        )

    cursor.execute("UPDATE tweet_articles SET article_id = %s WHERE article_id = %s", (owner_id, article_id))
    if stored_content:
        cursor.execute(
            """
            UPDATE web_articles
            SET canonical_url = %s, canonical_hash = NULL, domain = %s, scraping_status = 'COMPLETED', scraping_error = NULL,
                scraping_retry_at = NULL, http_status = 200, updated_at = UTC_TIMESTAMP()
            WHERE article_id = %s
            """,
            (result["canonical_url"], result["domain"], article_id),
        )
    elif stored_summary or (owner_has_content and owner_content_hash != result["content_hash"]):
        _save_content(cursor, article_id, result, None)
    else:
        _delete_empty_article(cursor, article_id)
    return "duplicates"


def process_pending_articles(conn, cursor, started):
    settings = ScrapeSettings.from_env()
    deadline = started + settings.max_seconds
    metrics = {
        "articles_processed": 0, "completed": 0, "failed": 0, "skipped": 0,
        "duplicates": 0, "superseded": 0, "recovered": 0, "scraping_errors": {},
    }
    scraper = ArticleScraper(settings, deadline)
    metrics["recovered"] = _recover_stale_articles(cursor, settings)
    pending = fetch_pending_articles(cursor, settings)
    conn.commit()

    for article_id, url, previous_attempts in pending:
        if time.monotonic() >= deadline:
            break

        claimed = _claim_article(cursor, article_id, previous_attempts)
        conn.commit()
        if not claimed:
            continue

        attempt = previous_attempts + 1
        try:
            result = scraper.scrape_article(url)
        except ScrapeError as error:
            outcome = _save_failure(cursor, article_id, attempt, error, settings)
            if outcome in {"failed", "skipped"}:
                errors = metrics["scraping_errors"]
                errors[error.code] = errors.get(error.code, 0) + 1
        else:
            outcome = save_article_result(cursor, article_id, attempt, result)

        conn.commit()
        metrics[outcome] += 1
        metrics["articles_processed"] += 1
    metrics["time_limit_reached"] = time.monotonic() >= deadline
    return metrics
