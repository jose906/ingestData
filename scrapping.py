import argparse
import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import closing, contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import unquote, unquote_plus, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from dateutil.parser import isoparse
from mysql.connector import DataError, IntegrityError, errorcode
from protego import Protego
from requests.adapters import HTTPAdapter
from requests.utils import requote_uri
from urllib3 import PoolManager
from urllib3.exceptions import HTTPError as TransportError
from urllib3.exceptions import ReadTimeoutError
from urllib3.util import Timeout

from database import get_db_connection, get_state, set_state


LOGGER = logging.getLogger(__name__)
BOT = "NetVoraNewsBot"
USER_AGENT = BOT + "/1.0 (+https://netvora.org/)"
ARTICLE_LOCK = "netvora_articles_scrape"
STATE_KEY = "web_articles_expanded_cursor"
REVIEW_INTERVAL = 24 * 60 * 60
MAX_ATTEMPTS = 3
MIN_WORDS = 40
MAX_HTML_BYTES = 2 * 1024 * 1024
REDIRECTS = {301, 302, 303, 307, 308}
BLOCKED_HTTP = {401, 403, 407, 451}
TRACKING_KEYS = {
    "fbclid", "gclid", "dclid", "msclkid", "igshid", "mc_cid", "mc_eid", "twclid",
}
IGNORED_DOMAINS = {
    "t.co", "x.com", "twitter.com", "twimg.com", "facebook.com", "fb.com", "fb.me", "fb.watch",
    "fbcdn.net", "instagram.com", "threads.net", "threads.com", "tiktok.com", "youtube.com",
    "youtu.be", "linkedin.com", "reddit.com", "twitch.tv", "pinterest.com", "bsky.app",
    "whatsapp.com", "t.me",
}
FILE_EXTENSIONS = set(
    "7z avi csv doc docx exe gif gz jpeg jpg json m4a mov mp3 mp4 pdf png ppt pptx rar rss "
    "svg tar tif tiff txt wav webm webp xls xlsx xml zip".split()
)
DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
NUMERIC_HOST = re.compile(r"(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*")
NON_PUBLIC_NETWORKS = tuple(map(ipaddress.ip_network, (
    "192.0.0.0/24", "192.88.99.0/24", "168.63.129.16/32", "2001::/23", "2002::/16",
)))
PUBLIC_IPV6 = ipaddress.ip_network("2000::/3")
DNS_WORKERS = ThreadPoolExecutor(max_workers=2, thread_name_prefix="article-dns")
RESTRICTED = re.compile(r"(?:^|[\s,:])(?:none|noarchive|nosnippet)(?=$|[\s,])", re.I)
BLOCKED_TITLE = re.compile(
    r"^(?:40[134]\b|access denied\b|acceso denegado\b|forbidden\b|just a moment\b|"
    r"attention required\b|verify (?:that )?you are human\b|consent required\b|"
    r"checking your browser\b|before you continue\b|antes de continuar\b|"
    r"page not found\b|página no encontrada\b)",
    re.I,
)
CLUTTER = (
    "script, style, nav, footer, header, aside, iframe, form, noscript, button, "
    "[hidden], [aria-hidden=true], [role=navigation], [role=complementary], "
    ".ad, .ads, .ad-container, .advertisement, .publicidad, .social-share, .share-buttons, "
    ".related-news, .related-posts, .related-articles, .related-content, "
    ".comments, .noticias-relacionadas, .newsletter"
)
ARTICLE_BODY = (
    '[itemprop="articleBody"], .article-body, .article-content, .article-text, '
    ".article__body, .article__content, .news-content, .news-body, .entry-content, "
    ".post-content, .post-body, .story-body, .story-content, .nota-contenido, "
    ".contenido-noticia, .field-name-body, .field--name-body"
)
CLUTTER_TEXT = re.compile(
    r"^(?:publicidad|advertisement|s[ií]guenos|suscr[ií]bete|noticias relacionadas|"
    r"te puede interesar|lee tambi[eé]n|lea tambi[eé]n|compartir en)\b",
    re.I,
)


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_url(url, *, strip_tracking=True):
    if not isinstance(url, str):
        return None
    url = url.strip()
    if not url or len(url) > 4096 or any(
        char.isspace() or not char.isprintable() or char == "\\" for char in url
    ):
        return None
    try:
        parts = urlsplit(url)
        default_port = {"http": 80, "https": 443}.get(parts.scheme)
        if not default_port or "@" in parts.netloc or parts.netloc.endswith(":"):
            return None
        if parts.port not in (None, default_port):
            return None

        host = (parts.hostname or "").rstrip(".").encode("idna").decode("ascii").lower()
        if (
            len(host) > 253 or "." not in host or "[" in parts.netloc
            or NUMERIC_HOST.fullmatch(host)
            or not all(DNS_LABEL.fullmatch(label) for label in host.split("."))
            or host == "metadata.goog"
            or host.endswith((".localhost", ".local", ".localdomain", ".internal", ".home.arpa"))
        ):
            return None

        query = parts.query
        if strip_tracking:
            kept = []
            for parameter in query.split("&"):
                key = unquote_plus(parameter.partition("=")[0]).lower()
                if not key.startswith("utm_") and key not in TRACKING_KEYS:
                    kept.append(parameter)
            query = "&".join(kept)
        normalized = requote_uri(urlunsplit((parts.scheme, host, parts.path or "/", query, "")))
        return normalized if len(normalized) <= 4096 else None
    except (ValueError, UnicodeError):
        return None


def _ignored_host(host):
    return any(host == domain or host.endswith("." + domain) for domain in IGNORED_DOMAINS)


def is_article_url(url):
    if not url:
        return False
    parts = urlsplit(url)
    if _ignored_host(parts.hostname):
        return False
    path = unquote(parts.path).lower().rstrip("/")
    extension = path.rsplit("/", 1)[-1].partition(";")[0].rpartition(".")[2]
    return bool(path or parts.query) and extension not in FILE_EXTENSIONS


def extract_expanded_urls(expanded_urls):
    """Return links keyed by their normalized source URL, with the expanded URL as target."""
    if isinstance(expanded_urls, (str, bytes)):
        try:
            expanded_urls = json.loads(expanded_urls)
        except (ValueError, RecursionError):
            return {}
    if not isinstance(expanded_urls, list):
        return {}

    links = {}
    for item in expanded_urls:
        if not isinstance(item, dict):
            continue
        target = normalize_url(item.get("expanded_url"))
        if not is_article_url(target):
            continue
        original = item.get("url")
        source = normalize_url(original)
        if source:
            links[source] = (original.strip(), target)
        else:
            links[target] = (target, target)
    return links


def _register_links(cursor, tweetid, expanded_urls, stats):
    links = extract_expanded_urls(expanded_urls)
    stats["urls_found"] += len(links)
    for source, (original, target) in links.items():
        url_hash = sha256(target)
        cursor.execute(
            """
            SELECT article_id FROM web_articles
            WHERE url_hash = %s OR canonical_hash = %s
            ORDER BY (url_hash = %s) DESC LIMIT 1
            """,
            (url_hash, url_hash, url_hash),
        )
        row = cursor.fetchone()
        if row:
            article_id = row[0]
            stats["articles_already_registered"] += 1
        else:
            cursor.execute(
                """
                INSERT INTO web_articles (url, url_hash, domain, scraping_status, created_at, updated_at)
                VALUES (%s, %s, %s, 'PENDING', UTC_TIMESTAMP(), UTC_TIMESTAMP())
                ON DUPLICATE KEY UPDATE article_id = LAST_INSERT_ID(article_id)
                """,
                (target, url_hash, urlsplit(target).hostname),
            )
            article_id = cursor.lastrowid
            stats["articles_registered"] += int(cursor.rowcount == 1)
            stats["articles_already_registered"] += int(cursor.rowcount == 0)

        cursor.execute(
            """
            INSERT INTO tweet_articles (tweetid, article_id, source_url, source_url_hash, created_at)
            VALUES (%s, %s, %s, %s, UTC_TIMESTAMP())
            ON DUPLICATE KEY UPDATE article_id = %s
            """,
            (tweetid, article_id, original, sha256(source), article_id),
        )
        stats["relations_created"] += int(cursor.rowcount == 1)


def _register_articles(conn, cursor, limit, deadline):
    stats = dict.fromkeys((
        "tweets_reviewed", "urls_found", "articles_registered",
        "relations_created", "articles_already_registered",
    ), 0)
    saved = json.loads(get_state(cursor, STATE_KEY, "{}"))
    state = {
        "last": 0, "review": 0, "review_until": 0,
        "review_at": int(time.time()) + REVIEW_INTERVAL, "review_turn": True,
        **saved,
    }
    cursor.execute("SELECT COALESCE(MAX(tweetid), 0) FROM Tweets")
    newest = max(state["last"], int(cursor.fetchone()[0]))
    conn.commit()

    scans = []
    if state["last"] and time.time() >= state["review_at"]:
        # X IDs follow publication time. An older ID can be inserted or receive links later.
        if not state["review_until"]:
            state["review_until"] = state["last"]
        if limit > 1 or state["review_turn"]:
            quota = limit if newest == state["last"] else max(1, limit // 4)
            scans.append(("review", state["review_until"], quota))
        if limit == 1:
            state["review_turn"] = not state["review_turn"]
    scans.append(("last", newest, limit))

    for checkpoint, upper, quota in scans:
        size = min(quota, limit - stats["tweets_reviewed"])
        if not size or time.monotonic() >= deadline:
            break
        if checkpoint == "last" and state["last"] >= upper:
            continue
        cursor.execute(
            """
            SELECT tweetid, expanded_urls FROM Tweets
            WHERE tweetid > %s AND tweetid <= %s
              AND JSON_TYPE(expanded_urls) = 'ARRAY' AND JSON_LENGTH(expanded_urls) > 0
            ORDER BY tweetid LIMIT %s
            """,
            (state[checkpoint], upper, size),
        )
        rows = cursor.fetchall()
        conn.commit()
        processed = 0
        for tweetid, expanded_urls in rows:
            if time.monotonic() >= deadline:
                break
            _register_links(cursor, tweetid, expanded_urls, stats)
            state[checkpoint] = int(tweetid)
            set_state(cursor, STATE_KEY, json.dumps(state))
            conn.commit()
            saved = state.copy()
            stats["tweets_reviewed"] += 1
            processed += 1

        if processed == len(rows) and (len(rows) < size or state[checkpoint] == upper):
            if checkpoint == "review":
                state.update(review=0, review_until=0, review_at=int(time.time()) + REVIEW_INTERVAL)
            else:
                # The remaining IDs in this range had no expanded links.
                state["last"] = upper

    if state != saved:
        set_state(cursor, STATE_KEY, json.dumps(state))
        conn.commit()
    return stats


def _remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("EXECUTION_LIMIT")
    return seconds


def _resolve_public_ip(host, timeout):
    future = DNS_WORKERS.submit(socket.getaddrinfo, host + ".", None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    try:
        addresses = [ipaddress.ip_address(row[4][0]) for row in future.result(timeout=timeout)]
    except FutureTimeout:
        future.cancel()
        raise requests.ConnectTimeout from None
    except OSError:
        raise requests.ConnectionError from None
    if not addresses:
        raise requests.ConnectionError
    for address in addresses:
        if (
            not address.is_global or address.is_multicast or address.is_reserved
            or (address.version == 6 and address not in PUBLIC_IPV6)
            or any(address in network for network in NON_PUBLIC_NETWORKS)
        ):
            raise ValueError("INVALID_URL")
    return str(min(addresses, key=lambda address: address.version))


def _open_url(session, url, timeout, deadline):
    parts = urlsplit(url)
    address = _resolve_public_ip(parts.hostname, min(timeout, _remaining(deadline)))
    authority = f"[{address}]" if ":" in address else address
    pinned_url = urlunsplit((parts.scheme, authority, parts.path, parts.query, ""))

    if parts.scheme == "https":
        origin = f"https://{parts.hostname}/"
        if origin not in session.adapters:
            adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1)
            adapter.poolmanager = PoolManager(
                num_pools=1, maxsize=1, server_hostname=parts.hostname, assert_hostname=parts.hostname,
            )
            session.mount(origin, adapter)

    prepared = session.prepare_request(requests.Request("GET", pinned_url, headers={"Host": parts.netloc}))
    # Send through the session's adapter so redirects are checked before their bodies are read.
    return session.get_adapter(url).send(
        prepared, stream=True, verify=True, proxies={},
        timeout=Timeout(total=min(timeout, _remaining(deadline)), connect=timeout, read=timeout),
    )


def _read_body(response, max_bytes, deadline):
    header = response.headers.get("Content-Length", "").strip()
    length = None
    if header.isdecimal():
        if len(header) > 10 or int(header) > max_bytes:
            raise ValueError("RESPONSE_TOO_LARGE", response.status_code)
        length = int(header)
    if response.headers.get("Content-Encoding", "identity").lower().strip() not in ("", "identity"):
        raise ValueError("UNSUPPORTED_ENCODING", response.status_code)

    body = bytearray()
    if hasattr(response.raw, "read1"):
        chunks = iter(lambda: response.raw.read1(16384, decode_content=False), b"")
    else:
        chunks = response.iter_content(chunk_size=16384)
    for chunk in chunks:
        _remaining(deadline)
        body.extend(chunk)
        if len(body) > max_bytes:
            raise ValueError("RESPONSE_TOO_LARGE", response.status_code)
    _remaining(deadline)
    if length is not None and len(body) != length:
        raise requests.ConnectionError
    return bytes(body)


@contextmanager
def _download_session(deadline, timeout):
    with requests.Session() as session:
        session.trust_env = False
        session.headers.update({
            "User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain;q=0.5",
            "Accept-Language": "es-BO,es;q=0.9", "Accept-Encoding": "identity",
        })
        robots_cache = {}
        last_request = {}
        cooldowns = {}
        domain_delays = {}
        run_seconds = deadline - time.monotonic()

        def download(url, *, robots_file=False):
            for _ in range(6):
                url = normalize_url(url, strip_tracking=False)
                if not url or (not robots_file and not is_article_url(url)):
                    raise ValueError("INVALID_URL")
                parts = urlsplit(url)
                if _ignored_host(parts.hostname):
                    raise ValueError("INVALID_URL")
                domain = parts.hostname[4:] if parts.hostname.startswith("www.") else parts.hostname
                if not robots_file:
                    origin = f"{parts.scheme}://{parts.netloc}"
                    if origin not in robots_cache:
                        try:
                            _, rules = download(origin + "/robots.txt", robots_file=True)
                        except requests.HTTPError as error:
                            if error.response.status_code in BLOCKED_HTTP:
                                raise PermissionError("ROBOTS_DISALLOWED", error.response.status_code) from None
                            raise
                        robots_cache[origin] = Protego.parse(rules.decode("utf-8-sig", errors="replace"))
                    rules = robots_cache[origin]
                    if not rules.can_fetch(url, BOT):
                        raise PermissionError("ROBOTS_DISALLOWED")
                    rate = rules.request_rate(BOT)
                    rate_delay = rate.seconds / rate.requests if rate and rate.requests else 0
                    domain_delays[domain] = max(
                        domain_delays.get(domain, 1), rules.crawl_delay(BOT) or 0, rate_delay,
                    )

                now = time.monotonic()
                delay = domain_delays.get(domain, 1)
                wait = max(0, last_request.get(domain, now - delay) + delay - now)
                if cooldowns.get(domain, 0) > now:
                    raise TimeoutError("DOMAIN_COOLDOWN")
                if delay > 1 and delay >= run_seconds:
                    raise PermissionError("ROBOTS_DELAY_EXCEEDS_BUDGET")
                if wait >= _remaining(deadline):
                    raise TimeoutError("EXECUTION_LIMIT")
                if wait:
                    time.sleep(wait)

                try:
                    response = _open_url(session, url, timeout, deadline)
                finally:
                    last_request[domain] = time.monotonic()
                with response:
                    status = response.status_code
                    if status in REDIRECTS:
                        location = response.headers.get("Location")
                        if not location:
                            raise ValueError("INVALID_REDIRECT")
                        url = urljoin(url, location)
                        continue
                    if robots_file and status in (204, 404, 410):
                        return url, b""
                    if status != 200:
                        if status in (429, 503):
                            cooldowns[domain] = time.monotonic() + max(1, _retry_after(response.headers.get("Retry-After")))
                        raise requests.HTTPError(response=response)

                    content_type = response.headers.get("Content-Type", "").partition(";")[0].strip().lower()
                    if robots_file:
                        body = _read_body(response, 512_000, deadline)
                        if content_type in ("text/html", "application/xhtml+xml") or body.lstrip().startswith(b"<"):
                            raise PermissionError("ROBOTS_UNAVAILABLE", status)
                        return url, body
                    if content_type and content_type not in ("text/html", "application/xhtml+xml"):
                        raise ValueError("NOT_HTML", status)
                    if (
                        RESTRICTED.search(response.headers.get("X-Robots-Tag", ""))
                        or response.headers.get("CF-Mitigated", "").lower() == "challenge"
                    ):
                        raise PermissionError("PAGE_BLOCKED", status)
                    return url, _read_body(response, MAX_HTML_BYTES, deadline)
            raise ValueError("TOO_MANY_REDIRECTS")

        yield download


def obtener_meta(soup, *names):
    for name in names:
        for attribute in ("property", "name", "itemprop"):
            tag = soup.find("meta", attrs={attribute: name})
            value = tag.get("content", "").strip() if tag else ""
            if value:
                return value
    return None


def _paragraphs(text):
    paragraphs = []
    seen = set()
    for line in text.splitlines():
        line = " ".join(line.split())
        key = line.casefold()
        if line and key not in seen and not CLUTTER_TEXT.match(line):
            paragraphs.append(line)
            seen.add(key)
    return "\n\n".join(paragraphs)


def _remove_clutter(soup):
    for element in soup.select(CLUTTER):
        element.decompose()
    for element in soup.find_all("br"):
        element.replace_with("\u2028")


def _has_article_body(content):
    return len(content.split()) >= MIN_WORDS and any(
        len(paragraph.split()) >= 15 for paragraph in content.split("\n\n")
    )


def extraer_contenido(soup):
    _remove_clutter(soup)
    for selector in (ARTICLE_BODY, "article", 'main, [role="main"]', "body"):
        best_in_group = ""
        containers = soup.select(selector)
        if selector == "article" and len(containers) > 1:
            containers = [container for container in containers if container.find("h1")]
        for container in containers:
            if len(container.find_all("article", limit=2)) > 1:
                continue
            blocks = []
            for paragraph in container.find_all("p"):
                text = paragraph.get_text().replace("\n", " ").replace("\r", " ")
                linked_words = sum(len(link.get_text().split()) for link in paragraph.find_all("a"))
                if linked_words * 2 > len(text.split()):
                    continue
                blocks.append(text)
            content = _paragraphs("\n".join(blocks))
            if _has_article_body(content) and len(content.split()) > len(best_in_group.split()):
                best_in_group = content

        if best_in_group:
            return best_in_group, selector
    return "", None


def _structured_article(soup, title):
    articles = []
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            nodes = [json.loads(script.string or "")]
        except (ValueError, RecursionError):
            continue
        while nodes:
            node = nodes.pop()
            if isinstance(node, list):
                nodes.extend(reversed(node))
            elif isinstance(node, dict):
                kinds = node.get("@type", [])
                kinds = kinds if isinstance(kinds, list) else [kinds]
                if any(isinstance(kind, str) and (
                    kind.lower().endswith("newsarticle") or re.split(r"[/#]", kind)[-1].lower() == "article"
                ) for kind in kinds):
                    articles.append(node)
                nodes.extend(node[key] for key in ("@graph", "mainEntity") if key in node)
    return max(
        articles, default={},
        key=lambda article: (bool(title and article.get("headline") == title), bool(article.get("articleBody"))),
    )


def _structured_text(value, *keys):
    if isinstance(value, list):
        return ", ".join(filter(None, (_structured_text(item, *keys) for item in value))) or None
    if isinstance(value, dict):
        for key in keys:
            if isinstance(value.get(key), str) and value[key].strip():
                return _structured_text(value[key])
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8").strip() or None
    return None


def _date(*values):
    for value in values:
        if not isinstance(value, str):
            continue
        try:
            date = isoparse(value.strip())
            if date.tzinfo is not None:
                date = date.astimezone(timezone.utc).replace(tzinfo=None)
            if date.year >= 1000:
                return date
        except (ValueError, OverflowError):
            continue
    return None


def _url_from_metadata(base, value, *, same_site=False):
    if not value:
        return None
    try:
        url = normalize_url(urljoin(base, value))
    except ValueError:
        return None
    if url and same_site:
        hosts = [urlsplit(address).hostname for address in (base, url)]
        hosts = [host[4:] if host.startswith("www.") else host for host in hosts]
        if hosts[0] != hosts[1]:
            return None
    return url


def _extract_article(html, url):
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.select_one("article h1, main h1") or soup.find("h1")
    heading = title_tag.get_text(" ", strip=True) if title_tag else None
    page_title = soup.title.get_text(" ", strip=True) if soup.title else None
    og_title = obtener_meta(soup, "og:title")
    article = _structured_article(soup, og_title or heading)
    access = [article.get("isAccessibleForFree")]
    parts = article.get("hasPart", [])
    access.extend(part.get("isAccessibleForFree") for part in (parts if isinstance(parts, list) else [parts])
                  if isinstance(part, dict))
    paywall = soup.select_one('#paywall:not([hidden]), .paywall:not([hidden]), [data-paywall="true"]')
    if (
        any(str(value).lower() == "false" for value in access)
        or (obtener_meta(soup, "article:content_tier") or "").lower() in ("locked", "metered")
        or RESTRICTED.search(obtener_meta(soup, "robots", BOT, BOT.lower()) or "")
        or any(BLOCKED_TITLE.match(value) for value in (heading, page_title, og_title) if value)
        or soup.select_one("#challenge-form, #cf-challenge-running, form[action*='captcha']")
        or (paywall and paywall.get_text(strip=True))
    ):
        raise PermissionError("PAGE_BLOCKED", 200)

    title = (
        _structured_text(article.get("headline")) or og_title or obtener_meta(soup, "headline", "title")
        or heading or page_title
    )
    description = _structured_text(article.get("description")) or obtener_meta(soup, "og:description", "description")
    author_tag = soup.select_one('[rel~="author"], [itemprop="author"] [itemprop="name"]')
    author = (
        _structured_text(article.get("author"), "name")
        or obtener_meta(soup, "article:author", "author")
        or (author_tag.get_text(" ", strip=True) if author_tag else None)
    )
    published_time = (
        soup.select_one('time[itemprop="datePublished"][datetime]')
        or soup.select_one('article time[datetime]:not([itemprop="dateModified"]), main time[datetime]:not([itemprop="dateModified"])')
    )
    updated_time = soup.select_one('time[itemprop="dateModified"][datetime]')
    published = _date(
        article.get("datePublished"),
        obtener_meta(soup, "article:published_time", "datePublished", "parsely-pub-date", "pubdate", "date"),
        published_time.get("datetime") if published_time else None,
    )
    updated = _date(
        article.get("dateModified"), obtener_meta(soup, "article:modified_time", "dateModified", "og:updated_time"),
        updated_time.get("datetime") if updated_time else None,
    )
    image = article.get("image")
    if isinstance(image, list):
        image = image[0] if image else None
    image = _structured_text(image, "url", "contentUrl", "@id")
    image_url = _url_from_metadata(url, image) or _url_from_metadata(url, obtener_meta(soup, "og:image", "image"))
    category = _structured_text(article.get("articleSection")) or obtener_meta(soup, "article:section")
    language = _structured_text(article.get("inLanguage")) or (soup.html.get("lang") if soup.html else None)

    canonical_tag = soup.select_one('link[rel~="canonical"][href]')
    canonical_url = None
    for candidate in (
        canonical_tag.get("href") if canonical_tag else None,
        _structured_text(article.get("mainEntityOfPage"), "@id", "url"),
        _structured_text(article.get("url")), obtener_meta(soup, "og:url"),
    ):
        canonical_url = _url_from_metadata(url, candidate, same_site=True)
        if canonical_url:
            break
    canonical_url = canonical_url or normalize_url(url)

    body = _structured_text(article.get("articleBody")) or ""
    if re.search(r"</?(?:p|div|br|em|strong|section|blockquote)\b", body, re.I):
        fragment = BeautifulSoup(body, "html.parser")
        _remove_clutter(fragment)
        for block in fragment.find_all(("p", "div", "section", "blockquote", "li", "h2", "h3")):
            block.append("\u2028")
        body = fragment.get_text().replace("\n", " ").replace("\r", " ")
    body = _paragraphs(body)
    content, selector = extraer_contenido(soup)
    if selector == "body" and not (heading or article or obtener_meta(soup, "og:type") == "article"):
        content = ""
    if (
        _has_article_body(body) and not body.endswith(("...", "…"))
        and " ".join(body.split()) != " ".join((description or "").split())
        and len(body.split()) >= 0.9 * len(content.split())
    ):
        content, selector = body, "json-ld"
    words = len(content.split())
    if not title or words < MIN_WORDS or " ".join(content.split()) == " ".join((description or "").split()):
        raise ValueError("INSUFFICIENT_CONTENT", 200)

    return {
        "canonical_url": canonical_url,
        "canonical_hash": sha256(canonical_url),
        "title": title[:8000],
        "description": description[:8000] if description else None,
        "content": content,
        "author": author[:255] if author else None,
        "published_at": published,
        "updated_at": updated,
        "image_url": image_url,
        "category": category,
        "language": language if language and len(language) <= 10 else None,
        "word_count": words,
        "content_hash": sha256(" ".join(content.split())),
        "selector_used": selector,
    }


def _retry_after(value):
    value = (value or "").strip()
    if not value:
        return 0
    if value.isascii() and value.isdecimal() and len(value) <= 9:
        return int(value)
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0, date.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError, OSError):
        return 0


def scrape_article(url, *, timeout=10, download=None):
    result = {
        "ok": False, "original_url": url, "final_url": None, "canonical_url": None,
        "canonical_hash": None, "domain": None, "title": None, "description": None,
        "content": None, "author": None, "published_at": None, "updated_at": None,
        "image_url": None, "category": None, "language": None, "word_count": 0,
        "content_hash": None, "selector_used": None, "scraping_status": "FAILED",
        "scraping_method": "beautifulsoup", "http_status": None, "error": None,
        "retryable": False, "retry_after": 0,
    }
    try:
        with (nullcontext(download) if download else _download_session(time.monotonic() + 60, timeout)) as get:
            final_url, html = get(url)
        result.update(final_url=final_url, domain=urlsplit(final_url).hostname, http_status=200)
        result.update(_extract_article(html, final_url))
        result.update(ok=True, scraping_status="COMPLETED")
    except requests.HTTPError as error:
        status = error.response.status_code
        temporary = status in (408, 425, 429) or (500 <= status < 600 and status not in (501, 505))
        result.update(
            http_status=status, error="PAGE_BLOCKED" if status in BLOCKED_HTTP else f"HTTP_{status}",
            retryable=temporary, retry_after=_retry_after(error.response.headers.get("Retry-After")),
            scraping_status="SKIPPED" if status in BLOCKED_HTTP else "FAILED",
        )
    except (requests.Timeout, ReadTimeoutError):
        result.update(error="TIMEOUT", retryable=True)
    except requests.exceptions.SSLError:
        result["error"] = "TLS_ERROR"
    except PermissionError as error:
        if error.args and isinstance(error.args[0], str):
            result.update(error=error.args[0], scraping_status="SKIPPED")
            if len(error.args) > 1:
                result["http_status"] = error.args[1]
        else:
            result.update(error="CONNECTION_ERROR", retryable=True)
    except TimeoutError as error:
        result["error"] = str(error)
    except (requests.RequestException, TransportError, OSError) as error:
        if error.args and isinstance(error.args[0], ReadTimeoutError):
            result.update(error="TIMEOUT", retryable=True)
        else:
            result.update(error="CONNECTION_ERROR", retryable=True)
    except ValueError as error:
        known_errors = {
            "INVALID_URL", "INVALID_REDIRECT", "TOO_MANY_REDIRECTS", "RESPONSE_TOO_LARGE",
            "UNSUPPORTED_ENCODING", "NOT_HTML", "INSUFFICIENT_CONTENT",
        }
        code = error.args[0] if error.args and error.args[0] in known_errors else "EXTRACTION_ERROR"
        result.update(error=code, scraping_status="FAILED" if code in {
            "INSUFFICIENT_CONTENT", "EXTRACTION_ERROR",
        } else "SKIPPED")
        if len(error.args) > 1 and code in known_errors:
            result["http_status"] = error.args[1]
    except Exception:
        LOGGER.exception("No se pudo extraer el artículo")
        result["error"] = "EXTRACTION_ERROR"

    result["scraped_at"] = datetime.now(timezone.utc).isoformat()
    return result


PENDING_SQL = """
    scraping_status = 'PENDING'
    OR (scraping_status = 'FAILED' AND scraping_retry_at <= UTC_TIMESTAMP())
    OR (scraping_status = 'PROCESSING'
        AND (scraping_started_at IS NULL OR scraping_started_at <= UTC_TIMESTAMP() - INTERVAL 15 MINUTE))
"""
SAVE_RESULT = """
    UPDATE web_articles
    SET scraping_attempts = scraping_attempts + 1, scraping_status = %(scraping_status)s,
        scraping_error = %(error)s, scraping_retry_at = %(retry_at)s, http_status = %(http_status)s,
        scraping_method = %(scraping_method)s, scraped_at = UTC_TIMESTAMP(), updated_at = UTC_TIMESTAMP()
"""
SAVE_CONTENT = """
    , canonical_url = %(canonical_url)s, canonical_hash = %(canonical_hash)s, domain = %(domain)s,
      title = %(title)s, description = %(description)s, content = %(content)s, author = %(author)s,
      published_at = %(published_at)s, image_url = %(image_url)s, language = %(language)s,
      word_count = %(word_count)s, content_hash = %(content_hash)s
"""
SAVE_WHERE = """
    WHERE article_id = %(article_id)s AND scraping_status = %(previous_status)s
      AND scraping_attempts = %(attempts)s
"""


def _save_result(cursor, article_id, previous_status, attempts, result):
    retry_at = None
    if result["retryable"] and attempts + 1 < MAX_ATTEMPTS:
        delay = max(60 * 2 ** attempts, result["retry_after"])
        retry_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=delay)
    params = {
        **result, "article_id": article_id, "previous_status": previous_status,
        "attempts": attempts, "retry_at": retry_at,
    }
    sql = SAVE_RESULT + (SAVE_CONTENT if result["ok"] else "") + SAVE_WHERE
    try:
        cursor.execute(sql, params)
    except IntegrityError as error:
        if not result["ok"] or error.errno != errorcode.ER_DUP_ENTRY:
            raise
        # Another source already owns this canonical hash. Keep both source rows and their links.
        cursor.execute(sql, {**params, "canonical_hash": None})
    return cursor.rowcount == 1


def _scrape_pending_articles(conn, cursor, limit, timeout, deadline):
    stats = {"articles_processed": 0, "articles_scraped": 0, "articles_failed": 0, "scraping_errors": {}}
    if time.monotonic() >= deadline:
        return stats
    cursor.execute(
        f"""
        SELECT article_id, url, scraping_status, scraping_attempts FROM web_articles
        WHERE scraping_attempts < %s AND ({PENDING_SQL})
        ORDER BY article_id LIMIT %s
        """,
        (MAX_ATTEMPTS, limit),
    )
    rows = cursor.fetchall()
    conn.commit()
    if not rows or time.monotonic() >= deadline:
        return stats

    with _download_session(deadline, timeout) as download:
        for article_id, url, previous_status, attempts in rows:
            if time.monotonic() >= deadline:
                break
            result = scrape_article(url, download=download)
            if result["error"] == "EXECUTION_LIMIT":
                stats["time_limit_reached"] = True
                break
            if result["error"] == "DOMAIN_COOLDOWN":
                continue
            try:
                saved = _save_result(cursor, article_id, previous_status, attempts, result)
            except DataError:
                conn.rollback()
                result.update(ok=False, scraping_status="FAILED", error="STORAGE_ERROR", retryable=False)
                saved = _save_result(cursor, article_id, previous_status, attempts, result)
            conn.commit()
            if saved:
                stats["articles_processed"] += 1
                if result["ok"]:
                    stats["articles_scraped"] += 1
                else:
                    stats["articles_failed"] += 1
                    errors = stats["scraping_errors"]
                    errors[result["error"]] = errors.get(result["error"], 0) + 1
    return stats


def _setting(name, value, default, maximum):
    env_name = "WEB_ARTICLES_" + name
    value = os.getenv(env_name, str(default)) if value is None else value
    try:
        value = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"{env_name} debe ser un entero") from None
    if not 1 <= value <= maximum:
        raise ValueError(f"{env_name} debe estar entre 1 y {maximum}")
    return value


def _run_job(job, max_seconds):
    started = time.monotonic()
    deadline = started + max_seconds
    with closing(get_db_connection()) as conn, closing(conn.cursor()) as cursor:
        cursor.execute("SELECT GET_LOCK(%s, 0)", (ARTICLE_LOCK,))
        acquired, = cursor.fetchone()
        if acquired is None:
            raise RuntimeError("No se pudo obtener el lock de artículos")
        # Closing the connection releases the MySQL lock and rolls back any uncommitted writes.
        stats = job(conn, cursor, deadline) if acquired else {"busy": True}
    finished = time.monotonic()
    return {
        "ok": True, **stats, "elapsed_seconds": round(finished - started, 2),
        "time_limit_reached": stats.get("time_limit_reached", False) or finished >= deadline,
    }


def register_articles(limit=None, *, max_seconds=None):
    limit = _setting("DISCOVER_LIMIT", limit, 200, 1000)
    seconds = _setting("DISCOVER_SECONDS", max_seconds, 20, 240)
    return _run_job(lambda conn, cursor, deadline: _register_articles(conn, cursor, limit, deadline), seconds)


def scrape_pending_articles(limit=None, *, max_seconds=None, timeout=None):
    limit = _setting("SCRAPE_LIMIT", limit, 20, 100)
    seconds = _setting("SCRAPE_SECONDS", max_seconds, 60, 240)
    timeout = _setting("TIMEOUT", timeout, 10, 60)
    return _run_job(
        lambda conn, cursor, deadline: _scrape_pending_articles(conn, cursor, limit, timeout, deadline),
        seconds,
    )


def run_scraping(tweet_limit=None, article_limit=None, *, max_seconds=None, timeout=None):
    tweet_limit = _setting("DISCOVER_LIMIT", tweet_limit, 200, 1000)
    article_limit = _setting("SCRAPE_LIMIT", article_limit, 20, 100)
    seconds = _setting("RUN_SECONDS", max_seconds, 60, 240)
    registration_seconds = _setting("DISCOVER_SECONDS", None, 20, 240)
    timeout = _setting("TIMEOUT", timeout, 10, 60)

    def run(conn, cursor, deadline):
        register_until = min(deadline, time.monotonic() + min(registration_seconds, seconds / 2))
        stats = _register_articles(conn, cursor, tweet_limit, register_until)
        stats.update(_scrape_pending_articles(conn, cursor, article_limit, timeout, deadline))
        return stats

    return _run_job(run, seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Registro y scraping de noticias de NetVora")
    parser.add_argument("job", choices=("register", "scrape", "run"), nargs="?", default="run")
    args = parser.parse_args()
    jobs = {"register": register_articles, "scrape": scrape_pending_articles, "run": run_scraping}
    print(json.dumps(jobs[args.job](), ensure_ascii=False))
