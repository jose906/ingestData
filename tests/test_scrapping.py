import json
import os
import socket
import time
import unittest
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import requests
from mysql.connector import IntegrityError

import scrapping


URL = "https://medio.example/noticia?id=7"
OTHER_URL = "https://otro.example/reportaje"
P1 = (
    "La alcaldía anunció que el nuevo sistema de distribución comenzará el lunes en los "
    "barrios del norte. Los vecinos podrán consultar el cronograma actualizado en el portal municipal."
)
P2 = (
    "Los técnicos explicaron que la primera fase incluye la instalación de cuatro tanques y "
    "la revisión de las tuberías. El servicio operará durante toda la temporada de lluvias."
)
BODY = P1 + "\n\n" + P2
HTML = (
    f"<html><head><title>Un nuevo sistema de distribución</title></head><body>"
    f"<article><h1>Un nuevo sistema de distribución</h1><p>{P1}</p><p>{P2}</p>"
    f"<p>{P2}</p><aside><p>No forma parte del artículo.</p></aside></article></body></html>"
).encode()


def expanded(short, url):
    return {"url": "https://t.co/" + short, "expanded_url": url}


def http(status, body=b"", **headers):
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = status
    response.headers = requests.structures.CaseInsensitiveDict(headers)
    response.raw = object()
    response.iter_content.return_value = iter([body] if body else [])
    return response


class ExtractionTest(unittest.TestCase):
    def test_json_ld_and_metadata(self):
        structured = {
            "@graph": [{
                "@type": ["NewsArticle"], "headline": "Nota del municipio",
                "articleBody": f"<div>{P1}</div><div>{P2}</div>",
                "author": [{"name": "Ana Rojas"}, {"name": "Luis Ponce"}],
                "datePublished": "2026-11-26T08:00:00-04:00",
                "dateModified": "2026-11-26T09:00:00-04:00",
                "articleSection": "Local", "image": {"url": "/foto.jpg"},
            }]
        }
        metadata = (
            '<link rel="canonical" href="/noticia?id=7&utm_campaign=ejemplo">'
            '<meta property="og:description" content="Cronograma municipal">'
            f'<script type="application/ld+json">{json.dumps(structured)}</script>'
        )
        html = HTML.replace(b"</head>", metadata.encode() + b"</head>")
        result = scrapping.scrape_article(URL, download=lambda url: (url, html))

        self.assertTrue(result["ok"])
        self.assertEqual(result["selector_used"], "json-ld")
        self.assertEqual(result["scraping_status"], "COMPLETED")
        self.assertEqual(result["canonical_url"], URL)
        self.assertEqual(result["author"], "Ana Rojas, Luis Ponce")
        self.assertEqual(result["published_at"], datetime(2026, 11, 26, 12))
        self.assertEqual(result["updated_at"], datetime(2026, 11, 26, 13))
        self.assertEqual(result["image_url"], "https://medio.example/foto.jpg")
        self.assertEqual(result["category"], "Local")
        self.assertEqual(result["description"], "Cronograma municipal")
        self.assertEqual(result["content"], BODY)
        self.assertEqual(result["word_count"], len(BODY.split()))
        self.assertEqual(result["content_hash"], scrapping.sha256(" ".join(BODY.split())))

    def test_excerpt_falls_back_to_clean_html(self):
        structured = {"@type": "Article", "headline": "La noticia", "articleBody": P1}
        metadata = f'<script type="application/ld+json">{json.dumps(structured)}</script>'
        html = HTML.replace(b"</head>", metadata.encode() + b"</head>")
        result = scrapping.scrape_article(URL, download=lambda url: (url, html))
        self.assertEqual(result["selector_used"], "article")
        self.assertEqual(result["content"], BODY)

    def test_blocking_page_and_navigation_are_not_articles(self):
        pages = (
            (b"<html><title>Just a moment</title><p>Verifica que eres humano</p></html>", "PAGE_BLOCKED"),
            (b"<html><title>Inicio</title><nav>" + HTML + b"</nav></html>", "INSUFFICIENT_CONTENT"),
            (HTML.replace(b"</head>", b'<meta name="robots" content="nosnippet"></head>'), "PAGE_BLOCKED"),
        )
        for html, expected in pages:
            with self.subTest(expected=expected):
                result = scrapping.scrape_article(URL, download=lambda url: (url, html))
                self.assertFalse(result["ok"])
                self.assertEqual(result["error"], expected)

    def test_robots_and_redirects(self):
        redirect = http(302, Location="/noticia-nueva?id=7")
        with patch.object(scrapping, "_open_url", side_effect=[http(404), redirect, http(200, HTML)]) as send:
            with patch.object(scrapping.time, "sleep"):
                result = scrapping.scrape_article(URL)
        self.assertTrue(result["ok"])
        self.assertEqual(result["final_url"], "https://medio.example/noticia-nueva?id=7")
        self.assertEqual([call.args[1] for call in send.call_args_list], [
            "https://medio.example/robots.txt", URL, result["final_url"],
        ])
        redirect.iter_content.assert_not_called()

        private_redirect = http(302, Location="http://metadata.google.internal/")
        with patch.object(scrapping, "_open_url", side_effect=[http(404), private_redirect]) as send:
            with patch.object(scrapping.time, "sleep"):
                result = scrapping.scrape_article(URL)
        self.assertEqual(result["error"], "INVALID_URL")
        self.assertEqual(send.call_count, 2)

        robots = http(200, b"User-agent: *\nDisallow: /noticia*\n")
        with patch.object(scrapping, "_open_url", return_value=robots) as send:
            result = scrapping.scrape_article(URL)
        self.assertEqual(result["error"], "ROBOTS_DISALLOWED")
        send.assert_called_once()

    def test_private_dns_and_pinned_tls(self):
        private = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.4", 0))]
        with patch.object(scrapping.socket, "getaddrinfo", return_value=private):
            with patch.object(scrapping.HTTPAdapter, "send") as send:
                self.assertEqual(scrapping.scrape_article(URL)["error"], "INVALID_URL")
                send.assert_not_called()

        with requests.Session() as session:
            with patch.object(scrapping, "_resolve_public_ip", return_value="8.8.8.8"):
                with patch.object(scrapping.HTTPAdapter, "send", autospec=True) as send:
                    scrapping._open_url(session, URL, 5, time.monotonic() + 5)
        adapter, prepared = send.call_args.args
        self.assertEqual(prepared.url, "https://8.8.8.8/noticia?id=7")
        self.assertEqual(prepared.headers["Host"], "medio.example")
        self.assertEqual(adapter.poolmanager.connection_pool_kw["server_hostname"], "medio.example")
        self.assertEqual(adapter.poolmanager.connection_pool_kw["assert_hostname"], "medio.example")
        self.assertTrue(send.call_args.kwargs["verify"])

    def test_network_errors_and_size_limit(self):
        for failure, code in (
            (requests.Timeout(), "TIMEOUT"), (requests.ConnectionError(), "CONNECTION_ERROR"),
        ):
            with self.subTest(code=code), patch.object(scrapping, "_open_url", side_effect=failure):
                result = scrapping.scrape_article(URL)
                self.assertEqual(result["error"], code)
                self.assertTrue(result["retryable"])

        cases = (
            (http(404), "HTTP_404", False),
            (http(429, **{"Retry-After": "120"}), "HTTP_429", True),
            (http(403), "PAGE_BLOCKED", False),
            (http(200, **{"Content-Length": str(scrapping.MAX_HTML_BYTES + 1)}), "RESPONSE_TOO_LARGE", False),
        )
        for response, code, retry in cases:
            with self.subTest(code=code), patch.object(scrapping, "_open_url", side_effect=[http(404), response]):
                with patch.object(scrapping.time, "sleep"):
                    result = scrapping.scrape_article(URL)
                self.assertEqual(result["error"], code)
                self.assertEqual(result["retryable"], retry)

    def test_download_does_not_start_when_pause_exceeds_budget(self):
        with patch.object(scrapping, "_open_url", return_value=http(404)) as send:
            with scrapping._download_session(time.monotonic() + 0.8, 10) as download:
                result = scrapping.scrape_article(URL, download=download)
        self.assertEqual(result["error"], "EXECUTION_LIMIT")
        send.assert_called_once()


class ArticleStore:
    """Simulate the MySQL transaction boundaries and unique keys used by these jobs."""

    def __init__(self, tweets):
        self.tweets = tweets
        self.data = ({}, {}, {})
        self.fail_tweet = None
        self.locked = False
        self.closed = False
        self.in_transaction = False

    def connect(self):
        self.state, self.articles, self.links = deepcopy(self.data)
        self.closed = False
        return self

    def cursor(self):
        return self

    def execute(self, sql, params=()):
        query = " ".join(sql.split())
        self.rows = []
        self.rowcount = 0
        if query.startswith("SELECT GET_LOCK"):
            self.rows = [(0 if self.locked else 1,)]
            return
        self.in_transaction = True
        if query.startswith("SELECT v FROM ingest_state"):
            if params[0] in self.state:
                self.rows = [(self.state[params[0]],)]
        elif query.startswith("INSERT INTO ingest_state"):
            key, value = params
            self.state[key] = value
        elif query.startswith("SELECT COALESCE(MAX(tweetid), 0) FROM Tweets"):
            self.rows = [(max(self.tweets, default=0),)]
        elif query.startswith("SELECT tweetid, expanded_urls"):
            start, end, limit = params
            self.rows = [(tid, json.dumps(urls)) for tid, urls in sorted(self.tweets.items())
                         if start < tid <= end and isinstance(urls, list) and urls][:limit]
        elif query.startswith("SELECT article_id FROM web_articles"):
            article = next(
                (item for item in self.articles.values()
                 if params[0] in (item["url_hash"], item.get("canonical_hash"))),
                None,
            )
            if article:
                self.rows = [(article["article_id"],)]
        elif query.startswith("INSERT INTO web_articles"):
            url, url_hash, domain = params
            self.lastrowid = len(self.articles) + 1
            self.articles[self.lastrowid] = dict(
                article_id=self.lastrowid, url=url, url_hash=url_hash, domain=domain,
                scraping_status="PENDING", attempts=0, summary_status="PENDING",
            )
            self.rowcount = 1
        elif query.startswith("INSERT INTO tweet_articles"):
            tweetid, article_id, original, source_hash, _ = params
            if tweetid == self.fail_tweet:
                raise IntegrityError("Simulated MySQL write failure")
            key = (tweetid, source_hash)
            value = (article_id, original)
            previous = self.links.get(key)
            self.rowcount = 1 if previous is None else 2 * (previous != value)
            self.links[key] = value
        elif query.startswith("SELECT article_id, url, scraping_status"):
            maximum, limit = params
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            self.rows = [
                (article_id, item["url"], item["scraping_status"], item["attempts"])
                for article_id, item in self.articles.items()
                if item["attempts"] < maximum and (
                    item["scraping_status"] == "PENDING" or (
                        item["scraping_status"] == "FAILED" and item.get("retry_at") and item["retry_at"] <= now
                    )
                )
            ][:limit]
        elif query.startswith("UPDATE web_articles"):
            item = self.articles[params["article_id"]]
            if "canonical_hash =" in query:
                if params["canonical_hash"] and any(
                    article_id != item["article_id"] and other.get("canonical_hash") == params["canonical_hash"]
                    for article_id, other in self.articles.items()
                ):
                    raise IntegrityError("Duplicate canonical hash", errno=1062)
                self.assert_no_summary_in_query(sql)
                item.update(canonical_hash=params["canonical_hash"], content=params["content"])
            item.update(
                scraping_status=params["scraping_status"], error=params["error"],
                attempts=item["attempts"] + 1, retry_at=params["retry_at"],
            )
            self.rowcount = 1
        else:
            raise AssertionError(query)

    @staticmethod
    def assert_no_summary_in_query(sql):
        if "summary" in sql:
            raise AssertionError(sql)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows

    def commit(self):
        self.data = deepcopy((self.state, self.articles, self.links))
        self.in_transaction = False

    def close(self):
        self.closed = True


class ArticleJobsTest(unittest.TestCase):
    def setUp(self):
        self.store = ArticleStore({
            1: [], 5: None,
            10: [expanded("one", URL + "&utm_medium=x#body"), expanded("two", OTHER_URL),
                 expanded("x", "https://x.com/user/status/123")],
            20: [expanded("three", URL + "&fbclid=abcd")],
            30: [expanded("four", "https://third.example/articulo")],
        })
        connection = patch.object(scrapping, "get_db_connection", self.store.connect)
        self.addCleanup(connection.stop)
        connection.start()

    def test_register_and_resume_with_shared_articles(self):
        first = scrapping.register_articles(limit=2)
        self.assertEqual((first["tweets_reviewed"], first["articles_registered"], first["relations_created"]), (2, 2, 3))
        self.assertEqual(first["articles_already_registered"], 1)
        state, articles, links = self.store.data
        self.assertEqual(json.loads(state[scrapping.STATE_KEY])["last"], 20)
        self.assertEqual(links[10, scrapping.sha256("https://t.co/one")][0],
                         links[20, scrapping.sha256("https://t.co/three")][0])
        self.assertEqual(articles[1]["url"], URL)
        self.assertEqual(scrapping.register_articles(limit=2)["articles_registered"], 1)
        self.assertEqual(scrapping.register_articles(limit=2)["relations_created"], 0)

    def test_failed_transaction_and_late_tweet(self):
        self.store.fail_tweet = 20
        with self.assertRaises(IntegrityError):
            scrapping.register_articles()
        state, articles, links = self.store.data
        self.assertTrue(self.store.closed)
        self.assertEqual(json.loads(state[scrapping.STATE_KEY])["last"], 10)
        self.assertEqual(len(links), 2)

        self.store.fail_tweet = None
        scrapping.register_articles()
        self.store.tweets[7] = [expanded("late", OTHER_URL)]
        self.store.tweets[10].append(expanded("edit", "https://edit.example/noticia"))
        review_at = json.loads(self.store.data[0][scrapping.STATE_KEY])["review_at"]
        with patch.object(scrapping.time, "time", return_value=review_at):
            result = scrapping.register_articles(limit=8)
        self.assertEqual(result["relations_created"], 2)
        self.assertEqual(len(self.store.data[2]), 6)

    def test_checkpoint_skips_empty_tail(self):
        self.store.tweets[90] = []
        first = scrapping.register_articles()
        self.assertEqual(first["tweets_reviewed"], 3)
        self.assertEqual(json.loads(self.store.data[0][scrapping.STATE_KEY])["last"], 90)
        self.assertEqual(scrapping.register_articles()["tweets_reviewed"], 0)

    def test_budget_retry_and_completed_articles(self):
        seen = []

        def download(url, **kwargs):
            self.assertFalse(self.store.in_transaction)
            seen.append(url)
            if url == URL and seen.count(URL) < 3:
                raise requests.Timeout
            return url, HTML

        with patch.object(scrapping, "_download_session", return_value=nullcontext(download)):
            stats = scrapping.run_scraping(tweet_limit=2, article_limit=2)
            self.assertEqual((stats["articles_processed"], stats["articles_scraped"], stats["articles_failed"]), (2, 1, 1))
            self.assertEqual(stats["scraping_errors"], {"TIMEOUT": 1})
            self.assertEqual(seen, [URL, OTHER_URL])
            self.assertEqual(scrapping.scrape_pending_articles()["articles_processed"], 0)
            due = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)
            self.store.data[1][1]["retry_at"] = due
            self.assertEqual(scrapping.scrape_pending_articles()["articles_failed"], 1)
            self.store.data[1][1]["retry_at"] = due
            self.assertEqual(scrapping.scrape_pending_articles()["articles_scraped"], 1)
            self.assertEqual(scrapping.scrape_pending_articles()["articles_processed"], 0)
        self.assertEqual(seen.count(OTHER_URL), 1)

        scrapping.register_articles()
        with patch.object(scrapping, "scrape_article", return_value={"error": "EXECUTION_LIMIT"}):
            self.assertTrue(scrapping.scrape_pending_articles()["time_limit_reached"])
        self.assertEqual(self.store.data[1][3]["scraping_status"], "PENDING")

    def test_http_batch_and_socket_limits(self):
        settings = {
            "WEB_ARTICLES_DISCOVER_SECONDS": "90", "WEB_ARTICLES_RUN_SECONDS": "120",
            "WEB_ARTICLES_SCRAPE_SECONDS": "3", "WEB_ARTICLES_TIMEOUT": "15",
        }
        register_mock = patch.object(scrapping, "_register_articles", return_value={})
        scrape_mock = patch.object(scrapping, "_scrape_pending_articles", return_value={})
        with patch.dict(os.environ, settings), patch.object(scrapping.time, "monotonic", return_value=100):
            with register_mock as register, scrape_mock as scrape:
                scrapping.run_scraping(tweet_limit=3, article_limit=2, via_http=True)
                self.assertEqual(register.call_args.args[2:], (3, 110))
                self.assertEqual(scrape.call_args.args[2:], (2, 5, 120))

                scrapping.run_scraping(tweet_limit=3, article_limit=2)
                self.assertEqual(register.call_args.args[2:], (3, 160))
                self.assertEqual(scrape.call_args.args[2:], (2, 15, 220))

                scrapping.register_articles(limit=3, via_http=True)
                self.assertEqual(register.call_args.args[2:], (3, 120))
                scrapping.scrape_pending_articles(limit=2, via_http=True)
                self.assertEqual(scrape.call_args.args[2:], (2, 5, 103))
                scrapping.scrape_pending_articles(limit=2, timeout=2, via_http=True)
                self.assertEqual(scrape.call_args.args[2:], (2, 2, 103))

    def test_http_batch_defers_article_if_socket_timeout_uses_remaining_budget(self):
        now = 100

        def download(url):
            nonlocal now
            if url == OTHER_URL:
                now = 120
                raise requests.Timeout
            return url, HTML

        with patch.object(scrapping.time, "monotonic", side_effect=lambda: now):
            with patch.object(scrapping, "_download_session", return_value=nullcontext(download)):
                result = scrapping.run_scraping(tweet_limit=2, via_http=True)

        self.assertTrue(result["time_limit_reached"])
        self.assertEqual((result["articles_scraped"], result["articles_failed"]), (1, 0))
        complete, pending = self.store.data[1].values()
        self.assertEqual(complete["scraping_status"], "COMPLETED")
        self.assertEqual((pending["scraping_status"], pending["attempts"]), ("PENDING", 0))

    def test_retry_limit_and_permanent_error(self):
        def download(url):
            if url == URL:
                raise requests.ConnectionError
            raise requests.HTTPError(response=http(404))

        scrapping.register_articles(limit=2)
        with patch.object(scrapping, "_download_session", return_value=nullcontext(download)):
            self.assertEqual(scrapping.scrape_pending_articles()["articles_failed"], 2)
            due = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)
            for _ in range(2):
                self.store.data[1][1]["retry_at"] = due
                self.assertEqual(scrapping.scrape_pending_articles()["articles_failed"], 1)
            self.assertEqual(scrapping.scrape_pending_articles()["articles_processed"], 0)
        temporary, permanent = self.store.data[1].values()
        self.assertEqual(temporary["attempts"], 3)
        self.assertEqual(permanent["attempts"], 1)
        self.assertIsNone(temporary["retry_at"])

    def test_canonical_collision_preserves_existing_rows(self):
        scrapping.register_articles(limit=2)
        self.store.data[1][1]["summary_status"] = "COMPLETED"
        canonical = "https://medio.example/canonica"
        with patch.object(scrapping, "_download_session", return_value=nullcontext(lambda _: (canonical, HTML))):
            self.assertEqual(scrapping.scrape_pending_articles()["articles_scraped"], 2)
        first, second = self.store.data[1].values()
        self.assertEqual(first["canonical_hash"], scrapping.sha256(canonical))
        self.assertIsNone(second["canonical_hash"])
        self.assertEqual(first["summary_status"], "COMPLETED")
        self.assertEqual(len(self.store.data[2]), 3)

    def test_busy_lock_leaves_state_untouched(self):
        self.store.locked = True
        self.assertTrue(scrapping.run_scraping()["busy"])
        self.assertEqual(self.store.data, ({}, {}, {}))
        self.assertTrue(self.store.closed)


if __name__ == "__main__":
    unittest.main()
