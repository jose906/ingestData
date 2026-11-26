import io
import json
import socket
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import MagicMock, Mock, call, patch
from urllib.parse import urlsplit

import requests
from mysql.connector import DataError, Error
from requests.structures import CaseInsensitiveDict
from urllib3 import HTTPSConnectionPool, HTTPResponse

import main
import web_scraper as scraper


NEWS = "https://medio.example/noticia/puentes"
SHORT = "https://t.co/puentes"
SECOND = "https://otro.example/cronicas/agua"
ARTICLE_HTML = """
<!doctype html>
<html lang="es-BO"><head>
<meta charset="utf-8">
<title>La Paz inaugura un puente peatonal</title>
<meta property="og:title" content="La Paz inaugura un puente peatonal">
<meta property="og:type" content="article">
<meta property="og:description" content="La obra conecta dos barrios de La Paz.">
<meta property="og:image" content="https://medio.example/fotos/puente.jpg">
<meta property="article:published_time" content="2026-11-25T09:30:00-04:00">
<meta name="author" content="Ana Pérez">
<link rel="canonical" href="/noticia/puentes?utm_source=x">
</head><body>
<nav><a href="/">Menú inicial</a><a href="/categorias">Categorías</a></nav>
<article><h1>La Paz inaugura un puente peatonal</h1>
<p>Los vecinos de dos barrios paceños utilizaron por primera vez el nuevo puente peatonal
el miércoles por la mañana. La estructura atraviesa un río que durante años complicó los
desplazamientos hacia la escuela, el mercado y el centro de salud de la zona. El municipio
informó que las obras incluyeron rampas, luminarias y barandas para mejorar la seguridad.</p>
<p>Una comisión vecinal seguirá la ejecución del cronograma de mantenimiento. La alcaldía
publicará el presupuesto de la obra y realizará una inspección con profesionales de varias
especialidades. Los comerciantes de la zona señalaron que el proyecto acorta el camino hacia
sus puestos y reduce los tiempos de entrega de los productos que llegan desde el altiplano.</p>
<p>El encuentro de inauguración reunió a estudiantes, familias y dirigentes barriales. Los
equipos municipales midieron el flujo de personas al inicio de la jornada y anunciaron que
recogerán opiniones durante las próximas semanas para ajustar la iluminación y señalización.</p>
</article><div id="comments"><p>Comentario irrelevante de un lector.</p></div>
<footer>Publicidad y navegación del sitio</footer></body></html>
"""


def wire_response(status, headers=None, body=b""):
    payload = io.BytesIO(body)
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = status
    response.headers = CaseInsensitiveDict(headers or {})
    response.raw.read1.side_effect = lambda size, **kwargs: payload.read(size)
    return response


class UrlTests(unittest.TestCase):
    def test_entity_destinations_take_precedence_over_shorteners_and_text(self):
        tweet = {"entities": {"urls": [
            {"url": SHORT, "expanded_url": NEWS + "?utm_source=x", "unwound_url": NEWS},
            {"url": "https://t.co/social", "expanded_url": "https://x.com/user/status/100"},
            {"url": "https://t.co/unresolved", "expanded_url": None},
        ]}, "text": f"Mira {SHORT} https://t.co/social y {SECOND}. {NEWS}?fbclid=123"}
        unresolved = "https://t.co/unresolved"
        self.assertEqual(scraper.extract_external_urls(tweet), [
            scraper.TweetLink(SHORT, NEWS),
            scraper.TweetLink("https://t.co/social", "https://x.com/user/status/100"),
            scraper.TweetLink(unresolved, unresolved),
            scraper.TweetLink(SECOND, SECOND),
        ])

    def test_note_tweet_and_base_entities_are_combined(self):
        tweet = {
            "text": f"Truncado {SHORT}",
            "entities": {"urls": [{"url": SHORT}, None]},
            "note_tweet": {"text": "Texto completo " + SECOND, "entities": {"urls": [{
                "url": SHORT, "expanded_url": NEWS,
            }]}},
        }
        self.assertEqual(scraper.extract_external_urls(tweet), [
            scraper.TweetLink(SHORT, NEWS), scraper.TweetLink(SECOND, SECOND),
        ])
        self.assertEqual(scraper.extract_external_urls({"note_tweet": None, "text": None, "entities": []}), [])

    def test_valid_expansion_wins_over_unwound_short_url_and_covers_text(self):
        original = SHORT.replace("https:", "http:")
        tweet = {
            "text": SHORT,
            "entities": {"urls": [{"url": original, "unwound_url": SHORT, "expanded_url": NEWS}]},
            "note_tweet": {"entities": {"urls": [{"url": SHORT, "expanded_url": NEWS}]}},
        }
        metrics = dict.fromkeys(("tweets_with_url_entities", "expanded_urls_received", "tweets_only_tco"), 0)
        links = scraper.extract_external_urls(tweet, metrics=metrics)
        self.assertEqual(links, [scraper.TweetLink(SHORT, NEWS)])
        self.assertEqual(metrics, {"tweets_with_url_entities": 1, "expanded_urls_received": 1, "tweets_only_tco": 0})
        del tweet["note_tweet"]
        self.assertEqual(scraper.extract_external_urls(tweet), [scraper.TweetLink(original, NEWS)])

    def test_only_tco_keeps_original_short_link_and_counts_it(self):
        metrics = dict.fromkeys(("tweets_with_url_entities", "expanded_urls_received", "tweets_only_tco"), 0)
        tweet = {"entities": {"urls": [{"url": SHORT, "unwound_url": "https://t.co/otro"}]}}
        self.assertEqual(scraper.extract_external_urls(tweet, metrics=metrics), [scraper.TweetLink(SHORT, SHORT)])
        self.assertEqual(metrics, {"tweets_with_url_entities": 1, "expanded_urls_received": 0, "tweets_only_tco": 1})

    def test_text_extracts_addresses_and_preserves_balanced_parentheses(self):
        self.assertEqual(scraper.extract_urls_from_text(
            f"Lee ({NEWS}). También www.diario.example/tema_(Bolivia)! y https://medio.example/foto.JPG?x=1"
        ), [NEWS, "www.diario.example/tema_(Bolivia)", "https://medio.example/foto.JPG?x=1"])

    def test_text_extracts_article_without_scheme_and_keeps_visible_source(self):
        link = "diario.com.bo/pais/nota?id=2"
        text = f"Leé ({link}). {SHORT} correo@diario.com.bo/pais/nota ftp://diario.com.bo/archivo"
        self.assertEqual(scraper.extract_urls_from_text(text), [link, SHORT])
        self.assertEqual(scraper.normalize_url(link), "https://" + link)

    def test_normalization_keeps_parameters_that_identify_the_article(self):
        source = " HTTPS://MEDIO.EXAMPLE:443/noticia?utm_medium=x&id=42&tag=a+b&tag=c%2Fd&FbClId=abc#foto "
        self.assertEqual(scraper.normalize_url(source), "https://medio.example/noticia?id=42&tag=a+b&tag=c%2Fd")
        self.assertEqual(scraper.normalize_url("www.diário.example"), "https://www.xn--dirio-yqa.example/")

    def test_invalid_non_article_and_non_public_destinations_are_rejected(self):
        for url in (
            "file:///etc/test", "https://user:pass@medio.example/article", "https://not a host/",
            "https://medio.example:8080/noticia", "https://[v1.a]/articulo", "https://youtube.com/watch?v=123",
            "https://images.example/path/file%2EPDF", "http://localhost/", "http://127.1/article",
            "http://0x7f000001/article", "http://169.254.169.254/", "http://metadata.google.internal/",
            "http://10.5.3.2/", "http://100.66.0.1/", "http://[::1]/", "http://[::ffff:10.0.0.1]/",
            "http://[fe80::1]/", "http://[2002:7f00:1::1]/", "http://224.0.0.1/",
        ):
            with self.subTest(url=url):
                self.assertFalse(scraper.is_valid_article_url(url))
        self.assertTrue(scraper.is_valid_article_url(SHORT))


class RegistrationTests(unittest.TestCase):
    def test_registers_public_links_for_later_scraping(self):
        cursor = Mock(rowcount=1, lastrowid=11)
        cursor.fetchone.return_value = None
        pdf = "https://medio.example/boletin.pdf"
        profile = "https://x.com/medio"
        private_url = "http://169.254.169.254/"
        self.assertEqual(scraper.register_tweet_urls(cursor, 1, [pdf, profile, private_url]), 2)
        inserts = [entry.args for entry in cursor.execute.call_args_list if "INSERT INTO web_articles" in entry.args[0]]
        self.assertCountEqual([params[0] for _, params in inserts], [pdf, profile])
        self.assertTrue(all("'PENDING'" in sql for sql, _ in inserts))

    def test_expansion_keeps_original_source_and_discovery_reuses_it(self):
        cursor = Mock(rowcount=1, lastrowid=11)
        source = SHORT + "?utm_medium=x"
        cursor.fetchone.side_effect = [None, None, None, (42,)]
        links = [scraper.TweetLink(source, NEWS + "?utm_source=x")]
        self.assertEqual(scraper.register_tweet_urls(cursor, 1, links), 1)
        self.assertEqual(scraper.register_tweet_urls(cursor, 1, [SHORT]), 0)

        inserts = [entry.args[1] for entry in cursor.execute.call_args_list if "INSERT INTO " in entry.args[0]]
        self.assertEqual(inserts, [
            (NEWS, scraper._sha256(NEWS)),
            (1, 11, source, scraper._sha256(SHORT), 11),
        ])

    def test_known_expansion_consolidates_an_unresolved_short_url(self):
        cursor = Mock(rowcount=1)
        cursor.fetchone.side_effect = [(11,), (11, None, None), (22,), (22,)]
        links = [scraper.TweetLink(SHORT, NEWS)]
        self.assertEqual(scraper.register_tweet_urls(cursor, 1, links), 1)

        queries = cursor.execute.call_args_list
        self.assertFalse(any("INSERT INTO web_articles" in entry.args[0] for entry in queries))
        self.assertIn("article_id = %s", queries[-4].args[0])
        move, delete = queries[-3:-1]
        self.assertEqual(move, call("UPDATE tweet_articles SET article_id = %s WHERE article_id = %s", (11, 22)))
        self.assertIn("NOT EXISTS (SELECT 1 FROM tweet_articles", delete.args[0])
        self.assertEqual(delete.args[1], (22, 22))

    def test_two_article_links_for_one_tweet_and_an_existing_article_for_another(self):
        cursor = Mock(rowcount=1)
        ids = iter((11, 12))

        def execute(sql, params=()):
            if "INSERT INTO web_articles" in sql:
                cursor.lastrowid = next(ids)

        cursor.execute.side_effect = execute
        cursor.fetchone.side_effect = [None, None, None, None, None, None, None, (11,), (11, None, None)]
        self.assertEqual(scraper.register_tweet_urls(cursor, 1, [SECOND, NEWS, NEWS + "?utm_source=x"]), 2)
        self.assertEqual(scraper.register_tweet_urls(cursor, 2, [NEWS]), 1)

        inserted = [params for sql, params in (entry.args for entry in cursor.execute.call_args_list if len(entry.args) == 2)
                    if "INSERT INTO tweet_articles" in sql]
        self.assertEqual([row[:3] for row in inserted], [(1, 11, NEWS), (1, 12, SECOND), (2, 11, NEWS)])
        self.assertEqual(inserted[0][3], inserted[2][3])

    def test_existing_tweet_link_and_canonical_alias_do_not_create_articles(self):
        cursor = Mock(rowcount=1)
        cursor.fetchone.side_effect = [{"id": 1}, None, None, {"id": 1}, {"article_id": 11}]
        self.assertEqual(scraper.register_tweet_urls(cursor, 1, [SHORT]), 0)
        self.assertEqual(scraper.register_tweet_urls(cursor, 2, [SHORT]), 1)
        queries = cursor.execute.call_args_list
        self.assertFalse(any("INSERT INTO web_articles" in entry.args[0] for entry in queries))
        self.assertTrue(any(entry.args[1][:3] == (2, 11, SHORT)
                            for entry in queries if "INSERT INTO tweet_articles" in entry.args[0]))

    def test_retained_duplicate_with_distinct_content_routes_new_links_to_owner(self):
        cursor = Mock(rowcount=1)
        cursor.fetchone.side_effect = [None, (22,), (22, None, NEWS), (11,)]
        self.assertEqual(scraper.register_tweet_urls(cursor, 3, [SHORT]), 1)
        relation = next(entry.args[1] for entry in cursor.execute.call_args_list
                        if "INSERT INTO tweet_articles" in entry.args[0])
        self.assertEqual(relation[:3], (3, 11, SHORT))

    def test_bad_link_data_rolls_back_its_writes_without_exposing_url(self):
        cursor = Mock(rowcount=1, lastrowid=11)
        cursor.fetchone.return_value = None

        def execute(sql, params=()):
            if "INSERT INTO tweet_articles" in sql:
                raise DataError("sensitive query string", errno=1406)

        cursor.execute.side_effect = execute
        with self.assertLogs(scraper.LOGGER, "WARNING") as log:
            self.assertEqual(scraper.register_tweet_urls(cursor, 1, [NEWS]), 0)
        cursor.execute.assert_any_call("ROLLBACK TO SAVEPOINT web_article_url")
        cursor.execute.assert_any_call("RELEASE SAVEPOINT web_article_url")
        self.assertNotIn("sensitive query string", str(log.output))

    def test_concurrent_registration_discards_any_orphan_article(self):
        cursor = Mock(rowcount=0, lastrowid=11)
        cursor.fetchone.return_value = None
        self.assertEqual(scraper.register_tweet_urls(cursor, 1, [NEWS]), 0)
        cursor.execute.assert_any_call("ROLLBACK TO SAVEPOINT web_article_url")

    def test_transaction_failure_propagates_to_ingest_without_sql_message(self):
        cursor = Mock()
        cursor.execute.side_effect = Error("sensitive query string", errno=1213)
        with self.assertRaisesRegex(RuntimeError, r"MySQL 1213") as caught:
            scraper.register_tweet_urls(cursor, 1, [NEWS])
        self.assertNotIn("sensitive query string", str(caught.exception))


class ScraperTests(unittest.TestCase):
    def crawler(self, **settings):
        return scraper.ArticleScraper(replace(scraper.ScrapeSettings(), **settings), time.monotonic() + 15)

    @patch.object(scraper.socket, "getaddrinfo")
    def test_dns_rejects_mixed_public_and_private_addresses(self, lookup):
        lookup.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fd00::1", 0, 0, 0)),
        ]
        with self.assertRaisesRegex(scraper.ScrapeError, "unsafe_address"):
            scraper._resolve_public_ip("medio.example", 1)

    @patch.object(scraper.socket, "getaddrinfo")
    def test_requests_uses_only_the_validated_ip_and_checks_the_original_tls_host(self, lookup):
        lookup.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0))]
        response = HTTPResponse(body=io.BytesIO(b"page"), headers={"Content-Type": "text/html"}, status=200,
                                preload_content=False)
        with patch.object(HTTPSConnectionPool, "urlopen", autospec=True, return_value=response) as send:
            _, _, body = self.crawler()._download(NEWS + "?id=42")
        lookup.assert_called_once_with("medio.example.", None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        pool, = send.call_args.args
        self.assertEqual(pool.host, "8.8.8.8")
        self.assertEqual(pool.assert_hostname, "medio.example")
        self.assertEqual(pool.conn_kw["server_hostname"], "medio.example")
        self.assertEqual(pool.cert_reqs, "CERT_REQUIRED")
        self.assertEqual(send.call_args.kwargs["url"], "/noticia/puentes?id=42")
        self.assertEqual(send.call_args.kwargs["headers"]["Host"], "medio.example")
        self.assertIs(send.call_args.kwargs["redirect"], False)
        self.assertEqual(body, b"page")

    def test_unsafe_article_or_robots_redirect_is_never_requested(self):
        target = {"Location": "http://169.254.169.254/"}
        for responses in ([(404, {}, b""), (302, target, b"")], [(302, target, b"")]):
            with self.subTest(responses=responses):
                crawler = self.crawler()
                with patch.object(crawler, "_download", side_effect=responses) as download:
                    with self.assertRaisesRegex(scraper.ScrapeError, "unsafe_or_unsupported_url"):
                        crawler.scrape_article(SHORT)
                self.assertEqual(download.call_count, len(responses))

    def test_media_and_x_redirects_are_discarded_before_fetching_the_destination(self):
        for target in ("https://media.example/foto.png", "https://media.example/video.mp4",
                       "https://x.com/usuario", "https://twitter.com/usuario/status/100"):
            with self.subTest(target=target):
                crawler = self.crawler()
                responses = [(404, {}, b""), (302, {"Location": target}, b"")]
                with patch.object(crawler, "_download", side_effect=responses) as download:
                    with self.assertRaisesRegex(scraper.ScrapeError, "unsafe_or_unsupported_url") as caught:
                        crawler.scrape_article(SHORT)
                self.assertTrue(caught.exception.skipped)
                self.assertEqual(download.call_count, 2)

    def test_scraper_discards_saved_profiles_and_files_before_http(self):
        for url in ("https://x.com/medio", "https://medio.example/boletin.pdf"):
            with self.subTest(url=url):
                crawler = self.crawler()
                with patch.object(crawler, "_download") as download:
                    with self.assertRaisesRegex(scraper.ScrapeError, "unsafe_or_unsupported_url") as caught:
                        crawler.scrape_article(url)
                self.assertTrue(caught.exception.skipped)
                download.assert_not_called()

    def test_tco_robots_can_block_resolution_before_the_shortened_url_is_requested(self):
        crawler = self.crawler()
        robots = b"User-agent: *\nDisallow: /\n"
        with patch.object(crawler, "_download", return_value=(200, {}, robots)) as download:
            with self.assertRaisesRegex(scraper.ScrapeError, "robots_disallowed") as caught:
                crawler.scrape_article(SHORT)
        download.assert_called_once_with("https://t.co/robots.txt", robots=True)
        self.assertEqual(caught.exception.domain, "t.co")

    def test_tco_interstitial_cannot_be_saved_as_the_publisher(self):
        crawler = self.crawler()
        responses = [(404, {}, b""), (200, {}, ARTICLE_HTML.encode())]
        with patch.object(crawler, "_download", side_effect=responses):
            with self.assertRaisesRegex(scraper.ScrapeError, "unresolved_short_url") as caught:
                crawler.scrape_article(SHORT)
        self.assertTrue(caught.exception.skipped)

    def test_repeated_wire_url_is_still_a_redirect_cycle(self):
        crawler = self.crawler()
        responses = [(404, {}, b""), (302, {"Location": SHORT}, b"")]
        with patch.object(crawler, "_download", side_effect=responses) as download:
            with self.assertRaisesRegex(scraper.ScrapeError, "redirect_cycle") as caught:
                crawler.scrape_article(SHORT)
        self.assertTrue(caught.exception.skipped)
        self.assertEqual(download.call_count, 2)

    def test_safe_redirect_checks_destination_robots_before_downloading(self):
        crawler = self.crawler()
        robots = b"User-agent: *\nDisallow: /noticia/*\nAllow: /noticia/puentes$\nCrawl-delay: 2\n"
        responses = [(404, {}, b""), (308, {"Location": NEWS}, b""), (200, {}, robots),
                     (200, {}, ARTICLE_HTML.encode())]
        with patch.object(crawler, "_download", side_effect=responses) as download:
            result = crawler.scrape_article(SHORT)
            with self.assertRaisesRegex(scraper.ScrapeError, "robots_disallowed") as caught:
                crawler.scrape_article("https://medio.example/noticia/otra")
        self.assertEqual(caught.exception.domain, "medio.example")
        self.assertEqual(download.call_args_list, [
            call("https://t.co/robots.txt", robots=True), call(SHORT, robots=False),
            call("https://medio.example/robots.txt", robots=True), call(NEWS, robots=False),
        ])
        self.assertEqual(result["canonical_url"], NEWS)
        self.assertEqual(crawler.domain_delays["medio.example"], 2)

    def test_redirected_domain_resolving_to_private_ip_is_rejected(self):
        crawler = self.crawler(domain_delay=0)
        robots_missing = wire_response(404)
        redirect = wire_response(302, {"Location": "https://private.example/noticia"})

        def dns(host, *args):
            address = "8.8.8.8" if host == "t.co." else "10.0.0.5"
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]

        with patch.object(scraper.socket, "getaddrinfo", side_effect=dns):
            with patch.object(scraper._PinnedHTTPSAdapter, "send", side_effect=[robots_missing, redirect]) as send:
                with self.assertRaisesRegex(scraper.ScrapeError, "unsafe_address"):
                    crawler.scrape_article(SHORT)
        self.assertEqual(send.call_count, 2)
        redirect.raw.read1.assert_not_called()

    def test_http_errors_permanent_and_transient(self):
        for status, retryable, skipped in ((403, False, True), (404, False, True), (429, True, False),
                                            (503, True, False), (522, True, False)):
            crawler = self.crawler()
            with patch.object(crawler, "_download", side_effect=[(404, {}, b""), (status, {"Retry-After": "120"}, b"")]):
                with self.assertRaisesRegex(scraper.ScrapeError, f"http_{status}") as caught:
                    crawler.scrape_article(NEWS)
            self.assertEqual(caught.exception.status, status)
            self.assertEqual(caught.exception.retryable, retryable)
            self.assertEqual(caught.exception.skipped, skipped)
            self.assertEqual(caught.exception.retry_after, 120)

    def test_robots_temporary_error_blocks_article_download(self):
        crawler = self.crawler()
        with patch.object(crawler, "_download", return_value=(503, {"Retry-After": "120"}, b"")) as download:
            with self.assertRaisesRegex(scraper.ScrapeError, "http_503") as caught:
                crawler.scrape_article(NEWS)
        self.assertEqual(caught.exception.retry_after, 120)
        download.assert_called_once_with("https://medio.example/robots.txt", robots=True)

    def test_redirect_retry_after_above_job_budget_is_deferred(self):
        crawler = self.crawler()
        responses = [(404, {}, b""), (302, {"Location": NEWS, "Retry-After": "120"}, b"")]
        with patch.object(crawler, "_download", side_effect=responses) as download:
            with self.assertRaisesRegex(scraper.ScrapeError, "redirect_delay") as caught:
                crawler.scrape_article(SHORT)
        self.assertEqual(download.call_count, 2)
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(caught.exception.retry_after, 120)

    def test_body_size_and_mime_are_checked_before_extraction(self):
        cases = (
            ({"Content-Type": "image/png"}, b"", "non_html"),
            ({"Content-Type": "text/html", "Content-Length": "300"}, b"", "response_too_large"),
            ({"Content-Type": "text/html"}, b"a" * 40, "response_too_large"),
            ({"Content-Type": "text/html", "Content-Encoding": "gzip"}, b"x", "unsupported_encoding"),
        )
        for headers, body, code in cases:
            with self.subTest(code=code):
                crawler = self.crawler(max_bytes=32)
                response = wire_response(200, headers, body)
                with patch.object(scraper, "_resolve_public_ip", return_value="8.8.8.8"):
                    with patch.object(scraper._PinnedHTTPSAdapter, "send", return_value=response):
                        with self.assertRaisesRegex(scraper.ScrapeError, code):
                            crawler._download(NEWS)

    def test_transport_errors_dont_reveal_url_parameters(self):
        with patch.object(scraper, "_resolve_public_ip", return_value="8.8.8.8"):
            with patch.object(scraper._PinnedHTTPSAdapter, "send", side_effect=requests.ConnectionError("secret=42")):
                with self.assertRaisesRegex(scraper.ScrapeError, "^network_error$") as error:
                    self.crawler()._download(NEWS + "?secret=42")
        self.assertTrue(error.exception.retryable)

    def test_trafilatura_extracts_content_and_metadata_from_article_html(self):
        result = scraper.extract_article(ARTICLE_HTML.encode(), NEWS + "?version=amp", 80)
        self.assertEqual(result["title"], "La Paz inaugura un puente peatonal")
        self.assertEqual(result["description"], "La obra conecta dos barrios de La Paz.")
        self.assertEqual(result["author"], "Ana Pérez")
        self.assertEqual(result["image_url"], "https://medio.example/fotos/puente.jpg")
        self.assertEqual(result["published_at"], datetime(2026, 11, 25, 13, 30))
        self.assertEqual(result["canonical_url"], NEWS)
        self.assertEqual(result["language"], "es-BO")
        self.assertEqual(result["domain"], "medio.example")
        self.assertGreater(result["word_count"], 100)
        self.assertIn("cronograma de mantenimiento", result["content"])
        for text in ("Comentario irrelevante", "Publicidad y navegación", "Menú inicial", "<article>"):
            self.assertNotIn(text, result["content"])

    def test_paywall_consent_and_thin_pages_are_rejected(self):
        paywall = ARTICLE_HTML.replace('</head>', '<script type="application/ld+json">'
                                       '{"@type":"NewsArticle","isAccessibleForFree":false}</script></head>')
        consent = ARTICLE_HTML.replace("<title>La Paz inaugura un puente peatonal", "<title>Before you continue")
        thin = '<!doctype html><html><head><title>Noticia</title></head><body><article>Corta</article></body></html>'
        for source, code in ((paywall, "restricted_content"), (consent, "error_or_consent_page"),
                             (thin, "insufficient_article_content")):
            with self.subTest(code=code):
                with self.assertRaisesRegex(scraper.ScrapeError, code):
                    scraper.extract_article(source.encode(), NEWS, 80)

    def test_canonical_hint_cannot_merge_independent_domains(self):
        first = scraper.extract_article(ARTICLE_HTML.encode(), NEWS, 80)
        other = ARTICLE_HTML.replace('href="/noticia/puentes?utm_source=x"', f'href="{NEWS}"')
        second = scraper.extract_article(other.encode(), SECOND, 80)
        self.assertNotEqual(first["canonical_hash"], second["canonical_hash"])
        self.assertEqual(first["content_hash"], second["content_hash"])
        self.assertEqual(second["canonical_url"], SECOND)


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.content = scraper.extract_article(ARTICLE_HTML.encode(), NEWS, 80)
        self.cursor = Mock(rowcount=1)
        self.settings = scraper.ScrapeSettings()

    def test_canonical_merge_preserves_source_urls_and_checks_before_delete(self):
        self.cursor.fetchone.side_effect = [(None, 0, 0), (11, self.content["content_hash"], 1)]
        self.assertEqual(scraper.save_article_result(self.cursor, 22, 1, self.content), "duplicates")
        move, remove = self.cursor.execute.call_args_list[-2:]
        self.assertIn("UPDATE tweet_articles SET article_id =", move.args[0])
        self.assertEqual(move.args[1], (11, 22))
        self.assertIn("NOT EXISTS (SELECT 1 FROM tweet_articles", remove.args[0])
        self.assertEqual(remove.args[1], (22, 22))
        self.assertIn("domain = %s", self.cursor.execute.call_args_list[-3].args[0])
        self.assertEqual(self.cursor.execute.call_args_list[-3].args[1][2], "medio.example")

    @patch.dict("os.environ", {"TWITTER_BEARER_TOKEN": "mock-x-token", "INGEST_JOB_TOKEN": "mock-job-token"})
    def test_shorteners_save_final_html_and_media_domain_after_one_destination_get(self):
        for short in (SHORT, "https://corto.example/go"):
            with self.subTest(short=short):
                parts = urlsplit(short)
                short_host = parts.hostname
                final_path = "/noticia/puentes?utm_source=handoff&id=9"
                replies = {
                    (short_host, "/robots.txt"): (404, {}, b""),
                    (short_host, parts.path): (302, {"Location": "?utm_source=handoff"}, b""),
                    (short_host, parts.path + "?utm_source=handoff"): (
                        307, {"Location": "//medio.example" + final_path, "Set-Cookie": "tracker=1"}, b"",
                    ),
                    ("medio.example", "/robots.txt"): (404, {}, b""),
                    ("medio.example", final_path): (200, {"Content-Type": "text/html"}, ARTICLE_HTML.encode()),
                }
                requests_seen = []

                def send(pool, method, url, headers, **kwargs):
                    host = headers["Host"]
                    self.assertEqual(method, "GET")
                    self.assertEqual(pool.host, "8.8.8.8")
                    self.assertEqual(pool.assert_hostname, host)
                    self.assertFalse({"authorization", "proxy-authorization", "cookie"} &
                                     {name.lower() for name in headers})
                    self.assertEqual(headers["User-Agent"], scraper.USER_AGENT)
                    requests_seen.append((host, url))
                    status, response_headers, body = replies[(host, url)]
                    return HTTPResponse(body=io.BytesIO(body), headers=response_headers, status=status,
                                        preload_content=False)

                cursor, conn = Mock(rowcount=1), Mock()
                cursor.fetchall.side_effect = [[], [(22, short, 0)]]
                cursor.fetchone.side_effect = [(None, 0, 0), None, None]
                with patch.object(HTTPSConnectionPool, "urlopen", autospec=True, side_effect=send):
                    with patch.object(scraper, "_resolve_public_ip", return_value="8.8.8.8"):
                        with patch.object(scraper.ScrapeSettings, "from_env", return_value=replace(self.settings, domain_delay=0)):
                            metrics = scraper.process_pending_articles(conn, cursor, time.monotonic())

                self.assertEqual(requests_seen, list(replies))
                self.assertEqual(metrics["completed"], 1)
                params = cursor.execute.call_args.args[1]
                self.assertEqual(params["article_id"], 22)
                self.assertEqual(params["domain"], "medio.example")
                self.assertEqual(params["canonical_url"], NEWS)
                self.assertEqual(params["canonical_hash"], scraper._sha256(NEWS))
                self.assertIn("cronograma de mantenimiento", params["content"])

    def test_legacy_redirect_loop_can_be_claimed_with_the_existing_attempt_budget(self):
        self.cursor.fetchall.return_value = [(22, SHORT, 1)]
        self.assertEqual(scraper.fetch_pending_articles(self.cursor, self.settings), [(22, SHORT, 1)])
        select = self.cursor.execute.call_args
        self.assertIn("scraping_error = 'redirect_loop' AND scraping_retry_at IS NULL", select.args[0])
        self.assertIn("COALESCE(scraping_attempts, 0) < %s", select.args[0])
        self.assertIn("ORDER BY (url LIKE %s OR url LIKE %s), article_id", select.args[0])
        self.assertEqual(select.args[1], (3, "https://t.co/%", "http://t.co/%", self.settings.limit))

        self.assertTrue(scraper._claim_article(self.cursor, 22, 1))
        update = self.cursor.execute.call_args
        self.assertIn(scraper.ELIGIBLE_FOR_SCRAPING, update.args[0])
        self.assertIn("COALESCE(scraping_attempts, 0) + 1", update.args[0])
        self.assertEqual(update.args[1], (22, 1))

    def test_canonical_merge_retains_new_content_when_it_differs(self):
        self.cursor.fetchone.side_effect = [(None, 0, 0), (11, "different", 1)]
        self.assertEqual(scraper.save_article_result(self.cursor, 22, 1, self.content), "duplicates")
        saved = self.cursor.execute.call_args.args[1]
        self.assertEqual(saved["article_id"], 22)
        self.assertEqual(saved["content"], self.content["content"])
        self.assertIsNone(saved["canonical_hash"])

    def test_claim_does_not_publish_a_result_after_attempt_ownership_is_lost(self):
        self.cursor.fetchone.return_value = None
        self.assertEqual(scraper.save_article_result(self.cursor, 22, 1, self.content), "superseded")
        self.cursor.execute.assert_called_once()

    def test_successful_extraction_fills_existing_canonical_placeholder(self):
        self.cursor.fetchone.side_effect = [(None, 0, 0), None, (11, None, 0)]
        self.assertEqual(scraper.save_article_result(self.cursor, 22, 1, self.content), "duplicates")
        saved = next(entry for entry in self.cursor.execute.call_args_list if "content_hash = %(content_hash)s" in entry.args[0])
        self.assertIn("'PENDING', summary_status", saved.args[0])
        self.assertEqual(saved.args[1]["article_id"], 11)
        self.assertEqual(saved.args[1]["canonical_hash"], self.content["canonical_hash"])

    def test_retry_delay_is_progressive_respects_retry_after_and_stops_at_third_attempt(self):
        before = datetime.now(timezone.utc).replace(tzinfo=None)
        error = scraper.ScrapeError("http_503", status=503, retryable=True)
        with self.assertLogs(scraper.LOGGER, "WARNING"):
            self.assertEqual(scraper._save_failure(self.cursor, 10, 1, error, self.settings), "failed")
            first = self.cursor.execute.call_args.args[1][3]
            scraper._save_failure(self.cursor, 10, 2, error, self.settings)
            second = self.cursor.execute.call_args.args[1][3]
            error.retry_after = 220
            scraper._save_failure(self.cursor, 10, 2, error, self.settings)
            delayed = self.cursor.execute.call_args.args[1][3]
            scraper._save_failure(self.cursor, 10, 3, error, self.settings)
            final = self.cursor.execute.call_args.args[1][3]
        self.assertGreaterEqual(first, before + timedelta(seconds=60))
        self.assertGreaterEqual(second, first + timedelta(seconds=60))
        self.assertGreaterEqual(delayed, before + timedelta(seconds=220))
        self.assertIsNone(final)
        self.assertGreater(scraper._retry_after(format_datetime(datetime.now(timezone.utc) + timedelta(minutes=5))), 295)

    def test_recovers_stale_processing_then_commits_claim_before_http(self):
        conn = Mock()
        self.cursor.fetchall.side_effect = [[(11,)], [(11, NEWS, 1)]]
        self.cursor.fetchone.side_effect = [(None, 0, 0), None, None]

        def scrape(url):
            self.assertEqual(url, NEWS)
            self.assertEqual(conn.commit.call_count, 2)
            self.assertIn("scraping_status = 'PROCESSING'", self.cursor.execute.call_args.args[0])
            return self.content

        with patch.object(scraper.ArticleScraper, "scrape_article", side_effect=scrape):
            result = scraper.process_pending_articles(conn, self.cursor, time.monotonic())
        self.assertEqual((result["recovered"], result["articles_processed"], result["completed"]), (1, 1, 1))
        recovery = self.cursor.execute.call_args_list[1]
        self.assertIn("< %s, UTC_TIMESTAMP(), NULL", recovery.args[0])
        self.assertEqual(recovery.args[1], (3, 11))
        self.assertEqual(conn.commit.call_count, 3)

    def test_scraping_metrics_show_why_an_article_did_not_complete(self):
        conn = Mock()
        self.cursor.fetchall.side_effect = [[], [(11, SHORT, 0), (22, NEWS, 0)]]
        errors = [scraper.ScrapeError("robots_disallowed", skipped=True, domain="t.co"), scraper.ScrapeError("tls_error")]
        with patch.object(scraper.ArticleScraper, "scrape_article", side_effect=errors):
            with self.assertLogs(scraper.LOGGER, "WARNING"):
                result = scraper.process_pending_articles(conn, self.cursor, time.monotonic())
        self.assertEqual((result["articles_processed"], result["skipped"], result["failed"]), (2, 1, 1))
        self.assertEqual(result["scraping_errors"], {"robots_disallowed": 1, "tls_error": 1})
        failures = [entry.args for entry in self.cursor.execute.call_args_list if "domain = COALESCE" in entry.args[0]]
        self.assertEqual([params[4] for _, params in failures], ["t.co", None])


class DiscoveryTests(unittest.TestCase):
    def test_discovery_reports_found_urls_even_if_they_were_already_registered(self):
        cursor, conn = Mock(), Mock()
        bare = "diario.com.bo/actualidad/nota"
        cursor.fetchall.return_value = [(100, f"{SHORT} o {bare}")]
        with patch.object(scraper, "register_tweet_urls", return_value=0) as register:
            result = scraper.discover_historical_urls(conn, cursor, datetime.now(), time.monotonic(),
                                                     Mock(return_value="0"), Mock())
        register.assert_called_once_with(cursor, 100, [SHORT, bare])
        self.assertEqual((result["urls_found"], result["tco_urls_found"], result["links_registered"]), (2, 1, 0))

    @patch.dict("os.environ", {"WEB_ARTICLES_DISCOVER_LIMIT": "4"})
    def test_older_tweet_ids_are_revisited_while_newer_tweets_keep_arriving(self):
        now = datetime(2026, 11, 26, 8)
        rows = [(tweetid, NEWS, now) for tweetid in range(100, 1500, 100)]
        rows.append((20, NEWS, now - timedelta(days=8)))
        cursor, conn = Mock(), Mock()
        state = {}

        def query(sql, params):
            cutoff, until, after = params[:3]
            self.assertEqual(cutoff, now - timedelta(days=7))
            self.assertEqual(until, now)
            self.assertNotIn("Tweets.url", sql)
            through, limit = params[3:] if len(params) == 5 else (float("inf"), params[3])
            cursor.fetchall.return_value = [(i, text) for i, text, created in sorted(rows)
                                             if after < i <= through and cutoff <= created <= until][:limit]

        cursor.execute.side_effect = query
        get_state = Mock(side_effect=lambda cur, key, default: state.get(key, default))
        set_state = Mock(side_effect=lambda cur, key, value: state.update({key: value}))
        with patch.object(scraper, "register_tweet_urls", return_value=1) as register:
            first = scraper.discover_historical_urls(conn, cursor, now, time.monotonic(), get_state, set_state)
            self.assertEqual(first["tweets_scanned"], 4)
            rows.append((150, NEWS, now - timedelta(days=1)))
            for _ in range(2):
                result = scraper.discover_historical_urls(conn, cursor, now, time.monotonic(), get_state, set_state)
                self.assertEqual((result["tweets_scanned"], result["tweets_reviewed"]), (3, 1))
        register.assert_any_call(cursor, 150, [NEWS])
        self.assertNotIn(call(cursor, 20, [NEWS]), register.call_args_list)
        self.assertEqual(json.loads(state[scraper.DISCOVERY_STATE_KEY]), {"last_tweetid": 1000, "review_tweetid": 150})
        self.assertEqual(conn.commit.call_count, 3)

    def test_error_does_not_advance_discovery_state(self):
        cursor, conn = Mock(), Mock()
        cursor.fetchall.return_value = [(100, NEWS)]
        set_state = Mock()
        with patch.object(scraper, "register_tweet_urls", side_effect=RuntimeError("DB failure")):
            with self.assertRaises(RuntimeError):
                scraper.discover_historical_urls(conn, cursor, datetime.now(), time.monotonic(),
                                                 Mock(return_value="0"), set_state)
        set_state.assert_not_called()
        conn.commit.assert_not_called()


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.conn = Mock()
        self.cursor = Mock(rowcount=1, lastrowid=50)
        self.lookup = Mock()
        self.get_state = Mock(return_value="0")
        self.set_state = Mock()
        self.prefix = scraper.RECOVERY_STATE_KEY

    def respond(self, payload, status=200, headers=None):
        response = Mock(status_code=status, headers=CaseInsensitiveDict(headers or {}))
        response.json.return_value = payload
        self.lookup.return_value = response
        return response

    def recover(self):
        return scraper.recover_historical_urls(self.conn, self.cursor, self.lookup, self.get_state, self.set_state)

    @patch.dict("os.environ", {"WEB_ARTICLES_RECOVER_LIMIT": "10"})
    def test_partial_lookup_uses_only_matching_entities_and_tracks_each_post(self):
        original = SHORT.replace("https:", "http:") + "?utm_source=x"
        other_short = "https://t.co/otra"
        self.cursor.fetchall.side_effect = [
            [(11, None), (12, None), (13, None), (14, None)],
            [(11, original, SHORT), (11, other_short, other_short), (13, SHORT, SHORT)],
        ]
        response = self.respond({
            "data": [
                {"id": "11", "text": SECOND, "note_tweet": {"entities": {"urls": [
                    {"url": SHORT, "expanded_url": NEWS}, {"url": "https://t.co/ajeno", "expanded_url": SECOND},
                ]}}},
                {"id": "13", "text": f"{SHORT} {SECOND}", "entities": {"urls": [{"url": SHORT}]}},
            ],
            "errors": [{"type": "https://api.x.com/2/problems/resource-not-found", "resource_id": "12"}],
        })

        def lookup(ids):
            self.conn.commit.assert_called_once()
            return response

        self.lookup.side_effect = lookup
        with patch.object(scraper, "register_tweet_urls", return_value=1) as register:
            metrics = self.recover()
        self.lookup.assert_called_once_with(["11", "12", "13", "14"])
        register.assert_called_once_with(self.cursor, "11", [scraper.TweetLink(original, NEWS, SHORT)])
        query, params = self.cursor.execute.call_args_list[0].args
        self.assertIn("SELECT DISTINCT ta.tweetid", query)
        self.assertIn("a.scraping_status IN ('PENDING', 'FAILED', 'SKIPPED')", query)
        self.assertIn("s.k IS NULL OR", query)
        self.assertEqual(params, (self.prefix + ":", "https://t.co/%", "http://t.co/%", 10))
        state = {entry.args[1]: entry.args[2] for entry in self.set_state.call_args_list}
        self.assertEqual(state[f"{self.prefix}:11"], "recovered")
        self.assertEqual(state[f"{self.prefix}:12"], "unavailable")
        self.assertEqual(state[f"{self.prefix}:13"], "without_expansion")
        self.assertRegex(state[f"{self.prefix}:14"], r"^retry:1:\d+$")
        self.assertEqual((metrics["posts_recovered"], metrics["posts_unavailable"], metrics["posts_retryable"]), (1, 1, 1))
        self.assertEqual(self.conn.commit.call_count, 2)

    def test_recovery_reassigns_relationships_to_an_existing_article(self):
        original = SHORT.replace("https:", "http:")
        self.cursor.fetchall.side_effect = [[(11, None)], [(11, original, SHORT)]]
        self.cursor.fetchone.side_effect = [(30,), (30, None, None), None, (40,), (40,)]
        self.cursor.rowcount = 2
        self.respond({"data": [{"id": "11", "entities": {"urls": [{"url": SHORT, "unwound_url": NEWS}]}}]})
        metrics = self.recover()
        self.assertEqual(metrics["expanded_urls_recovered"], 1)
        self.assertFalse(any("INSERT INTO web_articles" in entry.args[0] for entry in self.cursor.execute.call_args_list))
        self.cursor.execute.assert_any_call("UPDATE tweet_articles SET article_id = %s WHERE article_id = %s", (30, 40))
        self.set_state.assert_called_once_with(self.cursor, f"{self.prefix}:11", "recovered")

    def test_recovery_creates_pending_target_and_preserves_protected_provisional(self):
        self.cursor.fetchall.side_effect = [[(11, None)], [(11, SHORT, SHORT)]]
        self.cursor.fetchone.side_effect = [None, None, (40,), None]
        self.respond({"data": [{"id": "11", "entities": {"urls": [{"url": SHORT, "expanded_url": NEWS}]}}]})
        metrics = self.recover()
        self.assertEqual(metrics["posts_recovered"], 1)
        queries = [entry.args for entry in self.cursor.execute.call_args_list]
        insert = next(args for args in queries if "INSERT INTO web_articles" in args[0])
        self.assertIn("'PENDING', 'PENDING'", insert[0])
        self.assertEqual(insert[1][0], NEWS)
        link = next(args for args in queries if "INSERT INTO tweet_articles" in args[0])
        self.assertEqual(link[1][:3], ("11", 50, SHORT))
        guard = next(args[0] for args in queries if "FOR UPDATE" in args[0])
        self.assertIn("COALESCE(content, '') = ''", guard)
        self.assertIn("summary IS NULL", guard)
        self.assertFalse(any("DELETE FROM web_articles" in args[0] for args in queries))

    def test_rate_limits_persist_and_block_another_lookup_until_reset(self):
        self.cursor.fetchall.return_value = [(11, None)]
        now = int(time.time())
        self.respond({}, 429, {"Retry-After": "90", "x-rate-limit-reset": str(now + 200)})
        metrics = self.recover()
        self.assertEqual(metrics["status_code"], 429)
        self.assertGreaterEqual(metrics["retry_after_seconds"], 190)
        self.set_state.assert_called_once_with(self.cursor, self.prefix, str(now + 200))
        self.get_state.return_value = str(now + 200)
        self.assertEqual(self.recover()["posts_requested"], 0)
        self.lookup.assert_called_once()

    def test_credit_limit_does_not_mark_publications_as_checked(self):
        self.cursor.fetchall.return_value = [(11, None)]
        self.respond({"type": "https://api.x.com/2/problems/usage-capped"}, 429)
        result = self.recover()
        self.assertEqual((result["status_code"], result["error"]), (402, "x_usage_limit"))
        self.set_state.assert_not_called()

    def test_request_error_does_not_mark_publications_as_checked(self):
        self.cursor.fetchall.return_value = [(11, None)]
        self.lookup.side_effect = requests.Timeout("sensitive data")
        result = self.recover()
        self.assertEqual((result["status_code"], result["error"]), (502, "x_network_error"))
        self.set_state.assert_not_called()

    def test_repeated_partial_omission_exhausts_the_attempt_budget(self):
        self.cursor.fetchall.return_value = [(11, "retry:2:0")]
        self.respond({"data": []})
        self.assertEqual(self.recover()["posts_retry_exhausted"], 1)
        self.set_state.assert_called_once_with(self.cursor, f"{self.prefix}:11", "retry_exhausted")

    @patch.dict("os.environ", {"WEB_ARTICLES_RECOVER_LIMIT": "100"})
    def test_recovery_uses_the_configured_maximum_and_skips_empty_lookups(self):
        self.cursor.fetchall.return_value = []
        self.assertEqual(self.recover()["batch_limit"], 100)
        self.assertEqual(self.cursor.execute.call_args.args[1][-1], 100)
        self.lookup.assert_not_called()


class FlaskIngestionTests(unittest.TestCase):
    def setUp(self):
        self.client = main.app.test_client()

    @patch.dict("os.environ", {"INGEST_JOB_TOKEN": "unit-token"})
    @patch.object(main, "get_db_connection")
    def test_endpoints_require_configured_token_and_take_independent_locks(self, get_db):
        cursor = get_db.return_value.cursor.return_value
        for name, lock in (("discover", "discover"), ("scrape", "scrape"), ("recover_urls", "recover")):
            with self.subTest(name=name):
                route = f"/articles/{name}"
                self.assertEqual(self.client.get(route).status_code, 401)
                cursor.fetchone.return_value = (0,)
                response = self.client.get(route, headers={"Authorization": "Bearer unit-token"})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json["busy"])
                cursor.execute.assert_called_with("SELECT GET_LOCK(%s, 0)", (f"netvora_articles_{lock}",))
        self.assertEqual(get_db.call_count, 3)

    @patch.dict("os.environ", {"INGEST_JOB_TOKEN": ""})
    @patch.object(main, "get_db_connection")
    def test_recovery_head_has_no_side_effects_without_job_token(self, get_db):
        self.assertEqual(self.client.head("/articles/recover_urls").status_code, 204)
        get_db.assert_not_called()

    @patch.dict("os.environ", {"INGEST_JOB_TOKEN": "", "TWITTER_BEARER_TOKEN": "x-unit-token"})
    @patch.object(main, "get_db_connection")
    @patch.object(main.requests, "get")
    def test_recovery_route_uses_one_official_lookup_and_no_ingestion(self, http_get, get_db):
        conn = get_db.return_value
        cursor = conn.cursor.return_value
        cursor.fetchone.side_effect = [(1,), None, (1,)]
        cursor.fetchall.return_value = [(101, None), (102, None)]
        http_get.return_value.status_code = 429
        http_get.return_value.headers = CaseInsensitiveDict({"Retry-After": "60"})
        http_get.return_value.json.return_value = {}
        with patch.object(main, "insert_or_update_tweet") as insert:
            response = self.client.get("/articles/recover_urls")
        insert.assert_not_called()
        self.assertEqual(response.status_code, 429)
        self.assertFalse(response.json["ok"])
        self.assertEqual(response.json["posts_requested"], 2)
        self.assertEqual(response.headers["Retry-After"], str(response.json["retry_after_seconds"]))
        http_get.assert_called_once_with(
            "https://api.x.com/2/tweets", headers={"Authorization": "Bearer x-unit-token"},
            params={"ids": "101,102", "tweet.fields": "entities,note_tweet"}, timeout=20, allow_redirects=False,
        )
        cursor.execute.assert_called_with("SELECT RELEASE_LOCK(%s)", ("netvora_articles_recover",))

    @patch.dict("os.environ", {"INGEST_JOB_TOKEN": ""})
    @patch.object(main, "get_db_connection")
    def test_scrape_empty_queue_returns_metrics_and_releases_lock(self, get_db):
        conn = get_db.return_value
        cursor = conn.cursor.return_value
        cursor.fetchone.side_effect = [(1,), (1,)]
        cursor.fetchall.return_value = []
        response = self.client.get("/articles/scrape")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["articles_processed"], 0)
        cursor.execute.assert_called_with("SELECT RELEASE_LOCK(%s)", ("netvora_articles_scrape",))
        conn.commit.assert_called_once()
        conn.close.assert_called_once()

    @patch.dict("os.environ", {"INGEST_JOB_TOKEN": ""})
    @patch.object(main, "get_db_connection")
    def test_discovery_failure_rolls_back_and_releases_lock(self, get_db):
        conn = get_db.return_value
        cursor = conn.cursor.return_value
        cursor.fetchone.side_effect = [(1,), (1,)]
        with patch.object(main, "discover_historical_urls", side_effect=Error("secret=42", errno=1213)) as discover:
            with self.assertLogs(main.app.logger, "ERROR") as log:
                response = self.client.get("/articles/discover")
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("secret=42", str(log.output))
        self.assertIsNone(discover.call_args.args[2].tzinfo)
        conn.rollback.assert_called_once()
        cursor.execute.assert_called_with("SELECT RELEASE_LOCK(%s)", ("netvora_articles_discover",))

    @patch.object(main.requests, "get")
    def test_tweet_insert_and_update_keep_text_ml_and_post_url(self, http_get):
        tweet = {"id": "100", "created_at": "2026-11-26T13:00:00Z", "text": "Texto corto", "note_tweet": {
            "text": "Texto completo " + SHORT,
            "entities": {"urls": [{"url": SHORT, "expanded_url": NEWS}]},
        }}
        for exists in (False, True):
            cursor = Mock(rowcount=1)
            cursor.fetchone.side_effect = [{"tweetid": "100"} if exists else None, None, None, None]
            cursor.lastrowid = 11
            with patch.object(main.MLModel, "get_sentiment", return_value=["neutral"]) as sentiment:
                with patch.object(main.MLModel, "predecir_categoria", return_value=["Actualidad"]) as category:
                    self.assertEqual(main.insert_or_update_tweet(cursor, tweet, 5), 0 if exists else 1)
            self.assertEqual(sentiment.call_count, int(not exists))
            self.assertEqual(category.call_count, int(not exists))
            statements = [entry.args for entry in cursor.execute.call_args_list]
            _, write = statements[1]
            self.assertIn("Texto completo " + SHORT, write)
            self.assertIn("https://twitter.com/i/web/status/100", write)
            self.assertIn(datetime(2026, 11, 26, 9), write)
            relation = next(params for sql, params in (entry for entry in statements if len(entry) == 2)
                            if "INSERT INTO tweet_articles" in sql)
            self.assertEqual(relation[:3], ("100", 11, SHORT))
            article = next(params for sql, params in (entry for entry in statements if len(entry) == 2)
                           if "INSERT INTO web_articles" in sql)
            self.assertEqual(article[0], NEWS)
            self.assertFalse(any(sql.startswith("COMMIT") for sql, *_ in statements))
        http_get.assert_not_called()

    @patch.object(main.requests, "get")
    @patch.object(main, "x_headers", return_value={"Authorization": "Bearer mock"})
    def test_fetching_tweets_uses_the_existing_single_x_api_request(self, headers, http_get):
        http_get.return_value.status_code = 200
        http_get.return_value.json.return_value = {"meta": {}, "data": [{"id": "100", "text": NEWS}]}
        result = main.fetch_tweets_for_user("medio", last_tweetid="42")
        self.assertTrue(result["ok"])
        http_get.assert_called_once()
        self.assertEqual(http_get.call_args.kwargs["params"]["tweet.fields"], "created_at,text,entities,author_id,note_tweet")

    @patch.dict("os.environ", {"INGEST_JOB_TOKEN": "", "TWITTER_BEARER_TOKEN": "x-unit-token"})
    @patch.object(main, "get_db_connection")
    @patch.object(main, "get_users")
    @patch.object(main.requests, "get")
    def test_ingest_reports_url_metrics_from_its_existing_x_request(self, http_get, get_users, get_db):
        cursor = get_db.return_value.cursor.return_value
        cursor.fetchone.side_effect = [{"acquired": 1}, {"tweetid": 11}, {"tweetid": 12}, (1,)]
        get_users.return_value = [{"idTweetUser": 5, "TweetUser": "medio", "last_tweetid": "10", "pagination_token": None}]
        other_short = "https://t.co/otra"
        pdf = "https://medio.example/boletin.pdf"
        profile = "https://x.com/medio"
        profile_short = "https://t.co/perfil"
        http_get.return_value.status_code = 200
        http_get.return_value.json.return_value = {
            "data": [
                {"id": "11", "text": f"{SHORT} {profile_short} {pdf}", "entities": {"urls": [
                    {"url": SHORT, "expanded_url": NEWS}, {"url": profile_short, "expanded_url": profile},
                ]}},
                {"id": "12", "text": other_short},
            ],
            "meta": {"next_token": "unit-next-page"},
        }
        with patch.object(main, "register_tweet_urls") as register:
            with patch.object(scraper.ArticleScraper, "scrape_article") as scrape:
                response = self.client.get("/ingest")
        scrape.assert_not_called()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["article_urls"], {
            "tweets_with_url_entities": 1, "expanded_urls_received": 2, "tweets_only_tco": 1,
        })
        self.assertEqual(register.call_args_list, [
            call(cursor, "11", [scraper.TweetLink(SHORT, NEWS), scraper.TweetLink(profile_short, profile),
                                scraper.TweetLink(pdf, pdf)]),
            call(cursor, "12", [scraper.TweetLink(other_short, other_short)]),
        ])
        http_get.assert_called_once()
        self.assertEqual(http_get.call_args.args[0], main.TWEETS_URL)

    @patch.dict("os.environ", {"INGEST_JOB_TOKEN": ""})
    @patch.object(main, "get_db_connection")
    def test_existing_replies_route_uses_its_own_lock_and_state(self, get_db):
        cursor = get_db.return_value.cursor.return_value
        cursor.fetchone.side_effect = [(1,), None, None, (1,)]
        cursor.fetchall.return_value = []
        with patch.object(main, "register_tweet_urls") as register:
            response = self.client.get("/ingest_replies")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["msg"], "No hay tweets root recientes para procesar")
        cursor.execute.assert_any_call("SELECT v FROM ingest_state WHERE k=%s", ("replies_last_root_id",))
        cursor.execute.assert_called_with("SELECT RELEASE_LOCK(%s)", ("netvora_replies_ingest",))
        register.assert_not_called()


if __name__ == "__main__":
    unittest.main()
