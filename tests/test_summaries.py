import json
import os
import re
import unittest
from contextlib import nullcontext
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from google.genai.errors import ClientError
from mysql.connector import DataError

from crearTopic import get_gemini_client
import summaries


DAY = "2026-11-26"
LONG_CONTENT = (
    "El municipio anunció una nueva ruta de transporte para los barrios del norte. "
    "El proyecto incluye accesos para personas con movilidad reducida, ajustes de horario "
    "y una etapa de evaluación con las juntas vecinales. La autoridad informó que el servicio "
    "comenzará el lunes y que la tarifa se mantendrá durante la primera etapa del programa."
)
ARTICLE_REPLY = {
    "summary": "El municipio anunció una nueva ruta de transporte para los barrios del norte.",
    "key_points": ["La ruta comienza el lunes.", "Se mantienen las tarifas.", "Habrá una evaluación vecinal."],
}


def failed_response(code, message):
    return ClientError(code, {"error": {"code": code, "message": message}})


def tweet(tweetid, text, category, medium="Medio", when=None):
    return {
        "tweetid": tweetid, "text": text, "categoria": category, "medium": medium,
        "handle": medium.lower(), "account_id": 1 if medium == "Medio" else 2,
        "url": f"https://x.com/i/web/status/{tweetid}",
        "created": when or datetime(2026, 11, 26, 12),
    }


def article(article_id, *, status="PENDING", scraping="COMPLETED", content=LONG_CONTENT):
    return {
        "article_id": article_id, "title": "Nueva ruta", "domain": "medio.example",
        "published_at": datetime(2026, 11, 26, 15), "content": content, "content_hash": str(article_id),
        "url": f"https://medio.example/noticia/{article_id}", "canonical_url": None,
        "summary": None, "key_points": None, "summary_attempts": 0, "summary_status": status,
        "summary_started_at": None, "summary_retry_at": None, "scraping_status": scraping, "summary_error": None,
    }


class SummaryStore:
    """Emulate the row transitions and query filters without a live database."""

    def __init__(self):
        self.articles = {1: article(1, status="COMPLETED")}
        self.articles[1].update(summary=ARTICLE_REPLY["summary"], key_points=json.dumps(ARTICLE_REPLY["key_points"]))
        self.tweets = {
            101: tweet(101, "Ruta municipal", "Política"),
            102: tweet(102, "Nueva ruta", "Política"),
            103: tweet(103, "Presupuesto municipal", "Economía", "Economía Noticias"),
            104: tweet(104, "Convocatoria barrial 100% local", "Política"),
            105: tweet(105, "En el día siguiente", "Política", when=datetime(2026, 11, 27, 0)),
        }
        self.topics = {1: "Transporte público", 2: "Presupuesto"}
        self.topic_tweets = {101: {1}, 102: {1}, 103: {2}, 105: {1}}
        self.links = [(101, 1), (102, 1)]
        self.summaries = {}
        self.busy = False
        self.in_transaction = False

    def connect(self):
        return self

    def cursor(self, *, dictionary):
        assert dictionary
        return self

    def _matching_tweets(self, query, params):
        rows = list(self.tweets.values())
        values = iter(params)
        if "t.created >= %s" in query:
            start = next(values)
            rows = [row for row in rows if row["created"] >= start]
        if "t.created < %s" in query:
            end = next(values)
            rows = [row for row in rows if row["created"] < end]
        categories = re.search(r"LOWER\(TRIM\(t.categoria\)\) IN \(([^)]+)\)", query)
        if categories:
            allowed = {next(values) for _ in range(categories[1].count("%s"))}
            rows = [row for row in rows if row["categoria"].lower() in allowed]
        topics = re.search(r"tt.topic_id IN \(([^)]+)\)", query)
        if topics:
            allowed = {next(values) for _ in range(topics[1].count("%s"))}
            rows = [row for row in rows if self.topic_tweets.get(row["tweetid"], set()) & allowed]
        media = re.search(r"LOWER\(TRIM\(u.TweetUser\)\) IN \(([^)]+)\)", query)
        accounts = re.search(r"u.idTweetUser IN \(([^)]+)\)", query)
        handles, names, ids = set(), set(), set()
        if media:
            handles = {next(values) for _ in range(media[1].count("%s"))}
            names = {next(values) for _ in range(media[1].count("%s"))}
        if accounts:
            ids = {next(values) for _ in range(accounts[1].count("%s"))}
        if media or accounts:
            rows = [
                row for row in rows if row["handle"] in handles or row["medium"].lower() in names
                or row["account_id"] in ids
            ]
        def term():
            return next(values)[1:-1].replace("!%", "%").replace("!_", "_").replace("!!", "!")

        has_search = " AND LOWER(t.text) LIKE %s" in query
        if has_search:
            text_search = term()
            rows = [row for row in rows if text_search in row["text"].lower()]
        keyword_count = query.count("LOWER(t.text) LIKE %s") - has_search
        if keyword_count:
            terms = [term() for _ in range(keyword_count)]
            rows = [row for row in rows if any(term in row["text"].lower() for term in terms)]
        if "AND t.tweetid IN (" in query:
            ids = {int(value) for value in values}
            rows = [row for row in rows if row["tweetid"] in ids]
        rows.sort(key=lambda row: (row["created"], row["tweetid"]), reverse=True)
        if query.endswith("LIMIT %s"):
            rows = rows[:next(values)]
        return rows

    def _pending(self, max_attempts, stale):
        now = summaries._utc_now()
        return [
            row for row in sorted(self.articles.values(), key=lambda row: row["article_id"])
            if row["scraping_status"] == "COMPLETED" and (row["content"] or "").strip()
            and row["summary_attempts"] < max_attempts and (
                row["summary_status"] == "PENDING"
                or (row["summary_status"] == "FAILED" and row["summary_retry_at"] and row["summary_retry_at"] <= now)
                or (row["summary_status"] == "PROCESSING"
                    and (row["summary_started_at"] is None or row["summary_started_at"] <= stale))
            )
        ]

    def execute(self, sql, params=()):
        query = " ".join(sql.split())
        assert query.count("%s") == len(params), query
        self.rows, self.rowcount = [], 0
        if query.startswith("SELECT GET_LOCK"):
            self.rows = [{"acquired": int(not self.busy)}]
            return
        self.in_transaction = True
        if query.startswith("SELECT COUNT(*) AS total FROM web_articles"):
            self.rows = [{"total": len(self._pending(*params))}]
        elif query.startswith("SELECT article_id, title, domain, published_at"):
            size, max_attempts, stale, limit = params
            self.rows = [{**row, "content": row["content"][:size]} for row in self._pending(max_attempts, stale)[:limit]]
        elif query.startswith("UPDATE web_articles"):
            self._update_article(query, params)
        elif query.startswith("SELECT topic_name, main_target, target_description FROM topics"):
            if params[0] in self.topics:
                self.rows = [{"topic_name": self.topics[params[0]], "main_target": None, "target_description": "Contexto"}]
        elif query.startswith("SELECT COUNT(*) AS total FROM Tweets"):
            self.rows = [{"total": len(self._matching_tweets(query, params))}]
        elif query.startswith(("SELECT t.tweetid, t.text,", "SELECT t.tweetid FROM Tweets")):
            self.rows = self._matching_tweets(query, params)
        elif query.startswith("SELECT ta.tweetid, a.article_id"):
            ids = set(params[:-1])
            self.rows = [
                {**self.articles[article_id], "tweetid": tweetid}
                for tweetid, article_id in self.links if tweetid in ids
                and self.articles[article_id]["scraping_status"] == "COMPLETED"
                and self.articles[article_id]["summary_status"] == "COMPLETED"
            ][:params[-1]]
        elif query.startswith("SELECT * FROM generated_summaries WHERE scope_hash"):
            self.rows = [row.copy() for row in self.summaries.values() if row["scope_hash"] == params[0]]
        elif query.startswith("SELECT id, summary_type, topic_id, category"):
            kinds = set(params[:-3])
            lo, hi, cap = params[-3:]
            self.rows = sorted(
                [row.copy() for row in self.summaries.values()
                 if row["status"] == "COMPLETED" and row["summary_type"] in kinds
                 and row["start_date"] == lo and row["end_date"] == hi],
                key=lambda row: (row["summary_type"] != "CATEGORY", -row["total_tweets"], row["id"]),
            )[:cap]
        elif query.startswith("INSERT INTO generated_summaries"):
            self.lastrowid = len(self.summaries) + 1
            fields = ("summary_type", "topic_id", "category", "start_date", "end_date", "filters",
                      "scope_hash", "input_hash", "model", "prompt_version", "attempts")
            self.summaries[self.lastrowid] = dict(
                zip(fields, params), id=self.lastrowid, status="PROCESSING",
                processing_started_at=summaries._utc_now(), summary=None, error_message=None, next_retry_at=None,
            )
            self.rowcount = 1
        elif query.startswith("UPDATE generated_summaries"):
            self._update_summary(query, params)
        else:
            raise AssertionError(query)

    def _update_article(self, query, params):
        if "summary_error = 'PROCESSING_EXPIRED'" in query:
            maximum, stale, limit = params
            for row in self.articles.values():
                if (row["summary_status"] == "PROCESSING" and row["summary_attempts"] >= maximum
                        and row["summary_started_at"] <= stale and self.rowcount < limit):
                    row.update(summary_status="FAILED", summary_error="PROCESSING_EXPIRED", summary_retry_at=None)
                    self.rowcount += 1
            return
        if "SET summary_status = 'SKIPPED'" in query:
            article_id, status, attempt = params
            row = self.articles[article_id]
            assert (row["summary_status"], row["summary_attempts"]) == (status, attempt)
            row.update(summary_status="SKIPPED", summary_error="INSUFFICIENT_CONTENT")
        elif "SET summary_status = 'PROCESSING'" in query:
            attempt, article_id, previous_status, previous_attempt, content_hash = params
            row = self.articles[article_id]
            assert (row["summary_status"], row["summary_attempts"], row["content_hash"]) == (
                previous_status, previous_attempt, content_hash,
            )
            row.update(summary_status="PROCESSING", summary_attempts=attempt, summary_error=None, summary_retry_at=None)
        elif "SET summary = %s" in query:
            text, points, model, version, article_id, attempt, content_hash = params
            row = self.articles[article_id]
            assert (row["summary_status"], row["summary_attempts"], row["content_hash"]) == ("PROCESSING", attempt, content_hash)
            row.update(
                summary=text, key_points=points, summary_model=model, summary_prompt_version=version,
                summary_status="COMPLETED",
            )
        else:
            status, error, retry_at, article_id, attempt = params
            row = self.articles[article_id]
            assert row["summary_attempts"] == attempt and row["summary_status"] == "PROCESSING"
            row.update(summary_status=status, summary_error=error, summary_retry_at=retry_at)
        self.rowcount = 1

    def _update_summary(self, query, params):
        row = self.summaries[params[-1]]
        if "SET input_hash =" in query:
            digest, model, version, attempts, _ = params
            row.update(
                input_hash=digest, model=model, prompt_version=version, status="PROCESSING",
                attempts=attempts, processing_started_at=summaries._utc_now(),
            )
        elif "SET filters =" in query:
            fields = ("filters", "title", "summary", "key_points", "main_narratives", "timeline", "article_ids",
                      "total_articles", "total_tweets", "total_sources", "model", "generated_at")
            assert row["status"] == "PROCESSING"
            row.update(zip(fields, params[:-1]))
            row.update(status="COMPLETED", error_message=None, next_retry_at=None)
        elif "SET status = 'FAILED'" in query:
            assert row["status"] == "PROCESSING"
            row.update(status="FAILED", error_message=params[0], next_retry_at=params[1])
        elif "SET total_tweets" in query:
            row["total_tweets"] = params[0]
        else:
            raise AssertionError(query)
        self.rowcount = 1

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows

    def commit(self):
        self.in_transaction = False

    rollback = commit
    close = commit


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.store = SummaryStore()
        self.requests = []
        self.responses = []
        self.patch(summaries, "get_db_connection", self.store.connect)
        self.patch(summaries, "get_gemini_client", lambda _: nullcontext(SimpleNamespace(
            models=SimpleNamespace(generate_content=self.generate_content),
        )))
        settings = patch.dict(os.environ, {
            "GEMINI_SUMMARY_MODEL": "gemini-2.5-flash", "SUMMARY_SECONDS": "60", "SUMMARY_MAX_CALLS": "10",
        })
        settings.start()
        self.addCleanup(settings.stop)

    def patch(self, module, name, value):
        handle = patch.object(module, name, value)
        handle.start()
        self.addCleanup(handle.stop)

    def generate_content(self, *, model, contents, config):
        self.assertFalse(self.store.in_transaction)
        self.assertEqual(config.http_options.retry_options.attempts, 1)
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.thinking_config.thinking_budget, 0)
        payload = json.loads(contents)
        self.requests.append(payload)
        if self.responses:
            result = self.responses.pop(0)
            if isinstance(result, Exception):
                raise result
        elif "sources" in payload:
            ids = [source["id"] for source in payload["sources"]]
            result = {
                "title": "Noticias del periodo", "summary": ARTICLE_REPLY["summary"], "summary_source_ids": ids,
                "key_points": [{"text": "La ruta comienza el lunes.", "source_ids": ids[:1]}],
                "main_narratives": [], "timeline": [{"date": DAY, "event": "Anuncio", "source_ids": ids[:1]}],
            }
        else:
            result = ARTICLE_REPLY
        return SimpleNamespace(text=result if isinstance(result, str) else json.dumps(result), model_version=model)

    def test_article_storage_excludes_blocked_and_completed(self):
        self.store.articles.update({
            2: article(2), 3: article(3, scraping="SKIPPED"),
            4: article(4, content=""), 5: article(5, content="Demasiado breve."),
        })
        result = summaries.generate_pending_article_summaries()
        self.assertEqual((result["articles_pending"], result["gemini_requests"]), (2, 1))
        self.assertEqual((result["summaries_generated"], result["summaries_skipped"]), (1, 1))
        saved = self.store.articles[2]
        self.assertEqual(saved["summary_status"], "COMPLETED")
        self.assertEqual(json.loads(saved["key_points"]), ARTICLE_REPLY["key_points"])
        self.assertEqual(saved["summary_model"], "gemini-2.5-flash")
        self.assertEqual(saved["summary_prompt_version"], summaries.ARTICLE_VERSION)
        self.assertEqual(self.requests[0]["domain"], "medio.example")
        self.assertEqual(self.requests[0]["published_at"], "2026-11-26T15:00:00")
        self.assertEqual(summaries.generate_pending_article_summaries()["gemini_requests"], 0)

    def test_malformed_output_continues_and_reporting_of_excerpt(self):
        self.store.articles.update({2: article(2), 3: article(3, content=LONG_CONTENT * 100)})
        self.responses.append("no es JSON")
        result = summaries.generate_pending_article_summaries()
        self.assertEqual((result["summaries_failed"], result["summaries_generated"], result["articles_sampled"]), (1, 1, 1))
        self.assertEqual(result["errors"], {"GEMINI_INVALID_RESPONSE": 1})
        self.assertEqual(self.store.articles[2]["summary_error"], "GEMINI_INVALID_RESPONSE")
        self.assertIsNotNone(self.store.articles[2]["summary_retry_at"])
        self.assertTrue(self.requests[1]["selection_limited"])
        self.assertLessEqual(len(self.requests[1]["content"]), summaries.MAX_ARTICLE_CHARS)

    def test_quota_and_billing_stop_before_next_article(self):
        for status, message, expected in (
            (429, "RESOURCE_EXHAUSTED", "GEMINI_QUOTA"), (400, "Billing is disabled", "GEMINI_BILLING"),
        ):
            with self.subTest(status=status):
                self.store.articles.update({2: article(2), 3: article(3)})
                self.responses.append(failed_response(status, message))
                result = summaries.generate_pending_article_summaries()
                self.assertFalse(result["ok"])
                self.assertEqual((result["gemini_requests"], result["error"]), (1, expected))
                self.assertEqual(self.store.articles[3]["summary_attempts"], 0)

    def test_failure_storing_an_article_does_not_stop_batch(self):
        self.store.articles.update({2: article(2), 3: article(3)})
        update = self.store._update_article

        def fail_one(query, params):
            if "SET summary = %s" in query and params[4] == 2:
                raise DataError("Simulated MySQL data error")
            return update(query, params)

        with patch.object(self.store, "_update_article", side_effect=fail_one):
            result = summaries.generate_pending_article_summaries()
        self.assertEqual((result["summaries_failed"], result["summaries_generated"]), (1, 1))
        self.assertEqual(self.store.articles[2]["summary_error"], "MYSQL_DATA_ERROR")
        self.assertEqual(self.store.articles[3]["summary_status"], "COMPLETED")

    def test_article_limit_and_stale_processing(self):
        stale = summaries._utc_now() - timedelta(minutes=20)
        self.store.articles.update({2: article(2, status="PROCESSING"), 3: article(3, status="PROCESSING"), 4: article(4)})
        self.store.articles[2].update(summary_attempts=1, summary_started_at=stale)
        self.store.articles[3].update(summary_attempts=3, summary_started_at=stale)
        with patch.dict(os.environ, {"SUMMARY_MAX_CALLS": "1"}):
            result = summaries.generate_pending_article_summaries()
        self.assertEqual(result["gemini_requests"], 1)
        self.assertTrue(result["call_limit_reached"])
        self.assertEqual(self.store.articles[2]["summary_attempts"], 2)
        self.assertEqual(self.store.articles[3]["summary_error"], "PROCESSING_EXPIRED")
        self.assertEqual(self.store.articles[4]["summary_status"], "PENDING")

    def test_topic_deduplicates_reuses_and_refreshes_on_new_input(self):
        original = summaries.generate_topic_summary(1, DAY, DAY)
        self.assertEqual(original["article_ids"], [1])
        self.assertEqual(original["total_tweets"], 2)
        self.assertEqual([source["id"] for source in self.requests[0]["sources"]], ["article:1"])
        self.assertTrue(summaries.generate_topic_summary(1, DAY, DAY)["reused"])
        self.store.tweets[106] = tweet(106, "Anuncio complementario", "Política")
        self.store.topic_tweets[106] = {1}
        changed = summaries.generate_topic_summary(1, DAY, DAY)
        self.assertFalse(changed["reused"])
        self.assertEqual(changed["id"], original["id"])
        self.assertNotEqual(changed["input_hash"], original["input_hash"])
        self.assertIn("tweet:106", changed["summary_source_ids"])
        self.assertEqual(len(self.requests), 2)

    def test_topic_without_article_and_unknown_topic(self):
        self.store.articles[2] = article(2, scraping="SKIPPED")
        self.store.links.append((103, 2))
        result = summaries.generate_topic_summary(2)
        self.assertEqual(result["article_ids"], [])
        self.assertEqual(result["sources"][0]["id"], "tweet:103")
        self.assertEqual(result["sources"][0]["url"], self.store.tweets[103]["url"])
        self.assertEqual(summaries.generate_topic_summary(999)["error"], "TOPIC_NOT_FOUND")
        self.assertEqual(len(self.requests), 1)

    def test_category_and_general_prefer_compatible_rollups(self):
        topic = summaries.generate_topic_summary(1, DAY, DAY)
        summaries.generate_topic_summary(2, "2026-11-25", DAY)
        category = summaries.generate_category_summary("Política", DAY, DAY)
        self.assertEqual(category["summary_type"], "CATEGORY")
        self.assertEqual([source["id"] for source in self.requests[-1]["sources"]], [f"summary:{topic['id']}", "tweet:104"])
        general = summaries.generate_general_summary(DAY, DAY)
        self.assertEqual(general["summary_type"], "GENERAL")
        self.assertEqual([source["id"] for source in self.requests[-1]["sources"]], [f"summary:{category['id']}", "tweet:103"])
        self.assertEqual(general["article_ids"], [1])
        self.assertEqual(general["total_tweets"], 4)
        self.assertTrue(summaries.generate_general_summary(DAY, DAY)["reused"])

    def test_mixed_category_rollup_is_not_used(self):
        self.store.topic_tweets[103] = {1}
        summaries.generate_topic_summary(1, DAY, DAY)
        result = summaries.generate_category_summary("Política", DAY, DAY)
        self.assertTrue(result["ok"])
        self.assertEqual([source["id"] for source in self.requests[-1]["sources"]], ["article:1", "tweet:104"])

    def test_custom_search_normalizes_filters_and_escapes_like(self):
        self.store.tweets[106] = tweet(106, "Convocatoria barrial", "Política")
        filters = {"start_date": DAY, "end_date": DAY, "categories": ["Política"], "media": ["@Medio"],
                   "search": "100%", "keywords": ["convocatoria", "barrial"]}
        first = summaries.generate_custom_summary(filters)
        self.assertEqual(first["summary_type"], "CUSTOM")
        self.assertEqual([source["id"] for source in first["sources"]], ["tweet:104"])
        filters.update(media=["@MEDIO", "medio"], keywords=["barrial", "convocatoria"])
        self.assertTrue(summaries.generate_custom_summary(filters)["reused"])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(summaries._like("10%_!"), "%10!%!_!!%")

    def test_custom_supports_existing_account_ids(self):
        result = summaries.generate_custom_summary({"start_date": DAY, "end_date": DAY, "media": [2]})
        self.assertEqual(result["total_tweets"], 1)
        self.assertEqual(result["summary_source_ids"], ["tweet:103"])

    def test_unknown_citation_and_url_fail_validation(self):
        output = {
            "title": "Título", "summary": "Resumen", "summary_source_ids": ["article:9999"],
            "key_points": [{"text": "Hecho", "source_ids": ["article:1"]}], "main_narratives": [], "timeline": [],
        }
        self.responses.append(output)
        result = summaries.generate_topic_summary(1, DAY, DAY)
        self.assertEqual(result["error"], "GEMINI_INVALID_RESPONSE")
        self.assertEqual(summaries.generate_topic_summary(1, DAY, DAY)["gemini_requests"], 0)
        row = self.store.summaries[result["id"]]
        self.assertIsNone(row["summary"])
        row["next_retry_at"] = summaries._utc_now() - timedelta(seconds=1)
        output.update(summary_source_ids=["article:1"], summary="Falso enlace https://inventado.example")
        self.responses.append(output)
        self.assertEqual(summaries.generate_topic_summary(1, DAY, DAY)["error"], "GEMINI_INVALID_RESPONSE")

    def test_consolidated_processing_recovers_with_existing_row(self):
        self.responses.append(failed_response(503, "UNAVAILABLE"))
        first = summaries.generate_general_summary(DAY, DAY)
        row = self.store.summaries[first["id"]]
        row.update(status="PROCESSING", processing_started_at=summaries._utc_now())
        self.assertEqual(summaries.generate_general_summary(DAY, DAY)["error"], "BUSY")
        row["processing_started_at"] -= timedelta(minutes=20)
        result = summaries.generate_general_summary(DAY, DAY)
        self.assertEqual(result["id"], first["id"])
        self.assertEqual(self.store.summaries[result["id"]]["attempts"], 2)

    def test_sample_limits_are_reported_and_kept_when_reused(self):
        with patch.object(summaries, "MAX_SOURCES", 2):
            result = summaries.generate_general_summary(DAY, DAY)
            self.assertTrue(result["selection_limited"])
            self.assertEqual(result["sources_sent"], 2)
            self.assertEqual(result["total_tweets"], 4)
            cached = summaries.generate_general_summary(DAY, DAY)
            self.assertTrue(cached["selection_limited"])
            self.assertTrue(cached["reused"])

    def test_validation_and_empty_period_precede_api_calls(self):
        for function, args in (
            (summaries.generate_category_summary, ("Política", "2026-02-30", DAY)),
            (summaries.generate_general_summary, (DAY, "2026-11-25")),
            (summaries.generate_topic_summary, (False,)),
            (summaries.generate_custom_summary, ({"unknown": "x"},)),
            (summaries.generate_custom_summary, ({"media": "Medio"},)),
            (summaries.generate_custom_summary, ({},)),
            (summaries.generate_pending_article_summaries, (0,)),
        ):
            with self.subTest(function=function.__name__), self.assertRaises(ValueError):
                function(*args)
        result = summaries.generate_general_summary("2026-11-20", "2026-11-20")
        self.assertEqual(result["error"], "NO_SOURCES")
        self.assertFalse(self.requests)

    def test_locked_scope_skips_generation(self):
        self.store.busy = True
        self.assertEqual(summaries.generate_pending_article_summaries()["error"], "BUSY")
        self.assertEqual(summaries.generate_topic_summary(1, DAY, DAY)["error"], "BUSY")
        self.assertFalse(self.requests)

    def test_flask_endpoints_use_existing_token_and_return_job_results(self):
        from main import app

        client = app.test_client()
        with patch.dict(os.environ, {"INGEST_JOB_TOKEN": ""}):
            self.assertEqual(client.post("/summaries/articles").status_code, 503)
            self.assertEqual(client.head("/articles/discover").status_code, 204)

        self.store.articles[2] = article(2)
        with patch.dict(os.environ, {"INGEST_JOB_TOKEN": "summary-test-token"}):
            headers = {"Authorization": "Bearer summary-test-token"}
            self.assertEqual(client.post("/summaries/articles").status_code, 401)
            self.assertEqual(client.get("/summaries/topic", headers=headers).status_code, 405)
            invalid = client.post("/summaries/general", json={"start_date": DAY}, headers=headers)
            self.assertEqual(invalid.status_code, 400)
            self.assertFalse(self.requests)
            period = {"start_date": DAY, "end_date": DAY}
            for path, payload in (
                ("articles", {"limit": 1}),
                ("topic", {"topic_id": 1, **period}),
                ("category", {"category": "Política", **period}),
                ("general", period),
                ("custom", {"filters": {"categories": ["Economía"], **period}}),
            ):
                response = client.post("/summaries/" + path, json=payload, headers=headers)
                self.assertEqual(response.status_code, 200, response.json)
                self.assertTrue(response.json["ok"])
            response = client.post("/summaries/general", json=period, headers=headers)
            self.assertTrue(response.json["reused"])
            self.assertEqual(response.json["gemini_requests"], 0)


class GeminiSdkTest(unittest.TestCase):
    def test_request_contract_and_quota_does_not_trigger_sdk_retries(self):
        try:
            import httpx2 as transport
        except ImportError:
            import httpx as transport

        requests_sent = []

        def respond(request):
            requests_sent.append(json.loads(request.content))
            return transport.Response(429, json={
                "error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exhausted"},
            })

        with transport.Client(transport=transport.MockTransport(respond)) as http:
            options = summaries._api_options(2)
            options.httpx_client = http
            with patch.dict(os.environ, {"GEMINI_API_KEY": "mock-gemini-key"}):
                with get_gemini_client(options) as client:
                    with self.assertRaises(ClientError):
                        summaries.generate_article_summary(article(1), client=client, model="gemini-2.5-flash", timeout=2)

        self.assertEqual(len(requests_sent), 1)
        config = requests_sent[0]["generationConfig"]
        self.assertEqual(config["responseMimeType"], "application/json")
        thinking = config["thinkingConfig"]
        self.assertEqual(thinking.get("thinkingBudget", thinking.get("thinking_budget")), 0)


if __name__ == "__main__":
    unittest.main()
