from flask import Blueprint, jsonify, request
from mysql.connector import Error as MySQLError
from werkzeug.exceptions import HTTPException

from summaries import (
    generate_category_summary,
    generate_custom_summary,
    generate_general_summary,
    generate_pending_article_summaries,
    generate_topic_summary,
)


summary_routes = Blueprint("summaries", __name__, url_prefix="/summaries")
SUMMARY_HTTP_MAX_SECONDS = 20
SUMMARY_ERROR_STATUS = {
    "BUSY": 409,
    "TOPIC_NOT_FOUND": 404,
    "NO_SOURCES": 422,
    "GEMINI_QUOTA": 429,
    "GEMINI_BILLING": 503,
    "GEMINI_CONFIGURATION": 503,
    "EXECUTION_LIMIT": 503,
    "MYSQL_DATA_ERROR": 500,
}


@summary_routes.before_request
def _log_summary_request():
    print(f"[summaries] {request.method} {request.path} inicio", flush=True)


def _summary_params(required=(), optional=()):
    payload = request.get_json(silent=True) if request.get_data(cache=True) else {}
    if not isinstance(payload, dict) or set(payload) - set(required + optional):
        raise ValueError("Envía un objeto JSON con los parámetros del resumen")
    missing = set(required) - set(payload)
    if missing:
        raise ValueError("Faltan parámetros: " + ", ".join(sorted(missing)))
    return payload


def _summary_response(result):
    status = 200 if result["ok"] else SUMMARY_ERROR_STATUS.get(result["error"], 502)
    fields = (
        "error", "gemini_http_status", "gemini_requests", "reused", "articles_pending",
        "summaries_generated", "summaries_failed", "errors", "time_limit_reached",
    )
    details = {key: result[key] for key in fields if key in result}
    print(f"[summaries] {request.path} http={status} {details}", flush=True)
    return jsonify(result), status


@summary_routes.errorhandler(ValueError)
def _invalid_summary(error):
    print(f"[summaries] {request.path} http=400 parametros_o_configuracion_invalidos", flush=True)
    return jsonify(ok=False, error=str(error)), 400


@summary_routes.errorhandler(MySQLError)
def _summary_mysql_error(error):
    print(f"[summaries] {request.path} http=500 error=MYSQL_ERROR errno={error.errno}", flush=True)
    return jsonify(ok=False, error="MYSQL_ERROR"), 500


@summary_routes.errorhandler(Exception)
def _unexpected_summary_error(error):
    if isinstance(error, HTTPException):
        return error
    print(f"[summaries] {request.path} http=500 error=SUMMARY_ERROR tipo={type(error).__name__}", flush=True)
    return jsonify(ok=False, error="SUMMARY_ERROR"), 500


@summary_routes.get("/articles")
def summary_articles_handler():
    params = _summary_params(optional=("limit",))
    return _summary_response(generate_pending_article_summaries(**params, max_seconds=SUMMARY_HTTP_MAX_SECONDS))


@summary_routes.post("/topic")
def summary_topic_handler():
    params = _summary_params(("topic_id",), ("start_date", "end_date"))
    return _summary_response(generate_topic_summary(**params, max_seconds=SUMMARY_HTTP_MAX_SECONDS))


@summary_routes.post("/category")
def summary_category_handler():
    params = _summary_params(("category", "start_date", "end_date"))
    return _summary_response(generate_category_summary(**params, max_seconds=SUMMARY_HTTP_MAX_SECONDS))


@summary_routes.post("/general")
def summary_general_handler():
    params = _summary_params(("start_date", "end_date"))
    return _summary_response(generate_general_summary(**params, max_seconds=SUMMARY_HTTP_MAX_SECONDS))


@summary_routes.post("/custom")
def summary_custom_handler():
    params = _summary_params(("filters",))
    return _summary_response(generate_custom_summary(**params, max_seconds=SUMMARY_HTTP_MAX_SECONDS))
