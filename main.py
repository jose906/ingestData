from database import get_db_connection
# main.py
import os
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, request
import requests
import sklearn
import MLModel
from mysql.connector import Error
import time
import hmac
import base64
from selectTopics import classify_tweets
from crearTopic import insertar_nuevos_topicos
from stance import (
    ejecutar_stance_por_target,
    ejecutar_targets_topicos,
    ejecutar_stance_por_topico
)
from web_scraper import (
    discover_historical_urls,
    extract_external_urls,
    process_pending_articles,
    recover_historical_urls,
    register_tweet_urls,
)



# ================== CONFIG ==================

  



def x_headers():
    token = os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        raise RuntimeError("Falta TWITTER_BEARER_TOKEN")
    return {"Authorization": f"Bearer {token}"}


TWEETS_URL = "https://api.twitter.com/2/tweets/search/recent"
TWEET_LOOKUP_URL = "https://api.x.com/2/tweets"
#TWEETS_URL = "https://api.twitter.com/2/users/{}/tweets"


MAX_USERS_PER_RUN = 5
MAX_RESULTS_PER_CALL = 100  # máximo permitido por endpoint
REQUEST_SLEEP_SECONDS = 0.2 

BOLIVIA_TZ = timezone(timedelta(hours=-4))

app = Flask(__name__)

@app.before_request
def authorize_job():
    if request.endpoint not in {
        "ingest_handler", "ingest_replies_handler", "reprocess_handler",
        "ejecutar_stance_posts_handler", "ejecutar_stance_topics_handler",
        "select_topics", "create_topics", "discover_articles_handler", "scrape_articles_handler",
        "recover_article_urls_handler",
    }:
        return None

    expected = os.getenv("INGEST_JOB_TOKEN")
    if not expected:
        if request.endpoint == "recover_article_urls_handler":
            return jsonify(ok=False, error="Configura INGEST_JOB_TOKEN para recuperar URLs"), 503
        return None
    authorization = request.headers.get("Authorization", "")
    if not hmac.compare_digest(authorization.encode(), f"Bearer {expected}".encode()):
        return jsonify({"ok": False, "error": "No autorizado"}), 401
    return None


def rollback_job(conn):
    try:
        conn.rollback()
    except Error:
        app.logger.warning("Fallo revirtiendo transacción MySQL", exc_info=True)


def close_job_connection(conn, cursor, lock_name, acquired):
    if cursor is not None:
        if acquired:
            try:
                cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
                cursor.fetchone()
            except Exception:
                app.logger.warning("Fallo liberando %s", lock_name, exc_info=True)
        try:
            cursor.close()
        except Exception:
            app.logger.warning("Fallo cerrando cursor MySQL", exc_info=True)

    if conn is not None:
        try:
            conn.close()
        except Exception:
            app.logger.warning("Fallo cerrando conexión MySQL", exc_info=True)


# ================== FUNCIONES BD ==================


# ================== INSERT TWEETS ==================

def insert_or_update_tweet(cur, tweet, tweetuser_pk_id, *, url_metrics=None):
    tweet_id = tweet["id"]
    links = extract_external_urls(tweet, metrics=url_metrics)
    note_tweet = tweet.get("note_tweet") or {}
    text = note_tweet.get("text") or tweet.get("text", "")
    created_at = tweet.get("created_at")

    created_dt = None

    if created_at:
        try:
            created_dt = (
                datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                )
                .astimezone(BOLIVIA_TZ)
                .replace(tzinfo=None)
            )
        except Exception as e:
            print(f"Error convirtiendo fecha tweet {tweet_id}: {e}")
            created_dt = None

    url = f"https://twitter.com/i/web/status/{tweet_id}"

    # --------------------------------------------------
    # 1. COMPROBAR SI EL TWEET YA EXISTE
    # --------------------------------------------------

    cur.execute(
        """
        SELECT tweetid
        FROM Tweets
        WHERE tweetid = %s
        LIMIT 1
        """,
        (tweet_id,)
    )

    existe = cur.fetchone()

    # --------------------------------------------------
    # 2. SI YA EXISTE, SOLO ACTUALIZAMOS DATOS BÁSICOS
    # --------------------------------------------------

    if existe:

        cur.execute(
            """
            UPDATE Tweets
            SET
                text = %s,
                created = %s,
                url = %s,
                TweetUser_idTweetUser = %s
            WHERE tweetid = %s
            """,
            (
                text,
                created_dt,
                url,
                int(tweetuser_pk_id),
                tweet_id
            )
        )

        register_tweet_urls(cur, tweet_id, links)
        return 0

    # --------------------------------------------------
    # 3. SI ES NUEVO, RECIÉN EJECUTAMOS ML
    # --------------------------------------------------

    # --------------------------------------------------
# 3. EJECUTAR ML
#    Si falla, NO perdemos el tweet
# --------------------------------------------------

    ml_ok = True

    try:
        sentimiento = MLModel.get_sentiment(text)[0]
    except Exception as e:
        print(
            f"⚠️ Error sentimiento tweet {tweet_id}: {e}"
        )
        sentimiento = ""
        ml_ok = False

    try:
        categoria = MLModel.predecir_categoria(text)[0]
    except Exception as e:
        print(
            f"⚠️ Error categoría tweet {tweet_id}: {e}"
        )
        categoria = ""
        ml_ok = False

    # --------------------------------------------------
    # 4. GUARDAR EL TWEET IGUALMENTE
    # --------------------------------------------------

    cur.execute(
        """
        INSERT INTO Tweets
        (
            tweetid,
            text,
            created,
            url,
            sentimiento,
            categoria,
            Lugar,
            Persona,
            Organizacion,
            Locacion,
            Otros,
            TweetUser_idTweetUser, 
            ml_procesado
        )
        VALUES
        (
            %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s
        )
        """,
        (
            tweet_id,
            text,
            created_dt,
            url,
            sentimiento,
            categoria,
            "",
            "",
            "",
            "",
            "",
            int(tweetuser_pk_id),
            1 if ml_ok else 0
        )
    )

    register_tweet_urls(cur, tweet_id, links)
    return 1



def reprocess_pending_tweets(limit=100):
    conn = None
    cur = None
    acquired = False

    procesados = 0
    errores = 0

    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
        "SELECT GET_LOCK('netvora_ml_reprocess', 0) AS acquired"
    )

        lock_result = cur.fetchone()

        acquired = bool(lock_result and lock_result["acquired"] == 1)
        if not acquired:
            return {
                "ok": True,
                "busy": True,
                "procesados": 0,
                "message": "Ya existe otra ejecución de reprocesamiento ML."
            }
        # Buscar solamente tweets pendientes
        cur.execute(
            """
            SELECT
                tweetid,
                text
            FROM Tweets
            WHERE ml_procesado = 0
            ORDER BY created ASC
            LIMIT %s
            """,
            (limit,)
        )

        tweets = cur.fetchall()

        if not tweets:
            return {
                "ok": True,
                "procesados": 0,
                "errores": 0,
                "message": "No hay tweets pendientes de procesamiento."
            }

        for tweet in tweets:

            tweet_id = tweet["tweetid"]
            text = tweet["text"] or ""

            try:
                sentimiento = MLModel.get_sentiment(text)[0]
                categoria = MLModel.predecir_categoria(text)[0]

            except Exception:
                errores += 1
                app.logger.exception("Error clasificando tweet %s", tweet_id)
                continue

            cur.execute(
                """
                UPDATE Tweets
                SET sentimiento = %s, categoria = %s, ml_procesado = 1
                WHERE tweetid = %s
                """,
                (sentimiento, categoria, tweet_id),
            )
            procesados += 1

        conn.commit()

        return {
            "ok": errores == 0,
            "procesados": procesados,
            "errores": errores,
            "total_encontrados": len(tweets),
            
        }

    except Exception:
        if conn is not None:
            rollback_job(conn)

        app.logger.exception("Fallo reprocesando tweets")

        return {
            "ok": False,
            "procesados": 0,
            "errores": errores,
            "error": "Falló el reprocesamiento"
        }

    finally:
        close_job_connection(conn, cur, "netvora_ml_reprocess", acquired)



# ================== TIME HELPERS ==================

def bolivia_day_start_utc_iso():
    """
    Inicio del día en Bolivia convertido a UTC en ISO Z.
    """
    now_bo = datetime.now(BOLIVIA_TZ)
    start_bo = now_bo.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_bo.astimezone(timezone.utc)
    return start_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")



def invalid_x_pagination(error_text):
    return any(field in error_text for field in ("next_token", "pagination_token"))


def parse_x_page(data):
    if not isinstance(data, dict) or "meta" not in data:
        raise ValueError("Respuesta de búsqueda de X incompleta")
    tweets = data.get("data", [])
    meta = data["meta"]
    if not isinstance(tweets, list) or not isinstance(meta, dict):
        raise ValueError("Formato de página de X inválido")
    if data.get("errors"):
        raise ValueError("X devolvió una página con errores")
    return tweets, meta


def fetch_tweets_for_user(username: str,last_tweetid=None,pagination_token=None):
    params = {
        "query": f"from:{username}",
        "max_results": 100,
        "tweet.fields": "created_at,text,entities,author_id,note_tweet",
    }
   
    # Si ya estamos recorriendo páginas anteriores,
    # continuamos exactamente desde esa página.
    # Siempre mantenemos el since_id original
    # durante todas las páginas de esta búsqueda.
    if last_tweetid not in (None, "", "0", "1", 0, 1):
        params["since_id"] = str(last_tweetid)

    # Si estamos continuando una búsqueda,
    # agregamos además el token de paginación.
    if pagination_token:
        params["next_token"] = pagination_token

    try:
        response = requests.get(
            TWEETS_URL,
            headers=x_headers(),
            params=params,
            timeout=20
        )

    except requests.RequestException as e:
        raise Exception(
            f"Error de conexión con X para @{username}: {str(e)}"
        )

    # Rate limit
    if response.status_code == 429:
        retry_after = response.headers.get("retry-after")

        return {
            "ok": False,
            "rate_limit": True,
            "tweets": [],
            "next_token": pagination_token,
            "error": f"Rate limit. retry_after={retry_after}"
        }

    # since_id inválido
    if response.status_code == 400:
        try:
            error_data = response.json()
            error_text = str(error_data)
        except Exception:
            error_text = response.text

        if pagination_token and invalid_x_pagination(error_text):
            return {"ok": False, "reset_pagination": True}

        if "since_id" in params and "since_id" in error_text and "must be a tweet id created after" in error_text:
            return {
                "ok": False,
                "rate_limit": False,
                "tweets": [],
                "next_token": None,
                "reset_since_id": True,
                "error": error_text
            }

    if response.status_code != 200:
        return {
            "ok": False,
            "rate_limit": False,
            "tweets": [],
            "next_token": pagination_token,
            "error": f"X devolvió HTTP {response.status_code}"
        }

    tweets, meta = parse_x_page(response.json())
    next_token = meta.get("next_token")

    if (pagination_token and next_token and pagination_token == next_token):
        return {
            "ok": False,
            "rate_limit": False,
            "reset_pagination": True,
        }

    return {
        "ok": True,
        "rate_limit": False,
        "reset_since_id": False,
        "tweets": tweets,
        "next_token": next_token
    }
    
    
def fetch_tweets_by_ids(tweetids):
    return requests.get(
        TWEET_LOOKUP_URL,
        headers=x_headers(),
        params={"ids": ",".join(tweetids), "tweet.fields": "entities,note_tweet"},
        timeout=20,
        allow_redirects=False,
    )


def update_last_tweetid(cur, idTweetUser, last_tweetid):
    sql = """
        UPDATE TweetUser 
        SET last_tweetid = %s, tweets_procesados = 1 
        WHERE idTweetUser = %s
    """
    
    cur.execute(sql, (str(last_tweetid), int(idTweetUser)))

    if cur.rowcount > 0:
        print(f"✅ Usuario {idTweetUser} actualizado correctamente con last_tweetid={last_tweetid}")
        return True
    else:
        print(f"⚠️ No se actualizó ningún registro para idTweetUser={idTweetUser}")
        return False

  
def update_pagination_token(cursor, user_id, pagination_token):
    sql = """
        UPDATE TweetUser
        SET pagination_token = %s
        WHERE idTweetUser = %s
    """

    cursor.execute(
        sql,
        (
            pagination_token,
            user_id
        )
    )


def max_tweet_id(cursor, user_id):
    cursor.execute(
        """
        SELECT MAX(CAST(tweetid AS UNSIGNED)) AS max_id
        FROM Tweets
        WHERE TweetUser_idTweetUser = %s
        """,
        (user_id,)
    )

    row = cursor.fetchone()

    if not row:
        return None

    return row["max_id"]

def get_users(limit=1):
    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        sql = """
            SELECT
                idTweetUser,
                TweetUser,
                last_tweetid,
                tweets_procesados,
                pagination_token
            FROM TweetUser
            WHERE tweets_procesados = 0
            ORDER BY idTweetUser ASC
            LIMIT %s
        """

        cur.execute(sql, (limit,))
        users = cur.fetchall()

        return users

    except Exception as e:
        print(f"❌ Error obteniendo usuarios: {e}")
        raise

    finally:
        close_job_connection(conn, cur, None, False)


def update_all():
    conn = None
    cur = None

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            """
            UPDATE TweetUser
            SET tweets_procesados = 0
            WHERE tweets_procesados = 1
            AND pagination_token IS NULL
            """
        )

        conn.commit()

    except Exception as e:
        print(f"❌ Error reseteando usuarios: {e}")
        raise

    finally:
        if cur:
            try:
                cur.close()
            except Exception:
                pass

        if conn:
            try:
                conn.close()
            except Exception:
                pass

# ================== CLOUD RUN HANDLER ==================

@app.route("/ingest", methods=["GET"])
def ingest_handler():
    conn = None
    cur = None
    acquired = False

    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        # Evita que dos ejecuciones de /ingest corran al mismo tiempo
        cur.execute(
            "SELECT GET_LOCK('netvora_tweet_ingest', 0) AS acquired"
        )

        lock_result = cur.fetchone()

        acquired = bool(lock_result and lock_result["acquired"] == 1)
        if not acquired:
            return jsonify({
                "ok": True,
                "busy": True,
                "message": "Ya existe otra ejecución de ingest en proceso."
            }), 200

        users = get_users(limit=1)

        if not users:
            update_all()
            users = get_users(limit=1)

        if not users:
            return jsonify({
                "ok": True,
                "message": "No hay usuarios para procesar"
            }), 200

        user = users[0]

        user_id = user["idTweetUser"]
        username = user["TweetUser"]
        last_tweetid = user["last_tweetid"]
        pagination_token = user["pagination_token"]

        print(
            f"Procesando @{username} | "
            f"last_tweetid={last_tweetid} | "
            f"pagination={bool(pagination_token)}"
        )

        resultado = fetch_tweets_for_user(username=username,last_tweetid=last_tweetid,pagination_token=pagination_token)

        # ----------------------------------------
        # ERROR / RATE LIMIT
        # ----------------------------------------

        if not resultado["ok"]:

            if resultado.get("rate_limit"):
                return jsonify({
                    "ok": False,
                    "user": username,
                    "rate_limit": True,
                    "error": resultado.get("error")
                }), 429

            if resultado.get("reset_pagination"):
                update_pagination_token(cur, user_id, None)
                conn.commit()
                return jsonify({"ok": False, "retry": True, "user": username}), 502

            if resultado.get("reset_since_id"):

                cur.execute(
                    """
                    UPDATE TweetUser
                    SET
                        last_tweetid = 1,
                        pagination_token = NULL,
                        tweets_procesados = 0
                    WHERE idTweetUser = %s
                    """,
                    (user_id,)
                )

                conn.commit()

                return jsonify({
                    "ok": True,
                    "user": username,
                    "reset_since_id": True,
                    "message": (
                        "El since_id era inválido. "
                        "La cuenta fue reiniciada y se procesará nuevamente "
                        "desde la búsqueda reciente."
                    )
                    }), 200
            return jsonify({
                "ok": False,
                "user": username,
                "error": resultado.get("error")
            }), 502

        tweets = resultado["tweets"]
        next_token = resultado["next_token"]

        guardados = 0
        url_metrics = {"tweets_with_url_entities": 0, "expanded_urls_received": 0, "tweets_only_tco": 0}

        # ----------------------------------------
        # Confirmar las escrituras y el avance de página en la misma transacción.
        for tweet in tweets:
            guardados += insert_or_update_tweet(cur, tweet, user_id, url_metrics=url_metrics)

        # TODAVÍA QUEDAN MÁS PÁGINAS
        # ----------------------------------------

        if next_token:

            update_pagination_token(cur,user_id,next_token)

            conn.commit()

            return jsonify({
                "ok": True,
                "user": username,
                "tweets_recibidos": len(tweets),
                "tweets_guardados": guardados,
                "article_urls": url_metrics,
                "pagination_pending": True,
                "message": "Página guardada. La cuenta continuará en la siguiente ejecución."
            }), 200

        # ----------------------------------------
        # YA TERMINAMOS TODAS LAS PÁGINAS
        # ----------------------------------------

        nuevo_last_tweetid = max_tweet_id(
            cur,
            user_id
        )

        cur.execute(
            """
            UPDATE TweetUser
            SET
                last_tweetid = %s,
                pagination_token = NULL,
                tweets_procesados = 1
            WHERE idTweetUser = %s
            """,
            (
                str(nuevo_last_tweetid) if nuevo_last_tweetid else last_tweetid,
                user_id
            )
        )

        conn.commit()

        return jsonify({
            "ok": True,
            "user": username,
            "tweets_recibidos": len(tweets),
            "tweets_guardados": guardados,
            "article_urls": url_metrics,
            "pagination_pending": False,
            "last_tweetid": nuevo_last_tweetid,
            "message": "Cuenta procesada completamente."
        }), 200

    except Exception:
        if conn is not None:
            rollback_job(conn)

        app.logger.exception("Falló la ingesta de tweets")

        return jsonify({
            "ok": False,
            "error": "Falló el trabajo"
        }), 500

    finally:
        close_job_connection(conn, cur, "netvora_tweet_ingest", acquired)


@app.route("/test", methods=["GET"])
def health():
    conn = None
    try:
        conn = get_db_connection()
        if conn.is_connected():
            return "OK - DB Connected", 200
        return "DB Not Connected", 500

    except Exception:
        app.logger.exception("Falló la comprobación de MySQL")
        return "DB Not Connected", 500

    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            app.logger.warning("Fallo cerrando conexión MySQL", exc_info=True)


@app.route("/reprocess", methods=["GET"])
def reprocess_handler():
    try:
        resultado = reprocess_pending_tweets(limit=100)

        status_code = 200 if resultado.get("ok") else 500

        return jsonify(resultado), status_code

    except Exception:
        app.logger.exception("Fallo en /reprocess")

        return jsonify({
            "ok": False,
            "error": "Falló el trabajo"
        }), 500

@app.route("/", methods=["GET"])
def status():
    

    return f"scikit-learn version: {sklearn.__version__}", 200

@app.route("/predict", methods=["GET"])
def predict():
    texto = request.args.get("text", "").strip()
    if not texto:
        return jsonify({"error": "Falta el parámetro text"}), 400
    return jsonify({
        "categoria": str(MLModel.predecir_categoria(texto)[0]),
        "sentimiento": str(MLModel.get_sentiment(texto)[0]),
    })


# ================== REPLIES INGEST ==================

def get_state(cursor, key: str, default=None):
    cursor.execute("SELECT v FROM ingest_state WHERE k=%s", (key,))
    row = cursor.fetchone()
    value = row[0] if row else default
    if value == "":
        return default
    return value

def set_state(cursor, key: str, value):
    cursor.execute(
        "INSERT INTO ingest_state (k, v) VALUES (%s, %s) "
        "ON DUPLICATE KEY UPDATE v=VALUES(v)",
        (key, value)
    )


def run_article_job(lock_name, job):
    started = time.monotonic()
    conn = None
    cursor = None
    acquired = False
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT GET_LOCK(%s, 0)", (lock_name,))
        acquired = cursor.fetchone() == (1,)
        metrics = job(conn, cursor, started) if acquired else {"busy": True}
        status = metrics.pop("status_code", 200)
        response = jsonify(ok=status < 400, elapsed_seconds=round(time.monotonic() - started, 2), **metrics)
        if status == 429 and "retry_after_seconds" in metrics:
            response.headers["Retry-After"] = str(metrics["retry_after_seconds"])
        return response, status
    except Exception as error:
        if conn is not None:
            rollback_job(conn)
        app.logger.error("Falló %s (%s, MySQL %s)", lock_name, type(error).__name__, getattr(error, "errno", None))
        return jsonify(ok=False, error="Falló el trabajo de artículos"), 500
    finally:
        close_job_connection(conn, cursor, lock_name, acquired)


@app.route("/articles/discover", methods=["GET"])
def discover_articles_handler():
    def discover(conn, cursor, started):
        now = datetime.now(BOLIVIA_TZ).replace(tzinfo=None)
        return discover_historical_urls(conn, cursor, now, started, get_state, set_state)

    return run_article_job("netvora_articles_discover", discover)


@app.route("/articles/scrape", methods=["GET"])
def scrape_articles_handler():
    return run_article_job("netvora_articles_scrape", process_pending_articles)


@app.route("/articles/recover_urls", methods=["GET"])
def recover_article_urls_handler():
    if request.method == "HEAD":
        return "", 204

    def recover(conn, cursor, _started):
        return recover_historical_urls(conn, cursor, fetch_tweets_by_ids, get_state, set_state)

    return run_article_job("netvora_articles_recover", recover)


def fetch_recent_root_tweetids(cursor, hours_back: int = 48, cap: int = 5000):
    cursor.execute(
        """
        SELECT tweetid
        FROM Tweets
        WHERE created >= CONVERT_TZ(UTC_TIMESTAMP(), '+00:00', '-04:00') - INTERVAL %s HOUR
        ORDER BY created DESC
        LIMIT %s
        """,
        (hours_back, cap),
    )
    return set(str(r[0]) for r in cursor.fetchall())
def fetch_recent_root_tweets(cursor, cap=5, after_id=None, root_id=None):
    query = """
        SELECT
            t.tweetid,
            t.created,
            t.TweetUser_idTweetUser,
            tu.TweetUser
        FROM Tweets t
        JOIN TweetUser tu ON tu.idTweetUser = t.TweetUser_idTweetUser
        WHERE t.created >= CONVERT_TZ(UTC_TIMESTAMP(), '+00:00', '-04:00') - INTERVAL 48 HOUR
    """
    params = []
    if root_id is not None:
        query += " AND t.tweetid = %s"
        params.append(root_id)
    elif after_id is not None:
        query += " AND t.tweetid > %s"
        params.append(after_id)

    query += " ORDER BY t.tweetid ASC LIMIT %s"
    params.append(cap)
    cursor.execute(query, tuple(params))

    rows = cursor.fetchall()

    return [
        {
            "tweetid": str(row[0]),
            "created": row[1],
            "tweetuser_id": row[2],
            "username": row[3]
        }
        for row in rows
    ]

def insert_reply(cursor, root_tweetid: str, reply: dict):
    reply_id = reply["id"]
    text = reply.get("text", "")
    created_at = reply.get("created_at")

    created_dt = None

    if created_at:
        try:
            created_dt = (
                datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                .astimezone(BOLIVIA_TZ)
                .replace(tzinfo=None)
            )
        except Exception:
            created_dt = None

    author_id = reply.get("author_id")
    parent_tweetid = None

    for ref in reply.get("referenced_tweets", []):
        if ref.get("type") == "replied_to":
            parent_tweetid = ref.get("id")
            break

    sentimiento = None

    if text:
        try:
            sentimiento = MLModel.get_sentiment(text)[0]
        except Exception as e:
            print(f"⚠️ Error sentimiento reply {reply_id}: {e}")
            sentimiento = None

    sql = """
    INSERT INTO replies
    (
        replyid,
        tweetid,
        parent_tweetid,
        text,
        created,
        TweetUser_idTweetUser,
        sentimiento
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s)

    ON DUPLICATE KEY UPDATE
    text = VALUES(text),
    created = VALUES(created),
    parent_tweetid = VALUES(parent_tweetid)
    """

    cursor.execute(
        sql,
        (
            reply_id,
            root_tweetid,
            parent_tweetid,
            text,
            created_dt,
            author_id,
            sentimiento
        )
    )

    return 1 if cursor.rowcount == 1 else 0


def x_search_replies_to_username(username: str, since_id: str | None, next_token: str | None):
    url = "https://api.twitter.com/2/tweets/search/recent"

    base_params = {
        "query": f"to:{username} is:reply",
        "max_results": 100,
        "tweet.fields": "created_at,author_id,conversation_id",
    }

    def do_request(use_since=True):
        params = dict(base_params)

        if use_since and since_id:
            params["since_id"] = since_id
        if next_token:
            params["next_token"] = next_token

        return requests.get(url, headers=x_headers(), params=params, timeout=20)

    r = do_request(use_since=True)

    if r.status_code == 429:
        retry_after = r.headers.get("retry-after")
        return {"rate_limited": True, "retry_after": retry_after, "status": 429, "body": r.text}

    if r.status_code == 400 and since_id:
        body_text = r.text or ""
        if "since_id" in body_text and "must be a tweet id created after" in body_text:
            return {
                "since_id_expired": True,
                "status": 400,
                "body": body_text
            }

    if r.status_code != 200:
        return {"error": True, "status": r.status_code, "body": r.text}

    return r.json()

def x_search_conversation(root_tweetid: str,since_id: str | None,next_token: str | None):
    url = "https://api.twitter.com/2/tweets/search/recent"

    base_params = {
        "query": f"conversation_id:{root_tweetid} is:reply",
        "max_results": 100,
        "tweet.fields": (
            "created_at,"
            "author_id,"
            "conversation_id,"
            "in_reply_to_user_id,"
            "referenced_tweets"
        ),
    }

    params = dict(base_params)

    if since_id:
        params["since_id"] = str(since_id)

    if next_token:
        params["next_token"] = next_token

    try:
        r = requests.get(url,headers=x_headers(),params=params,timeout=20)

    except requests.RequestException as e:
        return {
            "error": True,
            "status": None,
            "body": str(e)
        }

    if r.status_code == 429:
        return {
            "rate_limited": True,
            "retry_after": r.headers.get("retry-after"),
            "status": 429,
            "body": r.text
        }

    if r.status_code == 400:
        body_text = r.text or ""
        if since_id and "since_id" in body_text and "must be a tweet id created after" in body_text:
            return {"since_id_expired": True}
        if next_token and invalid_x_pagination(body_text):
            return {"pagination_expired": True}

    if r.status_code != 200:
        return {
            "error": True,
            "status": r.status_code,
            "body": r.text
        }

    return r.json()

@app.route("/ingest_replies", methods=["GET"])
def ingest_replies_handler():
    """
    Extrae replies NUEVOS dirigidos a tus cuentas,
    y guarda solo los que pertenecen a tweets (root) de las últimas 48h.
    Diseñado para corridas frecuentes (cada 5-10 min) y para no reventar rate limits.
    """
    start_time = time.monotonic()
    MAX_RUNTIME_SECONDS = 40
    conn = None
    cursor = None
    acquired = False
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
        "SELECT GET_LOCK('netvora_replies_ingest', 0)"
        )

        lock_result = cursor.fetchone()

        acquired = bool(lock_result and lock_result[0] == 1)
        if not acquired:
            return jsonify({
                "ok": True,
                "busy": True,
                "message": "Ya existe otra ejecución de ingest_replies en proceso."
            }), 200

        
        cursor.execute("""
            DELETE s
            FROM ingest_state s
            LEFT JOIN Tweets t
                ON t.tweetid = CAST(
                    SUBSTRING_INDEX(s.k, ':', -1)
                    AS UNSIGNED
                )
                AND t.created >= CONVERT_TZ(UTC_TIMESTAMP(), '+00:00', '-04:00') - INTERVAL 48 HOUR
            WHERE s.k LIKE 'replies_%:%'
            AND t.tweetid IS NULL
        """)
        cursor.execute("""
            SELECT k
            FROM ingest_state
            WHERE k LIKE 'replies_pagination_token:%'
            AND v IS NOT NULL
            AND v <> ''
            ORDER BY k
            LIMIT 1
        """)

        pending_row = cursor.fetchone()
        selected = []
        if pending_row:
            pending_root_id = pending_row[0].split(":")[-1]
            selected = fetch_recent_root_tweets(cursor, cap=1, root_id=pending_root_id)
            if not selected:
                set_state(cursor, f"replies_pagination_token:{pending_root_id}", "")
                set_state(cursor, f"replies_max_seen:{pending_root_id}", "")

        if not selected:
            last_root_id = get_state(cursor, "replies_last_root_id")
            selected = fetch_recent_root_tweets(cursor, after_id=last_root_id)
            if not selected and last_root_id is not None:
                selected = fetch_recent_root_tweets(cursor)

        conn.commit()
        if not selected:
            return jsonify({
                "ok": True,
                "msg": "No hay tweets root recientes para procesar",
            }), 200

        saved = 0
        rate_limited = False
        upstream_failed = False
        details = []
        completed_roots = []
        fetched_replies = 0
        pages_fetched = 0

        for root in selected:
            if time.monotonic() - start_time >= MAX_RUNTIME_SECONDS:
                details.append({
                    "type": "runtime_limit",
                    "message": "Se alcanzó el límite interno de tiempo."
                })
                break

            root_tweetid = str(root["tweetid"])
            uname = root["username"]

            # Cada conversación mantiene su propio estado
            since_key = f"replies_since_id:{root_tweetid}"
            pagination_key = f"replies_pagination_token:{root_tweetid}"
            max_seen_key = f"replies_max_seen:{root_tweetid}"

            since_id = get_state(cursor, since_key)
            next_token = get_state(cursor, pagination_key)
            max_seen = get_state(cursor, max_seen_key)
            data = x_search_conversation(root_tweetid, since_id, next_token)

            if data.get("since_id_expired") or data.get("pagination_expired"):
                if data.get("since_id_expired"):
                    set_state(cursor, since_key, "")
                # Reiniciar desde el último since_id confirmado y reprocesar duplicados.
                set_state(cursor, pagination_key, "")
                set_state(cursor, max_seen_key, "")
                conn.commit()
                details.append({"root_tweetid": root_tweetid, "pagination_reset": True})
                upstream_failed = True
                break

            if data.get("rate_limited"):
                rate_limited = True
                details.append({
                    "user": uname,
                    "root_tweetid": root_tweetid,
                    "rate_limited": True,
                    "retry_after": data.get("retry_after"),
                })
                break

            if data.get("error"):
                upstream_failed = True
                details.append({
                    "user": uname,
                    "root_tweetid": root_tweetid,
                    "error": True,
                    "status": data.get("status"),
                })
                break

            tweets, meta = parse_x_page(data)
            response_next_token = meta.get("next_token")
            if next_token and response_next_token == next_token:
                set_state(cursor, pagination_key, "")
                set_state(cursor, max_seen_key, "")
                conn.commit()
                upstream_failed = True
                details.append({"root_tweetid": root_tweetid, "error": "Token de paginación repetido"})
                break

            pages_fetched += 1
            fetched_replies += len(tweets)

            for reply in tweets:
                if str(reply.get("conversation_id")) != root_tweetid:
                    raise ValueError(f"Respuesta fuera de la conversación {root_tweetid}")
                saved += insert_reply(cursor, root_tweetid, reply)
                if not max_seen or int(reply["id"]) > int(max_seen):
                    max_seen = str(reply["id"])

            if response_next_token:
                set_state(cursor, pagination_key, response_next_token)
                if max_seen:
                    set_state(cursor, max_seen_key, max_seen)
                conn.commit()
                break

            if max_seen:
                set_state(cursor, since_key, max_seen)
            set_state(cursor, pagination_key, "")
            set_state(cursor, max_seen_key, "")
            set_state(cursor, "replies_last_root_id", root_tweetid)
            conn.commit()
            completed_roots.append(root_tweetid)

        status_code = 429 if rate_limited else 502 if upstream_failed else 200
        return jsonify({
            "ok": status_code == 200,
            "saved": saved,
            "roots_processed": completed_roots,
            "rate_limited": rate_limited,
            "details": details,
            "fetched_replies": fetched_replies,
            "pages_fetched": pages_fetched,
            "elapsed_seconds": round(time.monotonic() - start_time, 2),
        }), status_code

    except Exception:
        if conn is not None:
            rollback_job(conn)
        app.logger.exception("Fallo en la ingesta de respuestas")
        return jsonify({"ok": False, "error": "Falló la ingesta de respuestas"}), 500
    finally:
        close_job_connection(conn, cursor, "netvora_replies_ingest", acquired)


@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    secret = os.getenv("CONSUMER_SECRET_KEY")
    if not secret:
        return jsonify({"error": "Webhook sin configurar"}), 503

    if request.method == "GET":
        crc_token = request.args.get("crc_token")
        if not crc_token:
            return jsonify({"error": "Falta crc_token"}), 400
        payload = crc_token.encode()
    else:
        payload = request.get_data()

    digest = hmac.digest(secret.encode(), payload, "sha256")
    signature = "sha256=" + base64.b64encode(digest).decode()
    if request.method == "GET":
        return jsonify({"response_token": signature})

    received = request.headers.get("X-Twitter-Webhooks-Signature", "")
    if not hmac.compare_digest(received.encode(), signature.encode()):
        return jsonify({"error": "Firma inválida"}), 401
    return "OK", 200


def ejecutar_job_stance(lock_name, tarea):
    conn = None
    cur = None
    adquirido = False

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            "SELECT GET_LOCK(%s, 0)",
            (lock_name,)
        )

        adquirido = cur.fetchone()[0] == 1

        if not adquirido:
            return jsonify({
                "ok": True,
                "busy": True
            }), 200

        resultado = tarea()
        failed = (
            resultado.get("fallidos", 0)
            if "fallidos" in resultado
            else sum(resumen["fallidos"] for resumen in resultado.values())
        )
        return jsonify({
            "ok": failed == 0,
            "resultado": resultado,
        }), 502 if failed else 200

    except Exception:
        app.logger.exception("Fallo en job Stance: %s", lock_name)
        return jsonify({"ok": False, "error": "Falló el trabajo de Stance"}), 500

    finally:
        close_job_connection(conn, cur, lock_name, adquirido)


@app.route("/stance/posts", methods=["GET"])
def ejecutar_stance_posts_handler():
    # Procesa pendientes entre todas las publicaciones almacenadas,
    # sin token propio ni filtro por fecha. Las tandas pequeñas evitan timeout.
    return ejecutar_job_stance(
        "netvora_stance_posts",
        lambda: ejecutar_stance_por_target(
            max_comentarios=10,
            tamano_lote=10,
            max_publicaciones=3
        )
    )


@app.route("/stance/topics", methods=["GET"])
def ejecutar_stance_topics_handler():
    # Primero calcula los targets de topicos que aún no los tengan.
    # Después clasifica comentarios pendientes por topico, sin filtro de fecha.
    def tarea():
        targets = ejecutar_targets_topicos(
            limit=2,
            tweets_por_topico=20
        )

        comentarios = ejecutar_stance_por_topico(
            max_comentarios=10,
            tamano_lote=10,
            max_topicos=2,
            max_publicaciones=3
        )

        return {
            "targets": targets,
            "comentarios": comentarios
        }

    return ejecutar_job_stance(
        "netvora_stance_topics",
        tarea
    )


@app.route("/selectTopics", methods=["GET"])
def select_topics():
    lock_name = "netvora_select_topics"
    conn = None
    cursor = None
    acquired = False
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT GET_LOCK(%s, 0)", (lock_name,))
        lock_result = cursor.fetchone()[0]
        if lock_result == 0:
            return jsonify({
                "ok": True,
                "busy": True,
                "tweets_no_asignados": 0,
                "tweets_asignados": 0,
            }), 200
        if lock_result != 1:
            raise RuntimeError("MySQL no pudo adquirir el bloqueo de tópicos")
        acquired = True

        df_no_asignados, df_asignados = classify_tweets()
        tweets_asignados = int(df_asignados["tweetid"].nunique()) if not df_asignados.empty else 0
        return jsonify({
            "ok": True,
            "tweets_no_asignados": len(df_no_asignados),
            "tweets_asignados": tweets_asignados,
        }), 200
    except Exception:
        app.logger.exception("Falló el trabajo de tópicos")
        return jsonify({"ok": False, "error": "Falló el trabajo de tópicos"}), 500
    finally:
        close_job_connection(conn, cursor, lock_name, acquired)
    
@app.route("/create_topics", methods=["GET"])
def create_topics():
    try:  
        a = insertar_nuevos_topicos(aplicar_cambios=True)
        if a is None:
            return jsonify({"ok": True, "msg": "No hay topics para nombrar."}), 200
        return jsonify({"ok": True}), 200
    except Exception:
        app.logger.exception("Falló el trabajo de tópicos")
        return jsonify({"ok": False, "error": "Falló el trabajo de tópicos"}), 500

if __name__ == "__main__":
    # Para correrlo localmente
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
