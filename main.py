# main.py
import os
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, request
import requests
import sklearn
import MLModel
import mysql.connector
from mysql.connector import Error
import time
from dateutil import parser as dtparser
import hmac
import hashlib
import base64
from selectTopics import classify_tweets
from crearTopic import insertar_nuevos_topicos



# ================== CONFIG ==================

DB_CONFIG = {
            # IP pública o nombre interno de Cloud SQL
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME"),
    "unix_socket": f"/cloudsql/{os.environ.get('INSTANCE_CONNECTION_NAME')}",
    "charset": "utf8mb4",
    "port": "3306",
}
  



#TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
CONSUMER_SECRET = os.environ.get("CONSUMER_SECRET")
CONSUMER_SECRET_KEY = os.environ.get("CONSUMER_SECRET_KEY")
if not TWITTER_BEARER_TOKEN:
    raise RuntimeError("Falta TWITTER_BEARER_TOKEN en variables de entorno")

headers = {
    "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"
}
TWEETS_URL = "https://api.twitter.com/2/tweets/search/recent"
#TWEETS_URL = "https://api.twitter.com/2/users/{}/tweets"


USER_IDS = [
    '65444625', '94438031', '935136477616443393', '209279715',
    '2800854409', '735321776', '44489439', '311132840',
    '525394081', '1742290477234208768', '118861947', '188870982'
]
MAX_USERS_PER_RUN = 5
MAX_RESULTS_PER_CALL = 100  # máximo permitido por endpoint
REQUEST_SLEEP_SECONDS = 0.2 

BOLIVIA_TZ = timezone(timedelta(hours=-4))

app = Flask(__name__)

# ================== FUNCIONES BD ==================

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)





# ================== INSERT TWEETS ==================

def insert_or_update_tweet(cur, tweet, tweetuser_pk_id):
    tweet_id = tweet["id"]
    text = tweet.get("text", "")
    created_at = tweet.get("created_at")

    created_dt = None

    if created_at:
        try:
            created_dt = datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            )
        except Exception:
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

    return 1



def reprocess_pending_tweets(limit=100):
    conn = None
    cur = None

    procesados = 0
    errores = 0

    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
        "SELECT GET_LOCK('netvora_ml_reprocess', 0) AS acquired"
    )

        lock_result = cur.fetchone()

        if not lock_result or lock_result["acquired"] != 1:
            return {
                "ok": True,
                "busy": True,
                "procesados": 0,
                "message": "Ya existe otra ejecución de reprocesamiento ML."
            }
        cur.execute("""
            SELECT
                DATABASE() AS database_name,
                @@hostname AS mysql_host,
                COUNT(*) AS pendientes
            FROM Tweets
            WHERE ml_procesado = 0
        """)

        debug_db = cur.fetchone()

        print(f"DEBUG DB: {debug_db}")

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

                cur.execute(
                    """
                    UPDATE Tweets
                    SET
                        sentimiento = %s,
                        categoria = %s,
                        ml_procesado = 1
                    WHERE tweetid = %s
                    """,
                    (
                        sentimiento,
                        categoria,
                        tweet_id
                    )
                )

                procesados += 1

            except Exception as e:
                errores += 1

                print(
                    f"❌ Error reprocesando tweet "
                    f"{tweet_id}: {e}"
                )

        conn.commit()

        return {
            "ok": True,
            "procesados": procesados,
            "errores": errores,
            "total_encontrados": len(tweets),
            "database": debug_db["database_name"],
            "mysql_host": debug_db["mysql_host"],
            "pendientes_antes": debug_db["pendientes"],
            
        }

    except Exception as e:

        if conn:
            conn.rollback()

        print(
            f"❌ Error general reprocesando tweets: {e}"
        )

        return {
            "ok": False,
            "procesados": procesados,
            "errores": errores,
            "error": str(e)
        }

    finally:

        if cur and conn:
            try:
                cur.execute(
                    "SELECT RELEASE_LOCK('netvora_ml_reprocess')"
                )
            except Exception:
                pass

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

# ================== TIME HELPERS ==================

def bolivia_day_start_utc_iso():
    """
    Inicio del día en Bolivia convertido a UTC en ISO Z.
    """
    now_bo = datetime.now(BOLIVIA_TZ)
    start_bo = now_bo.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_bo.astimezone(timezone.utc)
    return start_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")



def fetch_tweets_for_user(username: str,last_tweetid=None,pagination_token=None):
    params = {
        "query": f"from:{username}",
        "max_results": 100,
        "tweet.fields": "created_at,text,entities,author_id",
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
            headers=headers,
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

        if "since_id" in error_text:
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
            "error": (
                f"X API error {response.status_code}: "
                f"{response.text}"
            )
        }

    data = response.json()

    tweets = data.get("data", [])

    meta = data.get("meta", {})

    next_token = meta.get("next_token")
    print(
    f"@{username} | "
    f"token_entrada={pagination_token[:20] if pagination_token else None} | "
    f"token_salida={next_token[:20] if next_token else None} | "
    f"tweets={len(tweets)}")
    
    if (pagination_token and next_token and pagination_token == next_token):
        return {
            "ok": False,
            "rate_limit": False,
            "reset_since_id": False,
            "tweets": [],
            "next_token": pagination_token,
            "error": "X devolvió el mismo pagination_token. Se detuvo para evitar un ciclo infinito."
        }

    return {
        "ok": True,
        "rate_limit": False,
        "reset_since_id": False,
        "tweets": tweets,
        "next_token": next_token
    }
    
    
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
        if cur:
            cur.close()

        if conn:
            conn.close()


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
            WHERE pagination_token IS NULL
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

    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        # Evita que dos ejecuciones de /ingest corran al mismo tiempo
        cur.execute(
            "SELECT GET_LOCK('netvora_tweet_ingest', 0) AS acquired"
        )

        lock_result = cur.fetchone()

        if not lock_result or lock_result["acquired"] != 1:
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
            }), 500

        tweets = resultado["tweets"]
        next_token = resultado["next_token"]

        guardados = 0

        # ----------------------------------------
        # GUARDAR TWEETS DE ESTA PÁGINA
        # ----------------------------------------

        errores_tweets = 0

        for tweet in tweets:

            try:
                nuevo = insert_or_update_tweet(
                    cur,
                    tweet,
                    user_id
                )

                guardados += nuevo

            except Exception as e:
                errores_tweets += 1

                print(
                    f"❌ Error procesando tweet "
                    f"{tweet.get('id')}: {e}"
                )

        # ----------------------------------------
        # TODAVÍA QUEDAN MÁS PÁGINAS
        # ----------------------------------------

        if next_token:

            update_pagination_token(cur,user_id,next_token)

            conn.commit()

            return jsonify({
                "ok": True,
                "user": username,
                "tweets_recibidos": len(tweets),
                "tweets_con_error": errores_tweets,
                "tweets_guardados": guardados,
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
            "tweets_con_error": errores_tweets,
            "pagination_pending": False,
            "last_tweetid": nuevo_last_tweetid,
            "message": "Cuenta procesada completamente."
        }), 200

    except Exception as e:

        if conn:
            conn.rollback()

        print(f"❌ Error ingest: {e}")

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

    finally:

        if cur and conn:
            try:
                cur.execute(
                    "SELECT RELEASE_LOCK('netvora_tweet_ingest')"
                )
            except Exception:
                pass

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

@app.route("/test", methods=["GET"])
def health():
    try:
        conn = get_db_connection()
        if conn.is_connected():
            return "OK - DB Connected", 200
        return "DB Not Connected", 500

    except Exception as e:
        return f"DB Error: {e}", 500

    finally:
        try:
            conn.close()
        except:
            pass


@app.route("/reprocess", methods=["GET"])
def reprocess_handler():
    try:
        resultado = reprocess_pending_tweets(limit=100)

        status_code = 200 if resultado.get("ok") else 500

        return jsonify(resultado), status_code

    except Exception as e:
        print(f"❌ Error en /reprocess: {e}")

        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

@app.route("/", methods=["GET"])
def status():
    

    return f"scikit-learn version: {sklearn.__version__}", 200

@app.route("/predict", methods=["GET"])
def predict():
    texto = "Me encanta"
    print(MLModel.predecir_categoria(texto)[0])
    print(MLModel.get_sentiment(texto)[0])
    
    
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

def fetch_recent_root_tweetids(cursor, hours_back: int = 48, cap: int = 5000):
    cursor.execute(
        """
        SELECT tweetid
        FROM Tweets
        WHERE created >= (UTC_TIMESTAMP() - INTERVAL %s HOUR)
        ORDER BY created DESC
        LIMIT %s
        """,
        (hours_back, cap),
    )
    return set(str(r[0]) for r in cursor.fetchall())
def fetch_recent_root_tweets(
    cursor,
    hours_back: int = 48,
    cap: int = 5000
):
    cursor.execute(
        """
        SELECT
            t.tweetid,
            t.created,
            t.TweetUser_idTweetUser,
            tu.TweetUser
        FROM Tweets t
        JOIN TweetUser tu
            ON tu.idTweetUser = t.TweetUser_idTweetUser
        WHERE t.created >= (
            UTC_TIMESTAMP() - INTERVAL %s HOUR
        )
        ORDER BY t.created ASC, t.tweetid ASC
        LIMIT %s
        """,
        (
            hours_back,
            cap
        )
    )

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
            created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
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
        created = VALUES(created)
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

        return requests.get(url, headers=headers, params=params, timeout=20)

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

def x_search_conversation(
    root_tweetid: str,
    since_id: str | None,
    next_token: str | None
):
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
        r = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=20
        )

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

    if r.status_code == 400 and since_id:
        body_text = r.text or ""

        if (
            "since_id" in body_text
            and "must be a tweet id created after" in body_text
        ):
            return {
                "since_id_expired": True,
                "status": 400,
                "body": body_text
            }

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
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
        "SELECT GET_LOCK('netvora_replies_ingest', 0)"
        )

        lock_result = cursor.fetchone()

        if not lock_result or lock_result[0] != 1:
            return jsonify({
                "ok": True,
                "busy": True,
                "message": "Ya existe otra ejecución de ingest_replies en proceso."
            }), 200

        
        root_tweets = fetch_recent_root_tweets(cursor,hours_back=48,cap=5000)

        if not root_tweets:
            return jsonify({
                "ok": True,
                "msg": "No hay tweets root recientes para procesar"
            }), 200


        # 3) Procesar pocos tweets por ejecución
        batch_size = 1

        start_idx = int(
            get_state(cursor,"replies_root_idx","0") or "0")


        selected = []

        for i in range(batch_size):

            selected.append(root_tweets[(start_idx + i) % len(root_tweets)])

            

        saved = 0
        rate_limited = False
        details = []

        for root in selected:

            root_tweetid = str(root["tweetid"])
            uname = root["username"]

            # Cada conversación mantiene su propio estado
            since_key = f"replies_since_id:{root_tweetid}"
            pagination_key = f"replies_pagination_token:{root_tweetid}"
            max_seen_key = f"replies_max_seen:{root_tweetid}"

            since_id = get_state(cursor,since_key,None)

            next_token = get_state(cursor,pagination_key,None)

            max_seen_global = get_state(cursor,max_seen_key,None)
            processing_failed = False
            
            for page in range(2):
                data = x_search_conversation(root_tweetid,since_id,next_token)

                if isinstance(data, dict) and data.get("since_id_expired"):
                    print(f"⚠️ since_id expirado para {uname}: {since_id}")

                    set_state(cursor, since_key, "")
                    set_state(cursor, pagination_key, "")
                    set_state(cursor, max_seen_key, "")

                    since_id = None
                    next_token = None
                    max_seen_global = None

                    data = x_search_conversation(root_tweetid,None,None)

                if isinstance(data, dict) and data.get("rate_limited"):
                    rate_limited = True
                    processing_failed = True
                    details.append({
                        "user": uname,
                        "root_tweetid": root_tweetid,
                        "rate_limited": True,
                        "retry_after": data.get("retry_after")
                    })
                    break

                if isinstance(data, dict) and data.get("error"):
                    processing_failed = True
                    details.append({
                        "user": uname,
                        "root_tweetid": root_tweetid,
                        "error": True,
                        "status": data.get("status"),
                        "body": data.get("body")
                    })
                    break

                tweets = data.get("data", []) if isinstance(data, dict) else []
                meta = data.get("meta", {}) if isinstance(data, dict) else {}

                response_next_token = meta.get("next_token")

                if not tweets:

                    # Si X todavía entrega otra página, guardamos ese token.
                    if response_next_token:

                        set_state( cursor,pagination_key,response_next_token)

                        next_token = response_next_token

                    else:
                        if max_seen_global not in (None, ""):
                            set_state(cursor,since_key,str(max_seen_global))

                        set_state(cursor,pagination_key,"")
                        set_state(cursor,max_seen_key,"")
                        next_token = None

                    break

                for tw in tweets:
                    conv_id = str(tw.get("conversation_id") or "")

                    if conv_id == root_tweetid:

                        nuevo = insert_reply(cursor,root_tweetid,tw)
                        saved += nuevo

                    tid = str(tw["id"])

                    if max_seen_global in (None, ""):
                        max_seen_global = tid
                    elif int(tid) > int(max_seen_global):
                        max_seen_global = tid


                # Guardamos el mayor tweet ID encontrado hasta ahora
                if max_seen_global not in (None, ""):
                    set_state(cursor,max_seen_key,str(max_seen_global))


                meta = data.get("meta", {})
                next_token = meta.get("next_token")


                # ----------------------------------------
                # TODAVÍA QUEDAN MÁS PÁGINAS
                # ----------------------------------------
                if next_token:
                    set_state(cursor,pagination_key,next_token)

                    continue


                # ----------------------------------------
                # YA TERMINAMOS TODAS LAS PÁGINAS
                # ----------------------------------------

                if max_seen_global not in (None, ""):
                    set_state(cursor,since_key,str(max_seen_global))

                # Limpiamos estados temporales
                set_state(cursor, pagination_key, "")
                set_state(cursor, max_seen_key, "")

                break
        # ----------------------------------------
        # DECIDIR SI AVANZAMOS AL SIGUIENTE USUARIO
        # ----------------------------------------

        pagination_pending = get_state(
            cursor,
            f"replies_pagination_token:{root_tweetid}",
            None
        )

        if processing_failed:

            # Hubo error o rate limit.
            # No avanzamos para poder reintentar este mismo root.
            set_state(
                cursor,
                "replies_root_idx",
                str(start_idx)
            )

        elif pagination_pending:

            # Todavía quedan páginas de esta conversación.
            # Seguimos en el mismo tweet raíz.
            set_state(cursor,"replies_root_idx",str(start_idx))

        else:

            # La conversación terminó correctamente.
            # Avanzamos al siguiente tweet raíz.
            next_start = (start_idx + 1) % max(len(root_tweets), 1)

            set_state(cursor,"replies_root_idx",str(next_start))

        conn.commit()
        return jsonify({
            "ok": True,
            "saved": saved,
            "roots_processed": [
                str(root["tweetid"])
                for root in selected
            ],
            "rate_limited": rate_limited,
            "details": details
        }), 200

    except Error as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        if cursor and conn:
            try:
                cursor.execute(
                    "SELECT RELEASE_LOCK('netvora_replies_ingest')"
                )
            except Exception:
                pass
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        crc_token = request.args.get("crc_token")

        hash_digest = hmac.new(
            CONSUMER_SECRET_KEY.encode(),
            msg=crc_token.encode(),
            digestmod=hashlib.sha256
        ).digest()

        response_token = "sha256=" + base64.b64encode(hash_digest).decode()

        return jsonify({"response_token": response_token})

    if request.method == "POST":
        print(request.json)  # aquí llegan los eventos
        return "OK", 200


@app.route("/selectTopics", methods=["GET"])
def select_topics():
    try:
        df_no_asignados , df_asignados = classify_tweets()
        return jsonify({
            "ok": True,
            "tweets_no_asignados": len(df_no_asignados),
            "tweets_asignados": len(df_asignados),
            
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
@app.route("/create_topics", methods=["GET"])
def create_topics():
    try:  
        a = insertar_nuevos_topicos()
        if a is None:
            return jsonify({"ok": True, "msg": "No hay topics para nombrar."}), 200
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    # Para correrlo localmente
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
