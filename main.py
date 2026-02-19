# main.py
import os
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify
import requests
import sklearn
import MLModel
import mysql.connector
from mysql.connector import Error
import time
from dateutil import parser as dtparser


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
TWITTER_BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAN9WpgEAAAAAHarp9HjcuJFZ4wtx1DtpsP8Z93A%3DC3AEHMO2YXaGFFgblPEdkYTGhBne75WLUlG5Mc95FGKlR003vg'

if not TWITTER_BEARER_TOKEN:
    raise RuntimeError("Falta TWITTER_BEARER_TOKEN en variables de entorno")

headers = {
    "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"
}
TWEETS_URL = "https://api.twitter.com/2/tweets/search/recent"


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


def pick_users_to_process(conn, limit=MAX_USERS_PER_RUN):
    """
    Selecciona hasta N usuarios para procesar rotando por tweets_procesados.
    Prioridad:
      1) bootstrap: sin last_tweetid (NULL o '')
      2) menor tweets_procesados
      3) menor last_tweetid (desempate)
    """
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT idTweetUser, TweetUser, last_tweetid, tweets_procesados
        FROM TweetUser
        ORDER BY
          CASE WHEN last_tweetid IS NULL OR last_tweetid = '' THEN 0 ELSE 1 END ASC,
          tweets_procesados ASC,
          CAST(NULLIF(last_tweetid,'') AS UNSIGNED) ASC
        LIMIT %s
        """,
        (limit,),
    )
    rows = cur.fetchall()
    cur.close()
    return rows


def update_user_last_tweetid(conn, tweetuser_pk_id, last_tweetid_str):
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE TweetUser
        SET last_tweetid = %s
        WHERE idTweetUser = %s
        """,
        (str(last_tweetid_str), int(tweetuser_pk_id)),
    )
    cur.close()


def bump_user_processed_tweets(conn, tweetuser_pk_id, inc):
    """
    Suma al contador de tweets procesados.
    """
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE TweetUser
        SET tweets_procesados = tweets_procesados + %s
        WHERE idTweetUser = %s
        """,
        (int(inc), int(tweetuser_pk_id)),
    )
    cur.close()


# ================== INSERT TWEETS ==================

def insert_or_update_tweet(cur, tweet, tweetuser_pk_id):
    tweet_id = tweet["id"]
    text = tweet.get("text", "")
    created_at = tweet.get("created_at")

    created_dt = None
    if created_at:
        try:
            created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except Exception:
            created_dt = None

    url = f"https://twitter.com/i/web/status/{tweet_id}"

    sql = """
    INSERT INTO Tweets
      (tweetid, text, created, url, sentimiento, categoria,
       Lugar, Persona, Organizacion, Locacion, Otros, TweetUser_idTweetUser)
    VALUES
      (%s, %s, %s, %s, %s, %s,
       %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      text = VALUES(text),
      created = VALUES(created),
      url = VALUES(url),
      TweetUser_idTweetUser = VALUES(TweetUser_idTweetUser)
    """

    params = (
        tweet_id,
        text,
        created_dt,
        url,
        MLModel.get_sentiment(text)[0],
        MLModel.predecir_categoria(text)[0],
        "", "", "", "", "",
        int(tweetuser_pk_id),
    )
    cur.execute(sql, params)


# ================== TIME HELPERS ==================

def bolivia_day_start_utc_iso():
    """
    Inicio del día en Bolivia convertido a UTC en ISO Z.
    """
    now_bo = datetime.now(BOLIVIA_TZ)
    start_bo = now_bo.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_bo.astimezone(timezone.utc)
    return start_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ================== X API ==================

def request_with_retry(url, headers, params, max_retries=4):
    wait = 2
    last_exc = None
    for _ in range(max_retries):
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 200:
            return r

        if r.status_code == 429 or (500 <= r.status_code <= 599):
            time.sleep(wait)
            wait = min(wait * 2, 30)
            last_exc = Exception(f"X API error {r.status_code}: {r.text}")
            continue

        raise Exception(f"X API error {r.status_code}: {r.text}")

    raise last_exc if last_exc else Exception("X API error: retries agotados")


def fetch_tweets_for_user(username: str, last_tweetid: str | None):
    """
    Trae tweets para UN usuario (username).
    - Si last_tweetid vacío => start_time = inicio del día Bolivia (UTC)
    - Si last_tweetid existe => since_id = last_tweetid
    """
    params = {
        "query": f"from:{username}",
        "max_results": MAX_RESULTS_PER_CALL,
        "tweet.fields": "created_at,text,entities,author_id",
    }

    if not last_tweetid:
        params["start_time"] = bolivia_day_start_utc_iso()
    else:
        params["since_id"] = str(last_tweetid)

    all_tweets = []

    r = request_with_retry(TWEETS_URL, headers, params)
    data = r.json()
    all_tweets.extend(data.get("data", []))

    while True:
        meta = data.get("meta", {}) or {}
        next_token = meta.get("next_token")
        if not next_token:
            break
        params["pagination_token"] = next_token
        r = request_with_retry(TWEETS_URL, headers, params)
        data = r.json()
        all_tweets.extend(data.get("data", []))

    return all_tweets


def max_tweet_id(tweets):
    if not tweets:
        return None
    m = max(int(t["id"]) for t in tweets if "id" in t)
    return str(m)


# ================== CLOUD RUN HANDLER ==================

@app.route("/ingest", methods=["GET"])
def ingest_handler():
    """
    Una corrida:
      - Toma 5 usuarios con menor tweets_procesados (bootstrap primero)
      - Para cada usuario:
          - fetch tweets (from:username)
          - inserta/actualiza Tweets
          - actualiza last_tweetid
          - suma tweets_procesados += tweets_guardados
    """
    conn = None
    try:
        conn = get_db_connection()

        users = pick_users_to_process(conn, MAX_USERS_PER_RUN)
        if not users:
            return jsonify({"ok": True, "msg": "No hay usuarios en TweetUser", "procesados": 0}), 200

        total_saved = 0
        per_user_stats = []

        for u in users:
            tweetuser_pk_id = int(u["idTweetUser"])      # FK a Tweets.TweetUser_idTweetUser
            username = (u.get("TweetUser") or "").strip()  # para from:username
            last_tid = u.get("last_tweetid")
            last_tid = str(last_tid) if last_tid not in (None, "") else None

            if not username:
                per_user_stats.append({
                    "idTweetUser": tweetuser_pk_id,
                    "username": username,
                    "nuevos": 0,
                    "error": "TweetUser (username) vacío",
                })
                continue

            tweets = fetch_tweets_for_user(username, last_tid)

            if not tweets:
                per_user_stats.append({
                    "idTweetUser": tweetuser_pk_id,
                    "username": username,
                    "nuevos": 0,
                    "updated_last_tweetid": False
                })
                time.sleep(REQUEST_SLEEP_SECONDS)
                continue

            cur = conn.cursor()
            saved = 0
            for t in tweets:
                insert_or_update_tweet(cur, t, tweetuser_pk_id=tweetuser_pk_id)
                saved += 1

            new_last = max_tweet_id(tweets)
            if new_last:
                update_user_last_tweetid(conn, tweetuser_pk_id, new_last)

            # contador de rotación
            bump_user_processed_tweets(conn, tweetuser_pk_id, saved)

            conn.commit()
            cur.close()

            total_saved += saved
            per_user_stats.append({
                "idTweetUser": tweetuser_pk_id,
                "username": username,
                "nuevos": saved,
                "updated_last_tweetid": True,
                "last_tweetid": new_last,
            })

            time.sleep(REQUEST_SLEEP_SECONDS)

        return jsonify({
            "ok": True,
            "procesados": len(per_user_stats),
            "tweets_guardados": total_saved,
            "detalle": per_user_stats
        }), 200

    except Error as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return jsonify({"ok": False, "error": f"MySQL: {str(e)}"}), 500

    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return jsonify({"ok": False, "error": str(e)}), 500

    finally:
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
    return row[0] if row else default

def set_state(cursor, key: str, value: str):
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

    sql = """
    INSERT INTO replies (replyid, tweetid, text, created, TweetUser_idTweetUser, sentimiento)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      text = VALUES(text),
      created = VALUES(created)
    """
    cursor.execute(sql, (
        reply_id,
        root_tweetid,          # guardamos el tweet ORIGINAL (root) en tweetid
        text,
        created_dt,
        author_id,
        MLModel.get_sentiment(text)[0] if text else None,
    ))

def x_search_replies_to_username(username: str, since_id: str | None, next_token: str | None):
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": f"to:{username} is:reply",
        "max_results": 100,
        "tweet.fields": "created_at,author_id,conversation_id",
    }
    if since_id:
        params["since_id"] = since_id
    if next_token:
        params["next_token"] = next_token

    r = requests.get(url, headers=headers, params=params, timeout=20)

    # Manejo 429: no dormimos infinito (Cloud Run). Cortamos y reanudamos luego.
    if r.status_code == 429:
        retry_after = r.headers.get("retry-after")
        return {"rate_limited": True, "retry_after": retry_after, "status": 429, "body": r.text}

    if r.status_code != 200:
        return {"error": True, "status": r.status_code, "body": r.text}

    return r.json()

@app.route("/ingest_replies", methods=["GET"])
def ingest_replies_handler():
    """
    Extrae replies NUEVOS dirigidos a tus cuentas,
    y guarda solo los que pertenecen a tweets (root) de las últimas 48h.
    Diseñado para corridas frecuentes (cada 5-10 min) y para no reventar rate limits.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 1) tweets (root) recientes (48h)
        root_ids = fetch_recent_root_tweetids(cursor, hours_back=48, cap=5000)
        if not root_ids:
            return jsonify({"ok": True, "msg": "No hay tweets root en las últimas 48h"}), 200

        # 2) Obtener usernames (mejor que user_id para operador to:)
        #    Si tu TweetUser.TweetUser guarda username, úsalo:
        cursor.execute("SELECT TweetUser FROM TweetUser WHERE TweetUser IS NOT NULL AND TweetUser<>''")
        usernames = [r[0] for r in cursor.fetchall()]

        # 3) procesar pocas cuentas por corrida (anti-timeout)
        batch_size = int(os.environ.get("REPLIES_BATCH_USERS", "3"))
        start_idx = int(get_state(cursor, "replies_user_idx", "0") or "0")

        selected = []
        for i in range(batch_size):
            if not usernames:
                break
            selected.append(usernames[(start_idx + i) % len(usernames)])

        # guardo próximo inicio (round-robin)
        next_start = (start_idx + len(selected)) % max(len(usernames), 1)
        set_state(cursor, "replies_user_idx", str(next_start))

        saved = 0
        rate_limited = False
        details = []

        for uname in selected:
            since_key = f"replies_since_id:{uname}"
            since_id = get_state(cursor, since_key, None)

            # paginar máximo 2 páginas por usuario (corto)
            next_token = None
            for page in range(2):
                data = x_search_replies_to_username(uname, since_id, next_token)

                if isinstance(data, dict) and data.get("rate_limited"):
                    rate_limited = True
                    details.append({"user": uname, "rate_limited": True, "retry_after": data.get("retry_after")})
                    break

                if isinstance(data, dict) and data.get("error"):
                    details.append({"user": uname, "error": True, "status": data.get("status"), "body": data.get("body")})
                    break

                tweets = data.get("data", []) if isinstance(data, dict) else []
                if not tweets:
                    break

                # filtrar por conversation_id ∈ root_ids
                max_seen = since_id
                for tw in tweets:
                    conv_id = str(tw.get("conversation_id") or "")
                    if conv_id and conv_id in root_ids:
                        insert_reply(cursor, conv_id, tw)
                        saved += 1
                    # actualizar max id visto para since_id (son strings numéricas)
                    tid = str(tw["id"])
                    if (max_seen is None) or (int(tid) > int(max_seen)):
                        max_seen = tid

                if max_seen:
                    set_state(cursor, since_key, str(max_seen))

                meta = data.get("meta", {})
                next_token = meta.get("next_token")
                if not next_token:
                    break

            if rate_limited:
                break

        conn.commit()
        return jsonify({
            "ok": True,
            "saved": saved,
            "users_processed": selected,
            "rate_limited": rate_limited,
            "details": details
        }), 200

    except Error as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass



if __name__ == "__main__":
    # Para correrlo localmente
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
