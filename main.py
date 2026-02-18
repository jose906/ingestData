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

MAX_RULE_LEN = 512
SAFE_RULE_LEN = 500        # margen para no pegar justo al límite
TIME_BUDGET_S = 40         # corta paginación antes del timeout (ajusta)
DEFAULT_MINUTES_BACK = 15  # primera vez por batch
OVERLAP_MINUTES = 5  

# Si ML es pesado, pon True para NO hacerlo en ingesta (evita timeout)
DISABLE_ML_DURING_INGEST = False



#TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
TWITTER_BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAN9WpgEAAAAAHarp9HjcuJFZ4wtx1DtpsP8Z93A%3DC3AEHMO2YXaGFFgblPEdkYTGhBne75WLUlG5Mc95FGKlR003vg'

if not TWITTER_BEARER_TOKEN:
    raise RuntimeError("Falta TWITTER_BEARER_TOKEN en variables de entorno")

headers = {
    "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"
}

USER_IDS = [
    '65444625', '94438031', '935136477616443393', '209279715',
    '2800854409', '735321776', '44489439', '311132840',
    '525394081', '1742290477234208768', '118861947', '188870982'
]

BOLIVIA_TZ = timezone(timedelta(hours=-4))

app = Flask(__name__)

# ================== FUNCIONES BD ==================

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def get_state(conn, key: str, default: str = "") -> str:
    cur = conn.cursor()
    cur.execute("SELECT v FROM ingest_state WHERE k=%s", (key,))
    row = cur.fetchone()
    cur.close()
    return row[0] if row else default

def set_state(conn, key: str, value: str):
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO ingest_state (k, v) VALUES (%s, %s)
      ON DUPLICATE KEY UPDATE v=VALUES(v)
    """, (key, value))
    conn.commit()
    cur.close()

def get_last_tweet_id(cursor):
    cursor.execute("SELECT tweetid FROM Tweets ORDER BY tweetid DESC LIMIT 1")
    row = cursor.fetchone()
    if row:
        return str(row[0])
    return None

def get_users_id():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = []
        cursor.execute("SELECT idTweetUser FROM TweetUser")
        rows = cursor.fetchall()
        for row in rows:
            user_id.append(str(row[0]))
        cursor.close()
        conn.close()
        return user_id
    except Error as e:
        print(f"[INGEST][ERROR] MySQL: {e}")
        return []


# ================== BATCH BUILDER ==================

def build_query_batches(user_ids, max_len=SAFE_RULE_LEN):
    """
    Crea batches tipo: from:1 OR from:2 ... sin pasarse de max_len.
    """
    batches = []
    current = ""

    for uid in user_ids:
        term = f"from:{uid}"
        candidate = term if not current else (current + " OR " + term)

        if len(candidate) > max_len:
            if current:
                batches.append(current)
            current = term
        else:
            current = candidate

    if current:
        batches.append(current)

    return batches


# ================== INSERT / UPDATE ==================

def insert_or_update_tweet(cursor, tweet, tweetuser_id):
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

    if DISABLE_ML_DURING_INGEST:
        sent = None
        cat = None
    else:
        sent = MLModel.get_sentiment(text)[0]
        cat = MLModel.predecir_categoria(text)[0]

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
      TweetUser_idTweetUser = VALUES(TweetUser_idTweetUser),
      sentimiento = COALESCE(VALUES(sentimiento), sentimiento),
      categoria = COALESCE(VALUES(categoria), categoria)
    """

    params = (
        tweet_id,
        text,
        created_dt,
        url,
        sent,
        cat,
        "", "", "", "", "",
        tweetuser_id,
    )

    cursor.execute(sql, params)


# ================== TWITTER FETCH (CON TIME BUDGET) ==================

def fetch_new_tweets(query: str, start_time_utc: str | None = None, time_budget_s: int = TIME_BUDGET_S):
    tweets_url = "https://api.twitter.com/2/tweets/search/recent"
    t0 = time.time()

    params = {
        "query": query,
        "max_results": 100,
        "tweet.fields": "created_at,text,entities,author_id",
        "expansions": "attachments.media_keys,author_id",
        "media.fields": "url",
        "user.fields": "username",
    }
    if start_time_utc:
        params["start_time"] = start_time_utc

    all_tweets, all_users = [], []

    response = requests.get(tweets_url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Error Twitter API: {response.status_code} - {response.text}")

    data = response.json()
    all_tweets.extend(data.get("data", []))
    includes = data.get("includes", {})
    all_users.extend(includes.get("users", []))

    # paginación con corte por presupuesto de tiempo
    while "next_token" in data.get("meta", {}):
        if time.time() - t0 > time_budget_s:
            print("[INGEST] Cortando paginación por TIME_BUDGET_S para evitar timeout.")
            break

        params["pagination_token"] = data["meta"]["next_token"]
        response = requests.get(tweets_url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Error paginación Twitter API: {response.status_code} - {response.text}")

        data = response.json()
        all_tweets.extend(data.get("data", []))
        includes = data.get("includes", {})
        all_users.extend(includes.get("users", []))

    return all_tweets, all_users


# ================== CLOUD RUN HANDLER ==================

@app.route("/ingest", methods=["GET"])
def ingest_handler():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

    

        user_ids = get_users_id()
        if not user_ids:
            return jsonify({"ok": True, "msg": "No hay usuarios en TweetUser", "nuevos": 0}), 200

        batches = build_query_batches(user_ids, max_len=SAFE_RULE_LEN)
        if not batches:
            return jsonify({"ok": True, "msg": "No se pudo construir batches", "nuevos": 0}), 200

        # round-robin
        idx = int(get_state(conn, "batch_idx", "0"))
        idx = idx % len(batches)
        query = batches[idx]

        # start_time por batch (persistido) con overlap
        key_last = f"batch_last_utc_{idx}"
        last_utc = get_state(conn, key_last, "")

        if last_utc:
            last_dt = datetime.fromisoformat(last_utc.replace("Z", "+00:00"))
            start_dt = last_dt - timedelta(minutes=OVERLAP_MINUTES)
        else:
            start_dt = datetime.now(timezone.utc) - timedelta(minutes=DEFAULT_MINUTES_BACK)

        start_time_utc = start_dt.isoformat().replace("+00:00", "Z")

        print(f"[INGEST] batches={len(batches)} idx={idx} query_len={len(query)} start_time={start_time_utc}")

        tweets, users = fetch_new_tweets(query=query, start_time_utc=start_time_utc, time_budget_s=TIME_BUDGET_S)

        if not tweets:
            # avanzar batch aunque no haya datos
            set_state(conn, "batch_idx", str(idx + 1))
            set_state(conn, key_last, datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
            return jsonify({"ok": True, "batch": idx, "batches": len(batches), "nuevos": 0}), 200

        users_map = {u["id"]: u.get("username") for u in users}

        count = 0
        for t in tweets:
            author_id = t.get("author_id")
            _username = users_map.get(author_id)  # por si lo necesitas después
            insert_or_update_tweet(cursor, t, author_id)
            count += 1

        conn.commit()

        # guardar estado
        set_state(conn, "batch_idx", str(idx + 1))
        set_state(conn, key_last, datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

        return jsonify({
            "ok": True,
            "batch": idx,
            "batches": len(batches),
            "query_len": len(query),
            "time_budget_s": TIME_BUDGET_S,
            "nuevos": count
        }), 200

    except Error as e:
        print(f"[INGEST][ERROR] MySQL: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        print(f"[INGEST][ERROR] General: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        try:
            cursor.close()
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
