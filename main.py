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

USER_IDS = [
    '65444625', '94438031', '935136477616443393', '209279715',
    '2800854409', '735321776', '44489439', '311132840',
    '525394081', '1742290477234208768', '118861947', '188870982'
]

BOLIVIA_TZ = timezone(timedelta(hours=-4))

app = Flask(__name__)

# ================== FUNCIONES BD ==================


def chunk_list(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def get_last_tweet_id_for_user(conn, user_id: str):
    cur = conn.cursor()
    cur.execute("SELECT last_tweet_id FROM tweet_ingest_state WHERE user_id=%s", (user_id,))
    row = cur.fetchone()
    cur.close()
    return str(row[0]) if row and row[0] else None

def upsert_last_tweet_id_for_user(conn, user_id: str, last_tweet_id: str):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tweet_ingest_state (user_id, last_tweet_id)
        VALUES (%s, %s)
        ON DUPLICATE KEY UPDATE last_tweet_id=VALUES(last_tweet_id)
        """,
        (user_id, last_tweet_id)
    )
    conn.commit()
    cur.close()


def get_last_tweet_id(cursor):
    cursor.execute("SELECT tweetid FROM Tweets ORDER BY tweetid DESC LIMIT 1")
    row = cursor.fetchone()
    if row:
        return str(row[0])
    return None


def get_or_create_tweet_user(cursor, username, id_tweetuser):
    if not username:
        return None

    cursor.execute(
        "SELECT idTweetUser FROM TweetUser WHERE TweetUser = %s",
        (username,)
    )
    row = cursor.fetchone()
    if row:
        return row[0]

    cursor.execute(
        "INSERT INTO TweetUser (idTweetUser, TweetUser) VALUES (%s, %s)",
        (id_tweetuser, username or "")
    )
    return cursor.lastrowid


def insert_or_update_tweet(cursor, tweet, tweetuser_id):
    tweet_id = tweet["id"]
    text = tweet.get("text", "")
    created_at = tweet.get("created_at")

    created_dt = None
    if created_at:
        try:
            created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except Exception:
            created_dt = created_at  # fallback string

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
        MLModel.predecir_categoria(text)[0],   # categoria (llenarás con otro worker o script)
        "", "", "", "", "",
        tweetuser_id,
    )

    cursor.execute(sql, params)
def get_users_id():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = []
        cursor.execute("SELECT idTweetUser FROM TweetUser")
        rows = cursor.fetchall()
        for row in rows:
            user_id.append(str(row[0]))
    except Error as e:
        print(f"[INGEST][ERROR] MySQL: {e}")
    return user_id
    
# ================== TWITTER ==================

def fetch_new_tweets(last_tweet_id=None):
    tweets_url = "https://api.twitter.com/2/tweets/search/recent"
    user_ids = get_users_id()

    now_bolivia = datetime.now(BOLIVIA_TZ)

    all_tweets = []
    all_users = []

    for user_batch in chunk_list(user_ids, 20):  # 20 usuarios por query
        query = " OR ".join([f"from:{uid}" for uid in user_batch])

        params = {
            "query": query,
            "max_results": 100,
            "tweet.fields": "created_at,text,entities,author_id",
            "expansions": "attachments.media_keys,author_id",
            "media.fields": "url",
            "user.fields": "username",
        }

        # filtro temporal (se aplica por batch)
        if last_tweet_id is None:
            start_time = (
                now_bolivia
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .astimezone(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
            params["start_time"] = start_time
            print(f"[INGEST] BD vacía -> start_time={start_time} (batch={len(user_batch)})")
        else:
            params["since_id"] = last_tweet_id
            print(f"[INGEST] Usando since_id={last_tweet_id} (batch={len(user_batch)})")

        # 1ra llamada
        response = requests.get(tweets_url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Error Twitter API: {response.status_code} - {response.text}")

        data = response.json()
        all_tweets.extend(data.get("data", []))

        includes = data.get("includes", {})
        all_users.extend(includes.get("users", []))

        # paginación por batch
        while "next_token" in data.get("meta", {}):
            params["pagination_token"] = data["meta"]["next_token"]

            response = requests.get(tweets_url, headers=headers, params=params)
            if response.status_code != 200:
                raise Exception(f"Error paginación Twitter API: {response.status_code} - {response.text}")

            data = response.json()
            all_tweets.extend(data.get("data", []))

            includes = data.get("includes", {})
            all_users.extend(includes.get("users", []))

    return all_tweets, all_users

def fetch_new_tweets_per_user():
    tweets_url = "https://api.twitter.com/2/tweets/search/recent"
    user_ids = get_users_id()

    conn = get_db_connection()

    all_tweets = []
    all_users = []

    for uid in user_ids:
        since_id = get_last_tweet_id_for_user(conn, uid)

        params = {
            "query": f"from:{uid}",
            "max_results": 100,
            "tweet.fields": "created_at,text,entities,author_id",
            "expansions": "attachments.media_keys,author_id",
            "media.fields": "url",
            "user.fields": "username",
        }

        if since_id:
            params["since_id"] = since_id
            print(f"[INGEST] user={uid} since_id={since_id}")
        else:
            print(f"[INGEST] user={uid} sin since_id (primera vez)")

        newest_seen = None

        while True:
            r = requests.get(tweets_url, headers=headers, params=params)
            if r.status_code != 200:
                raise Exception(f"Error Twitter API: {r.status_code} - {r.text}")

            data = r.json()

            tweets = data.get("data", [])
            all_tweets.extend(tweets)

            includes = data.get("includes", {})
            all_users.extend(includes.get("users", []))

            # actualizar newest visto para este usuario
            # (tweets vienen en orden reciente -> el primero suele ser el más nuevo)
            if tweets:
                newest_seen = max(newest_seen or "0", max(t["id"] for t in tweets), key=int)

            meta = data.get("meta", {})
            if "next_token" not in meta:
                break
            params["pagination_token"] = meta["next_token"]

        # guardamos el last_tweet_id por usuario
        if newest_seen:
            upsert_last_tweet_id_for_user(conn, uid, newest_seen)

        # opcional: pequeño sleep para no reventar rate limit
        time.sleep(0.2)

    conn.close()
    return all_tweets, all_users


# ================== CLOUD RUN HANDLER ==================
@app.route("/ingesta", methods=["GET"])
def ingest():
    tweets, users = fetch_new_tweets_per_user()
    return {
        "ok": True,
        "tweets": len(tweets),
        "users": len(users)
    }


@app.route("/ingest", methods=["GET"])
def ingest_handler():
    """
    Endpoint que ejecuta UNA sola pasada de ingesta.
    Cloud Scheduler llamará a /ingest cada X minutos.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        last_tweet_id = get_last_tweet_id(cursor)
        print(f"[INGEST] Último tweetid en BD: {last_tweet_id}")

        tweets, users = fetch_new_tweets(last_tweet_id)

        if not tweets:
            print("[INGEST] No hay tweets nuevos.")
            return jsonify({"ok": True, "nuevos": 0}), 200

        users_map = {u["id"]: u.get("username") for u in users}

        count = 0
        for t in tweets:
            author_id = t.get("author_id")
            username = users_map.get(author_id)
            tweetuser_id = author_id
            insert_or_update_tweet(cursor, t, tweetuser_id)
            count += 1

        conn.commit()
        print(f"[INGEST] Guardados {count} tweets nuevos.")
        return jsonify({"ok": True, "nuevos": count}), 200

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
