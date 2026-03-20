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
    return 1



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


def fetch_tweets_for_user(username: str, last_tweetid, userid=None ):
    params = {
        "query": f"from:{username}",
        "max_results": 100,
        "tweet.fields": "created_at,text,entities,author_id",
    }
    a = False
    

    if not last_tweetid or last_tweetid in (None, "") or last_tweetid == 1:
        a = True
    else:
        if int(last_tweetid) > 2031390546013978624:
            params["since_id"] = str(last_tweetid)
        else: 
            a = True

    tweets_data = []
    tweets_response = requests.get(TWEETS_URL, headers=headers, params=params)
    if tweets_response.status_code != 200:

        if tweets_response.status_code == 400:
            try:
                error_json = tweets_response.json()

                errors = error_json.get("errors", [])
                if errors:
                    message = errors[0].get("message", "")

                    if "since_id" in message:
                        print(f"⚠️ since_id inválido para user {userid}, reseteando...")

                        conn = get_db_connection()
                        cur = conn.cursor()

                        update_last_tweetid(cur, userid, 1)

                        conn.commit()
                        cur.close()
                        conn.close()

                        return []  # 👈 IMPORTANTE: no romper el flujo

            except Exception as e:
                print("Error parsing JSON:", e)

    # otros errores reales
    raise Exception(f"Error: {tweets_response.status_code} - {tweets_response.text}")

    j = tweets_response.json()
    tweets_data.extend(j.get("data", []))

   

    return tweets_data
    
'''
def fetch_tweets_for_user(user_id: str, last_tweetid):

    params = {
        "max_results": 100,
        "tweet.fields": "created_at,text,entities,author_id"
    }

    if last_tweetid not in (None, "", "1"):
        params["since_id"] = str(last_tweetid)

    url = TWEETS_URL.format(user_id)

    tweets_data = []
    next_token = None

    while True:

        if next_token:
            params["pagination_token"] = next_token

        r = requests.get(url, headers=headers, params=params, timeout=20)

        if r.status_code != 200:
            raise Exception(f"Error {r.status_code}: {r.text}")

        j = r.json()

        tweets = j.get("data", [])
        tweets_data.extend(tweets)

        meta = j.get("meta", {})
        next_token = meta.get("next_token")

        if not next_token:
            break

    return tweets_data
'''
    
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

    

def max_tweet_id(tweets):
    if not tweets:
        return None
    m = max(int(t["id"]) for t in tweets if "id" in t)
    return str(m)

def get_users(limit=1):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT idTweetUser, TweetUser, last_tweetid, tweets_procesados
            FROM TweetUser
            WHERE tweets_procesados = 0
            ORDER BY idTweetUser ASC
            LIMIT %s
        """, (limit,))
        return cur.fetchall()
    except Exception as e:
        print(f"Error fetching users: {e}")
        return []
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def update_all():
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE TweetUser SET tweets_procesados = 0")
        conn.commit()
    except Exception as e:
        print(f"Error resetting users: {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

# ================== CLOUD RUN HANDLER ==================

@app.route("/ingest", methods=["GET"])
def ingest_handler():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()  # cursor normal para inserts/updates

        users = get_users()
        if not users:
            update_all()
            users = get_users()

        total_saved = 0
        per_user_stats = []

        for u in users:
            tweetuser_pk_id = int(u["idTweetUser"])
            username = (u.get("TweetUser") or "").strip()
            last_tid = u.get("last_tweetid")
            last_tid = str(last_tid) if last_tid not in (None, "") else None
            user_id = str(u["idTweetUser"])

            #tweets = fetch_tweets_for_user(user_id, last_tid)

            tweets = fetch_tweets_for_user(username, last_tid,user_id)
            if not tweets:
                # si no hay tweets, igual marca procesado si quieres evitar loop infinito
                
                update_last_tweetid(cur, u["idTweetUser"], last_tid )
                conn.commit()
                continue

            saved = 0
            for t in tweets:
                saved += insert_or_update_tweet(cur, t, tweetuser_pk_id)

            # actualiza last_tweetid
            max_tid = max_tweet_id(tweets)
            if max_tid:
                update_last_tweetid(cur, u["idTweetUser"], max_tid)

            conn.commit()

            total_saved += saved
            per_user_stats.append({
                "username": username,
                "tweets_guardados": saved,
                "last_tweetid": max_tid
            })

        return jsonify({
            "ok": True,
            "procesados": len(per_user_stats),
            "tweets_guardados": total_saved,
            "detalle": per_user_stats
        }), 200

    except Exception as e:
        if conn:
            try: conn.rollback()
            except Exception: pass
        return jsonify({"ok": False, "error": str(e)}), 500

    finally:
        if conn:
            try: conn.close()
            except Exception: pass

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
'''
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

'''
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
        batch_size = 1
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
                

                if isinstance(data, dict) and data.get("since_id_expired"):
                    print(f"⚠️ since_id expirado para {uname}: {since_id}")
                    set_state(cursor, since_key, "")   # o None si tu tabla lo maneja
                    since_id = None
                    next_token = None
                    data = x_search_replies_to_username(uname, None, None)

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
