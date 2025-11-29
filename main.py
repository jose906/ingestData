# main.py
import os
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify
import requests
import MLModel
import mysql.connector
from mysql.connector import Error

# ================== CONFIG ==================

DB_CONFIG = {
    "host": "34.69.57.221",         # IP pública o nombre interno de Cloud SQL
    "user": "admin",
    "password": "Admin123!",
    "database": "Analisis",
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

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def get_last_tweet_id(cursor):
    cursor.execute("SELECT tweetid FROM Tweets ORDER BY tweetid DESC LIMIT 1")
    row = cursor.fetchone()
    if row:
        return str(row[0])
    return None


def get_or_create_tweet_user(cursor, username):
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
        "INSERT INTO TweetUser (TweetUser) VALUES (%s)",
        (username,)
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
       Lugar, Persona, Organizacion, Locacion, Otros, TweetUse)
    VALUES
      (%s, %s, %s, %s, %s, %s,
       %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
      text = VALUES(text),
      created = VALUES(created),
      url = VALUES(url),
      TweetUse = VALUES(TweetUse)
    """

    params = (
        tweet_id,
        text,
        created_dt,
        url,
        MLModel.get_sentiment(text)[0],    # sentimiento (llenarás después)
        MLModel.predecir_categoria(text)[0],    # categoria (llenarás con otro worker o script)
        None, None, None, None, None,
        tweetuser_id,
    )

    cursor.execute(sql, params)

# ================== TWITTER ==================

def fetch_new_tweets(last_tweet_id=None):
    tweets_url = 'https://api.twitter.com/2/tweets/search/recent'
    query = ' OR '.join([f'from:{uid}' for uid in USER_IDS])

    now_bolivia = datetime.now(BOLIVIA_TZ)

    params = {
        'query': query,
        'max_results': 100,
        'tweet.fields': 'created_at,text,entities,author_id',
        'expansions': 'attachments.media_keys,author_id',
        'media.fields': 'url',
        'user.fields': 'username'
    }

    if last_tweet_id is None:
        start_time = (
            now_bolivia
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .astimezone(timezone.utc)
            .isoformat()
            .replace('+00:00', 'Z')
        )
        params['start_time'] = start_time
        print(f"[INGEST] BD vacía -> start_time={start_time}")
    else:
        params['since_id'] = last_tweet_id
        print(f"[INGEST] Usando since_id={last_tweet_id}")

    all_tweets = []
    all_users = []

    response = requests.get(tweets_url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Error Twitter API: {response.status_code} - {response.text}")
    data = response.json()

    all_tweets.extend(data.get('data', []))
    includes = data.get('includes', {})
    all_users.extend(includes.get('users', []))

    # paginación
    while 'next_token' in data.get('meta', {}):
        params['pagination_token'] = data['meta']['next_token']
        response = requests.get(tweets_url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Error paginación Twitter API: {response.status_code} - {response.text}")
        data = response.json()
        all_tweets.extend(data.get('data', []))
        includes = data.get('includes', {})
        all_users.extend(includes.get('users', []))

    return all_tweets, all_users



# ================== CLOUD RUN HANDLER ==================

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
            tweetuser_id = get_or_create_tweet_user(cursor, username)
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
    return "OK - DB Connected", 200

if __name__ == "__main__":
    # Para correrlo localmente
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
