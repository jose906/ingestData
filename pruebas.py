import requests
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta, timezone

# ================== CONFIG BD ==================
DB_CONFIG = {
    
     "host": "34.69.57.221",      # o la IP de tu contenedor / Cloud SQL
    "user": "admin",
    "password": "Admin123!",
    "database": "Analisis",
    "port": 3306,
    
}


# ================== CONFIG TWITTER ==================
TWITTER_BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAN9WpgEAAAAAHarp9HjcuJFZ4wtx1DtpsP8Z93A%3DC3AEHMO2YXaGFFgblPEdkYTGhBne75WLUlG5Mc95FGKlR003vg'
TWEETS_URL = "https://api.twitter.com/2/tweets/search/recent"

SEARCH_URL = "https://api.x.com/2/tweets/search/recent"
# o tu token fijo mientras pruebas:
# TWITTER_BEARER_TOKEN = "AAAAAAAA..."

headers = {
    "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def fetch_tweets_for_user2(username:str, user_id: str, last_tweetid: str):
    
    params = {
        "query": f"from:{username}",
        "max_results": 100,
        "tweet.fields": "created_at,text,entities,author_id",
        
    }
    now = datetime.now(timezone.utc)
    safe_now = now - timedelta(seconds=60)

    start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if last_tweetid == '1':
        
        params["start_time"] = start_today.isoformat().replace("+00:00", "Z"),
        params["end_time"] = safe_now.isoformat().replace("+00:00", "Z"),
    elif last_tweetid == '0':
        pass
    else:
        params["since_id"] = str(last_tweetid)
        
    tweets_data = []
    tweets_response = requests.get(TWEETS_URL, headers=headers, params=params)
    if tweets_response.status_code != 200:
        return []

        

    j = tweets_response.json()
    tweets_data.extend(j.get("data", []))

   

    return tweets_data
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
        if cur:
            cur.close()
        if conn:
            conn.close()
            
def get_lasts_posts():
    con = None
    try:
        con = get_db_connection()
        cur = con.cursor(dictionary=True)
        cur.execute("""
            SELECT 
                t.tweetid,
                t.procesado,
                t.created AS tweet_created,
                r.replyid,
                r.created AS reply_created
            FROM Tweets t
            LEFT JOIN (
                SELECT tweetid, MAX(replyid) AS max_replyid
                FROM replies
                GROUP BY tweetid
            ) r_max ON t.tweetid = r_max.tweetid
            LEFT JOIN replies r 
                ON r.tweetid = r_max.tweetid 
                AND r.replyid = r_max.max_replyid
            WHERE t.created >= NOW() - INTERVAL 2 DAY AND t.procesado = 0;
        """)
        return cur.fetchall()
    except Exception as e:
        print(f"Error fetching last posts: {e}")
        return []
    finally:
        if cur:
            cur.close()
        if con:
            con.close()





def extraer_comentarios(tweet_ids, ultimo_reply_id=None, max_por_tweet=100):
    headers = {
        "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"
    }

    todos = []

    for tweet_id in tweet_ids:
        params = {
            "query": f"conversation_id:{tweet_id} is:reply",
            "tweet.fields": "id,text,created_at,author_id,conversation_id,in_reply_to_user_id,public_metrics",
            "expansions": "author_id",
            "user.fields": "id,name,username,verified,public_metrics",
            "max_results": min(max_por_tweet, 100),
            "sort_order": "recency"
        }

        if ultimo_reply_id:
            params["since_id"] = str(ultimo_reply_id)

        while True:
            response = requests.get(SEARCH_URL, headers=headers, params=params)

            if response.status_code != 200:
                print("Error:", response.status_code, response.text)
                break

            data = response.json()

            replies = data.get("data", [])
            users = {
                u["id"]: u for u in data.get("includes", {}).get("users", [])
            }

            for r in replies:
                autor = users.get(r["author_id"], {})

                todos.append({
                    "tweet_original_id": tweet_id,
                    "replyid": r["id"],
                    "text": r["text"],
                    "created": r["created_at"],
                    "author_id": r["author_id"],
                    "username": autor.get("username"),
                    "name": autor.get("name"),
                    "verified": autor.get("verified"),
                    "metrics": r.get("public_metrics", {})
                })

            next_token = data.get("meta", {}).get("next_token")

            if not next_token:
                break

            params["next_token"] = next_token

    return todos

def save_replies_to_db(replies):
    if not replies:
        return

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        for r in replies:
            cur.execute("""
                INSERT INTO replies (tweetid, replyid, text, created, author_id, username, name, verified, metrics)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    text = VALUES(text),
                    created = VALUES(created),
                    author_id = VALUES(author_id),
                    username = VALUES(username),
                    name = VALUES(name),
                    verified = VALUES(verified),
                    metrics = VALUES(metrics)
            """, (
                r["tweet_original_id"],
                r["replyid"],
                r["text"],
                r["created"],
                r["author_id"],
                r["username"],
                r["name"],
                r["verified"],
                str(r["metrics"])
            ))

        conn.commit()
    except Exception as e:
        print(f"Error saving replies: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()    
            
 
if __name__ == "__main__":
    tweets = get_lasts_posts()
    a = [t for t in tweets[0:10]]
    
    print(a)
        
        