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
            
 
if __name__ == "__main__":
    users = get_users()
    if not users:
            update_all()
            users = get_users()# Obtener los primeros 5 usuarios
    for user in users:
        print(f"Procesando usuario: {user['TweetUser']} (ID: {user['idTweetUser']})")
        tweets = fetch_tweets_for_user2(user['TweetUser'], user['idTweetUser'], user['last_tweetid'])
        print(f"Tweets obtenidos para {user['TweetUser']}: {len(tweets)}")
        
        