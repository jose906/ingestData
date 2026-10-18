import requests
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta, timezone
import pandas as pd

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

def get_topics(limit_topics=2):
    connection = get_db_connection()

    query = """
    SELECT
        tp.topic_id,
        tp.topic_name,
        tp.topic_keywords,
        tp.first_seen,
        tp.last_seen,
        tp.total_tweets,
        tt.tweetid,
        tt.similarity,
        tw.text
    FROM (
        SELECT *
        FROM topics
        WHERE topic_name LIKE %s
        ORDER BY topic_id
        LIMIT %s
    ) tp
    JOIN topic_tweets tt
        ON tp.topic_id = tt.topic_id
    JOIN Tweets tw
        ON tt.tweetid = tw.tweetid
    ORDER BY tp.topic_id, tt.similarity DESC
    """

    try:
        return pd.read_sql(
            query,
            connection,
            params=("%NUEVO_CLUSTER%", limit_topics)
        )
    finally:
        connection.close()



print(get_topics() )