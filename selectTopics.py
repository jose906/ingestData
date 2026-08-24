import re
import json
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np
import os


URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
UMBRAL = 0.8

"""DB_CONFIG = {
    "host": "34.69.57.221",
    "user": "admin",
    "password": "Admin123!",
    "database": "Analisis",
    "port": 3306,
}"""

DB_CONFIG = {
            # IP pública o nombre interno de Cloud SQL
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME"),
    "unix_socket": f"/cloudsql/{os.environ.get('INSTANCE_CONNECTION_NAME')}",
    "charset": "utf8mb4",
    "port": "3306",
}
  

TOPICS_ESPECIALES = [
    "__SOLO_LINK__",
    "__SALUDO__",
    "__PORTADA__",
    "__EPAPER__",
    "__PROMOCION__",
    "__PROGRAMACION__",
    "__RESUMEN__"
]


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def get_topic_embeddings():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT topic_id, embedding_vector
        FROM topic_embeddings
        ORDER BY topic_id
    """
    cursor.execute(query)
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    topics = []

    for row in results:
        topics.append({
            "topic_id": row["topic_id"],
            "embedding": json.loads(row["embedding_vector"])
        })

    return topics


def get_tweet_embedding():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT 
            t.tweetid,
            t.text,
            e.embedding_vector
        FROM Tweets t
        JOIN tweet_embeddings e 
            ON t.tweetid = e.tweetid
        LEFT JOIN topic_tweets tt 
            ON t.tweetid = tt.tweetid
        WHERE tt.tweetid IS NULL
          AND t.created >= NOW() - INTERVAL 14 DAY
    """

    cursor.execute(query)
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    tweets = []

    for row in results:
        tweets.append({
            "tweetid": row["tweetid"],
             "text": row["text"],
            "embedding": json.loads(row["embedding_vector"])
        })

    return tweets




def insert_tweets_topic(df):
    if df.empty:
        print("No hay tweets para insertar")
        return

    

    connection = get_db_connection()
    cursor = connection.cursor()

    insert_query = """
        INSERT INTO topic_tweets 
            (topic_id, tweetid, similarity, assigned_at)
        VALUES 
            (%s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            similarity = VALUES(similarity),
            assigned_at = NOW()
    """

    tweets_topic = df.to_dict(orient="records")
    data_to_insert = [
        (item["topic_id"], item["tweetid"], item["similarity"])
        for item in tweets_topic
    ]

    cursor.executemany(insert_query, data_to_insert)

    conteo_topics = Counter(item["topic_id"] for item in tweets_topic)

    update_query = """
        UPDATE topics
        SET 
            total_tweets = total_tweets + %s,
            last_seen = NOW()
        WHERE topic_id = %s
    """

    data_update = [
        (cantidad, topic_id)
        for topic_id, cantidad in conteo_topics.items()
    ]

    cursor.executemany(update_query, data_update)

    connection.commit()

    cursor.close()
    connection.close()

    print("Tweets insertados:", len(tweets_topic))
    print("Topics actualizados:", len(conteo_topics))


def classify_tweets():
    df_topics = pd.DataFrame(get_topic_embeddings())
    df_tweets = pd.DataFrame(get_tweet_embedding())

    if df_topics.empty:
        print("No hay topics con embeddings")
        return

    if df_tweets.empty:
        print("No hay tweets nuevos para clasificar")
        return

    topic_matrix = np.array(df_topics["embedding"].tolist(), dtype=np.float32)
    tweet_matrix = np.array(df_tweets["embedding"].tolist(), dtype=np.float32)

    sim_matrix = cosine_similarity(tweet_matrix, topic_matrix)

    best_idx = np.argmax(sim_matrix, axis=1)
    best_score = np.max(sim_matrix, axis=1)

    asignados = []
    no_asignados = []

    for i in range(len(df_tweets)):
        tweetid = df_tweets.iloc[i]["tweetid"]
        score = float(best_score[i])

        if score >= UMBRAL:
            topic_id = int(df_topics.iloc[best_idx[i]]["topic_id"])

            asignados.append({
                "tweetid": tweetid,
                "topic_id": topic_id,
                "similarity": score
            })

        else:
            no_asignados.append({
            "tweetid": tweetid,
            "text": df_tweets.iloc[i]["text"],
            "embedding": df_tweets.iloc[i]["embedding"],
            "similarity": score
            })

    print("--------------------------------")
    print("Asignados:", len(asignados))
    print("No asignados:", len(no_asignados))
    print("--------------------------------")

    print(pd.DataFrame(asignados).head())
    print(pd.DataFrame(no_asignados).head())

    insert_tweets_topic(pd.DataFrame(asignados))
    
    return pd.DataFrame(no_asignados), pd.DataFrame(asignados)
    







