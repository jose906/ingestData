from contextlib import closing
from database import get_db_connection
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from itertools import islice
import helper


UMBRAL = 0.8
MARGEN_MINIMO = float(os.getenv("TOPIC_MIN_MARGIN", "0.03"))
UMBRAL_CENTROIDE = float(os.getenv("TOPIC_CENTROID_MIN_SIMILARITY", "0.85"))

TOPICS_ESPECIALES = [
    "__SOLO_LINK__",
    "__SALUDO__",
    "__PORTADA__",
    "__EPAPER__",
    "__PROMOCION__",
    "__PROGRAMACION__",
    "__RESUMEN__"
]



def get_topic_embeddings():
    with closing(get_db_connection()) as connection, closing(connection.cursor(dictionary=True)) as cursor:
        placeholders = ",".join(["%s"] * len(TOPICS_ESPECIALES))

        query = f"""
            SELECT te.topic_id, te.embedding_vector
            FROM topic_embeddings te
            JOIN topics t
                ON t.topic_id = te.topic_id
            WHERE t.topic_name NOT IN ({placeholders})
            ORDER BY te.topic_id
        """

        cursor.execute(query, TOPICS_ESPECIALES)
        results = cursor.fetchall()

    topics = []

    for row in results:
        topics.append({
            "topic_id": row["topic_id"],
            "embedding": json.loads(row["embedding_vector"])
        })

    return topics


def get_tweet_embedding():
    with closing(get_db_connection()) as connection, closing(connection.cursor(dictionary=True)) as cursor:
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
              AND t.created >= CONVERT_TZ(UTC_TIMESTAMP(), '+00:00', '-04:00') - INTERVAL 14 DAY
        """

        cursor.execute(query)
        results = cursor.fetchall()

    tweets = []

    for row in results:
        tweets.append({
            "tweetid": row["tweetid"],
             "text": row["text"],
            "embedding": json.loads(row["embedding_vector"])
        })

    return tweets

def get_especial_topics():

    with closing(get_db_connection()) as connection, closing(connection.cursor(dictionary=True)) as cursor:
        placeholders = ",".join(["%s"] * len(TOPICS_ESPECIALES))

        query = f"""
            SELECT topic_id, topic_name
            FROM topics
            WHERE topic_name IN ({placeholders})
        """

        cursor.execute(query, TOPICS_ESPECIALES)
        results = cursor.fetchall()

    topicos = {}

    for row in results:
        topicos[row["topic_name"]] = row["topic_id"]

    return topicos


def insert_tweets_topic(df, ids_especiales):
    if df.empty:
        print("No hay tweets para insertar")
        return

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        insert_query = """
            INSERT INTO topic_tweets 
                (topic_id, tweetid, similarity, assigned_at)
            VALUES 
                (%s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                tweetid = VALUES(tweetid)
        """

        tweets_topic = df.to_dict(orient="records")

        data_to_insert = [
            (
                item["topic_id"],
                item["tweetid"],
                item["similarity"]
            )
            for item in tweets_topic
        ]

        cursor.executemany(
            insert_query,
            data_to_insert
        )

        conteo_topics = Counter(
            item["topic_id"]
            for item in tweets_topic
        )

        update_query = """
            UPDATE topics
            SET
                total_tweets = (
                    SELECT COUNT(*)
                    FROM topic_tweets tt
                    WHERE tt.topic_id = topics.topic_id
                ),
                last_seen = NOW()
            WHERE topic_id = %s
        """

        data_update = [
            (topic_id,)
            for topic_id in conteo_topics.keys()
        ]

        cursor.executemany(
            update_query,
            data_update
        )

        for topic_id in sorted(conteo_topics.keys() - ids_especiales):
            recalcular_centroide_topic(cursor, topic_id)

        connection.commit()

        print("Tweets insertados:", len(tweets_topic))
        print("Topics actualizados:", len(conteo_topics))

    except Exception as e:

        if connection:
            connection.rollback()

        print(f"Error insertando tweets en topics: {e}")

        raise

    finally:

        try:
            if cursor is not None:
                cursor.close()
        finally:
            if connection is not None:
                connection.close()


def recalcular_centroide_topic(cursor, topic_id):
    query = """
        SELECT e.embedding_vector
        FROM topic_tweets tt
        JOIN tweet_embeddings e
            ON tt.tweetid = e.tweetid
        WHERE tt.topic_id = %s
          AND tt.similarity >= %s
    """

    cursor.execute(query, (topic_id, UMBRAL_CENTROIDE))
    results = cursor.fetchall()

    if not results:
        return

    embeddings = np.array(
        [
            json.loads(row["embedding_vector"])
            for row in results
        ],
        dtype=np.float32
    )

    centroide = np.mean(embeddings, axis=0)

    norma = np.linalg.norm(centroide)

    if norma > 0:
        centroide = centroide / norma

    update_query = """
    INSERT INTO topic_embeddings (
        topic_id,
        embedding_vector,
        updated_at
    )
    VALUES (%s, %s, NOW())
    ON DUPLICATE KEY UPDATE
        embedding_vector = VALUES(embedding_vector),
        updated_at = NOW()
"""

    cursor.execute(
        update_query,
        (
            topic_id,
            json.dumps(centroide.tolist())
        )
    )


    print(f"Centroide actualizado para topic {topic_id}")

def classify_tweets():
    df_topics = pd.DataFrame(get_topic_embeddings())
    df_tweets = pd.DataFrame(get_tweet_embedding())
    topicos_especiales = get_especial_topics()

    if df_tweets.empty:
        print("No hay tweets nuevos para clasificar")
        return pd.DataFrame(), pd.DataFrame()

    asignados = []
    no_asignados = []

    tweets_normales = []

    for tweet in df_tweets.to_dict(orient="records"):
        topic_especial = helper.detectar_topic_especial(tweet["text"])
        topic_id = topicos_especiales.get(topic_especial)
        if topic_id is None:
            tweets_normales.append(tweet)
        else:
            asignados.append({
                "tweetid": tweet["tweetid"],
                "topic_id": topic_id,
                "similarity": 1.0,
            })

    if tweets_normales and df_topics.empty:
        no_asignados = [
            {**tweet, "similarity": 0.0, "reason": "no_topics"}
            for tweet in tweets_normales
        ]
    elif tweets_normales:
        topic_ids = df_topics["topic_id"].tolist()
        topic_matrix = np.asarray(df_topics["embedding"].tolist(), dtype=np.float32)
        batch_size = 1000

        for start in range(0, len(tweets_normales), batch_size):
            batch = tweets_normales[start:start + batch_size]
            tweet_matrix = np.asarray([tweet["embedding"] for tweet in batch], dtype=np.float32)
            similarities = cosine_similarity(tweet_matrix, topic_matrix)
            rows = np.arange(len(batch))
            best_indices = similarities.argmax(axis=1)
            best_scores = similarities[rows, best_indices]
            second_indices = None

            if len(topic_ids) > 1:
                similarities[rows, best_indices] = -np.inf
                second_indices = similarities.argmax(axis=1)

            for row, tweet in enumerate(batch):
                topic_id = topic_ids[best_indices[row]]
                score = float(best_scores[row])
                second_topic_id = second_score = margin = None
                if second_indices is not None:
                    second_idx = second_indices[row]
                    second_topic_id = topic_ids[second_idx]
                    second_score = float(similarities[row, second_idx])
                    margin = score - second_score

                if score >= UMBRAL and (margin is None or margin >= MARGEN_MINIMO):
                    asignados.append({
                        "tweetid": tweet["tweetid"],
                        "topic_id": topic_id,
                        "similarity": score,
                    })
                else:
                    no_asignados.append({
                        **tweet,
                        "similarity": score,
                        "candidate_topic_id": topic_id,
                        "second_topic_id": second_topic_id,
                        "second_similarity": second_score,
                        "margin": margin,
                        "reason": "below_threshold" if score < UMBRAL else "ambiguous",
                    })

    print("Asignados:", len(asignados))
    print("No asignados por motivo:", dict(Counter(tweet["reason"] for tweet in no_asignados)))
    ambiguos = (tweet for tweet in no_asignados if tweet["reason"] == "ambiguous")
    for tweet in islice(ambiguos, 10):
        print("Asignación ambigua:", {
            key: tweet[key] for key in (
                "tweetid", "candidate_topic_id", "similarity",
                "second_topic_id", "second_similarity", "margin",
            )
        })

    insert_tweets_topic(pd.DataFrame(asignados), set(topicos_especiales.values()))

    return (
        pd.DataFrame(no_asignados),
        pd.DataFrame(asignados)
    )
