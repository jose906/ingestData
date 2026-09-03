import re
import json
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import os
import helper


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
    required_vars = [
        "DB_USER",
        "DB_PASS",
        "DB_NAME",
        "INSTANCE_CONNECTION_NAME"
    ]

    missing = [
        var
        for var in required_vars
        if not os.environ.get(var)
    ]

    if missing:
        raise RuntimeError(
            f"Faltan variables de entorno: {', '.join(missing)}"
        )

    return mysql.connector.connect(**DB_CONFIG)


def get_topic_embeddings():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

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

def get_especial_topics():

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    placeholders = ",".join(["%s"] * len(TOPICS_ESPECIALES))

    query = f"""
        SELECT topic_id, topic_name
        FROM topics
        WHERE topic_name IN ({placeholders})
    """

    cursor.execute(query, TOPICS_ESPECIALES)
    results = cursor.fetchall()

    cursor.close()
    connection.close()

    topicos = {}

    for row in results:
        topicos[row["topic_name"]] = row["topic_id"]

    return topicos


def insert_tweets_topic(df):
    if df.empty:
        print("No hay tweets para insertar")
        return

    connection = None
    cursor = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

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

        connection.commit()

        print("Tweets insertados:", len(tweets_topic))
        print("Topics actualizados:", len(conteo_topics))

    except Exception as e:

        if connection:
            connection.rollback()

        print(f"Error insertando tweets en topics: {e}")

        raise

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


def recalcular_centroide_topic(topic_id):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    query = """
        SELECT e.embedding_vector
        FROM topic_tweets tt
        JOIN tweet_embeddings e
            ON tt.tweetid = e.tweetid
        WHERE tt.topic_id = %s
    """

    cursor.execute(query, (topic_id,))
    results = cursor.fetchall()

    if not results:
        cursor.close()
        connection.close()
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

    connection.commit()
    cursor.close()
    connection.close()

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

    # -----------------------------------------
    # 1. Detectar tópicos especiales primero
    # -----------------------------------------

    indices_normales = []

    for i in range(len(df_tweets)):
        tweetid = df_tweets.iloc[i]["tweetid"]
        texto = df_tweets.iloc[i]["text"]

        topic_especial = helper.detectar_topic_especial(texto)

        if topic_especial is not None:
            topic_especial_id = topicos_especiales.get(topic_especial)

            if topic_especial_id is not None:
                asignados.append({
                    "tweetid": tweetid,
                    "topic_id": topic_especial_id,
                    "similarity": 1.0
                })

                print("--------------------------------")
                print("TÓPICO ESPECIAL DETECTADO")
                print("Tipo:", topic_especial)
                print("Tweet:", texto[:150])
                print("--------------------------------")

                continue

        # Solo estos tweets necesitan similitud semántica
        indices_normales.append(i)

    # -----------------------------------------
    # 2. Clasificar solamente tweets normales
    # -----------------------------------------

    if indices_normales:

        if df_topics.empty:
            print("No hay topics normales con embeddings")

            for i in indices_normales:
                no_asignados.append({
                    "tweetid": df_tweets.iloc[i]["tweetid"],
                    "text": df_tweets.iloc[i]["text"],
                    "embedding": df_tweets.iloc[i]["embedding"],
                    "similarity": 0.0
                })

        else:
            df_tweets_normales = df_tweets.iloc[indices_normales].reset_index(
                drop=True
            )

            topic_matrix = np.array(
                df_topics["embedding"].tolist(),
                dtype=np.float32
            )

            tweet_matrix = np.array(
                df_tweets_normales["embedding"].tolist(),
                dtype=np.float32
            )

            BATCH_SIZE = 1000

            best_idx = []
            best_score = []

            for start in range(0, len(tweet_matrix), BATCH_SIZE):
                end = start + BATCH_SIZE

                batch = tweet_matrix[start:end]

                sim_batch = cosine_similarity(
                    batch,
                    topic_matrix
                )

                best_idx.extend(
                    np.argmax(sim_batch, axis=1)
                )

                best_score.extend(
                    np.max(sim_batch, axis=1)
                )

            best_idx = np.array(best_idx)
            best_score = np.array(best_score)

            # -----------------------------------------
            # 3. Aplicar umbral 0.80
            # -----------------------------------------

            for i in range(len(df_tweets_normales)):
                tweetid = df_tweets_normales.iloc[i]["tweetid"]
                score = float(best_score[i])

                if score >= UMBRAL:
                    topic_id = int(
                        df_topics.iloc[best_idx[i]]["topic_id"]
                    )

                    asignados.append({
                        "tweetid": tweetid,
                        "topic_id": topic_id,
                        "similarity": score
                    })

                else:
                    no_asignados.append({
                        "tweetid": tweetid,
                        "text": df_tweets_normales.iloc[i]["text"],
                        "embedding": df_tweets_normales.iloc[i]["embedding"],
                        "similarity": score
                    })

    # -----------------------------------------
    # 4. Resultados
    # -----------------------------------------

    print("--------------------------------")
    print("Asignados:", len(asignados))
    print("No asignados:", len(no_asignados))
    print("--------------------------------")

    print(pd.DataFrame(asignados).head())
    print(pd.DataFrame(no_asignados).head())

    # -----------------------------------------
    # 5. Guardar asignaciones
    # -----------------------------------------

    insert_tweets_topic(
        pd.DataFrame(asignados)
    )

    # -----------------------------------------
    # 6. Actualizar centroides solo normales
    # -----------------------------------------

    if asignados:
        ids_especiales = set(
            topicos_especiales.values()
        )

        topics_actualizados = {
            item["topic_id"]
            for item in asignados
            if item["topic_id"] not in ids_especiales
        }

        for topic_id in topics_actualizados:
            recalcular_centroide_topic(topic_id)

    return (
        pd.DataFrame(no_asignados),
        pd.DataFrame(asignados)
    )







