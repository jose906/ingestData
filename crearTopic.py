import mysql.connector
import os
import time
import pandas as pd
from google import genai





DB_CONFIG = {
            # IP pública o nombre interno de Cloud SQL
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME"),
    "unix_socket": f"/cloudsql/{os.environ.get('INSTANCE_CONNECTION_NAME')}",
    "charset": "utf8mb4",
    "port": "3306",
}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def get_topics():
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
    FROM topics tp
    JOIN topic_tweets tt
        ON tp.topic_id = tt.topic_id
    JOIN Tweets tw
        ON tt.tweetid = tw.tweetid
    WHERE tp.topic_name LIKE '%NUEVO_CLUSTER%'
    ORDER BY tp.topic_id, tt.similarity DESC;
    """

    df_topics = pd.read_sql(query, connection)

    connection.close()

    return df_topics




def generar_nombre_topic(tweets):
    texto_tweets = "\n\n".join([
        f"Tweet {i+1}: {t}" for i, t in enumerate(tweets)
    ])

    prompt = f"""
    Eres un analista de noticias bolivianas.

    A partir de estos tweets, genera un nombre corto para el tema principal.

    Reglas:
    - Máximo 6 palabras.
    - No uses comillas.
    - No uses hashtags.
    - No inventes información.
    - No escribas explicación.
    - Devuelve solo el nombre del topic.

    Tweets:
    {texto_tweets}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()

def insert_tweets_topic(df):
    sql = """
    UPDATE topics
    SET topic_name = %s
    WHERE topic_id = %s
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    data = list(
        zip(
            df["topic_name_generado"],
            df["topic_id"].astype(int)
        )
    )

    cursor.executemany(sql, data)

    conn.commit()

    cursor.close()
    conn.close()

    print(f"{len(data)} nombres actualizados.")


def insertar_nuevos_topicos():
    df_topics = get_topics()
    df_validos = df_topics

    print("Tweets válidos:", len(df_validos))
    print("Topics válidos:", df_validos["topic_id"].nunique())
    df_representativos = (df_validos.sort_values(["topic_id", "similarity"], ascending=[True, False]).groupby("topic_id").head(5).copy())

    print("Tweets representativos:", len(df_representativos))
    print("Topics a nombrar:", df_representativos["topic_id"].nunique())
    resultados_nombres = []
    topics_para_nombrar = {}

    for topic_id, grupo in df_representativos.groupby("topic_id"):
        topics_para_nombrar[topic_id] = grupo["text"].tolist()

    for i, (topic_id, tweets) in enumerate(topics_para_nombrar.items(), 1):
        try:
            nombre = generar_nombre_topic(tweets)

            resultados_nombres.append({
                "topic_id": topic_id,
                "topic_name_generado": nombre
            })

            print(f"{i}/{len(topics_para_nombrar)} | Topic {topic_id}: {nombre}")

            time.sleep(1)

        except Exception as e:
            print(f"Error en topic {topic_id}: {e}")

    df_nombres = pd.DataFrame(resultados_nombres)
    insert_tweets_topic(df_nombres)
    