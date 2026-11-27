from contextlib import closing
import os
import json
import pandas as pd
import numpy as np
from collections import Counter
from database import get_db_connection
from helper import leer_embedding
import helper


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


def vector_unitario(embedding):
    try:
        vector = np.asarray(embedding, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None

    if vector.ndim != 1 or not vector.size:
        return None

    norma = np.linalg.norm(vector)
    if not np.isfinite(norma) or norma <= 0:
        return None

    return vector / norma


def get_topic_embeddings():
    with closing(get_db_connection()) as connection, closing(connection.cursor(dictionary=True)) as cursor:
        placeholders = ",".join(["%s"] * len(TOPICS_ESPECIALES))

        query = f"""
            SELECT te.topic_id, te.embedding_vector, t.topic_name, t.active
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
            "embedding": leer_embedding(row["embedding_vector"]),
            "topic_name": row["topic_name"],
            "active": row["active"],
        })

    return topics


def get_tweet_embedding():
    with closing(get_db_connection()) as connection, closing(connection.cursor(dictionary=True)) as cursor:
        query = """
            SELECT
                t.tweetid,
                t.text,
                e.tweetid AS embedding_tweetid,
                e.embedding_vector
            FROM Tweets t
            LEFT JOIN tweet_embeddings e
                ON t.tweetid = e.tweetid
                AND e.model_name = %s
            LEFT JOIN topic_tweets tt
                ON t.tweetid = tt.tweetid
            WHERE tt.tweetid IS NULL
              AND t.created >= CONVERT_TZ(UTC_TIMESTAMP(), '+00:00', '-04:00') - INTERVAL 14 DAY
        """

        cursor.execute(query, (helper.MODELO_EMBEDDINGS,))
        results = cursor.fetchall()

    tweets = []

    for row in results:
        tweets.append({
            "tweetid": row["tweetid"],
            "text": row["text"],
            "embedding": leer_embedding(row["embedding_vector"]),
            "has_embedding": row["embedding_tweetid"] is not None,
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
    if df is None or df.empty:
        return set()

    filas = [
        (int(fila.topic_id), int(fila.tweetid), float(fila.similarity))
        for fila in df.itertuples(index=False)
    ]
    with closing(get_db_connection()) as connection, closing(connection.cursor(dictionary=True)) as cursor:
        try:
            nuevas = helper.guardar_relaciones_nuevas(cursor, filas)
            principales = helper.topics_principales(nuevas) - set(ids_especiales)
            for topic_id in sorted(principales):
                recalcular_centroide_topic(cursor, topic_id)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    return {tweetid for _, tweetid, _ in nuevas}


def recalcular_centroide_topic(cursor, topic_id):

    topic_id = int(topic_id)

    # 1. Obtener centroide actual
    cursor.execute(
        """
        SELECT embedding_vector
        FROM topic_embeddings
        WHERE topic_id = %s
        FOR UPDATE
        """,
        (topic_id,)
    )

    actual = cursor.fetchone()

    anterior = None
    if actual is not None:
        anterior = vector_unitario(leer_embedding(actual["embedding_vector"]))
        if anterior is None:
            print(f"Topic {topic_id}: centroide anterior inválido")
            return False

    # 2. Obtener asignaciones principales de alta similitud.
    cursor.execute(
        f"""
        SELECT
            tt.tweetid,
            t.text,
            e.embedding_vector
        FROM topic_tweets tt
        JOIN Tweets t
            ON t.tweetid = tt.tweetid
        JOIN tweet_embeddings e
            ON e.tweetid = tt.tweetid
        WHERE tt.topic_id = %s AND tt.similarity >= %s AND e.model_name = %s
          AND {helper.TOPIC_PRINCIPAL_SQL}
        ORDER BY tt.tweetid
        """,
        (topic_id, UMBRAL_CENTROIDE, helper.MODELO_EMBEDDINGS)
    )

    resultados = cursor.fetchall()

    if not resultados:
        print(f"Topic {topic_id}: sin embeddings")
        return False

    # 3. Preparar los embeddings
    df = pd.DataFrame([
        {
            "tweetid": row["tweetid"],
            "text": row["text"],
            "embedding": leer_embedding(row["embedding_vector"])
        }
        for row in resultados
    ])

    # 4. Calcular centroide robusto
    centroide, utilizados, mediana = (
        calcular_centroide_robusto(df, anterior.size if anterior is not None else None)
    )

    if centroide is None or utilizados < 10:
        print(
            f"Topic {topic_id}: "
            "embeddings insuficientes"
        )
        return False

    nuevo = np.asarray(
        centroide,
        dtype=np.float64
    )

    # 5. Si no existe centroide anterior,
    # utilizar directamente el robusto
    if anterior is None:
        centroide_final = nuevo

    else:
        # 6. Medir desplazamiento
        similitud = float(anterior @ nuevo)

        print(
            f"Topic {topic_id} | "
            f"Similitud antiguo/nuevo: "
            f"{similitud:.4f}"
        )

        if similitud < 0.97:
            print(
                f"Topic {topic_id}: "
                "actualización bloqueada"
            )
            return False

        # 7. Actualización gradual 80/20
        centroide_final = (
            0.80 * anterior
            + 0.20 * nuevo
        )

        centroide_final /= np.linalg.norm(
            centroide_final
        )

    # 8. Guardar el nuevo centroide
    cursor.execute(
        """
        INSERT INTO topic_embeddings (
            topic_id,
            embedding_vector,
            updated_at
        )
        VALUES (%s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            embedding_vector = VALUES(embedding_vector),
            updated_at = NOW()
        """,
        (
            topic_id,
            json.dumps(centroide_final.tolist())
        )
    )

    print(
        f"Topic {topic_id} actualizado | "
        f"Tweets: {len(df)} | "
        f"Embeddings únicos: {utilizados} | "
        f"Similitud mediana: {mediana:.4f}"
    )

    return True


def calcular_centroide_robusto(grupo, dimension=None):

    grupo = grupo.drop_duplicates(
        subset="tweetid"
    ).copy()

    # 1. Normalizar textos para evitar duplicados.
    textos = (
        grupo["text"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.replace(
            r"https?://\S+",
            "",
            regex=True
        )
        .str.replace(
            r"^rt\s+@\w+:\s*",
            "",
            regex=True
        )
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    grupo["_texto_normalizado"] = textos

    # 2. Eliminar textos vacíos y repetidos.
    grupo = grupo[
        grupo["_texto_normalizado"] != ""
    ].drop_duplicates(
        subset="_texto_normalizado"
    )

    if grupo.empty:
        return None, 0, 0.0

    # Agrupar los vectores válidos por dimensión antes de construir la matriz.
    por_dimension = {}
    for embedding in grupo["embedding"]:
        vector = vector_unitario(embedding)
        if vector is not None:
            por_dimension.setdefault(vector.size, []).append(vector)

    if not por_dimension:
        return None, 0, 0.0
    if dimension is None:
        dimension = max(por_dimension, key=lambda size: len(por_dimension[size]))

    vectores = por_dimension.get(dimension)
    if not vectores:
        return None, 0, 0.0
    matriz = np.stack(vectores)

    # 6. Calcular centroide inicial.
    centroide = matriz.mean(axis=0)

    norma = np.linalg.norm(centroide)

    if norma < 1e-12:
        return None, 0, 0.0

    centroide /= norma

    # 7. Refinamiento iterativo del centroide.
    for _ in range(3):

        similitudes = matriz @ centroide

        p20, p80 = np.percentile(
            similitudes,
            [20, 80]
        )

        if p80 - p20 < 1e-8:

            pesos = np.ones(
                len(matriz),
                dtype=np.float64
            )

        else:

            pesos = np.clip(
                (similitudes - p20)
                / (p80 - p20),
                0.25,
                1.0
            )

        # Dar mayor importancia a los tweets
        # cercanos al centro del acontecimiento.
        nuevo_centroide = np.average(
            matriz,
            axis=0,
            weights=pesos
        )

        norma = np.linalg.norm(
            nuevo_centroide
        )

        if norma < 1e-12:
            break

        centroide = nuevo_centroide / norma

    # 8. Evaluar la cohesión de los embeddings.
    similitudes_finales = matriz @ centroide

    similitud_mediana = float(
        np.median(similitudes_finales)
    )

    return (
        centroide.astype(np.float32).tolist(),
        len(matriz),
        similitud_mediana
    )


def classify_tweets(guardar=True):
    tweets = get_tweet_embedding()
    if not tweets:
        return pd.DataFrame(), pd.DataFrame()

    especiales = get_especial_topics()
    asignados = []
    tweets_normales = []

    for tweet in tweets:
        especial = helper.detectar_topic_especial(tweet["text"])
        topic_id = especiales.get(especial)
        if topic_id is None:
            tweets_normales.append(tweet)
        else:
            asignados.append({
                "tweetid": tweet["tweetid"],
                "topic_id": int(topic_id),
                "similarity": 1.0,
            })

    pendientes = []
    if tweets_normales:
        clasificados, pendientes = helper.clasificar_por_centroide(
            tweets_normales,
            get_topic_embeddings(),
            cargar_referencias=None,
            margen=MARGEN_MINIMO,
        )
        asignados.extend(clasificados)

    df_asignados = pd.DataFrame(asignados, columns=["tweetid", "topic_id", "similarity"])
    df_asignados = df_asignados.drop_duplicates(subset=["tweetid", "topic_id"])
    ids_asignados = set(df_asignados["tweetid"])
    df_pendientes = pd.DataFrame(
        tweet for tweet in pendientes if tweet["tweetid"] not in ids_asignados
    )
    if guardar and not df_asignados.empty:
        insertados = insert_tweets_topic(df_asignados, especiales.values())
        df_asignados = df_asignados[df_asignados["tweetid"].map(int).isin(insertados)]

    total = len(tweets)
    asignados_unicos = int(df_asignados["tweetid"].nunique())
    especiales_unicos = int(df_asignados.loc[
        df_asignados["topic_id"].isin(especiales.values()), "tweetid"
    ].nunique())
    df_asignados.attrs.update(
        tweets_analizados=total,
        tweets_asignados_a_topicos=asignados_unicos - especiales_unicos,
    )
    print("Tweets analizados:", total)
    print("Tweets insertados:" if guardar else "Tweets asignables:", asignados_unicos)
    print("Asociaciones guardadas:" if guardar else "Asociaciones propuestas:", len(df_asignados))
    if not df_pendientes.empty:
        print("Motivos de rechazo:", dict(Counter(df_pendientes["reason"])))

    return df_pendientes, df_asignados
