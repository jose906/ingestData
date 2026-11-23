from contextlib import closing
from database import get_db_connection
import os
import json
import pandas as pd
import numpy as np
from collections import Counter
import helper
from mysql.connector.errors import IntegrityError


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


def leer_embedding(valor):
    try:
        return json.loads(valor)
    except (TypeError, ValueError):
        return None


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
            "embedding": leer_embedding(row["embedding_vector"])
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
            "embedding": leer_embedding(row["embedding_vector"])
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
        print("No hay tweets para insertar")
        return set()

    connection = None
    cursor = None

    nuevos_por_topic = Counter()
    duplicados = 0
    tweets_insertados = set()

    ids_especiales = set(
        int(x) for x in ids_especiales
    )

    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # Evitar duplicados dentro del lote
        df = df.drop_duplicates(
            subset="tweetid"
        ).copy()

        insert_query = """
            INSERT INTO topic_tweets (
                topic_id,
                tweetid,
                similarity,
                assigned_at
            )
            VALUES (%s, %s, %s, NOW())
        """

        # 1. Insertar solamente tweets nuevos
        for fila in df.itertuples(index=False):

            topic_id = int(fila.topic_id)
            tweetid = int(fila.tweetid)
            similarity = float(fila.similarity)

            if not np.isfinite(similarity):
                raise ValueError(
                    f"Similitud inválida para tweet {tweetid}"
                )

            try:
                cursor.execute(
                    insert_query,
                    (
                        topic_id,
                        tweetid,
                        similarity
                    )
                )

                nuevos_por_topic[topic_id] += 1
                tweets_insertados.add(tweetid)

            except IntegrityError as error:

                # Código MySQL 1062: clave duplicada.
                # No cambiar un tweet ya asignado.
                if error.errno == 1062:
                    duplicados += 1
                    continue

                raise

        # 2. Actualizar solamente tópicos afectados
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

        for topic_id in nuevos_por_topic.keys():

            cursor.execute(
                update_query,
                (int(topic_id),)
            )

        # 3. Recalcular centroides robustos
        # solamente si ingresaron nuevos tweets
        for topic_id in sorted(nuevos_por_topic):

            if topic_id in ids_especiales:
                continue

            recalcular_centroide_topic(
                cursor,
                topic_id
            )

        # 4. Confirmar todos los cambios
        connection.commit()

        print("\n" + "=" * 60)
        print("RESULTADOS DE GUARDADO")
        print("=" * 60)

        print(
            "Tweets nuevos insertados:",
            sum(nuevos_por_topic.values())
        )

        print("Duplicados omitidos:", duplicados)

        print(
            "Tópicos afectados:",
            len(nuevos_por_topic)
        )

        print(
            "Centroides recalculados solicitados:",
            len(
                set(nuevos_por_topic)
                - ids_especiales
            )
        )

        return tweets_insertados

    except Exception as error:

        if connection is not None:
            connection.rollback()

        print(
            f"Error guardando asignaciones: {error}"
        )

        raise

    finally:

        try:
            if cursor is not None:
                cursor.close()
        finally:
            if connection is not None:
                connection.close()

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

    # 2. Obtener tweets con asignaciones de alta similitud.
    cursor.execute(
        """
        SELECT
            tt.tweetid,
            t.text,
            e.embedding_vector
        FROM topic_tweets tt
        JOIN Tweets t
            ON t.tweetid = tt.tweetid
        JOIN tweet_embeddings e
            ON e.tweetid = tt.tweetid
        WHERE tt.topic_id = %s AND tt.similarity >= %s
        ORDER BY tt.tweetid
        """,
        (topic_id, UMBRAL_CENTROIDE)
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


def clasificar_por_centroide(tweets, topics, asignados, pendientes):
    topicos_por_dimension = {}
    for topic in topics:
        vector = vector_unitario(topic["embedding"])
        if vector is not None:
            topicos_por_dimension.setdefault(vector.size, []).append(
                (int(topic["topic_id"]), vector)
            )

    tweets_por_dimension = {}
    for tweet in tweets:
        vector = vector_unitario(tweet["embedding"])
        if vector is None:
            pendientes.append({**tweet, "similarity": 0.0, "reason": "invalid_embedding"})
        elif vector.size not in topicos_por_dimension:
            reason = "no_compatible_topics" if topics else "no_topics"
            pendientes.append({**tweet, "similarity": 0.0, "reason": reason})
        else:
            tweets_por_dimension.setdefault(vector.size, []).append((tweet, vector))

    batch_size = 1000
    for dimension, grupo in tweets_por_dimension.items():
        topic_ids, vectores = zip(*topicos_por_dimension[dimension])
        matriz_topics = np.stack(vectores).T
        for start in range(0, len(grupo), batch_size):
            lote = grupo[start:start + batch_size]
            similares = np.stack([vector for _, vector in lote]) @ matriz_topics
            mejores = similares.argmax(axis=1)
            for fila, ((tweet, _), mejor) in enumerate(zip(lote, mejores)):
                score = float(np.clip(similares[fila, mejor], -1.0, 1.0))
                topic_id = topic_ids[mejor]
                segundo_id = segundo_score = margen = None
                if len(topic_ids) > 1:
                    similares[fila, mejor] = -np.inf
                    segundo = similares[fila].argmax()
                    segundo_id = topic_ids[segundo]
                    segundo_score = float(np.clip(similares[fila, segundo], -1.0, 1.0))
                    margen = score - segundo_score

                if score >= UMBRAL and (margen is None or margen >= MARGEN_MINIMO):
                    asignados.append({
                        "tweetid": tweet["tweetid"],
                        "topic_id": topic_id,
                        "similarity": score,
                    })
                else:
                    pendientes.append({
                        **tweet,
                        "similarity": score,
                        "candidate_topic_id": topic_id,
                        "second_topic_id": segundo_id,
                        "second_similarity": segundo_score,
                        "margin": margen,
                        "reason": "below_threshold" if score < UMBRAL else "ambiguous",
                    })


def classify_tweets(guardar=True):
    tweets = get_tweet_embedding()
    if not tweets:
        return pd.DataFrame(), pd.DataFrame()

    especiales = get_especial_topics()
    asignados = []
    pendientes = []
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

    if tweets_normales:
        clasificar_por_centroide(
            tweets_normales, get_topic_embeddings(), asignados, pendientes
        )

    df_asignados = pd.DataFrame(asignados, columns=["tweetid", "topic_id", "similarity"])
    df_pendientes = pd.DataFrame(pendientes)
    if guardar and not df_asignados.empty:
        df_asignados = df_asignados.drop_duplicates(subset="tweetid")
        insertados = insert_tweets_topic(df_asignados, especiales.values())
        df_asignados = df_asignados[df_asignados["tweetid"].map(int).isin(insertados)]

    print("Tweets analizados:", len(tweets))
    print("Tweets insertados:" if guardar else "Asignaciones propuestas:", len(df_asignados))
    if not df_pendientes.empty:
        print("Motivos de rechazo:", dict(Counter(df_pendientes["reason"])))

    return df_pendientes, df_asignados
