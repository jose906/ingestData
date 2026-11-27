import json
import re
import unicodedata
from contextlib import closing

import numpy as np

URL_REGEX = re.compile(r"https?://\S+|www\.\S+")

SEPARADOR_AFIRMACIONES = re.compile(
    r"(?:[!?;\n]+|\.(?=\s|$)|\s+y\s+(?=(?:el|la|los|las|un|una)\s))",
    re.IGNORECASE,
)
CONTEXTO_INCIDENTAL = re.compile(
    r"\b(?:en (?:el marco|el contexto|medio) de|a ra[ií]z de|"
    r"tras|durante|respecto (?:a|de)|sobre|"
    r"mencion\w*|record\w*|aludi\w*|habl\w* de|"
    r"desminti\w*|desmient\w*|neg[oó]|no)\b",
    re.IGNORECASE,
)
VERBO_AFIRMACION = re.compile(
    r"\b(?:\w{2,}(?:ó|aron|ieron|ará|erá|irá|arán|erán|irán)|"
    r"hay|fue|son|anuncia|anuncian|confirma|confirman|"
    r"registra|registran|sube|aumenta|baja|inicia|comienza|"
    r"aprueba|rechaza|convoca|declara|elimina|suspende)\b",
    re.IGNORECASE,
)
PALABRAS_COMUNES = frozenset(
    "a al ante con de del el en entre la las lo los o para por se su un una y e".split()
)
SIMILITUD_TOPICOS_DUPLICADOS = 0.97
MODELO_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
UMBRAL_TOPIC = 0.79
UMBRAL_CANDIDATO_REFERENCIAS = 0.65
MARGEN_REFERENCIAS = 0.03
REFERENCIAS_POR_TOPIC = 150
CANDIDATOS_REFERENCIAS = 600
MIN_REFERENCIAS_TOPIC = 10
UMBRAL_REFERENCIA = 0.81
MIN_COINCIDENCIAS_REFERENCIA = 3
UMBRAL_PROMEDIO_TOP5 = 0.82
SIMILITUD_MAX_ENTRE_REFERENCIAS = 0.98
TOPIC_PRINCIPAL_SQL = """
    NOT EXISTS (
        SELECT 1 FROM topic_tweets anterior
        WHERE anterior.tweetid = tt.tweetid AND anterior.id < tt.id
    )
"""


def vector_referencia(embedding):
    try:
        vector = np.asarray(embedding, dtype=np.float32)
    except (TypeError, ValueError, OverflowError):
        return None
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        return None
    norma = np.linalg.norm(vector, axis=0)
    return vector / norma if np.isfinite(norma) and norma > 0 else None


def leer_embedding(valor):
    try:
        return json.loads(valor)
    except (TypeError, ValueError):
        return None


def cargar_referencias_topics(abrir_conexion, topic_ids, topics):
    ids_solicitados = {int(topic_id) for topic_id in topic_ids}
    centroides = {}
    for topic in topics:
        topic_id = int(topic["topic_id"])
        if topic_id in ids_solicitados:
            centroide = vector_referencia(topic["embedding"])
            if centroide is not None:
                centroides[topic_id] = centroide
    if not centroides:
        return {}

    ids = sorted(centroides)
    placeholders = ", ".join(["%s"] * len(ids))
    with closing(abrir_conexion()) as connection, closing(connection.cursor(dictionary=True)) as cursor:
        cursor.execute(
            f"""
            SELECT topic_id, text, embedding_vector
            FROM (
                SELECT tt.topic_id, t.text, e.embedding_vector,
                       ROW_NUMBER() OVER (
                           PARTITION BY tt.topic_id
                           ORDER BY tt.similarity DESC, tt.tweetid DESC
                       ) AS rn
                FROM topic_tweets tt
                JOIN Tweets t ON t.tweetid = tt.tweetid
                JOIN tweet_embeddings e ON e.tweetid = tt.tweetid
                WHERE tt.topic_id IN ({placeholders})
                  AND e.model_name = %s
                  AND t.text IS NOT NULL
                  AND {TOPIC_PRINCIPAL_SQL}
            ) AS ranking
            WHERE rn <= %s
            """,
            (*ids, MODELO_EMBEDDINGS, CANDIDATOS_REFERENCIAS),
        )
        filas = cursor.fetchall()

    candidatos = {topic_id: [] for topic_id in ids}
    for fila in filas:
        topic_id = int(fila["topic_id"])
        centroide = centroides[topic_id]
        try:
            vector = vector_referencia(json.loads(fila["embedding_vector"]))
        except (TypeError, ValueError):
            continue
        if vector is None or vector.shape != centroide.shape:
            continue
        texto = re.sub(r"https?://\S+", "", fila["text"])
        texto = re.sub(r"^RT\s+@\w+:\s*", "", texto, flags=re.IGNORECASE)
        texto = " ".join(texto.casefold().split())
        if len(texto) >= 20:
            candidatos[topic_id].append((float(vector @ centroide), texto, vector))

    referencias = {}
    for topic_id, elementos in candidatos.items():
        elementos.sort(key=lambda elemento: elemento[0], reverse=True)
        seleccionados = []
        textos_vistos = set()
        for _, texto, vector in elementos:
            if texto in textos_vistos:
                continue
            if seleccionados and np.max(np.asarray(seleccionados) @ vector) >= SIMILITUD_MAX_ENTRE_REFERENCIAS:
                continue
            seleccionados.append(vector)
            textos_vistos.add(texto)
            if len(seleccionados) == REFERENCIAS_POR_TOPIC:
                break
        referencias[topic_id] = np.asarray(seleccionados, dtype=np.float32).reshape(
            -1, len(centroides[topic_id])
        )
    return referencias


def puntuar_candidatos(vector, candidatos, referencias):
    aprobados = []
    if not candidatos or not np.isfinite(vector).all() or not np.any(vector):
        return aprobados

    for topic_id, similitud in candidatos:
        refs = referencias.get(topic_id)
        if refs is None or len(refs) < MIN_REFERENCIAS_TOPIC or refs.shape[1] != len(vector):
            continue
        similitudes = refs @ vector
        promedio_top5 = float(np.mean(np.sort(similitudes)[-5:]))
        coincidencias = np.count_nonzero(similitudes >= UMBRAL_REFERENCIA)
        if coincidencias >= MIN_COINCIDENCIAS_REFERENCIA and promedio_top5 >= UMBRAL_PROMEDIO_TOP5:
            aprobados.append({
                "topic_id": topic_id,
                "similarity": similitud,
                "score_referencias": promedio_top5,
            })

    aprobados.sort(key=lambda opcion: opcion["score_referencias"], reverse=True)
    return aprobados


def terminos_evento(texto):
    normalizado = unicodedata.normalize("NFKD", texto).casefold()
    sin_acentos = "".join(c for c in normalizado if not unicodedata.combining(c))
    return frozenset(re.findall(r"[^\W_]+", sin_acentos)) - PALABRAS_COMUNES


def es_topico_publicado(topic):
    nombre = str(topic.get("topic_name") or "").strip()
    return (
        bool(topic.get("active"))
        and bool(nombre)
        and not nombre.startswith("__")
        and "NUEVO_CLUSTER" not in nombre.upper()
    )


def eventos_de_topics(topics):
    eventos = {}
    for topic in topics:
        if not es_topico_publicado(topic):
            continue
        terminos = terminos_evento(topic["topic_name"])
        if len(terminos) >= 2:
            eventos[int(topic["topic_id"])] = (terminos, topic["embedding"])
    return eventos


def clasificar_por_centroide(tweets, topics, cargar_referencias, margen, limite=1, umbral_secundario=0.84):
    publicados = [topic for topic in topics if es_topico_publicado(topic)]
    por_dimension = {}
    for topic in publicados:
        vector = vector_referencia(topic["embedding"])
        if vector is not None:
            por_dimension.setdefault(vector.size, []).append((int(topic["topic_id"]), vector))

    eventos = eventos_de_topics(publicados) if cargar_referencias is not None and limite > 1 else {}
    asignados = []
    pendientes = []
    grupos = {}
    for tweet in tweets:
        vector = vector_referencia(tweet["embedding"])
        if vector is None:
            motivo = "missing_embedding" if not tweet.get("has_embedding", True) else "invalid_embedding"
        elif vector.size not in por_dimension:
            motivo = "no_compatible_topics" if publicados else "no_topics"
        else:
            grupos.setdefault(vector.size, []).append((tweet, vector))
            continue
        pendientes.append({**tweet, "reason": motivo})

    referencias = {}
    for dimension, grupo in grupos.items():
        ids, vectores = zip(*por_dimension[dimension])
        matriz = np.stack(vectores).T
        for inicio in range(0, len(grupo), 500):
            lote = grupo[inicio:inicio + 500]
            similitudes = np.clip(np.stack([vector for _, vector in lote]) @ matriz, -1, 1)
            candidatos_lote = []
            ids_referencias = set()
            for fila in similitudes:
                indices = np.argsort(-fila, kind="stable")
                primero = float(fila[indices[0]])
                segundo = float(fila[indices[1]]) if len(ids) > 1 else -1.0
                directo = primero >= UMBRAL_TOPIC and primero - segundo >= margen
                principales = [
                    (ids[i], float(fila[i])) for i in indices[:2] if fila[i] >= UMBRAL_CANDIDATO_REFERENCIAS
                ]
                # Revisar referencias si el centroide no decide o si hay dos candidatos fuertes.
                candidatos = {}
                if cargar_referencias is not None and (not directo or segundo >= UMBRAL_TOPIC):
                    candidatos = dict(principales)
                if eventos and segundo >= umbral_secundario:
                    candidatos.update(
                        (ids[i], float(fila[i]))
                        for i in np.flatnonzero(fila >= umbral_secundario) if ids[i] in eventos
                    )
                if directo and set(candidatos) == {ids[indices[0]]}:
                    candidatos.clear()
                ids_referencias.update(candidatos)
                candidatos_lote.append((principales, candidatos, directo, primero))

            faltantes = ids_referencias - referencias.keys()
            if faltantes:
                referencias.update(dict.fromkeys(faltantes))
                referencias.update(cargar_referencias(faltantes, publicados))

            for (tweet, vector), (principales, candidatos, directo, primero) in zip(lote, candidatos_lote):
                aprobados = puntuar_candidatos(vector, list(candidatos.items()), referencias)
                ids_principales = {topic_id for topic_id, _ in principales}
                comprobados = [opcion for opcion in aprobados if opcion["topic_id"] in ids_principales]
                competidor = None
                if comprobados:
                    principal = comprobados[0]["topic_id"], comprobados[0]["similarity"]
                    if len(comprobados) > 1 and (
                        comprobados[0]["score_referencias"] - comprobados[1]["score_referencias"] < MARGEN_REFERENCIAS
                    ):
                        competidor = comprobados[1]["topic_id"]
                elif directo:
                    principal = principales[0]
                else:
                    pendientes.append({
                        **tweet,
                        "similarity": primero,
                        "reason": "below_threshold" if primero < UMBRAL_TOPIC else "ambiguous",
                    })
                    continue

                destino, _ = principal
                opciones = [
                    (opcion["topic_id"], opcion["similarity"])
                    for opcion in aprobados
                    if opcion["topic_id"] != destino and opcion["similarity"] >= umbral_secundario
                ]
                if competidor is not None:
                    opciones.sort(key=lambda opcion: opcion[0] != competidor)
                secundarios = elegir_topics_secundarios(tweet["text"], destino, opciones, eventos, limite - 1)
                if competidor is not None and competidor not in {topic_id for topic_id, _ in secundarios}:
                    pendientes.append({**tweet, "similarity": primero, "reason": "ambiguous"})
                    continue
                asignados.extend(
                    {"tweetid": tweet["tweetid"], "topic_id": topic_id, "similarity": similitud}
                    for topic_id, similitud in [principal, *secundarios]
                )
    return asignados, pendientes


def elegir_topics_secundarios(texto, principal, opciones, eventos, limite):
    if limite < 1 or not opciones or principal not in eventos or detectar_topic_especial(texto):
        return []

    texto = URL_REGEX.sub("", str(texto or ""))
    ids = [principal, *(topic_id for topic_id, _ in opciones if topic_id in eventos)]
    confirmados = set()
    for afirmacion in SEPARADOR_AFIRMACIONES.split(texto):
        # Solo la afirmación aporta evidencia. Una referencia de contexto no basta.
        afirmacion = CONTEXTO_INCIDENTAL.split(afirmacion, maxsplit=1)[0]
        if VERBO_AFIRMACION.search(afirmacion):
            palabras = terminos_evento(afirmacion)
            coincidencias = [topic_id for topic_id in ids if eventos[topic_id][0] <= palabras]
            if len(coincidencias) == 1:
                confirmados.add(coincidencias[0])

    if principal not in confirmados:
        return []

    seleccionados = []
    vectores = [np.asarray(eventos[principal][1])]
    for topic_id, score in opciones:
        if topic_id not in confirmados:
            continue
        vector = np.asarray(eventos[topic_id][1])
        norma = np.linalg.norm(vector)
        if any(
            float(vector @ previo) >= SIMILITUD_TOPICOS_DUPLICADOS * norma * np.linalg.norm(previo)
            for previo in vectores
        ):
            continue
        seleccionados.append((topic_id, score))
        if len(seleccionados) == limite:
            break
        vectores.append(vector)
    return seleccionados


def guardar_relaciones_nuevas(cursor, filas):
    """Inserta el lote en orden, con la asociación principal primero por tuit."""
    ids = sorted({tweetid for _, tweetid, _ in filas})
    ya_asignados = set()
    for inicio in range(0, len(ids), 1000):
        lote = ids[inicio:inicio + 1000]
        placeholders = ", ".join(["%s"] * len(lote))
        cursor.execute(
            f"""
            SELECT t.tweetid, tt.id AS asignacion
            FROM Tweets t
            LEFT JOIN topic_tweets tt ON tt.tweetid = t.tweetid
            WHERE t.tweetid IN ({placeholders})
            ORDER BY t.tweetid
            FOR UPDATE
            """,
            tuple(lote),
        )
        ya_asignados.update(
            int(row["tweetid"]) for row in cursor.fetchall() if row["asignacion"] is not None
        )

    nuevas = []
    pares = set()
    for topic_id, tweetid, score in filas:
        if tweetid not in ya_asignados and (tweetid, topic_id) not in pares:
            if not np.isfinite(score):
                raise ValueError(f"Similitud inválida para el tweet {tweetid}")
            nuevas.append((topic_id, tweetid, score))
            pares.add((tweetid, topic_id))
    if not nuevas:
        return nuevas

    # Bloquear los contadores antes de insertar evita competir por su actualización.
    topics = sorted({topic_id for topic_id, _, _ in nuevas})
    placeholders = ", ".join(["%s"] * len(topics))
    cursor.execute(
        f"SELECT topic_id FROM topics WHERE topic_id IN ({placeholders}) ORDER BY topic_id FOR UPDATE",
        tuple(topics),
    )
    cursor.fetchall()

    # Una infracción de integridad aborta la transacción del llamador.
    cursor.executemany(
        """
        INSERT INTO topic_tweets (topic_id, tweetid, similarity, assigned_at)
        VALUES (%s, %s, %s, NOW())
        """,
        nuevas,
    )
    cursor.executemany(
        """
        UPDATE topics
        SET total_tweets = (
            SELECT COUNT(DISTINCT tt.tweetid)
            FROM topic_tweets tt
            WHERE tt.topic_id = topics.topic_id
        ),
        last_seen = NOW()
        WHERE topic_id = %s
        """,
        [(topic_id,) for topic_id in topics],
    )
    return nuevas


def topics_principales(filas):
    primer_topic = {}
    for topic_id, tweetid, _ in filas:
        primer_topic.setdefault(tweetid, topic_id)
    return set(primer_topic.values())


def detectar_topic_especial(texto):
    texto_original = str(texto).strip()
    texto_lower = texto_original.lower()

    # 1. Resumen de noticias
    if (
        "resumen de noticias" in texto_lower
        or "resumen informativo" in texto_lower
        or "resumen de la jornada" in texto_lower
    ):
        return "__RESUMEN__"

    # 2. Portada de periódico
    if (
        "portada de la edición impresa" in texto_lower
        or "portada de hoy" in texto_lower
        or "nuestra portada" in texto_lower
        or "te presentamos la edición impresa" in texto_lower
    ):
        return "__PORTADA__"

    # 3. E-paper o edición digital
    if (
        "epaper" in texto_lower
        or "e-paper" in texto_lower
        or "edición digital" in texto_lower
        or "edicion digital" in texto_lower
    ):
        return "__EPAPER__"

    # 4. Programación
    if (
        re.search(r"\b(?:nuestra programaci[oó]n|programaci[oó]n de hoy|sintoniza)\b", texto_lower)
        or re.match(r"en vivo:\s*(?:te esperamos|acompaña[nm]os|mira|mire)\b", texto_lower)
    ):
        return "__PROGRAMACION__"

    # 5. Promoción
    if re.search(
        r"\b(?:participa y gana|participa en (?:el|nuestro) sorteo|"
        r"nuestro sorteo|promoci[oó]n exclusiva|suscr[ií]bete)\b",
        texto_lower,
    ):
        return "__PROMOCION__"

    # 6. Saludos
    if (
        texto_lower.startswith("buenos días")
        or texto_lower.startswith("buenos dias")
        or texto_lower.startswith("buenas tardes")
        or texto_lower.startswith("buenas noches")
        or texto_lower.startswith("feliz lunes")
        or texto_lower.startswith("feliz martes")
        or texto_lower.startswith("feliz miércoles")
        or texto_lower.startswith("feliz miercoles")
        or texto_lower.startswith("feliz jueves")
        or texto_lower.startswith("feliz viernes")
    ):
        return "__SALUDO__"

    # 7. Solo link
    texto_sin_urls = URL_REGEX.sub("", texto_original)
    texto_sin_urls = re.sub(r"[\s\W_]+", "", texto_sin_urls)

    if URL_REGEX.search(texto_original) and texto_sin_urls == "":
        return "__SOLO_LINK__"

    return None
