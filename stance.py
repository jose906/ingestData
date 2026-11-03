
import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import mysql.connector
import requests


# ============================================================
# CONFIGURACION
# ============================================================

DB_CONFIG = {
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME"),
    "port": 3306,
    "charset": "utf8mb4",
    "connection_timeout": 10,
}

INSTANCE = os.environ.get("INSTANCE_CONNECTION_NAME")
DB_HOST = os.environ.get("DB_HOST")

if INSTANCE:
    DB_CONFIG["unix_socket"] = f"/cloudsql/{INSTANCE}"
elif DB_HOST:
    DB_CONFIG["host"] = DB_HOST

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

logger = logging.getLogger("netvora.stance")

VALIDOS = {
    "FAVOR",
    "CONTRA",
    "MIXTO",
    "NEUTRAL",
    "NO_CLARO"
}


def get_db_connection():
    if not (INSTANCE or DB_HOST):
        raise RuntimeError(
            "Falta INSTANCE_CONNECTION_NAME o DB_HOST"
        )
    return mysql.connector.connect(**DB_CONFIG)


def consultar_uno(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(sql, params)
            return cur.fetchone()
        finally:
            cur.close()
    finally:
        conn.close()


def consultar_todos(sql, params=()):
    conn = get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(sql, params)
            return cur.fetchall()
        finally:
            cur.close()
    finally:
        conn.close()


def ejecutar_sql(sql, params, muchos=False):
    conn = get_db_connection()

    try:
        cur = conn.cursor()
        try:
            if muchos:
                cur.executemany(sql, params)
            else:
                cur.execute(sql, params)

            total = cur.rowcount
            ultimo_id = cur.lastrowid

            conn.commit()
            return total, ultimo_id
        finally:
            cur.close()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# GEMINI
# ============================================================

def llamar_gemini(prompt):
    if not GEMINI_API_KEY:
        raise RuntimeError("Falta GEMINI_API_KEY")

    config = {
        "responseMimeType": "application/json"
    }

    if MODEL.startswith("gemini-2.5"):
        config["thinkingConfig"] = {
            "thinkingBudget": 0
        }
    elif MODEL.startswith("gemini-3"):
        config["thinkingConfig"] = {
            "thinkingLevel": "low"
        }

    respuesta = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        headers={
            "x-goog-api-key": GEMINI_API_KEY
        },
        json={
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": config
        },
        timeout=90
    )

    if not respuesta.ok:
        logger.error(
            "Error Gemini HTTP %s: %s",
            respuesta.status_code,
            respuesta.text[:1000]
        )
        respuesta.raise_for_status()

    data = respuesta.json()
    candidatos = data.get("candidates") or []

    if not candidatos:
        raise ValueError(
            f"Gemini no devolvio candidatos: {str(data)[:500]}"
        )

    partes = candidatos[0].get(
        "content", {}
    ).get("parts", [])

    texto = "".join(
        p.get("text", "")
        for p in partes
    )

    if not texto:
        raise ValueError(
            "Gemini devolvio una respuesta vacia"
        )

    return json.loads(texto)


def validar_target(resultado):
    if not isinstance(resultado, dict):
        raise ValueError(
            "El target no es un objeto JSON"
        )

    tipo = resultado.get("target_type")

    if tipo not in {
        "STANCE",
        "INFORMATIVO",
        "MIXTO"
    }:
        raise ValueError("target_type invalido")

    confidence = float(
        resultado.get("confidence", -1)
    )

    if not 0 <= confidence <= 1:
        raise ValueError(
            "Confianza del target invalida"
        )

    if tipo == "STANCE":
        if (
            not resultado.get("main_target")
            or not resultado.get("target_description")
        ):
            raise ValueError(
                "Target STANCE incompleto"
            )
    else:
        resultado["main_target"] = None
        resultado["target_description"] = None

    resultado["confidence"] = confidence

    return resultado


def analizar_target_publicacion(texto):
    return validar_target(
        llamar_gemini(f"""
Eres un analista de conversaciones publicas de Bolivia.

Analiza UNICAMENTE esta publicacion:

{texto}

Decide si existe UNA afirmacion, propuesta, decision
o accion clara sobre la que tenga sentido medir
apoyo o rechazo.

- STANCE: existe un objetivo unico y concreto
  que permite FAVOR o CONTRA.

- INFORMATIVO: describe hechos, noticias,
  denuncias o acontecimientos sin una cuestion
  central adecuada para medir apoyo y rechazo.

- MIXTO: existen varias cuestiones independientes,
  sin un objetivo unico.

No inventes objetivos ni conviertas automaticamente
una noticia en STANCE.

Si es STANCE, redacta main_target con una cuestion
concreta y target_description de modo neutral
explicando que significa FAVOR frente a CONTRA.

En INFORMATIVO/MIXTO ambos deben ser null.

Devuelve SOLO un objeto JSON:

{{
  "target_type": "STANCE|INFORMATIVO|MIXTO",
  "main_target": null,
  "target_description": null,
  "confidence": 0.95
}}
""")
    )


def analizar_target_topic(nombre, tweets):
    textos = "\n\n".join(
        f"PUBLICACION {i+1}: {t}"
        for i, t in enumerate(tweets)
    )

    return validar_target(
        llamar_gemini(f"""
Eres un analista de conversaciones publicas de Bolivia.

TOPICO:
{nombre}

PUBLICACIONES REPRESENTATIVAS:
{textos}

Determina si el topico es:

STANCE:
Una cuestion concreta, unica y debatible sobre
la cual se puede expresar apoyo o rechazo.

INFORMATIVO:
Noticias, hechos o informacion sin un
objetivo debatible.

MIXTO:
Multiples cuestiones independientes
sin un objetivo unico.

No inventes propuestas ni posiciones.

Para STANCE, redacta main_target con UNA cuestion
precisa y target_description neutral que defina
FAVOR y CONTRA.

Para otros tipos devuelve null en ambos campos.

Devuelve SOLO un objeto JSON:

{{
  "target_type": "STANCE|INFORMATIVO|MIXTO",
  "main_target": null,
  "target_description": null,
  "confidence": 0.95
}}
""")
    )


def clasificar_stance_target(
    publicacion,
    target,
    comentarios
):
    if target["target_type"] != "STANCE":
        return []

    lista = "\n\n".join(
        f"ID: {str(c['replyid'])}\n"
        f"COMENTARIO: {c['comentario']}"
        for c in comentarios
    )

    respuesta = llamar_gemini(f"""
Eres un clasificador de posturas de comentarios
publicos de Bolivia.

PUBLICACION ORIGINAL:
{publicacion}

OBJETIVO CENTRAL:
{target['main_target']}

DEFINICION DE POSTURAS:
{target['target_description']}

COMENTARIOS:
{lista}

Clasifica SOLO respecto al objetivo central,
no por sentimiento.

FAVOR:
Apoyo explicito al objetivo.

CONTRA:
Rechazo explicito al objetivo.

MIXTO:
Apoyo y rechazo claros al mismo objetivo.

NEUTRAL:
Relacionado con el objetivo, pero sin
apoyo ni rechazo.

NO_CLARO:
Referencia ambigua o falta contexto
para determinar postura.

No confundas criticas al autor, procedimiento
o institucion con rechazo al objetivo.

Los insultos y sarcasmos ambiguos son NO_CLARO;
las preguntas informativas relacionadas
suelen ser NEUTRAL.

No inventes contenido de imagenes o enlaces.

Evalua cada comentario individualmente
y devuelve exactamente un resultado por ID.

confidence es un numero entre 0 y 1.

Devuelve SOLO un array JSON como:

[
  {{
    "replyid": "123",
    "stance": "FAVOR",
    "confidence": 0.95
  }}
]
""")

    return validar_resultados_stance(
        comentarios,
        respuesta
    )


def validar_resultados_stance(
    comentarios,
    respuesta
):
    if not isinstance(respuesta, list):
        raise ValueError(
            "Gemini no devolvio una lista"
        )

    esperados = {
        str(c["replyid"])
        for c in comentarios
    }

    vistos = set()
    validos = []
    duplicados = set()

    for item in respuesta:
        if not isinstance(item, dict):
            continue

        rid = str(item.get("replyid"))

        if rid in vistos:
            duplicados.add(rid)
            continue

        vistos.add(rid)

        if (
            rid not in esperados
            or item.get("stance") not in VALIDOS
        ):
            continue

        try:
            confianza = float(
                item["confidence"]
            )
        except (KeyError, TypeError, ValueError):
            continue

        if 0 <= confianza <= 1:
            validos.append({
                "replyid": rid,
                "stance": item["stance"],
                "confidence": confianza
            })

    return [
        item
        for item in validos
        if item["replyid"] not in duplicados
    ]


# ============================================================
# STANCE POR PUBLICACION
# RECIENTES PRIMERO
# ============================================================

def get_siguiente_publicacion_target(
    ultimo_tweetid=None,
    desde=None
):
    # "desde" se conserva por compatibilidad.
    # No se aplica ningun filtro de fecha.

    return consultar_uno("""
        SELECT
            t.tweetid,
            t.text
        FROM Tweets t

        LEFT JOIN post_targets pt
          ON pt.tweetid = t.tweetid

        WHERE (
            %s IS NULL
            OR t.tweetid < %s
        )

        AND (
            (
                pt.id IS NULL
                AND EXISTS (
                    SELECT 1
                    FROM replies r
                    WHERE r.tweetid = t.tweetid
                )
            )
            OR
            (
                pt.target_type = 'STANCE'
                AND EXISTS (
                    SELECT 1
                    FROM replies r
                    WHERE r.tweetid = t.tweetid
                      AND NOT EXISTS (
                          SELECT 1
                          FROM reply_target_stance s
                          WHERE s.replyid = r.replyid
                            AND s.post_target_id = pt.id
                      )
                )
            )
        )

        ORDER BY t.tweetid DESC
        LIMIT 1
    """, (
        ultimo_tweetid,
        ultimo_tweetid
    ))


def get_post_target(tweetid):
    return consultar_uno("""
        SELECT
            id,
            tweetid,
            target_type,
            main_target,
            target_description
        FROM post_targets
        WHERE tweetid = %s
        LIMIT 1
    """, (tweetid,))


def guardar_post_target(tweetid, target):
    _, target_id = ejecutar_sql("""
        INSERT INTO post_targets (
            tweetid,
            target_type,
            main_target,
            target_description,
            confidence,
            model,
            prompt_version
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)

        ON DUPLICATE KEY UPDATE
            id = LAST_INSERT_ID(id)
    """, (
        tweetid,
        target["target_type"],
        target.get("main_target"),
        target.get("target_description"),
        target["confidence"],
        MODEL,
        "v1"
    ))

    return target_id


def get_comentarios_target_pendientes(
    tweetid,
    post_target_id,
    limit=20
):
    return consultar_todos("""
        SELECT
            r.replyid,
            r.text AS comentario
        FROM replies r

        WHERE r.tweetid = %s

          AND NOT EXISTS (
              SELECT 1
              FROM reply_target_stance s
              WHERE s.replyid = r.replyid
                AND s.post_target_id = %s
          )

        ORDER BY r.created DESC, r.id DESC
        LIMIT %s
    """, (
        tweetid,
        post_target_id,
        limit
    ))


def guardar_stance_target(
    post_target_id,
    resultados
):
    if not resultados:
        return 0

    valores = [
        (
            int(r["replyid"]),
            post_target_id,
            r["stance"],
            r["confidence"],
            MODEL,
            "v1"
        )
        for r in resultados
    ]

    guardados, _ = ejecutar_sql("""
        INSERT INTO reply_target_stance (
            replyid,
            post_target_id,
            stance,
            confidence,
            model,
            prompt_version
        )
        VALUES (%s, %s, %s, %s, %s, %s)

        ON DUPLICATE KEY UPDATE
            id = id
    """, valores, muchos=True)

    return guardados


def ejecutar_stance_por_target(
    max_comentarios=500,
    tamano_lote=20,
    max_publicaciones=200,
    desde=None
):
    if min(
        max_comentarios,
        tamano_lote,
        max_publicaciones
    ) <= 0:
        raise ValueError(
            "Los limites deben ser positivos"
        )

    resumen = {
        "intentados": 0,
        "guardados": 0,
        "fallidos": 0,
        "publicaciones": 0
    }

    ultimo_tweetid = None

    while (
        resumen["intentados"] < max_comentarios
        and resumen["publicaciones"] < max_publicaciones
    ):
        publicacion = get_siguiente_publicacion_target(
            ultimo_tweetid
        )

        if not publicacion:
            break

        tweetid = publicacion["tweetid"]

        # El siguiente tweet sera mas antiguo.
        ultimo_tweetid = tweetid

        resumen["publicaciones"] += 1

        try:
            target = get_post_target(tweetid)

            if target is None:
                guardar_post_target(
                    tweetid,
                    analizar_target_publicacion(
                        publicacion["text"]
                    )
                )

                target = get_post_target(tweetid)

            if (
                not target
                or target["target_type"] != "STANCE"
            ):
                continue

            while (
                resumen["intentados"] < max_comentarios
            ):
                n = min(
                    tamano_lote,
                    max_comentarios
                    - resumen["intentados"]
                )

                comentarios = (
                    get_comentarios_target_pendientes(
                        tweetid,
                        target["id"],
                        n
                    )
                )

                if not comentarios:
                    break

                resumen["intentados"] += len(
                    comentarios
                )

                try:
                    validos = clasificar_stance_target(
                        publicacion["text"],
                        target,
                        comentarios
                    )

                    guardados = guardar_stance_target(
                        target["id"],
                        validos
                    )

                    resumen["guardados"] += guardados

                    resumen["fallidos"] += (
                        len(comentarios) - guardados
                    )

                    logger.info(
                        "Tweet %s: enviados=%s guardados=%s",
                        tweetid,
                        len(comentarios),
                        guardados
                    )

                    if guardados < len(comentarios):
                        break

                except Exception:
                    resumen["fallidos"] += len(
                        comentarios
                    )

                    logger.exception(
                        "Error clasificando comentarios del tweet %s",
                        tweetid
                    )
                    break

        except Exception:
            resumen["fallidos"] += 1

            logger.exception(
                "Error analizando la publicacion %s",
                tweetid
            )

    return resumen


# ============================================================
# TARGETS Y STANCE POR TOPICO
# RECIENTES PRIMERO
# ============================================================

def get_topics_pending(
    limit=10,
    desde=None
):
    return consultar_todos("""
        SELECT
            topic_id,
            topic_name
        FROM topics
        WHERE target_type IS NULL

        ORDER BY last_seen DESC, topic_id DESC
        LIMIT %s
    """, (limit,))


def get_tweets_topic(
    topic_id,
    limit=20
):
    return consultar_todos("""
        SELECT
            t.tweetid,
            t.text,
            tt.similarity
        FROM topic_tweets tt

        JOIN Tweets t
          ON t.tweetid = tt.tweetid

        WHERE tt.topic_id = %s

        ORDER BY tt.similarity DESC
        LIMIT %s
    """, (
        topic_id,
        limit
    ))


def update_topic_target(
    topic_id,
    resultado
):
    ejecutar_sql("""
        UPDATE topics
        SET
            target_type = %s,
            main_target = %s,
            target_description = %s,
            target_confidence = %s
        WHERE topic_id = %s
    """, (
        resultado["target_type"],
        resultado.get("main_target"),
        resultado.get("target_description"),
        resultado["confidence"],
        topic_id
    ))


def ejecutar_targets_topicos(
    limit=10,
    tweets_por_topico=20,
    desde=None
):
    resumen = {
        "procesados": 0,
        "stance": 0,
        "informativos": 0,
        "mixtos": 0,
        "fallidos": 0
    }

    for topico in get_topics_pending(limit):
        try:
            tweets = get_tweets_topic(
                topico["topic_id"],
                tweets_por_topico
            )

            if not tweets:
                logger.warning(
                    "Topico %s sin publicaciones",
                    topico["topic_id"]
                )

                resumen["fallidos"] += 1
                continue

            target = analizar_target_topic(
                topico["topic_name"],
                [
                    t["text"]
                    for t in tweets
                ]
            )

            update_topic_target(
                topico["topic_id"],
                target
            )

            resumen["procesados"] += 1

            nombre = target["target_type"]

            clave = {
                "STANCE": "stance",
                "INFORMATIVO": "informativos",
                "MIXTO": "mixtos"
            }[nombre]

            resumen[clave] += 1

            logger.info(
                "Topico %s: %s",
                topico["topic_id"],
                nombre
            )

        except Exception:
            resumen["fallidos"] += 1

            logger.exception(
                "Error analizando target de topico %s",
                topico["topic_id"]
            )

    return resumen


def get_topicos_stance_pendientes(
    desde=None,
    limit=3
):
    return consultar_todos("""
        SELECT
            tp.topic_id,
            tp.target_type,
            tp.main_target,
            tp.target_description
        FROM topics tp

        WHERE tp.target_type = 'STANCE'

          AND NULLIF(
              TRIM(tp.main_target),
              ''
          ) IS NOT NULL

          AND NULLIF(
              TRIM(tp.target_description),
              ''
          ) IS NOT NULL

          AND EXISTS (
              SELECT 1
              FROM topic_tweets tt

              JOIN replies r
                ON r.tweetid = tt.tweetid

              WHERE tt.topic_id = tp.topic_id

                AND NOT EXISTS (
                    SELECT 1
                    FROM topic_reply_stance s
                    WHERE s.replyid = r.replyid
                      AND s.topic_id = tp.topic_id
                )
          )

        ORDER BY tp.last_seen DESC, tp.topic_id DESC
        LIMIT %s
    """, (limit,))


def get_topic_replies_pending(
    topic_id,
    limit=20,
    desde=None
):
    return consultar_todos("""
        SELECT
            r.replyid,
            r.tweetid,
            r.text AS comentario,
            t.text AS publicacion_original
        FROM topic_tweets tt

        JOIN Tweets t
          ON t.tweetid = tt.tweetid

        JOIN replies r
          ON r.tweetid = t.tweetid

        WHERE tt.topic_id = %s

          AND NOT EXISTS (
              SELECT 1
              FROM topic_reply_stance s
              WHERE s.replyid = r.replyid
                AND s.topic_id = tt.topic_id
          )

        ORDER BY r.created DESC, r.id DESC
        LIMIT %s
    """, (
        topic_id,
        limit
    ))


def guardar_stance_topico(
    topic_id,
    resultados
):
    if not resultados:
        return 0

    valores = [
        (
            int(r["replyid"]),
            topic_id,
            r["stance"],
            r["confidence"],
            MODEL,
            "v1"
        )
        for r in resultados
    ]

    guardados, _ = ejecutar_sql("""
        INSERT INTO topic_reply_stance (
            replyid,
            topic_id,
            stance,
            confidence,
            model,
            prompt_version
        )
        VALUES (%s, %s, %s, %s, %s, %s)

        ON DUPLICATE KEY UPDATE
            id = id
    """, valores, muchos=True)

    return guardados


def ejecutar_stance_por_topico(
    desde=None,
    max_comentarios=10,
    tamano_lote=10,
    max_topicos=2,
    max_publicaciones=3
):
    if min(
        max_comentarios,
        tamano_lote,
        max_topicos,
        max_publicaciones
    ) <= 0:
        raise ValueError(
            "Los limites deben ser positivos"
        )

    resumen = {
        "intentados": 0,
        "guardados": 0,
        "fallidos": 0,
        "topicos": 0,
        "publicaciones": 0
    }

    topicos = get_topicos_stance_pendientes(
        limit=max_topicos
    )

    for topico in topicos:
        if (
            resumen["intentados"] >= max_comentarios
            or resumen["publicaciones"] >= max_publicaciones
        ):
            break

        n = min(
            tamano_lote,
            max_comentarios - resumen["intentados"]
        )

        comentarios = get_topic_replies_pending(
            topico["topic_id"],
            n
        )

        if not comentarios:
            continue

        resumen["topicos"] += 1

        grupos = defaultdict(list)

        for comentario in comentarios:
            grupos[comentario["tweetid"]].append(
                comentario
            )

        for lote in grupos.values():
            if (
                resumen["intentados"] >= max_comentarios
                or resumen["publicaciones"] >= max_publicaciones
            ):
                break

            resumen["publicaciones"] += 1
            resumen["intentados"] += len(lote)

            try:
                validos = clasificar_stance_target(
                    lote[0]["publicacion_original"],
                    topico,
                    lote
                )

                guardados = guardar_stance_topico(
                    topico["topic_id"],
                    validos
                )

                resumen["guardados"] += guardados

                resumen["fallidos"] += (
                    len(lote) - guardados
                )

                logger.info(
                    "Topico %s: enviados=%s guardados=%s",
                    topico["topic_id"],
                    len(lote),
                    guardados
                )

            except Exception:
                resumen["fallidos"] += len(lote)

                logger.exception(
                    "Error clasificando topico %s",
                    topico["topic_id"]
                )

    return resumen


# ============================================================
# CLOUD RUN JOB
# INDEPENDIENTE DE FLASK Y main.py
# ============================================================

def ejecutar_job_stance(
    lock_name,
    tarea
):
    conn = get_db_connection()
    cur = None
    adquirido = False

    try:
        cur = conn.cursor()

        cur.execute(
            "SELECT GET_LOCK(%s, 0)",
            (lock_name,)
        )

        fila = cur.fetchone()

        adquirido = bool(
            fila and fila[0] == 1
        )

        if not adquirido:
            raise RuntimeError(
                f"El proceso {lock_name} ya esta en ejecucion"
            )

        logger.info(
            "Iniciando %s",
            lock_name
        )

        resultado = tarea()

        logger.info(
            "Termino %s: %s",
            lock_name,
            json.dumps(
                resultado,
                ensure_ascii=False
            )
        )

        return resultado

    finally:
        if cur is not None:
            if adquirido:
                try:
                    cur.execute(
                        "SELECT RELEASE_LOCK(%s)",
                        (lock_name,)
                    )
                    cur.fetchone()

                except Exception:
                    logger.exception(
                        "No se pudo liberar el bloqueo %s",
                        lock_name
                    )

            try:
                cur.close()
            except Exception:
                logger.exception(
                    "Error cerrando cursor"
                )

        conn.close()


