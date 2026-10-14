import os
import mysql.connector
import json
import requests
import json

DB_CONFIG = {
    "host": "34.69.57.221",
    "user": "admin",
    "password": "Admin123!",
    "database": "Analisis",
    "port": 3306,
}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

#client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)






def get_replies_pending(limit=20):
    """
    Obtiene hasta 'limit' comentarios pendientes
    pertenecientes a una misma publicación.

    Utiliza el índice:
    idx_replies_stance_tweetid_id
    """

    connection = get_db_connection()
    cursor = None

    try:
        cursor = connection.cursor(dictionary=True)

        # 1. Encontrar una publicación con comentarios pendientes
        cursor.execute("""
            SELECT tweetid
            FROM replies
            WHERE stance_processed = 0
            ORDER BY tweetid ASC, id ASC
            LIMIT 1
        """)

        publicacion = cursor.fetchone()

        if not publicacion:
            return []

        tweetid = publicacion["tweetid"]

        # 2. Obtener sus comentarios pendientes
        cursor.execute("""
            SELECT
                r.id,
                r.replyid,
                r.tweetid,
                r.text AS comentario,
                r.sentimiento,
                t.text AS publicacion_original
            FROM replies r
            INNER JOIN Tweets t
                ON t.tweetid = r.tweetid
            WHERE r.stance_processed = 0
              AND r.tweetid = %s
            ORDER BY r.id ASC
            LIMIT %s
        """, (tweetid, limit))

        return cursor.fetchall()

    finally:
        if cursor is not None:
            cursor.close()
        connection.close()
        

def get_replies_pending_batch(
    last_tweetid=0,
    limit=20
):
    """
    Obtiene comentarios pendientes de una publicación,
    avanzando por tweetid.

    Devuelve (comentarios, tweetid).
    """

    connection = get_db_connection()
    cursor = None

    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT tweetid
            FROM replies
            WHERE stance_processed = 0
              AND tweetid > %s
            ORDER BY tweetid ASC, id ASC
            LIMIT 1
        """, (last_tweetid,))

        row = cursor.fetchone()

        if not row:
            return [], None

        tweetid = row["tweetid"]

        cursor.execute("""
            SELECT
                r.id,
                r.replyid,
                r.tweetid,
                r.text AS comentario,
                r.sentimiento,
                t.text AS publicacion_original
            FROM replies r
            INNER JOIN Tweets t
                ON t.tweetid = r.tweetid
            WHERE r.stance_processed = 0
              AND r.tweetid = %s
            ORDER BY r.id ASC
            LIMIT %s
        """, (tweetid, limit))

        comentarios = cursor.fetchall()

        return comentarios, tweetid

    finally:
        if cursor is not None:
            cursor.close()
        connection.close()



def procesar_lote_stance(comentarios, max_intentos=3):
    """
    Clasifica comentarios con Gemini.
    Reintenta resultados faltantes o inválidos.
    No guarda datos en MySQL.
    """

    if not comentarios:
        return [], []

    publicacion = comentarios[0]["publicacion_original"]

    pendientes = {
        str(c["replyid"]): c
        for c in comentarios
    }

    resultados_finales = {}

    for intento in range(1, max_intentos + 1):

        if not pendientes:
            break

        print(
            f"\nIntento {intento}: "
            f"{len(pendientes)} comentarios"
        )

        lote = list(pendientes.values())

        try:
            respuesta = clasificar_stance(
                publicacion,
                lote
            )

            validacion = validar_resultados_stance(
                lote,
                respuesta
            )

            # No aceptar IDs duplicados o desconocidos
            ids_rechazados = set(
                validacion["duplicados"]
            ) | set(
                validacion["desconocidos"]
            )

            for r in validacion["validos"]:

                replyid = str(r["replyid"])

                if replyid in ids_rechazados:
                    continue

                resultados_finales[replyid] = r
                pendientes.pop(replyid, None)

            print(
                "Válidos acumulados:",
                len(resultados_finales)
            )

            print(
                "Pendientes:",
                len(pendientes)
            )

        except Exception as e:
            print(
                f"Error en intento {intento}: {e}"
            )

    return (
        list(resultados_finales.values()),
        list(pendientes.values())
    )


def validar_resultados_stance(comentarios, resultados):

    if not isinstance(resultados, list):
        raise ValueError(
            "Gemini no devolvió una lista JSON"
        )

    ids_esperados = {
        str(c["replyid"]) for c in comentarios
    }

    ids_recibidos = set()
    duplicados = set()
    desconocidos = set()
    invalidos = []

    etiquetas_validas = {
        "FAVOR",
        "CONTRA",
        "MIXTO",
        "NEUTRAL",
        "NO_CLARO"
    }

    validos = []

    for r in resultados:

        if not isinstance(r, dict):
            invalidos.append(r)
            continue

        replyid = str(r.get("replyid"))

        if replyid not in ids_esperados:
            desconocidos.add(replyid)
            continue

        if replyid in ids_recibidos:
            duplicados.add(replyid)
            continue

        ids_recibidos.add(replyid)

        try:
            confidence = float(r["confidence"])

            if (
                r.get("stance") not in etiquetas_validas
                or not 0 <= confidence <= 1
            ):
                invalidos.append(replyid)
                continue

            validos.append(r)

        except (KeyError, TypeError, ValueError):
            invalidos.append(replyid)

    ids_validos = {
        str(r["replyid"]) for r in validos
    }

    faltantes = ids_esperados - ids_validos

    return {
        "validos": validos,
        "faltantes": sorted(faltantes),
        "duplicados": sorted(duplicados),
        "desconocidos": sorted(desconocidos),
        "invalidos": invalidos
    }




def get_tweets_topic(topic_id, limit=20):
    connection = get_db_connection()

    query = """
        SELECT
            t.tweetid,
            t.text,
            tt.similarity
        FROM topic_tweets tt
        INNER JOIN Tweets t
            ON t.tweetid = tt.tweetid
        WHERE tt.topic_id = %s
        ORDER BY tt.similarity DESC
        LIMIT %s
    """

    cursor = connection.cursor(dictionary=True)
    cursor.execute(query, (topic_id, limit))
    resultados = cursor.fetchall()

    cursor.close()
    connection.close()

    return resultados

def llamar_gemini(prompt, thinking_budget=0):

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:generateContent"
    )

    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
            
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "thinkingConfig": {
                "thinkingBudget": thinking_budget
            }
        }

    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        print("ERROR GEMINI:")
        print("Status:", response.status_code)
        print("Respuesta:", response.text)

        response.raise_for_status()

    data = response.json()
    
    usage = data.get("usageMetadata", {})

    tokens_entrada = usage.get("promptTokenCount", 0)
    tokens_salida = usage.get("candidatesTokenCount", 0)
    tokens_razonamiento = usage.get("thoughtsTokenCount", 0)

    costo_entrada = tokens_entrada * 0.30 / 1_000_000

    costo_salida = (
        tokens_salida + tokens_razonamiento
    ) * 2.50 / 1_000_000

    costo_total = costo_entrada + costo_salida

    print("\n===== CONSUMO GEMINI =====")
    print("Entrada:", tokens_entrada)
    print("Salida:", tokens_salida)
    print("Razonamiento:", tokens_razonamiento)
    print("Costo estimado USD:", round(costo_total, 6))


    texto = data["candidates"][0]["content"]["parts"][0]["text"]

    return texto

def analizar_target_topic(topic_name, tweets):

    texto_tweets = "\n\n".join([
        f"PUBLICACIÓN {i+1}:\n{tweet}"
        for i, tweet in enumerate(tweets)
    ])

    prompt = f"""
        Eres un analista de conversaciones en redes sociales.

        Analiza el siguiente tópico y sus publicaciones representativas.

        TOPIC:
        {topic_name}

        PUBLICACIONES:
        {texto_tweets}

        Debes determinar qué tipo de tópico es:

        STANCE:
        Existe una cuestión, propuesta, decisión, acción o posición central
        sobre la cual tiene sentido medir si los comentarios están
        a FAVOR o EN CONTRA.

        INFORMATIVO:
        El tópico principalmente describe hechos, acontecimientos,
        accidentes, clima, resultados, sucesos o información donde
        FAVOR/CONTRA no representa adecuadamente la conversación.

        MIXTO:
        Las publicaciones agrupan varios asuntos diferentes y no existe
        un único objetivo suficientemente claro para medir stance.

        Si es STANCE:

        main_target debe describir UNA cuestión concreta sobre la que
        pueda existir apoyo o rechazo.

        Ejemplo correcto:
        "eliminación de la subvención a los combustibles"

        Ejemplo incorrecto:
        "combustibles"

        target_description debe definir únicamente cómo interpretar FAVOR
        y CONTRA respecto al main_target.

        Debe ser neutral, breve y simétrica.

        No agregues:
        - beneficios de la medida
        - consecuencias posibles
        - justificaciones
        - opiniones
        - argumentos a favor o en contra
        - información que no sea necesaria para definir el target

        Ejemplo:

        main_target:
        "aprobación de la ley para designación de vocales del TSE"

        target_description:
        "FAVOR significa apoyar la aprobación de la ley.
        CONTRA significa oponerse a la aprobación de la ley."

        No inventes información que no aparezca en las publicaciones.

        Devuelve exclusivamente JSON válido:

        {{
        "target_type": "STANCE|INFORMATIVO|MIXTO",
        "main_target": null,
        "target_description": null,
        "confidence": 0.00
        }}
        """

    respuesta = llamar_gemini(prompt)
    return json.loads(respuesta)

def clasificar_stance(publicacion,comentarios,thinking_budget=0):

    texto_comentarios = "\n".join([
            f"""
                ID: {c["replyid"]}
                COMENTARIO: {c["comentario"]}
                """
                        for c in comentarios
                    ])

    
    prompt = f"""
        Eres un analista especializado en STANCE de
        comentarios sobre publicaciones de Bolivia.

        Tu tarea es identificar la postura de cada comentario
        respecto a la AFIRMACIÓN, DECISIÓN, PROPUESTA o
        ACCIÓN CENTRAL de la publicación original.

        PUBLICACIÓN ORIGINAL:
        {publicacion}

        COMENTARIOS:
        {texto_comentarios}

        ETIQUETAS:

        FAVOR:
        El comentario apoya, respalda o está de acuerdo
        con la afirmación, decisión, propuesta o acción
        central de la publicación.

        CONTRA:
        El comentario rechaza, cuestiona o contradice
        directamente ese objetivo central.

        MIXTO:
        Expresa simultáneamente apoyo y rechazo
        al mismo objetivo, con evidencia clara.

        NEUTRAL:
        Habla del objetivo central, pero no expresa
        apoyo ni rechazo.

        NO_CLARO:
        No existe evidencia suficiente para establecer
        la postura respecto al objetivo central.

        REGLAS OBLIGATORIAS:

        1. Identifica internamente el objetivo central
        antes de clasificar los comentarios.

        2. Si la publicación solo informa un acontecimiento,
        no inventes una posición que el medio no expresó.

        3. Una crítica al autor, periodista, político
        o institución no implica automáticamente CONTRA.

        4. Un comentario negativo no necesariamente es CONTRA.
        Un comentario positivo no necesariamente es FAVOR.

        5. Diferencia cuestionar una afirmación de
        rechazar a la persona que la pronuncia.

        6. Si el comentario critica una acción que
        constituye el objetivo central, puede ser CONTRA,
        aunque utilice sarcasmo.

        7. Si el comentario apoya una afirmación factual
        claramente identificable, puede ser FAVOR.

        8. Si una publicación contiene varias afirmaciones
        y no existe una central claramente identificable,
        utiliza NO_CLARO cuando la postura sea ambigua.

        9. No infieras el contenido de imágenes, videos
        o enlaces que no estén disponibles en el texto.

        10. Insultos genéricos sin vínculo claro con
        el objetivo central deben ser NO_CLARO.

        11. Si un comentario introduce otro problema
        sin apoyar ni rechazar el objetivo central,
        utiliza NEUTRAL o NO_CLARO según el contexto.

        12. No clasifiques según sentimiento.

        13. Evalúa cada comentario individualmente.
        No infieras su postura a partir de otros comentarios.

        14. No inventes información ni intenciones.

        15. confidence debe representar la certeza
        de la etiqueta asignada, entre 0 y 1.

        Devuelve exclusivamente un ARRAY JSON válido:

        [
        {{
            "replyid": "123",
            "stance": "FAVOR",
            "confidence": 0.95
        }}
        ]

        Incluye exactamente un resultado por cada
        comentario recibido, conservando su replyid.
        """


    texto = llamar_gemini(prompt,thinking_budget=thinking_budget)

    return json.loads(texto)


from collections import defaultdict

def comparar_stance_modelos(limit=50):

    muestra = get_muestra_stance(limit)

    grupos = defaultdict(list)

    for comentario in muestra:
        grupos[comentario["tweetid"]].append(comentario)

    comparaciones = []

    for tweetid, comentarios in grupos.items():

        publicacion = comentarios[0]["publicacion_original"]

        print(
            f"\nAnalizando tweet {tweetid}: "
            f"{len(comentarios)} comentarios"
        )

        resultados = {}

        for budget in [0, 1024]:

            try:
                clasificaciones = clasificar_stance(
                    publicacion,
                    comentarios,
                    thinking_budget=budget
                )

                validacion = validar_resultados_stance(
                    comentarios,
                    clasificaciones
                )

                rechazados = set(
                    validacion["duplicados"]
                )

                resultados[budget] = {
                    str(r["replyid"]): r
                    for r in validacion["validos"]
                    if str(r["replyid"]) not in rechazados
                }

            except Exception as e:
                print("Error:", tweetid, budget, e)
                resultados[budget] = {}

        for comentario in comentarios:

            replyid = str(comentario["replyid"])

            sin = resultados[0].get(replyid, {})
            con = resultados[1024].get(replyid, {})

            stance_sin = sin.get("stance")
            stance_con = con.get("stance")

            comparaciones.append({
                "tweetid": str(tweetid),
                "replyid": replyid,
                "publicacion": publicacion,
                "comentario": comentario["comentario"],
                "stance_actual": comentario["stance_actual"],
                "sin_razonamiento": stance_sin,
                "con_razonamiento": stance_con,
                "coinciden": (
                    stance_sin == stance_con
                    if stance_sin is not None
                    and stance_con is not None
                    else None
                )
            })

    return comparaciones



def get_topics_pending(limit=10):
    connection = get_db_connection()

    query = """
        SELECT
            topic_id,
            topic_name
        FROM topics
        WHERE target_type IS NULL
          AND active = 1
        ORDER BY total_tweets DESC
        LIMIT %s
    """

    cursor = connection.cursor(dictionary=True)
    cursor.execute(query, (limit,))
    resultados = cursor.fetchall()

    cursor.close()
    connection.close()

    return resultados

def update_topic_target(topic_id, resultado):
    connection = get_db_connection()

    query = """
        UPDATE topics
        SET
            target_type = %s,
            main_target = %s,
            target_description = %s,
            target_confidence = %s
        WHERE topic_id = %s
    """

    cursor = connection.cursor()

    cursor.execute(
        query,
        (
            resultado["target_type"],
            resultado.get("main_target"),
            resultado.get("target_description"),
            resultado.get("confidence"),
            topic_id
        )
    )

    connection.commit()
    cursor.close()
    connection.close()


def guardar_stance(resultados):
    """
    Guarda las clasificaciones de Gemini en replies.
    """

    if not resultados:
        return 0

    etiquetas_validas = {
        "FAVOR",
        "CONTRA",
        "MIXTO",
        "NEUTRAL",
        "NO_CLARO"
    }

    datos = []
    ids_vistos = set()

    for resultado in resultados:
        try:
            replyid = int(resultado["replyid"])
            stance = resultado["stance"]
            confidence = float(resultado["confidence"])

            if stance not in etiquetas_validas:
                continue

            if not 0 <= confidence <= 1:
                continue

            if replyid in ids_vistos:
                continue

            ids_vistos.add(replyid)

            datos.append((
                stance,
                confidence,
                replyid
            ))

        except (KeyError, TypeError, ValueError):
            continue

    if not datos:
        return 0

    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        query = """
            UPDATE replies
            SET
                stance = %s,
                stance_confidence = %s,
                stance_processed = 1
            WHERE replyid = %s
              AND stance_processed = 0
        """

        cursor.executemany(query, datos)
        actualizados = cursor.rowcount
        connection.commit()

        return actualizados

    except Exception:
        connection.rollback()
        raise

    finally:
        if 'cursor' in locals():
            cursor.close()
        connection.close()



def ejecutar_procesamiento_stance(
    max_comentarios=100,
    tamano_lote=20
):
    """
    Procesa comentarios pendientes de forma incremental.
    Un lote fallido queda pendiente para una futura ejecución.
    """

    total_intentados = 0
    total_guardados = 0
    total_fallidos = 0
    total_lotes = 0

    ultimo_tweetid = 0

    while total_intentados < max_comentarios:

        limite = min(
            tamano_lote,
            max_comentarios - total_intentados
        )

        comentarios, tweetid = get_replies_pending_batch(
            last_tweetid=ultimo_tweetid,
            limit=limite
        )

        if tweetid is None:
            print("Fin del recorrido.")
            break

        if not comentarios:
            print("Sin publicación original:", tweetid)
            ultimo_tweetid = tweetid
            continue

        total_lotes += 1
        total_intentados += len(comentarios)

        try:
            resultados, pendientes = procesar_lote_stance(
                comentarios,
                max_intentos=3
            )

            ids_permitidos = {
                str(c["replyid"]) for c in comentarios
            }

            resultados = [
                r for r in resultados
                if str(r.get("replyid")) in ids_permitidos
            ]

            guardados = guardar_stance(resultados)

            total_guardados += guardados

            fallidos_lote = len(comentarios) - guardados
            total_fallidos += fallidos_lote

            print(
                f"Lote {total_lotes} | "
                f"Tweet {tweetid} | "
                f"Enviados {len(comentarios)} | "
                f"Guardados {guardados} | "
                f"Fallidos {fallidos_lote}"
            )

            # Si falló algún comentario, avanzar
            # para evitar repetir el mismo lote.
            if fallidos_lote > 0:
                ultimo_tweetid = tweetid
                continue

            # Si todo salió bien, repetir este tweetid
            # para recuperar sus siguientes comentarios.
            ultimo_tweetid = tweetid - 1

        except Exception as e:
            total_fallidos += len(comentarios)
            ultimo_tweetid = tweetid

            print(
                f"ERROR en publicación {tweetid}: {e}"
            )

    print("\n========== RESUMEN ==========")
    print("Intentados:", total_intentados)
    print("Guardados:", total_guardados)
    print("Fallidos:", total_fallidos)
    print("Lotes:", total_lotes)

    return {
        "intentados": total_intentados,
        "guardados": total_guardados,
        "fallidos": total_fallidos,
        "lotes": total_lotes
    }




def get_muestra_stance(limit=50):
    """
    Obtiene comentarios ya clasificados,
    agrupados por publicación.
    """

    connection = get_db_connection()

    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                r.replyid,
                r.tweetid,
                r.text AS comentario,
                r.stance AS stance_actual,
                r.stance_confidence AS confidence_actual,
                t.text AS publicacion_original
            FROM replies r
            INNER JOIN Tweets t
                ON t.tweetid = r.tweetid
            WHERE r.stance_processed = 1
              AND r.stance IS NOT NULL
            ORDER BY RAND()
            LIMIT %s
        """, (limit,))

        return cursor.fetchall()

    finally:
        cursor.close()
        connection.close()




def analizar_target_publicacion(publicacion):
    """
    Identifica si una publicación tiene un objetivo
    central adecuado para medir Stance.
    """

    prompt = f"""
Eres un analista de conversaciones públicas de Bolivia.

Analiza exclusivamente la siguiente publicación:

PUBLICACIÓN:
{publicacion}

Determina si existe un objetivo central sobre el cual
tenga sentido medir apoyo o rechazo de los comentarios.

TIPOS:

STANCE:
Existe una afirmación, propuesta, decisión o acción
concreta respecto a la cual se puede expresar apoyo
o rechazo.

INFORMATIVO:
Describe principalmente un acontecimiento o hecho
sin una cuestión central adecuada para medir apoyo
o rechazo.


TIPOS:

STANCE:
La publicación presenta UNA cuestión central
concreta y evaluable.

Puede ser:
- Una propuesta o medida.
- Una decisión política o institucional.
- Una acción específica cuestionada.
- Una afirmación central claramente debatible.
- Una exigencia explícita.

La postura FAVOR o CONTRA debe tener
un significado inequívoco.

INFORMATIVO:
La publicación principalmente informa sobre:
- Un hecho o acontecimiento.
- Una denuncia sin propuesta central.
- Un accidente o delito.
- Una actividad o evento.
- Una declaración meramente descriptiva.

No conviertas automáticamente cualquier noticia,
denuncia o acontecimiento en STANCE.

MIXTO:
La publicación presenta dos o más cuestiones
evaluables independientes.

No existe una sola postura que represente
adecuadamente todas las cuestiones.

REGLAS OBLIGATORIAS:

1. Identifica el objetivo central explícito.

2. No inventes una medida, decisión, postura
o conducta que el texto no establezca.

3. No transformes una denuncia en una propuesta
de apoyo o rechazo.

4. Distingue una acusación de un hecho comprobado.

5. No confundas críticas al autor con apoyo
o rechazo al contenido.

6. Si existen dos objetivos independientes,
clasifica MIXTO.

7. Si la publicación simplemente informa
sobre un hecho sin una cuestión central
adecuada, clasifica INFORMATIVO.

8. Para STANCE, main_target debe describir
exactamente UNA cuestión.

9. target_description debe ser neutral,
breve y simétrica.

10. No incluyas argumentos, consecuencias
ni justificaciones inventadas.

11. Si no existe un target único y claro,
no fuerces STANCE.

12. Devuelve null en main_target y
target_description para INFORMATIVO y MIXTO.


Devuelve exclusivamente JSON válido:

{{
    "target_type": "STANCE",
    "main_target": "objetivo concreto",
    "target_description": "FAVOR significa apoyar ... CONTRA significa rechazar ...",
    "confidence": 0.95
}}
"""

    respuesta = llamar_gemini(
        prompt,
        thinking_budget=0
    )

    resultado = json.loads(respuesta)

    tipos_validos = {
        "STANCE",
        "INFORMATIVO",
        "MIXTO"
    }

    if resultado.get("target_type") not in tipos_validos:
        raise ValueError("Tipo de target inválido")

    if resultado["target_type"] == "STANCE":
        if (
            not resultado.get("main_target")
            or not resultado.get("target_description")
        ):
            raise ValueError("Target incompleto")
    else:
        resultado["main_target"] = None
        resultado["target_description"] = None

    confianza = float(resultado["confidence"])

    if not 0 <= confianza <= 1:
        raise ValueError("Confianza inválida")

    resultado["confidence"] = confianza

    return resultado



def guardar_post_target(tweetid, resultado):
    """
    Guarda el target identificado para una publicación.
    Si ya existe, devuelve su ID sin sobrescribirlo.
    """

    tipos_validos = {"STANCE", "INFORMATIVO", "MIXTO"}

    tipo = resultado.get("target_type")

    if tipo not in tipos_validos:
        raise ValueError("target_type inválido")

    if tipo == "STANCE":
        main_target = resultado.get("main_target")
        description = resultado.get("target_description")

        if not main_target or not description:
            raise ValueError("Target STANCE incompleto")
    else:
        main_target = None
        description = None

    confidence = float(resultado["confidence"])

    if not 0 <= confidence <= 1:
        raise ValueError("Confianza inválida")

    connection = get_db_connection()
    cursor = None

    try:
        cursor = connection.cursor()

        cursor.execute("""
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
            tipo,
            main_target,
            description,
            confidence,
            "gemini-2.5-flash",
            "v1"
        ))

        target_id = cursor.lastrowid
        connection.commit()

        return target_id

    except Exception:
        connection.rollback()
        raise

    finally:
        if cursor is not None:
            cursor.close()
        connection.close()



def get_post_target(tweetid):
    """
    Devuelve el target de una publicación
    o None si todavía no fue analizada.
    """

    connection = get_db_connection()
    cursor = None

    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                id,
                tweetid,
                target_type,
                main_target,
                target_description,
                confidence,
                model,
                prompt_version
            FROM post_targets
            WHERE tweetid = %s
            LIMIT 1
        """, (tweetid,))

        return cursor.fetchone()

    finally:
        if cursor is not None:
            cursor.close()
        connection.close()



def clasificar_stance_target(publicacion, target, comentarios):
    """
    Clasifica comentarios respecto a un target explícito.
    No guarda resultados en MySQL.
    """

    if target["target_type"] != "STANCE":
        return []

    texto_comentarios = "\n\n".join([
        f"ID: {c['replyid']}\nCOMENTARIO: {c['comentario']}"
        for c in comentarios
    ])

    prompt = f"""
Eres un clasificador especializado en STANCE
para conversaciones públicas de Bolivia.

PUBLICACIÓN ORIGINAL:
{publicacion}

OBJETIVO CENTRAL:
{target["main_target"]}

DEFINICIÓN DE POSTURAS:
{target["target_description"]}

COMENTARIOS:
{texto_comentarios}

Clasifica la postura de cada comentario
EXCLUSIVAMENTE respecto al OBJETIVO CENTRAL.

ETIQUETAS:

FAVOR:
Apoya explícitamente el objetivo central.

CONTRA:
Rechaza explícitamente el objetivo central.

MIXTO:
Contiene apoyo y rechazo claros respecto
al mismo objetivo central.


NEUTRAL:
El comentario es comprensible y está relacionado
con el objetivo o su contexto inmediato, pero
no expresa apoyo ni rechazo al objetivo.

Incluye preguntas informativas, observaciones
descriptivas y comentarios sobre el procedimiento
que no implican una postura sobre el objetivo.

NO_CLARO:
No existe información suficiente para determinar
qué postura expresa el comentario, o no es posible
establecer su relación con el objetivo.

Incluye insultos genéricos, sarcasmo ambiguo,
menciones, enlaces sin contexto y expresiones
cuyo referente no puede identificarse.

IMPORTANTE:
No utilices NO_CLARO únicamente porque un
comentario carezca de apoyo o rechazo.

Si el comentario es comprensible y está
claramente relacionado con el tema, pero no
expresa postura, utiliza NEUTRAL.

REGLAS:

1. No confundas sentimiento con postura.

2. Una crítica al autor no implica automáticamente
rechazo al objetivo.

3. Una opinión negativa sobre otro asunto
no necesariamente significa CONTRA.

4. No inventes el contenido de enlaces,
imágenes ni videos.

5. Si existen varias interpretaciones razonables
con posturas diferentes, utiliza NO_CLARO.

6. Los insultos genéricos deben clasificarse
NO_CLARO si no existe una relación clara
con el objetivo.

7. No interpretes automáticamente emojis,
risas o sarcasmo como rechazo.

8. Usa el contexto de la publicación para
interpretar referencias indirectas.

9. FAVOR o CONTRA requieren evidencia
textual suficiente.

10. confidence debe estar entre 0 y 1.


REGLAS DE DESAMBIGUACIÓN DEL OBJETIVO:

- Distingue estrictamente el objetivo central
  del procedimiento utilizado para decidirlo.

- Una crítica al método de consulta, votación,
  comunicación o implementación NO implica
  automáticamente rechazar el objetivo central.

- Una crítica a una autoridad o institución
  tampoco implica rechazar el objetivo.

- Si el comentario critica un aspecto relacionado,
  pero no permite determinar apoyo o rechazo
  al objetivo central, utiliza NO_CLARO.

- Para clasificar CONTRA, identifica evidencia
  de rechazo al objetivo concreto.

- Para clasificar FAVOR, identifica evidencia
  de apoyo al objetivo concreto.

- No conviertas palabras negativas genéricas,
  insultos o sarcasmo ambiguo en CONTRA.

- Si el comentario cuestiona el procedimiento
  sin expresar una postura sobre el resultado
  buscado, utiliza NO_CLARO.

- Si el comentario cuestiona explícitamente
  los efectos o la conveniencia del objetivo,
  puedes clasificar CONTRA.

- Si existen dos interpretaciones razonables
  respecto al objetivo, prioriza NO_CLARO.


Devuelve exclusivamente un array JSON
con exactamente un resultado por comentario:

[
  {{
    "replyid": "123",
    "stance": "FAVOR",
    "confidence": 0.95
  }}
]
"""

    respuesta = llamar_gemini(
        prompt,
        thinking_budget=0
    )

    return json.loads(respuesta)



def guardar_stance_target(post_target_id, resultados):

    if not resultados:
        return 0

    etiquetas_validas = {
        "FAVOR", "CONTRA", "MIXTO",
        "NEUTRAL", "NO_CLARO"
    }

    datos = []
    ids_vistos = set()

    for r in resultados:

        replyid = int(r["replyid"])
        stance = r["stance"]
        confidence = float(r["confidence"])

        if replyid in ids_vistos:
            raise ValueError("replyid duplicado")

        if stance not in etiquetas_validas:
            raise ValueError("Stance inválido")

        if not 0 <= confidence <= 1:
            raise ValueError("Confidence inválida")

        ids_vistos.add(replyid)

        datos.append((
            replyid,
            post_target_id,
            stance,
            confidence,
            "gemini-2.5-flash",
            "v1"
        ))

    connection = get_db_connection()
    cursor = None

    try:
        cursor = connection.cursor()

        cursor.executemany("""
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
        """, datos)

        guardados = cursor.rowcount
        connection.commit()

        return guardados

    except Exception:
        connection.rollback()
        raise

    finally:
        if cursor is not None:
            cursor.close()
        connection.close()




def get_siguiente_publicacion_target(last_tweetid=0):

    connection = get_db_connection()
    cursor = None

    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                t.tweetid,
                t.text
            FROM Tweets t
            LEFT JOIN post_targets pt
                ON pt.tweetid = t.tweetid
            WHERE t.tweetid > %s
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
                                FROM reply_target_stance rs
                                WHERE rs.replyid = r.replyid
                                  AND rs.post_target_id = pt.id
                            )
                      )
                  )
              )
            ORDER BY t.tweetid ASC
            LIMIT 1
        """, (last_tweetid,))

        return cursor.fetchone()

    finally:
        if cursor is not None:
            cursor.close()
        connection.close()


def get_comentarios_target_pendientes(
    tweetid,
    post_target_id,
    limit=20
):
    """
    Recupera comentarios sin clasificación
    para el target indicado.
    """

    connection = get_db_connection()
    cursor = None

    try:
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                r.replyid,
                r.text AS comentario
            FROM replies r
            WHERE r.tweetid = %s
              AND NOT EXISTS (
                  SELECT 1
                  FROM reply_target_stance rs
                  WHERE rs.replyid = r.replyid
                    AND rs.post_target_id = %s
              )
            ORDER BY r.id ASC
            LIMIT %s
        """, (
            tweetid,
            post_target_id,
            limit
        ))

        return cursor.fetchall()

    finally:
        if cursor is not None:
            cursor.close()
        connection.close()




def ejecutar_stance_por_target(
    max_comentarios=500,
    tamano_lote=20,
    max_publicaciones=200
):
    """
    Procesa comentarios utilizando el target
    explícito de cada publicación.
    """

    if max_comentarios <= 0 or tamano_lote <= 0:
        raise ValueError("Los límites deben ser positivos")

    intentados = 0
    guardados = 0
    fallidos = 0
    publicaciones = 0
    ultimo_tweetid = 0

    while (intentados < max_comentarios and publicaciones < max_publicaciones):

        publicacion = get_siguiente_publicacion_target(
            ultimo_tweetid
        )

        if not publicacion:
            break

        tweetid = publicacion["tweetid"]
        publicaciones += 1

        try:
            target = get_post_target(tweetid)

            if target is None:
                resultado = analizar_target_publicacion(
                    publicacion["text"]
                )

                guardar_post_target(tweetid, resultado)
                target = get_post_target(tweetid)

            if target["target_type"] != "STANCE":
                ultimo_tweetid = tweetid
                continue

            while intentados < max_comentarios:

                limite = min(
                    tamano_lote,
                    max_comentarios - intentados
                )

                comentarios = get_comentarios_target_pendientes(
                    tweetid,
                    target["id"],
                    limite
                )

                if not comentarios:
                    break

                intentados += len(comentarios)

                try:
                    respuesta = clasificar_stance_target(
                        publicacion["text"],
                        target,
                        comentarios
                    )

                    validacion = validar_resultados_stance(
                        comentarios,
                        respuesta
                    )

                    ids_rechazados = set(
                        validacion["duplicados"]
                    )

                    resultados = [
                        r for r in validacion["validos"]
                        if str(r["replyid"]) not in ids_rechazados
                    ]

                    nuevos = guardar_stance_target(
                        target["id"],
                        resultados
                    )

                    guardados += nuevos
                    fallidos += len(comentarios) - nuevos

                    print(
                        f"Tweet {tweetid} | "
                        f"Enviados {len(comentarios)} | "
                        f"Guardados {nuevos}"
                    )

                    if nuevos != len(comentarios):
                        # No repetir inmediatamente un lote incompleto.
                        break

                except Exception as e:
                    fallidos += len(comentarios)
                    print("Error en lote:", tweetid, e)
                    break

        except Exception as e:
            print("Error en publicación:", tweetid, e)

        ultimo_tweetid = tweetid

    print("\n===== RESUMEN =====")
    print("Comentarios intentados:", intentados)
    print("Guardados:", guardados)
    print("Fallidos:", fallidos)
    print("Publicaciones recorridas:", publicaciones)

    return {
        "intentados": intentados,
        "guardados": guardados,
        "fallidos": fallidos,
        "publicaciones": publicaciones
    }
    
    
    
    
if __name__ == "__main__":
    resumen = ejecutar_stance_por_target(
        max_comentarios=500,
        tamano_lote=20,
        max_publicaciones=200
    )

    print(resumen)




