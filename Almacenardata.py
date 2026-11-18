from contextlib import closing
from datetime import datetime, timedelta, timezone
import os

import requests
from mysql.connector import Error

from database import get_db_connection
import MLModel


def get_last_created_from_db():
    """
    Devuelve el último datetime (MAX(created)) de la tabla Tweets.
    Si la BD está vacía, devuelve None.
    """
    try:
        with closing(get_db_connection()) as conn, closing(conn.cursor()) as cursor:
            cursor.execute("SELECT MAX(created) FROM Tweets")
            row = cursor.fetchone()

        if row and row[0]:
            return row[0]
        return None

    except Error as e:
        print(f"Error al consultar última fecha en MySQL: {e}")
        raise


def getTweets():
    # Lista de cuentas a monitorear
    user_ids = get_user_ids_from_db()
    """user_ids = [
        '65444625', '94438031', '935136477616443393', '209279715',
        '2800854409', '735321776', '44489439', '311132840',
        '525394081', '1742290477234208768', '118861947', '188870982'
    ]"""
    query = ' OR '.join([f'from:{uid}' for uid in user_ids])

    # Zona horaria Bolivia (UTC-4)
    BOLIVIA_TZ = timezone(timedelta(hours=-4))
    now_bolivia = datetime.now(BOLIVIA_TZ)

    # Dejar 60 segundos de margen para evitar problemas de reloj
    safe_now = now_bolivia - timedelta(seconds=60)

    # ====== OBTENER ÚLTIMA FECHA DE LA BD ======
    last_created_db = get_last_created_from_db()

    if last_created_db:
        start_dt_utc = last_created_db.replace(tzinfo=BOLIVIA_TZ).astimezone(timezone.utc)
    else:
        # Si la BD está vacía, usamos los últimos 5 días
        start_5d_bo = (now_bolivia - timedelta(days=5)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start_dt_utc = start_5d_bo.astimezone(timezone.utc)
        print("BD vacía o sin fecha válida. Usando últimos 5 días como inicio.")
        print(f"start_5d (BO): {start_5d_bo}")

    # Convertir a ISO8601 UTC para Twitter
    start_time = start_dt_utc.isoformat().replace('+00:00', 'Z')
    end_time = safe_now.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')

    print(f"Descargando tweets desde (UTC): {start_dt_utc} hasta {safe_now.astimezone(timezone.utc)}")
    print(f"start_time (UTC): {start_time}")
    print(f"end_time   (UTC): {end_time}")

    url = 'https://api.twitter.com/2/tweets/search/recent'
    params = {
        'query': query,
        'start_time': start_time,
        'end_time': end_time,
        'max_results': 100,  # máximo por request
        'tweet.fields': 'created_at,text,entities,author_id',
        'user.fields': 'username',
        'expansions': 'author_id',
    }

    all_tweets = []

    # ====== DESCARGAR TODOS LOS TWEETS PAGINANDO ======
    token = os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        raise RuntimeError("Falta TWITTER_BEARER_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}
    seen_tokens = set()
    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=20)

        if resp.status_code != 200:
            raise Exception(f"Error Twitter API: {resp.status_code} - {resp.text}")

        data = resp.json()
        tweets = data.get('data', [])
        includes = data.get('includes', {})
        users_includes = includes.get('users', [])

        # Mapear author_id -> username
        author_map = {u['id']: u.get('username') for u in users_includes}

        # Guardar tweets + username
        for t in tweets:
            t['username'] = author_map.get(t['author_id'])
            all_tweets.append(t)

        meta = data.get('meta', {})
        next_token = meta.get('next_token')

        print(f"Fetched {len(tweets)} tweets, total acumulado {len(all_tweets)}")

        if not next_token:
            break

        # Agregar token de paginación para la siguiente vuelta
        if next_token in seen_tokens:
            raise RuntimeError("X devolvió un token de paginación repetido")
        seen_tokens.add(next_token)
        params['pagination_token'] = next_token

    # ====== GUARDAR EN LA BASE DE DATOS ======
    if not all_tweets:
        print("No se encontraron tweets nuevos desde la última fecha registrada.")
        return []

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert para TweetUser
        sql_insert_user = """
            INSERT INTO TweetUser (idTweetUser, TweetUser)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE TweetUser = VALUES(TweetUser)
        """

        # Insert para Tweets
        sql_insert_tweet = """
            INSERT IGNORE INTO Tweets (
                tweetid,
                text,
                created,
                url,
                sentimiento,
                categoria,
                Lugar,
                Persona,
                Organizacion,
                Locacion,
                Otros,
                TweetUser_idTweetUser
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        for t in all_tweets:
            tweetid = int(t['id'])
            text = t.get('text', '')
            created_at_str = t.get('created_at')
            author_id = int(t.get('author_id'))
            username = t.get('username')

            created_at = (
                datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                .astimezone(BOLIVIA_TZ)
                .replace(tzinfo=None)
            )

            # URL del tweet (si tenemos username)
            if username:
                url_tweet = f"https://twitter.com/{username}/status/{tweetid}"
            else:
                url_tweet = None

            # Modelo de ML para sentimiento y categoría
            sentimiento = MLModel.get_sentiment(text)[0]
            categoria = MLModel.predecir_categoria(text)[0]
            lugar = ''
            persona = ''
            organizacion = ''
            locacion = ''
            otros = ''

            # 1) Insertar/actualizar usuario
            if username:
                cursor.execute(sql_insert_user, (author_id, username))

            # 2) Insertar tweet (si ya existe lo ignora por PRIMARY KEY)
            cursor.execute(sql_insert_tweet, (
                tweetid,
                text,
                created_at,
                url_tweet,
                sentimiento,
                categoria,
                lugar,
                persona,
                organizacion,
                locacion,
                otros,
                author_id
            ))

        conn.commit()
        print(f"Se guardaron {len(all_tweets)} tweets (con duplicates ignorados).")

    except Error as e:
        print(f"Error al guardar en MySQL: {e}")
        raise
    finally:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            if conn is not None:
                conn.close()

    return all_tweets

def get_user_ids_from_db():
    """
    Devuelve una lista de IDs de usuario (strings) desde la tabla TweetUser.idTweetUser.
    Si la tabla está vacía, devuelve una lista vacía.
    """
    try:
        with closing(get_db_connection()) as conn, closing(conn.cursor()) as cursor:
            cursor.execute("SELECT idTweetUser FROM TweetUser")
            rows = cursor.fetchall()

        # rows es una lista de tuplas, ej: [(65444625,), (94438031,), ...]
        user_ids = [str(r[0]) for r in rows]

        print(f"Usuarios obtenidos de BD: {user_ids}")
        return user_ids

    except Error as e:
        print(f"Error al consultar usuarios en MySQL: {e}")
        raise


