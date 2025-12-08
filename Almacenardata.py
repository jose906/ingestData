import os
from datetime import datetime, timedelta, timezone
import requests
import mysql.connector
from mysql.connector import Error
import MLModel

# ================== CONFIG BD ==================
DB_CONFIG = {
    
     "host": "34.69.57.221",      # o la IP de tu contenedor / Cloud SQL
    "user": "admin",
    "password": "Admin123!",
    "database": "Analisis",
    "port": 3306,
    
}

# ================== CONFIG TWITTER ==================
TWITTER_BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAN9WpgEAAAAAHarp9HjcuJFZ4wtx1DtpsP8Z93A%3DC3AEHMO2YXaGFFgblPEdkYTGhBne75WLUlG5Mc95FGKlR003vg'

# o tu token fijo mientras pruebas:
# TWITTER_BEARER_TOKEN = "AAAAAAAA..."

headers = {
    "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"
}

# ================== FUNCIÓN PRINCIPAL ==================

from datetime import datetime, timedelta, timezone
import mysql.connector
from mysql.connector import Error
import requests
import MLModel

def get_last_created_from_db():
    """
    Devuelve el último datetime (MAX(created)) de la tabla Tweets.
    Si la BD está vacía o hay error, devuelve None.
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(created) FROM Tweets")
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row and row[0]:
            return row[0]  # datetime de MySQL (naive, asumimos UTC)
        return None

    except Error as e:
        print(f"Error al consultar última fecha en MySQL: {e}")
        return None


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
        # Asumimos que 'created' en MySQL está en UTC (sin tz),
        # así que le ponemos tzinfo=UTC explícitamente.
        start_dt_utc = last_created_db.replace(tzinfo=timezone.utc)
        print(f"Último tweet en BD con created = {last_created_db} (UTC asumido)")
    else:
        # Si la BD está vacía o hubo error, usamos últimos 5 días (como antes)
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
    while True:
        resp = requests.get(url, headers=headers, params=params)

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
        params['pagination_token'] = next_token

    # ====== GUARDAR EN LA BASE DE DATOS ======
    if not all_tweets:
        print("No se encontraron tweets nuevos desde la última fecha registrada.")
        return []

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
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

            # Parsear fecha ISO de Twitter a DATETIME (UTC) para MySQL
            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))

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
            cursor.close()
            conn.close()
        except:
            pass

    return all_tweets

def get_user_ids_from_db():
    """
    Devuelve una lista de IDs de usuario (strings) desde la tabla TweetUser.idTweetUser.
    Si hay error o tabla vacía, devuelve lista vacía.
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT idTweetUser FROM TweetUser")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # rows es una lista de tuplas, ej: [(65444625,), (94438031,), ...]
        user_ids = [str(r[0]) for r in rows]

        print(f"Usuarios obtenidos de BD: {user_ids}")
        return user_ids

    except Error as e:
        print(f"Error al consultar usuarios en MySQL: {e}")
        return []


