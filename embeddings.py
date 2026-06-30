from sentence_transformers import SentenceTransformer
import re ,os,json
import mysql.connector

# 1. Cargar modelo
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)
"""DB_CONFIG = {
            # IP pública o nombre interno de Cloud SQL
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASS"),
    "database": os.environ.get("DB_NAME"),
    "unix_socket": f"/cloudsql/{os.environ.get('INSTANCE_CONNECTION_NAME')}",
    "charset": "utf8mb4",
    "port": "3306",
}"""
DB_CONFIG = {
    
     "host": "34.69.57.221",      # o la IP de tu contenedor / Cloud SQL
    "user": "admin",
    "password": "Admin123!",
    "database": "Analisis",
    "port": 3306,
    
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


  


def limpiar_tweet(texto):
    texto = str(texto)
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    texto = re.sub(r'[★☆◆◉•▪🔴🟡🟠✅✔️✳️🔹🔸▶️🔻🔺]+', ' ', texto)
    texto = re.sub(r'\s[-–—]{2,}\s', ' ', texto)
    text = re.sub(r'(?i)\bRT\b\s*:?', ' ', texto)

    # menciones
    texto = re.sub(r'(?<!\w)@\w+', ' ', texto)

    # URLs
    texto = re.sub(r'https?://\S+|www\.\S+', ' ', texto, flags=re.IGNORECASE)

    return texto


def get_tweets():
    sql = """SELECT t.tweetid, t.text
        FROM Tweets t
        LEFT JOIN tweet_embeddings e
            ON t.tweetid = e.tweetid
        WHERE e.tweetid IS NULL """
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(sql)
    return cur.fetchall()

def data_to_insert(embeddings,texts,tweet_ids):
    data = []

    for tweetid, texto, embedding in zip(tweet_ids, texts, embeddings):
        data.append((
            tweetid,
            model_name,
            len(embedding),
            json.dumps(embedding.tolist()),
            texto
        ))
    



def insert_embeddings():
    tweets = get_tweets()
    tweet_ids = []
    texts_texts = []
    if not tweets:
        print("No hay tweets sin embeddings")
        return 0
    else:
        print(f"Hay {len(tweets)} tweets sin embeddings")
        for tweet in tweets:
            tweet_ids.append(tweet["tweetid"])
            texts_texts.append(limpiar_tweet(tweet["text"]))
            print(f"Tweet ID: {tweet['tweetid']}, Cleaned Text: {limpiar_tweet(tweet['text'])}")
            #embedding = model.encode(texto).tolist()
        
        embeddings = model.encode(texts_texts, batch_size=32,show_progress_bar=True,normalize_embeddings=True)
        data = data_to_insert(embeddings, texts_texts, tweet_ids)
        query = """
        INSERT INTO tweet_embeddings (
            tweetid,
            model_name,
            embedding_dim,
            embedding_vector,
            texto_preprocesado
        )
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            embedding_vector = VALUES(embedding_vector),
            texto_preprocesado = VALUES(texto_preprocesado),
            embedding_dim = VALUES(embedding_dim),
            model_name = VALUES(model_name),
            created_at = CURRENT_TIMESTAMP
        """

        conn = get_db_connection()
        cur = conn.cursor()

        try:
            cur.executemany(query, data)
            conn.commit()

            print(f"Embeddings guardados/actualizados: {cur.rowcount}")

        except Exception as e:
            conn.rollback()
            print("Error guardando embeddings:", e)

        finally:
            cur.close()
            conn.close()
                
        print(f"Embeddings creados para {len(tweets)} tweets")
        return len(tweets)
