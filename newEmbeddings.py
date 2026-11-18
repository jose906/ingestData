from database import get_db_connection
# 1. Cargar modelo

import json
import re
from sentence_transformers import SentenceTransformer
   
    

model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"



  
      
      


def limpiar_tweet(texto):
    texto = str(texto)
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    texto = re.sub(r'[★☆◆◉•▪🔴🟡🟠✅✔️✳️🔹🔸▶️🔻🔺]+', ' ', texto)
    texto = re.sub(r'\s[-–—]{2,}\s', ' ', texto)
    texto = re.sub(r'(?i)\bRT\b\s*:?', ' ', texto)

    # menciones
    texto = re.sub(r'(?<!\w)@\w+', ' ', texto)

    # URLs
    texto = re.sub(r'https?://\S+|www\.\S+', ' ', texto, flags=re.IGNORECASE)

    return texto


def get_tweets():
    sql = """
        SELECT t.tweetid, t.text
        FROM Tweets t
        LEFT JOIN tweet_embeddings e
            ON t.tweetid = e.tweetid
        WHERE e.tweetid IS NULL
        ORDER BY t.tweetid
        LIMIT 50
    """

    conn = get_db_connection()

    try:
        cur = conn.cursor(dictionary=True)
        try:
            cur.execute(sql)
            return cur.fetchall()
        finally:
            cur.close()
    finally:
        conn.close()

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

    return data
    




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
            
            #embedding = model.encode(texto).tolist()
        
        model = SentenceTransformer(model_name)
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
        cur = None

        try:
            cur = conn.cursor()
            cur.executemany(query, data)
            conn.commit()

            print(f"Embeddings guardados/actualizados: {cur.rowcount}")

        except Exception:
            conn.rollback()
            raise

        finally:
            try:
                if cur is not None:
                    cur.close()
            finally:
                conn.close()
                
        print(f"Embeddings creados para {len(tweets)} tweets")
        return len(tweets)



if __name__ == "__main__":
    a = insert_embeddings()
    print(f"Embeddings creados para {a} tweets")
    