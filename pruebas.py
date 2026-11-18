import pandas as pd

from database import get_db_connection


def get_topics(limit_topics=2):
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
    FROM (
        SELECT *
        FROM topics
        WHERE topic_name LIKE %s
        ORDER BY topic_id
        LIMIT %s
    ) tp
    JOIN topic_tweets tt
        ON tp.topic_id = tt.topic_id
    JOIN Tweets tw
        ON tt.tweetid = tw.tweetid
    ORDER BY tp.topic_id, tt.similarity DESC
    """

    try:
        return pd.read_sql(
            query,
            connection,
            params=("%NUEVO_CLUSTER%", limit_topics)
        )
    finally:
        connection.close()



print(get_topics() )