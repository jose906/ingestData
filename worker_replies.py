# worker_replies.py
import os
from datetime import timezone
from dateutil import parser as dtparser
import MLModel

# Importa funciones desde tu main.py (donde ya las tienes)
from main import (
    get_db_connection,
    fetch_tweets_last_days,
    get_last_reply_id_for_tweet,
    fetch_replies_for_tweet,
    insert_replies,
)

def run():
    days_back = int(os.getenv("DAYS_BACK", "1"))
    cap = int(os.getenv("CAP", "100"))

    conn = get_db_connection()
    try:
        tweets = fetch_tweets_last_days(conn, days_back, cap)
        total_new = 0

        for tw in tweets:
            tweetid = int(tw["tweetid"])
            last_reply = get_last_reply_id_for_tweet(conn, tweetid)

            replies = fetch_replies_for_tweet(tweetid, since_id=last_reply)
            if not replies:
                continue

            rows = []
            for r in replies:
                rid = int(r["id"])
                created_at = r.get("created_at")
                created_dt = (
                    dtparser.isoparse(created_at)
                    .astimezone(timezone.utc)
                    .replace(tzinfo=None)
                    if created_at else None
                )

                text = r.get("text") or ""
                sent = MLModel.get_sentiment(text)[0]

                rows.append({
                    "replyid": rid,
                    "tweetid": tweetid,
                    "text": text,
                    "created": created_dt,
                    "sentimiento": sent,
                    "TweetUser_idTweetUser": None,
                })

            total_new += insert_replies(conn, rows)

        print(f"[JOB] OK - replies_inserted_or_updated={total_new}")
    finally:
        conn.close()

if __name__ == "__main__":
    run()
