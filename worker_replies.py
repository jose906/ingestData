# worker_replies.py
# job_main.py
import os
import sys
from main import run_ingest_users, run_ingest_replies  # vas a crear estas funciones “core”

if __name__ == "__main__":
    mode = os.environ.get("JOB_MODE", "ingest")  # ingest | replies | both
    try:
        if mode in ("ingest", "both"):
            run_ingest_users()
        if mode in ("replies", "both"):
            run_ingest_replies()
        sys.exit(0)
    except Exception as e:
        print("JOB ERROR:", str(e))
        sys.exit(1)