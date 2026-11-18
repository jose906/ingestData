import os

from main import app, ingest_handler, ingest_replies_handler


def main():
    jobs = {
        "ingest": (ingest_handler,),
        "replies": (ingest_replies_handler,),
        "both": (ingest_handler, ingest_replies_handler),
    }
    mode = os.getenv("JOB_MODE", "ingest")
    if mode not in jobs:
        app.logger.error("JOB_MODE inválido: %s", mode)
        return 2

    failed = False
    with app.app_context():
        for job in jobs[mode]:
            _, status = job()
            if status >= 400:
                app.logger.error("%s falló con estado %s", job.__name__, status)
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
