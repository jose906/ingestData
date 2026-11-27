import os

import mysql.connector


def get_db_connection():
    instance = os.getenv("INSTANCE_CONNECTION_NAME")
    host = os.getenv("DB_HOST")
    if not (instance or host):
        raise RuntimeError("Configura INSTANCE_CONNECTION_NAME o DB_HOST")

    config = {
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASS") or os.getenv("DB_PASSWORD"),
        "database": os.getenv("DB_NAME"),
        "charset": "utf8mb4",
        "connection_timeout": 10,
    }
    if instance:
        config["unix_socket"] = f"/cloudsql/{instance}"
    else:
        config["host"] = host
        config["port"] = int(os.getenv("DB_PORT", "3306"))
    return mysql.connector.connect(**config)


def get_state(cursor, key: str, default=None):
    cursor.execute("SELECT v FROM ingest_state WHERE k=%s", (key,))
    row = cursor.fetchone()
    value = row[0] if row else default
    return default if value == "" else value


def set_state(cursor, key: str, value):
    cursor.execute(
        "INSERT INTO ingest_state (k, v) VALUES (%s, %s) "
        "ON DUPLICATE KEY UPDATE v=VALUES(v)",
        (key, value)
    )
