"""Postgres connection + schema setup (L1, L2, L10, L11). Run `python -m app.db.connection` to apply schema.sql."""
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL
from app.log import get_logger

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
log = get_logger("app.db.connection")


def connect() -> psycopg.Connection:
    try:
        conn = psycopg.connect(DATABASE_URL)
        register_vector(conn)
    except Exception:
        log.exception("DB connect failed (is the Docker db container running?)")
        raise
    return conn


def init_schema() -> None:
    # Plain connection: register_vector needs the extension, which schema.sql creates.
    log.info("applying %s", SCHEMA_PATH.name)
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    except Exception:
        log.exception("schema apply failed")
        raise


if __name__ == "__main__":
    init_schema()
    with connect() as conn:
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY 1"
        ).fetchall()
        ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    log.info("schema applied; tables=%s pgvector=%s", [t[0] for t in tables], ext[0])
