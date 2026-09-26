"""Shared pytest fixtures.

DB tests run against a real Postgres + pgvector **test database** (default: the app database name + "_test" on the
same server, e.g. transcripts_test; override with TEST_DATABASE_URL). SQLite/mocks can't stand in for tsvector,
pgvector or FOR UPDATE SKIP LOCKED, so if the test DB isn't reachable, DB tests are skipped instead of faked.

Fixtures:
  test_db_url  (session)  - creates the test database if missing and applies app/db/schema.sql
  db_conn      (function) - connection with pgvector types; everything is rolled back after the test
  clean_db     (function) - for code that commits (e.g. ingest): truncates the tables before and after the test
"""
import os
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector
from psycopg.conninfo import conninfo_to_dict, make_conninfo

SCHEMA_SQL = Path(__file__).resolve().parent.parent / "app" / "db" / "schema.sql"


def _resolve_test_url() -> str | None:
    """TEST_DATABASE_URL, else the app's DB with '_test' appended to the name (None if no DB settings exist)."""
    if os.getenv("TEST_DATABASE_URL"):
        return os.environ["TEST_DATABASE_URL"]
    try:
        from app.config import DATABASE_URL  # imported lazily: needs .env / POSTGRES_* settings
    except RuntimeError:
        return None
    return make_conninfo(DATABASE_URL, dbname=conninfo_to_dict(DATABASE_URL)["dbname"] + "_test")


def _require_test_db(url: str) -> str:
    # Guard: fixtures create, write to and truncate this database, so never point them at a real one.
    name = conninfo_to_dict(url).get("dbname", "")
    if not name.endswith("_test"):
        raise pytest.UsageError(f"refusing to use database {name!r} for tests: its name must end with '_test'")
    return name


@pytest.fixture(scope="session")
def test_db_url() -> str:
    url = _resolve_test_url()
    if url is None:
        pytest.skip("no DB settings (.env / POSTGRES_* or TEST_DATABASE_URL); DB tests skipped")
    name = _require_test_db(url)
    try:
        # CREATE DATABASE can't run in a transaction: use an autocommit connection to the maintenance DB.
        with psycopg.connect(make_conninfo(url, dbname="postgres"), autocommit=True, connect_timeout=5) as admin:
            if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
                admin.execute(f'CREATE DATABASE "{name}"')
        with psycopg.connect(url, connect_timeout=5) as conn:
            conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))  # idempotent (IF NOT EXISTS)
    except psycopg.OperationalError as exc:
        pytest.skip(f"test database not reachable ({exc.__class__.__name__}); is `docker compose up -d` running?")
    return url


@pytest.fixture
def db_conn(test_db_url):
    """Connection to the test DB; all changes are rolled back after the test."""
    with psycopg.connect(test_db_url) as conn:
        register_vector(conn)
        try:
            yield conn
        finally:
            conn.rollback()


@pytest.fixture
def clean_db(test_db_url):
    """For code under test that commits: start and end with empty tables."""
    _require_test_db(test_db_url)

    def truncate():
        with psycopg.connect(test_db_url) as conn:
            conn.execute("TRUNCATE recordings, chunks RESTART IDENTITY CASCADE")

    truncate()
    yield test_db_url
    truncate()
