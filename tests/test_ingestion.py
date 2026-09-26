"""Ingestion tests against the Postgres test DB (see conftest: `<db>_test`, tables truncated per test).

The real ingest()/recover_stale() SQL runs unchanged; only file inputs are stubbed:
- load_rows -> 3 synthetic chunks with 384-d vectors (no pipeline outputs / .npz needed),
- the "audio" is a few bytes in tmp_path (its SHA-256 is the idempotency key), ingest.ROOT -> tmp_path.
"""
import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector

from app.db.connection import embedding_dim
from app.pipeline import ingest as ing

pytestmark = pytest.mark.db

N_CHUNKS = 3


@pytest.fixture
def conn(clean_db):
    with psycopg.connect(clean_db) as c:
        register_vector(c)
        yield c


@pytest.fixture
def dim(conn):
    d = embedding_dim(conn)
    conn.commit()
    return d


@pytest.fixture(autouse=True)
def stub_inputs(monkeypatch, tmp_path):
    def fake_load_rows(recording, db_dim):
        vec = np.full(db_dim, 0.05, np.float32)
        return 12.0, [(recording, i, "A", float(i), float(i + 1), f"turn {i}", vec) for i in range(N_CHUNKS)]

    monkeypatch.setattr(ing, "load_rows", fake_load_rows)
    monkeypatch.setattr(ing, "ROOT", tmp_path)


def wav(tmp_path, name, content: bytes):
    path = tmp_path / f"{name}.wav"
    path.write_bytes(content)
    return path


def counts(conn):
    r = conn.execute("SELECT (SELECT count(*) FROM recordings), (SELECT count(*) FROM chunks)").fetchone()
    conn.commit()
    return r


def row(conn, rec_id):
    r = conn.execute("SELECT status, content_sha256, error FROM recordings WHERE id = %s", (rec_id,)).fetchone()
    conn.commit()
    return r


# ---------------------------------------------------------------- idempotency (content hash)

def test_rerun_on_same_file_creates_no_duplicates(conn, dim, tmp_path):
    audio = wav(tmp_path, "rec_same", b"audio-v1")

    assert ing.ingest(conn, audio, dim) == "loaded"
    assert ing.ingest(conn, audio, dim) == "skipped"
    assert ing.ingest(conn, audio, dim) == "skipped"

    assert counts(conn) == (1, N_CHUNKS)
    assert row(conn, "rec_same")[0] == "completed"


def test_same_content_under_another_name_is_skipped(conn, dim, tmp_path):
    ing.ingest(conn, wav(tmp_path, "rec_original", b"identical-bytes"), dim)

    assert ing.ingest(conn, wav(tmp_path, "rec_renamed_copy", b"identical-bytes"), dim) == "skipped"
    assert counts(conn) == (1, N_CHUNKS)  # content_sha256 is the key, not the file name


# ---------------------------------------------------------------- K1: stale processing recovery

def test_stale_processing_row_is_recovered_and_reingested(conn, dim, tmp_path):
    audio = wav(tmp_path, "rec_stale", b"crashed-mid-ingest")
    conn.execute(
        "INSERT INTO recordings (id, file_path, content_sha256, duration_s, status, started_at, updated_at) "
        "VALUES ('rec_stale', 'x', %s, 12.0, 'processing', now() - interval '2 hours', now() - interval '2 hours')",
        (ing.sha256_file(audio),),
    )
    conn.commit()
    assert ing.ingest(conn, audio, dim) == "skipped"  # before recovery: 'processing' is skipped (K1 bug)

    assert ing.recover_stale(conn, minutes=30) == ["rec_stale"]
    status, _, error = row(conn, "rec_stale")
    assert status == "failed" and "stale" in error

    assert ing.ingest(conn, audio, dim) == "loaded"  # failed -> retried
    assert row(conn, "rec_stale")[0] == "completed" and counts(conn) == (1, N_CHUNKS)


def test_recent_processing_row_is_not_recovered(conn, tmp_path):
    conn.execute(
        "INSERT INTO recordings (id, file_path, content_sha256, duration_s, status, started_at) "
        "VALUES ('rec_busy', 'x', 'sha-busy', 1.0, 'processing', now() - interval '5 minutes')"
    )
    conn.commit()

    assert ing.recover_stale(conn, minutes=30) == []
    assert row(conn, "rec_busy")[0] == "processing"


# ---------------------------------------------------------------- D4: same id, different content

def test_reupload_with_different_content_fails_cleanly(conn, dim, tmp_path, caplog):
    audio = wav(tmp_path, "rec_conflict", b"first-version")
    ing.ingest(conn, audio, dim)
    original_sha = row(conn, "rec_conflict")[1]

    audio.write_bytes(b"different-second-version")  # same file name -> same recording id, new hash
    with pytest.raises(ing.RecordingIdConflict):  # not psycopg.errors.UniqueViolation
        ing.ingest(conn, audio, dim)

    assert "id conflict" in caplog.text and "existing recording left unchanged" in caplog.text
    status, sha, _ = row(conn, "rec_conflict")
    assert status == "completed" and sha == original_sha  # the stored recording is untouched
    assert counts(conn) == (1, N_CHUNKS)
