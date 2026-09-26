"""C5: load processed chunks + embeddings into Postgres, idempotent by audio content hash (L23, L24)."""
import argparse
import hashlib
import json
import sys
import traceback

import numpy as np

from app.config import ASR_MODEL, AUDIO_DIR, EMBED_MODEL, INGEST_STALE_MINUTES, PROCESSED_DIR, ROOT
from app.db.connection import connect, embedding_dim
from app.log import get_logger

ERROR_MAX_CHARS = 2000
log = get_logger("app.pipeline.ingest")


class RecordingIdConflict(Exception):
    """Same recording id (file name) already stored with different audio content."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(recording: str, db_dim: int) -> tuple[float, list[tuple]]:
    asr = json.loads((PROCESSED_DIR / "asr" / f"{recording}.{ASR_MODEL}.json").read_text(encoding="utf-8"))
    chunks = json.loads((PROCESSED_DIR / "chunks" / f"{recording}.json").read_text(encoding="utf-8"))["chunks"]
    emb = np.load(PROCESSED_DIR / "embeddings" / EMBED_MODEL / f"{recording}.npz")
    if emb["chunk_index"].tolist() != [c["chunk_index"] for c in chunks]:
        raise ValueError(f"embeddings out of sync with chunks for {recording}; re-run embed")
    if len(emb["vectors"]) and emb["vectors"].shape[1] != db_dim:
        raise ValueError(f"{recording}: embeddings are {emb['vectors'].shape[1]}-d ({EMBED_MODEL}) but "
                         f"chunks.embedding is vector({db_dim}); update schema.sql and migrate the column")
    rows = [
        (recording, c["chunk_index"], c["speaker"], c["start"], c["end"], c["text"], v)
        for c, v in zip(chunks, emb["vectors"])
    ]
    return asr["audio_duration_s"], rows


def recover_stale(conn, minutes: int) -> list[str]:
    """K1: a recording left in 'processing' (worker crashed) past the timeout becomes 'failed', so it gets retried."""
    stale = conn.execute(
        "UPDATE recordings SET status = 'failed', error = %s, updated_at = now() "
        "WHERE status = 'processing' AND COALESCE(started_at, updated_at) < now() - make_interval(mins => %s) "
        "RETURNING id",
        (f"stale: in 'processing' for more than {minutes} min; marked failed at ingest startup", minutes),
    ).fetchall()
    conn.commit()
    return [r[0] for r in stale]


def ingest(conn, audio_path, db_dim: int) -> str:
    recording = audio_path.stem
    sha = sha256_file(audio_path)
    try:
        # K2: SKIP LOCKED means a row another worker is claiming right now is invisible here instead of blocking.
        existing = conn.execute(
            "SELECT id, status FROM recordings WHERE content_sha256 = %s FOR UPDATE SKIP LOCKED", (sha,)
        ).fetchone()
        if existing and existing[1] in ("completed", "processing"):
            conn.commit()
            log.info("claim %s: skip, same content already %s as %s", recording, existing[1], existing[0])
            return "skipped"

        duration, rows = load_rows(recording, db_dim)
        if existing:
            conn.execute(
                "UPDATE recordings SET status = 'processing', error = NULL, started_at = now(), updated_at = now() "
                "WHERE id = %s",
                (existing[0],),
            )
            log.info("claim %s: retry (previous run failed)", recording)
        else:
            # No conflict target: covers both unique keys (content_sha256 and the id primary key).
            inserted = conn.execute(
                "INSERT INTO recordings (id, file_path, content_sha256, duration_s, status, started_at) "
                "VALUES (%s, %s, %s, %s, 'processing', now()) ON CONFLICT DO NOTHING RETURNING id",
                (recording, audio_path.relative_to(ROOT).as_posix(), sha, duration),
            ).fetchone()
            if not inserted:
                same_id = conn.execute("SELECT content_sha256 FROM recordings WHERE id = %s", (recording,)).fetchone()
                conn.commit()
                if same_id and same_id[0] != sha:
                    log.error("claim %s: id conflict, a recording with this id already exists with different content "
                              "(stored sha %s…, new sha %s…); existing recording left unchanged. Rename the file or "
                              "delete the old recording first", recording, same_id[0][:12], sha[:12])
                    raise RecordingIdConflict(recording)
                log.info("claim %s: skip, claimed by another worker", recording)
                return "skipped"
            log.info("claim %s: new", recording)
        conn.commit()
    except RecordingIdConflict:
        raise  # already logged cleanly above; no traceback noise
    except Exception:
        conn.rollback()
        log.exception("could not start ingest for %s", recording)
        raise

    try:
        conn.execute("DELETE FROM chunks WHERE recording_id = %s", (recording,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (recording_id, chunk_index, speaker, start_s, end_s, text, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                rows,
            )
        conn.execute(
            "UPDATE recordings SET status = 'completed', error = NULL, updated_at = now() WHERE id = %s",
            (recording,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        conn.execute(
            "UPDATE recordings SET status = 'failed', error = %s, updated_at = now() WHERE id = %s",
            (traceback.format_exc()[:ERROR_MAX_CHARS], recording),
        )
        conn.commit()
        log.exception("ingest failed for %s", recording)
        raise

    log.info("loaded %s: %d chunks, %.1fs audio", recording, len(rows), duration)
    return "loaded"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="audio file names in data/audio (default: all .wav)")
    args = parser.parse_args()

    files = [AUDIO_DIR / f for f in args.files] or sorted(AUDIO_DIR.glob("*.wav"))
    log.info("ingest run: files=%d embed_model=%s stale_timeout=%dmin", len(files), EMBED_MODEL, INGEST_STALE_MINUTES)
    with connect() as conn:
        db_dim = embedding_dim(conn)
        conn.commit()
        log.info("chunks.embedding is vector(%d); each recording's embeddings must match", db_dim)
        stale = recover_stale(conn, INGEST_STALE_MINUTES)
        log.info("stale recovery: %s", f"marked failed for retry: {stale}" if stale else "none")
        results, failed = [], []
        for f in files:
            try:
                results.append(ingest(conn, f, db_dim))
            except Exception:
                # ingest() already logged the traceback and marked the recording failed; keep going with the rest.
                conn.rollback()
                failed.append(f.stem)
                log.warning("continuing after failure: %s", f.stem)
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    log.info("done: loaded=%d skipped=%d failed=%d%s; chunks in DB=%d",
             results.count("loaded"), results.count("skipped"), len(failed),
             f" {failed}" if failed else "", total)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
