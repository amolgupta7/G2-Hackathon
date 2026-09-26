"""C5: load processed chunks + embeddings into Postgres, idempotent by audio content hash (L23, L24)."""
import argparse
import hashlib
import json
import traceback

import numpy as np

from app.config import ASR_MODEL, AUDIO_DIR, EMBED_MODEL, PROCESSED_DIR, ROOT
from app.db.connection import connect
from app.log import get_logger

ERROR_MAX_CHARS = 2000
log = get_logger("app.pipeline.ingest")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(recording: str) -> tuple[float, list[tuple]]:
    asr = json.loads((PROCESSED_DIR / "asr" / f"{recording}.{ASR_MODEL}.json").read_text(encoding="utf-8"))
    chunks = json.loads((PROCESSED_DIR / "chunks" / f"{recording}.json").read_text(encoding="utf-8"))["chunks"]
    emb = np.load(PROCESSED_DIR / "embeddings" / EMBED_MODEL / f"{recording}.npz")
    if emb["chunk_index"].tolist() != [c["chunk_index"] for c in chunks]:
        raise ValueError(f"embeddings out of sync with chunks for {recording}; re-run embed")
    rows = [
        (recording, c["chunk_index"], c["speaker"], c["start"], c["end"], c["text"], v)
        for c, v in zip(chunks, emb["vectors"])
    ]
    return asr["audio_duration_s"], rows


def ingest(conn, audio_path) -> str:
    recording = audio_path.stem
    sha = sha256_file(audio_path)
    existing = conn.execute(
        "SELECT id, status FROM recordings WHERE content_sha256 = %s", (sha,)
    ).fetchone()

    if existing and existing[1] in ("completed", "processing"):
        log.info("skip %s: same content already %s as %s", recording, existing[1], existing[0])
        return "skipped"

    try:
        duration, rows = load_rows(recording)
        if existing:
            log.info("retry %s (previous run failed)", recording)
            conn.execute(
                "UPDATE recordings SET status = 'processing', error = NULL, updated_at = now() WHERE id = %s",
                (existing[0],),
            )
        else:
            conn.execute(
                "INSERT INTO recordings (id, file_path, content_sha256, duration_s, status) "
                "VALUES (%s, %s, %s, %s, 'processing')",
                (recording, audio_path.relative_to(ROOT).as_posix(), sha, duration),
            )
        conn.commit()
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
    log.info("ingest run: files=%d embed_model=%s", len(files), EMBED_MODEL)
    with connect() as conn:
        results = [ingest(conn, f) for f in files]
        total = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    log.info("done: loaded=%d skipped=%d; chunks in DB=%d",
             results.count("loaded"), results.count("skipped"), total)


if __name__ == "__main__":
    main()
