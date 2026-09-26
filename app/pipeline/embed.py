"""C4: embed chunk text with a MiniLM model -> data/processed/embeddings/<model>/<recording>.npz (L7, L20)."""
import argparse
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import PROCESSED_DIR
from app.log import get_logger

CHUNK_DIR = PROCESSED_DIR / "chunks"
EMB_DIR = PROCESSED_DIR / "embeddings"
MODELS = ["all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1"]
log = get_logger("app.pipeline.embed")


def embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    # Unit-length vectors, so cosine distance in pgvector (<=>) matches the models' training objective.
    return model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("recordings", nargs="*", help="recording stems (default: all chunked)")
    args = parser.parse_args()

    recordings = args.recordings or sorted(p.stem for p in CHUNK_DIR.glob("*.json"))
    out_dir = EMB_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("embed run: model=%s recordings=%d", args.model, len(recordings))
    model = SentenceTransformer(args.model, device="cpu")

    for rec in recordings:
        try:
            chunks = json.loads((CHUNK_DIR / f"{rec}.json").read_text(encoding="utf-8"))["chunks"]
            t0 = time.perf_counter()
            vectors = embed(model, [c["text"] for c in chunks])
            np.savez(
                out_dir / f"{rec}.npz",
                vectors=vectors,
                chunk_index=np.array([c["chunk_index"] for c in chunks]),
            )
        except Exception:
            log.exception("embedding failed for %s (model=%s)", rec, args.model)
            raise
        log.info("%s: %d chunks -> %s in %.2fs", rec, len(chunks), vectors.shape, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
