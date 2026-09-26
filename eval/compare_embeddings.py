"""Compare embedding models on one recording: Hit@1, Hit@3, MRR over labeled queries (L26)."""
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.log import get_logger
from app.pipeline.embed import EMB_DIR, MODELS, embed

log = get_logger("app.eval.compare_embeddings")


def score(model_name: str, recording: str, queries: list[dict]) -> dict:
    data = np.load(EMB_DIR / model_name / f"{recording}.npz")
    vectors, chunk_index = data["vectors"], data["chunk_index"]
    q_vecs = embed(SentenceTransformer(model_name, device="cpu"), [q["q"] for q in queries])
    ranks = []
    for q, qv in zip(queries, q_vecs):
        order = chunk_index[np.argsort(-(vectors @ qv))]
        rank = int(np.where(order == q["relevant"])[0][0]) + 1
        ranks.append(rank)
        log.info("[%s] rank=%d top3=%s  q=%r", model_name, rank, order[:3].tolist(), q["q"])
    ranks = np.array(ranks)
    return {
        "hit@1": float(np.mean(ranks == 1)),
        "hit@3": float(np.mean(ranks <= 3)),
        "mrr": float(np.mean(1.0 / ranks)),
        "ranks": ranks.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=str(Path(__file__).with_name("queries_rec01.json")))
    args = parser.parse_args()

    spec = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    log.info("comparing %s on %s with %d queries", MODELS, spec["recording"], len(spec["queries"]))
    for m in MODELS:
        r = score(m, spec["recording"], spec["queries"])
        log.info("RESULT %-28s hit@1=%.2f hit@3=%.2f MRR=%.3f ranks=%s", m, r["hit@1"], r["hit@3"], r["mrr"], r["ranks"])


if __name__ == "__main__":
    main()
