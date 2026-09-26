"""C9: evaluate retrieval configs on dataset/queries.json -> eval/results.json.

Strict metrics count a result only if it is itself labeled relevant. Secondary "+nb" metrics also count a result
whose previous/next turn is relevant (e.g. a question turn whose answer is the next turn, shown as context).
"""
import json
import math
import statistics
import time
from collections import defaultdict

from app.config import ROOT
from app.db.connection import connect
from app.log import get_logger
from app.search.hybrid import search

QUERIES = ROOT / "dataset" / "queries.json"
OUT = ROOT / "eval" / "results.json"
N = 10
# name -> (mode, keyword_op); vector ignores keyword_op
CONFIGS = {
    "hybrid": ("hybrid", "and"),
    "hybrid_or": ("hybrid", "or"),
    "keyword": ("keyword", "and"),
    "keyword_or": ("keyword", "or"),
    "vector": ("vector", "and"),
}
METRICS = ["hit@1", "hit@5", "recall@5", "recall@10", "mrr@10", "ndcg@10", "hit@1+nb", "hit@5+nb", "mrr@10+nb"]
log = get_logger("app.eval.run_eval")


def first_hit(flags: list[bool]) -> int | None:
    return next((i for i, h in enumerate(flags, start=1) if h), None)


def score(ranked: list[tuple], relevant: set[tuple]) -> dict:
    hits = [doc in relevant for doc in ranked[:N]]
    nb_hits = [
        doc in relevant or (doc[0], doc[1] - 1) in relevant or (doc[0], doc[1] + 1) in relevant
        for doc in ranked[:N]
    ]
    first, first_nb = first_hit(hits), first_hit(nb_hits)
    dcg = sum(1 / math.log2(i + 1) for i, h in enumerate(hits, start=1) if h)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), N) + 1))
    return {
        "hit@1": float(first == 1),
        "hit@5": float(first is not None and first <= 5),
        "recall@5": sum(hits[:5]) / len(relevant),
        "recall@10": sum(hits) / len(relevant),
        "mrr@10": 1 / first if first else 0.0,
        "ndcg@10": dcg / idcg,
        "hit@1+nb": float(first_nb == 1),
        "hit@5+nb": float(first_nb is not None and first_nb <= 5),
        "mrr@10+nb": 1 / first_nb if first_nb else 0.0,
        "first_rank": first,
    }


def mean_metrics(rows: list[dict]) -> dict:
    return {m: round(statistics.mean(r[m] for r in rows), 3) for m in METRICS}


def main():
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    log.info("eval: %d queries x configs %s, top %d", len(queries), list(CONFIGS), N)
    per_query = defaultdict(dict)
    summary = {}

    with connect() as conn:
        search("warm up", conn=conn)
        for name, (mode, op) in CONFIGS.items():
            rows, latencies = [], []
            for q in queries:
                t0 = time.perf_counter()
                results = search(q["query"], speaker=q.get("speaker"), recording=q.get("recording"),
                                 n=N, conn=conn, mode=mode, keyword_op=op)
                latencies.append((time.perf_counter() - t0) * 1e3)
                ranked = [(r["recording_id"], r["chunk_index"]) for r in results]
                relevant = {(r["recording_id"], r["chunk_index"]) for r in q["relevant"]}
                s = score(ranked, relevant) | {"type": q["type"]}
                rows.append(s)
                per_query[q["query_id"]][name] = s["first_rank"]

            by_type = defaultdict(list)
            for r in rows:
                by_type[r["type"]].append(r)
            summary[name] = {
                "mode": mode, "keyword_op": op,
                "overall": mean_metrics(rows),
                "by_type": {t: mean_metrics(rs) | {"n": len(rs)} for t, rs in sorted(by_type.items())},
                "latency_ms": {
                    "p50": round(statistics.median(latencies), 1),
                    "p95": round(sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1], 1),
                },
            }

    for name in CONFIGS:
        o = summary[name]["overall"]
        log.info("RESULT %-10s %s | p50=%sms p95=%sms", name,
                 " ".join(f"{m}={o[m]:.3f}" for m in METRICS), *summary[name]["latency_ms"].values())

    misses = {}
    for name in ("hybrid", "hybrid_or"):
        misses[name] = [
            {"query_id": q["query_id"], "type": q["type"], "query": q["query"], "first_rank": per_query[q["query_id"]]}
            for q in queries
            if per_query[q["query_id"]][name] is None or per_query[q["query_id"]][name] > 5
        ]
        log.info("MISS@5 %s: %s", name, [m["query_id"] for m in misses[name]])

    OUT.write_text(json.dumps({"n": N, "queries": len(queries), "configs": CONFIGS, "summary": summary,
                               "misses_at_5": misses, "first_rank_per_query": per_query}, indent=2),
                   encoding="utf-8")
    log.info("wrote %s", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
