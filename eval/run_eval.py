"""C9: evaluate retrieval configs on dataset/queries.json -> eval/results.json.

Strict metrics count a result only if it is itself labeled relevant. Secondary "+nb" metrics also count a result
whose previous/next turn is relevant (e.g. a question turn whose answer is the next turn, shown as context).
"""
import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from app.config import RERANK, ROOT, RRF_K, RRF_KEYWORD_WEIGHT
from app.db.connection import connect
from app.log import get_logger
from app.search.hybrid import MIN_VECTOR_SIM, search

QUERIES = ROOT / "dataset" / "queries.json"
NEGATIVES = ROOT / "dataset" / "negative_queries.json"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-vector-sim", type=float, default=MIN_VECTOR_SIM, help="-1 disables the threshold")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--rerank", choices=["on", "off"], default="on" if RERANK else "off")
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--kw-weight", type=float, default=RRF_KEYWORD_WEIGHT)
    args = parser.parse_args()
    min_sim, out = args.min_vector_sim, Path(args.out)
    ranking = {"rerank": args.rerank == "on", "rrf_k": args.rrf_k, "keyword_weight": args.kw_weight}

    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    negatives = json.loads(NEGATIVES.read_text(encoding="utf-8"))
    log.info("eval: %d queries + %d negatives x configs %s, top %d, min_vector_sim=%.2f, ranking=%s",
             len(queries), len(negatives), list(CONFIGS), N, min_sim, ranking)
    per_query = defaultdict(dict)
    negative_counts = defaultdict(dict)
    summary = {}

    with connect() as conn:
        search("warm up", conn=conn)
        for name, (mode, op) in CONFIGS.items():
            rows, latencies, zero_results = [], [], 0
            for q in queries:
                t0 = time.perf_counter()
                results = search(q["query"], speaker=q.get("speaker"), recording=q.get("recording"),
                                 n=N, conn=conn, mode=mode, keyword_op=op, min_vector_sim=min_sim, **ranking)
                latencies.append((time.perf_counter() - t0) * 1e3)
                zero_results += not results
                ranked = [(r["recording_id"], r["chunk_index"]) for r in results]
                relevant = {(r["recording_id"], r["chunk_index"]) for r in q["relevant"]}
                s = score(ranked, relevant) | {"type": q["type"]}
                rows.append(s)
                per_query[q["query_id"]][name] = s["first_rank"]

            # Negative queries have no relevant chunk anywhere: the correct answer is an empty result.
            for q in negatives:
                negative_counts[q["query_id"]][name] = len(
                    search(q["query"], n=N, conn=conn, mode=mode, keyword_op=op, min_vector_sim=min_sim, **ranking))
            neg_by_type = defaultdict(list)
            for q in negatives:
                neg_by_type[q["type"]].append(negative_counts[q["query_id"]][name] == 0)

            by_type = defaultdict(list)
            for r in rows:
                by_type[r["type"]].append(r)
            summary[name] = {
                "mode": mode, "keyword_op": op,
                "overall": mean_metrics(rows),
                "by_type": {t: mean_metrics(rs) | {"n": len(rs)} for t, rs in sorted(by_type.items())},
                "zero_result_rate": round(zero_results / len(queries), 3),
                "true_negative_rate": {
                    "all": round(statistics.mean(v for vs in neg_by_type.values() for v in vs), 3),
                    **{t: round(statistics.mean(vs), 3) for t, vs in sorted(neg_by_type.items())},
                },
                "latency_ms": {
                    "p50": round(statistics.median(latencies), 1),
                    "p95": round(sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1], 1),
                },
            }

    for name in CONFIGS:
        o = summary[name]["overall"]
        log.info("RESULT %-10s %s | zero_result_rate=%.3f | p50=%sms p95=%sms", name,
                 " ".join(f"{m}={o[m]:.3f}" for m in METRICS), summary[name]["zero_result_rate"],
                 *summary[name]["latency_ms"].values())
        log.info("NEGATIVES %-10s true_negative_rate=%s counts=%s", name, summary[name]["true_negative_rate"],
                 {qid: c[name] for qid, c in negative_counts.items()})

    misses = {}
    for name in ("hybrid", "hybrid_or"):
        misses[name] = [
            {"query_id": q["query_id"], "type": q["type"], "query": q["query"], "first_rank": per_query[q["query_id"]]}
            for q in queries
            if per_query[q["query_id"]][name] is None or per_query[q["query_id"]][name] > 5
        ]
        log.info("MISS@5 %s: %s", name, [m["query_id"] for m in misses[name]])

    out.write_text(json.dumps({"n": N, "queries": len(queries), "negatives": len(negatives),
                               "min_vector_sim": min_sim, "ranking": ranking, "configs": CONFIGS, "summary": summary,
                               "misses_at_5": misses, "first_rank_per_query": per_query,
                               "negative_result_counts": negative_counts}, indent=2),
                   encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
