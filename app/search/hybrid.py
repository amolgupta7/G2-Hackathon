"""C6: hybrid search = keyword (tsvector, ts_rank_cd) + vector (exact cosine), fused with RRF (L8, L10-L12, L31)."""
import argparse
import time
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import EMBED_MODEL
from app.db.connection import connect
from app.log import get_logger
from app.pipeline.embed import embed

RRF_K = 60
log = get_logger("app.search.hybrid")

FILTERS = "(%(speaker)s::text IS NULL OR speaker = %(speaker)s) AND (%(rec)s::text IS NULL OR recording_id = %(rec)s)"

KEYWORD_SQL_TEMPLATE = """
SELECT id FROM chunks, (SELECT {tsquery} AS query) AS tq
WHERE tsv @@ tq.query AND {filters}
ORDER BY ts_rank_cd(tsv, tq.query) DESC, id
LIMIT %(k)s
"""
# "and": every term must match (websearch syntax). "or": same parse, but any term may match (phrases stay phrases).
KEYWORD_SQL = {
    "and": KEYWORD_SQL_TEMPLATE.format(
        tsquery="websearch_to_tsquery('english', %(q)s)", filters=FILTERS),
    "or": KEYWORD_SQL_TEMPLATE.format(
        tsquery="replace(websearch_to_tsquery('english', %(q)s)::text, ' & ', ' | ')::tsquery", filters=FILTERS),
}

VECTOR_SQL = f"""
SELECT id FROM chunks
WHERE {FILTERS}
ORDER BY embedding <=> %(qvec)s, id
LIMIT %(k)s
"""

RESULT_SQL = """
SELECT c.id, c.recording_id, c.chunk_index, c.speaker, c.start_s, c.end_s, c.text,
       p.speaker, p.text, n.speaker, n.text
FROM chunks c
LEFT JOIN chunks p ON p.recording_id = c.recording_id AND p.chunk_index = c.chunk_index - 1
LEFT JOIN chunks n ON n.recording_id = c.recording_id AND n.chunk_index = c.chunk_index + 1
WHERE c.id = ANY(%(ids)s)
"""


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def rrf(ranked_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


MODES = ("hybrid", "keyword", "vector")


def search(query: str, speaker: str | None = None, recording: str | None = None,
           top_k: int = 50, n: int = 10, conn=None, mode: str = "hybrid", keyword_op: str = "and") -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if keyword_op not in KEYWORD_SQL:
        raise ValueError(f"keyword_op must be one of {tuple(KEYWORD_SQL)}")
    # Speaker labels A/B are assigned per recording, so "speaker A" only identifies a person within one recording.
    if speaker and not recording:
        raise ValueError("speaker filter requires a recording filter (speaker labels are per recording)")
    params = {"q": query, "speaker": speaker, "rec": recording, "k": top_k}
    t0 = time.perf_counter()
    params["qvec"] = embed(_model(), [query])[0]
    t_embed = time.perf_counter()

    own_conn = conn is None
    conn = conn or connect()
    try:
        keyword_ids = [r[0] for r in conn.execute(KEYWORD_SQL[keyword_op], params).fetchall()]
        t_kw = time.perf_counter()
        vector_ids = [r[0] for r in conn.execute(VECTOR_SQL, params).fetchall()]
        t_vec = time.perf_counter()

        arms = {"hybrid": [keyword_ids, vector_ids], "keyword": [keyword_ids], "vector": [vector_ids]}[mode]
        scores = rrf(arms)
        top = sorted(scores, key=lambda i: (-scores[i], i))[:n]
        rows = {r[0]: r for r in conn.execute(RESULT_SQL, {"ids": top}).fetchall()}
    except Exception:
        log.exception("search failed: q=%r speaker=%s recording=%s", query, speaker, recording)
        raise
    finally:
        if own_conn:
            conn.close()

    kw_rank = {i: r for r, i in enumerate(keyword_ids, start=1)}
    vec_rank = {i: r for r, i in enumerate(vector_ids, start=1)}
    results = []
    for doc_id in top:
        _, rec, idx, spk, start, end, text, p_spk, p_text, n_spk, n_text = rows[doc_id]
        results.append({
            "chunk_id": doc_id, "recording_id": rec, "chunk_index": idx, "speaker": spk,
            "start_s": start, "end_s": end, "text": text, "score": round(scores[doc_id], 5),
            "keyword_rank": kw_rank.get(doc_id), "vector_rank": vec_rank.get(doc_id),
            "prev": {"speaker": p_spk, "text": p_text} if p_text else None,
            "next": {"speaker": n_spk, "text": n_text} if n_text else None,
        })

    log.info(
        "mode=%s keyword_op=%s q=%r speaker=%s rec=%s | keyword_hits=%d vector_hits=%d -> %d results | "
        "embed=%.0fms keyword=%.0fms vector=%.0fms total=%.0fms",
        mode, keyword_op, query, speaker, recording, len(keyword_ids), len(vector_ids), len(results),
        (t_embed - t0) * 1e3, (t_kw - t_embed) * 1e3, (t_vec - t_kw) * 1e3, (time.perf_counter() - t0) * 1e3,
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--speaker", choices=["A", "B"])
    parser.add_argument("--recording")
    parser.add_argument("-n", type=int, default=5)
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--keyword-op", choices=tuple(KEYWORD_SQL), default="and")
    args = parser.parse_args()
    if args.speaker and not args.recording:
        parser.error("--speaker requires --recording (speaker labels are per recording)")

    results = search(args.query, args.speaker, args.recording, n=args.n, mode=args.mode, keyword_op=args.keyword_op)
    for i, r in enumerate(results, start=1):
        print(f"{i}. [{r['recording_id']} #{r['chunk_index']} {r['speaker']} "
              f"{r['start_s']:.1f}-{r['end_s']:.1f}s] score={r['score']} "
              f"kw={r['keyword_rank']} vec={r['vector_rank']}\n   {r['text'][:160]}")


if __name__ == "__main__":
    main()
