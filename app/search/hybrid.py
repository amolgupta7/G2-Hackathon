"""C6: hybrid search = keyword (tsvector, ts_rank_cd) + vector (exact cosine), fused with RRF (L8, L10-L12, L31)."""
import argparse
import re
import time
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import (EMBED_MODEL, RERANK, RERANK_MODEL, RERANK_TOP, RRF_K, RRF_KEYWORD_WEIGHT,
                        TORCH_NUM_THREADS)
from app.db.connection import connect
from app.log import get_logger
from app.pipeline.embed import embed

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

# Vectors are unit-length, so cosine distance <=> = 1 - cosine similarity; hits below the similarity floor never enter fusion.
MIN_VECTOR_SIM = 0.3
# Queries without letters (e.g. "1234 5678") embed close to any number-heavy turn, so they get a stricter floor.
MIN_VECTOR_SIM_NO_LETTERS = 0.45
VECTOR_SQL = f"""
SELECT id FROM chunks
WHERE {FILTERS} AND (embedding <=> %(qvec)s) <= %(max_dist)s
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


# websearch syntax exclusions: -word or -"a phrase", at the start or after whitespace (not hyphenated words).
EXCLUDE_RE = re.compile(r'(?<!\S)-(?:"([^"]+)"|([^\s"]+))')


def excluded_terms(query: str) -> list[str]:
    return [(phrase or word).strip().lower() for phrase, word in EXCLUDE_RE.findall(query) if (phrase or word).strip()]


def positive_text(query: str) -> str:
    """Query without its -exclusions: what the embedder should see (excluded words would pull the vector arm toward them)."""
    return " ".join(EXCLUDE_RE.sub(" ", query).split())


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    torch.set_num_threads(TORCH_NUM_THREADS)
    log.info("query embedder: %s, torch threads=%d (TORCH_NUM_THREADS)", EMBED_MODEL, torch.get_num_threads())
    return SentenceTransformer(EMBED_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    log.info("reranker: %s (top %d fused candidates)", RERANK_MODEL, RERANK_TOP)
    return CrossEncoder(RERANK_MODEL, device="cpu")


def rrf(ranked_lists: list[list[int]], k: int = RRF_K, weights: list[float] | None = None) -> dict[int, float]:
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[int, float] = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return scores


MODES = ("hybrid", "keyword", "vector")


def search(query: str, speaker: str | None = None, recording: str | None = None,
           top_k: int = 50, n: int = 10, conn=None, mode: str = "hybrid", keyword_op: str = "and",
           min_vector_sim: float = MIN_VECTOR_SIM, rerank: bool = RERANK, rrf_k: int = RRF_K,
           keyword_weight: float = RRF_KEYWORD_WEIGHT) -> list[dict]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if keyword_op not in KEYWORD_SQL:
        raise ValueError(f"keyword_op must be one of {tuple(KEYWORD_SQL)}")
    # Speaker labels A/B are assigned per recording, so "speaker A" only identifies a person within one recording.
    if speaker and not recording:
        raise ValueError("speaker filter requires a recording filter (speaker labels are per recording)")
    embed_q = positive_text(query)
    if not embed_q:
        # Only exclusions ("-invoice"): websearch_to_tsquery would give "!invoice", matching nearly every chunk.
        log.info("mode=%s q=%r speaker=%s rec=%s | no positive terms (only exclusions) -> 0 results",
                 mode, query, speaker, recording)
        return []
    # The similarity floor only guards unscoped searches (gibberish/off-topic); inside a user-chosen recording,
    # weakly-worded but correct turns (e.g. role-word questions, sim 0.12-0.27) must still be returned.
    no_letters = not re.search(r"[^\W\d_]", embed_q)
    threshold = None if recording else (max(min_vector_sim, MIN_VECTOR_SIM_NO_LETTERS) if no_letters else min_vector_sim)
    max_dist = 2.0 if threshold is None else 1.0 - threshold  # cosine distance range is [0, 2]
    params = {"q": query, "speaker": speaker, "rec": recording, "k": top_k, "max_dist": max_dist}
    t0 = time.perf_counter()
    params["qvec"] = embed(_model(), [embed_q])[0]
    t_embed = time.perf_counter()

    own_conn = conn is None
    conn = conn or connect()
    try:
        keyword_ids = [r[0] for r in conn.execute(KEYWORD_SQL[keyword_op], params).fetchall()]
        t_kw = time.perf_counter()
        vector_ids = [r[0] for r in conn.execute(VECTOR_SQL, params).fetchall()]
        t_vec = time.perf_counter()

        arms, weights = {"hybrid": ([keyword_ids, vector_ids], [keyword_weight, 1.0]),
                         "keyword": ([keyword_ids], [1.0]), "vector": ([vector_ids], [1.0])}[mode]
        scores = rrf(arms, k=rrf_k, weights=weights)
        ranked = sorted(scores, key=lambda i: (-scores[i], i))
        excluded = excluded_terms(query)
        dropped = 0
        if excluded and ranked:
            # The keyword arm already honors -terms; the vector arm can't, so enforce them on the fused list.
            texts = dict(conn.execute("SELECT id, lower(text) FROM chunks WHERE id = ANY(%s)", (ranked,)).fetchall())
            # Word-boundary prefix: "-invoice" drops "invoices", but "-art" doesn't drop "start".
            patterns = [re.compile(rf"\b{re.escape(term)}") for term in excluded]
            kept = [i for i in ranked if not any(p.search(texts[i]) for p in patterns)]
            dropped, ranked = len(ranked) - len(kept), kept
        t_rr = time.perf_counter()
        reranked = rerank and len(ranked) > 1
        if reranked:
            # Cross-encoder reads query + turn together, so it can fix fusion order (e.g. paraphrases, exact matches).
            head = ranked[:RERANK_TOP]
            texts_raw = dict(conn.execute("SELECT id, text FROM chunks WHERE id = ANY(%s)", (head,)).fetchall())
            ce = _reranker().predict([(embed_q, texts_raw[i]) for i in head])
            ranked = [i for _, i in sorted(zip(ce, head), key=lambda p: -p[0])] + ranked[RERANK_TOP:]
        t_rr_done = time.perf_counter()
        top = ranked[:n]
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
        "mode=%s keyword_op=%s q=%r embedded=%r speaker=%s rec=%s | keyword_hits=%d vector_hits(threshold=%s)=%d "
        "excluded=%s dropped=%d rrf(k=%d,kw_w=%.2f) rerank=%s -> %d results | "
        "embed=%.0fms keyword=%.0fms vector=%.0fms rerank=%.0fms total=%.0fms",
        mode, keyword_op, query, embed_q, speaker, recording, len(keyword_ids),
        "off:recording-scoped" if threshold is None else f"{threshold:.2f}{'(no-letters)' if no_letters else ''}",
        len(vector_ids), excluded, dropped, rrf_k, keyword_weight,
        f"on({min(len(ranked), RERANK_TOP)})" if reranked else "off", len(results),
        (t_embed - t0) * 1e3, (t_kw - t_embed) * 1e3, (t_vec - t_kw) * 1e3, (t_rr_done - t_rr) * 1e3,
        (time.perf_counter() - t0) * 1e3,
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
