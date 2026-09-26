# Evaluation Results

Retrieval quality of the prototype, measured on a hand-labeled query set.
Run date: 2026-09-26 (run 2) · Code: [`eval/run_eval.py`](../eval/run_eval.py) · Raw output: [`eval/results.json`](../eval/results.json) · Queries: [`dataset/queries.json`](../dataset/queries.json)

Reproduce: `python -m eval.run_eval` (needs the DB running and ingested).

**Changes since run 1**
- **O25:** a speaker filter now **requires** a recording filter, because speaker labels `A`/`B` are assigned per recording. The 6 speaker queries carry their `recording`.
- **O26:** added **secondary "+nb" metrics**, where a result also counts if its previous/next turn is relevant. The strict metrics are unchanged.
- **O27:** A/B test of an **OR-mode keyword arm** (`hybrid_or`, `keyword_or`) against the default AND mode.

---

## 1. Setup

| | |
|---|---|
| **Corpus** | 6 recordings, 344 speaker-turn chunks |
| **Query set** | 30 queries, 5 per recording: 13 keyword · 11 paraphrase · 6 speaker-specific (speaker + recording filter) |
| **Labels** | Relevant chunks per query, labeled by reading the transcripts (not from search output), stored as `(recording_id, chunk_index)` pairs. 82 relevant pairs in total. |
| **Relevance rule (strict)** | A chunk is relevant if its text contains information that answers or matches the query. Pure question turns and passing mentions are excluded. |
| **Settings** | 50 candidates per arm, top 10 scored, RRF k = 60, CPU, model warm |

**Configurations**

| Name | Keyword arm | Vector arm | Fusion |
|---|---|---|---|
| **hybrid** (current default) | AND: `websearch_to_tsquery` (every term must match) | all-MiniLM-L6-v2, exact cosine | RRF |
| hybrid_or | OR: same parse, `&` → `\|` (any term may match; phrases stay phrases) | same | RRF |
| keyword | AND | — | — |
| keyword_or | OR | — | — |
| vector | — | same | — |

**Query types**
- **keyword:** a short term found in the transcript (`refund`, `r-8812`, `4,420`, `Marcus`). Some relevant chunks are on-topic without the literal word.
- **paraphrase:** a natural-language question worded differently from the transcript.
- **speaker:** what one speaker said inside one recording (speaker + recording filter).

## 2. Metrics

| Metric | Meaning |
|---|---|
| Hit@k | Share of queries with at least one relevant chunk in the top k |
| Recall@k | Share of a query's relevant chunks found in the top k, averaged over queries |
| MRR@10 | Mean of 1 / (rank of the first relevant result), 0 if none in the top 10 |
| nDCG@10 | Ranking quality with binary relevance; rewards relevant chunks near the top |
| **Hit@k+nb, MRR@10+nb** | *Secondary.* Same, but a result also counts if the turn **right before or after it** is relevant. The UI shows those neighbors, so this measures "the answer is on screen". |
| Latency | Wall time of `search()` per query, including query embedding (p50 / p95) |

## 3. Overall results

| Config | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Hit@1+nb | Hit@5+nb | MRR@10+nb | p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| **hybrid** | 0.600 | **0.867** | **0.653** | **0.733** | **0.724** | **0.665** | **0.733** | **0.933** | **0.815** | 23 / 29 ms |
| hybrid_or | **0.633** | 0.767 | 0.575 | 0.675 | 0.710 | 0.616 | **0.733** | 0.867 | 0.800 | 26 / 53 ms |
| keyword | 0.433 | 0.433 | 0.303 | 0.309 | 0.433 | 0.335 | 0.467 | 0.467 | 0.467 | 23 / 32 ms |
| keyword_or | **0.633** | 0.667 | 0.488 | 0.543 | 0.660 | 0.545 | 0.700 | 0.800 | 0.735 | 25 / 33 ms |
| vector | 0.600 | **0.867** | 0.614 | 0.722 | 0.716 | 0.644 | **0.733** | **0.933** | 0.807 | 22 / 29 ms |

**Run 1 → run 2, hybrid (effect of the O25 recording filter on the 6 speaker queries):** Hit@5 0.800 → 0.867 · Recall@10 0.689 → 0.733 · MRR@10 0.708 → 0.724 · nDCG@10 0.635 → 0.665.

## 4. Results by query type

| Type (n) | Config | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | Hit@5+nb |
|---|---|---|---|---|---|---|---|
| **keyword (13)** | hybrid | 0.846 | 1.000 | 0.821 | 0.923 | 0.821 | 1.000 |
| | hybrid_or | 0.846 | 1.000 | **0.846** | 0.923 | **0.829** | 1.000 |
| | keyword | **1.000** | 1.000 | 0.713 | **1.000** | 0.774 | 1.000 |
| | keyword_or | **1.000** | 1.000 | 0.818 | **1.000** | **0.841** | 1.000 |
| | vector | 0.846 | 1.000 | 0.795 | 0.904 | 0.774 | 1.000 |
| **paraphrase (11)** | hybrid | 0.364 | **0.727** | **0.606** | **0.543** | **0.490** | **0.909** |
| | hybrid_or | **0.455** | 0.455 | 0.417 | 0.481 | 0.326 | 0.727 |
| | keyword | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.091 |
| | keyword_or | 0.182 | 0.273 | 0.212 | 0.255 | 0.179 | 0.636 |
| | vector | 0.364 | **0.727** | **0.606** | **0.543** | **0.490** | **0.909** |
| **speaker (6)** | hybrid | 0.500 | **0.833** | **0.778** | 0.625 | 0.645 | **0.833** |
| | hybrid_or | 0.500 | **0.833** | **0.778** | 0.667 | **0.687** | **0.833** |
| | keyword | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| | keyword_or | **0.667** | 0.667 | 0.556 | **0.667** | 0.573 | 0.667 |
| | vector | 0.500 | **0.833** | **0.778** | 0.625 | 0.645 | **0.833** |

## 5. Findings

1. **Hybrid (AND) is the best overall configuration.** It leads or ties on Hit@5, Recall, MRR, nDCG and every +nb metric.
2. **The recording filter (O25) fixed the speaker queries.** Speaker Hit@5 went from 0.500 to 0.833: q10 improved from rank 9 to 4, and q20 from 7 to 2. Filtering `A`/`B` across recordings had been mixing different people.
3. **The neighbor-aware view (O26) is much higher than the strict one.** Hybrid Hit@5+nb is 0.933 vs 0.800–0.867 strict, and paraphrase Hit@5+nb is 0.909 vs 0.727. Many "misses" land on the question turn next to the answer, which the UI shows. q2 is the clearest case: #1 is "How long was the total customer impact window?", and the answer is the following turn.
4. **OR mode (O27) is a trade-off, not a win.**
   - **Keyword-only benefits a lot:** MRR 0.433 → 0.660, because natural questions now get keyword hits.
   - **Inside hybrid it hurts paraphrase queries:** Hit@5 0.727 → 0.455, nDCG 0.490 → 0.326. Common words ("long", "time", "year", "night", "getting worse") match many unrelated turns, and those keyword hits push the correct vector hits down: q9 moved 2 → 6, q14 1 → miss, q24 5 → miss, q29 2 → 8.
   - **It helps a few queries:** q4 improved 6 → 1 and q28 2 → 1, plus slightly better keyword/speaker nDCG.
   - **Net:** hybrid_or loses on Hit@5 (0.767 vs 0.867) and nDCG (0.616 vs 0.665), and wins only on Hit@1 (0.633 vs 0.600).
5. **With AND, hybrid equals vector on paraphrase and speaker queries.** The keyword arm returns nothing there, so hybrid's advantage comes from keyword-type queries (Recall@10 0.821 vs 0.795, nDCG 0.821 vs 0.774).
6. **Fusion can demote an exact match:** hybrid Hit@1 on keyword queries is 0.846 vs 1.000 for keyword-only (`connection pool` and `refund` came in at rank 2).
7. **Tricky tokens work:** a number (`4,420`), a hyphenated reference (`r-8812`) and a name (`Marcus`) are all found at rank 1.
8. **Latency is negligible at this scale:** ~22–26 ms p50 for all configurations, mostly query embedding.

## 6. Per-query first relevant rank

`–` = no relevant chunk in the top 10 (strict rule).

| ID | Type | Query | hybrid | hybrid_or | keyword | keyword_or | vector |
|---|---|---|---|---|---|---|---|
| q1 | keyword | connection pool | 2 | 2 | 1 | 1 | 4 |
| q2 | paraphrase | how long were customers affected by the outage | 9 | – | – | – | 9 |
| q3 | keyword | runbook | 1 | 1 | 1 | 1 | 1 |
| q4 | paraphrase | what caused the incident in the first place | 6 | 1 | – | 6 | 6 |
| q5 | speaker | what did the on-call engineer do to reduce the damage (B, rec01) | – | – | – | – | – |
| q6 | keyword | burnout | 1 | 1 | 1 | 1 | 1 |
| q7 | keyword | employee assistance program | 1 | 1 | 1 | 1 | 1 |
| q8 | keyword | Marcus | 1 | 1 | 1 | 1 | 1 |
| q9 | paraphrase | trouble sleeping at night | 2 | 6 | – | – | 2 |
| q10 | speaker | which tasks did the manager offer to hand off or postpone (A, rec02) | 4 | 2 | – | 1 | 4 |
| q11 | keyword | quarterly billing | 1 | 1 | 1 | 1 | 1 |
| q12 | keyword | 4,420 | 1 | 1 | 1 | 1 | 1 |
| q13 | paraphrase | can we cancel if the product is not adopted | – | – | – | – | – |
| q14 | paraphrase | how long does setup take after signing the contract | 1 | – | – | – | 1 |
| q15 | speaker | what did the customer say about their bad experience with a previous tool (B, rec03) | 1 | 1 | – | 1 | 1 |
| q16 | keyword | refund | 2 | 2 | 1 | 1 | 2 |
| q17 | keyword | r-8812 | 1 | 1 | 1 | 1 | 1 |
| q18 | paraphrase | why was the customer billed twice | 1 | 1 | – | 2 | 1 |
| q19 | paraphrase | how do I change the email address on my invoices | 1 | 1 | – | 1 | 1 |
| q20 | speaker | what compensation did the agent offer (A, rec04) | 2 | 2 | – | – | 2 |
| q21 | keyword | chorizo | 1 | 1 | 1 | 1 | 1 |
| q22 | keyword | custard tarts | 1 | 1 | 1 | 1 | 1 |
| q23 | paraphrase | the town with a castle they never got to visit | 1 | 1 | – | 1 | 1 |
| q24 | paraphrase | best time of year to travel there | 5 | – | – | – | 5 |
| q25 | speaker | problems with the hotel room (A, rec05) | 1 | 1 | – | 1 | 1 |
| q26 | keyword | carbon tax | 1 | 1 | 1 | 1 | 1 |
| q27 | keyword | solar | 1 | 1 | 1 | 1 | 1 |
| q28 | paraphrase | why is heavy industry hard to decarbonize | 2 | 1 | – | 7 | 2 |
| q29 | paraphrase | why is flooding getting worse | 2 | 8 | – | – | 2 |
| q30 | speaker | what did the expert say about coal towns and workers (B, rec06) | 1 | 1 | – | 1 | 1 |

## 7. Failure analysis (hybrid, strict)

Hybrid misses the top 5 on **4 of 30** queries (run 1: 6). The two speaker misses (q10, q20) were fixed by the recording filter.

| Query | Bucket | What happens |
|---|---|---|
| q2 | **Question-turn hit (label artifact)** | #1 is the question "How long was the total customer impact window?"; the answer is the next turn. Counted as a hit by the +nb metric. |
| q4 | **Near-miss** | #1 is "two separate issues collided" (the lead's summary), and the root-cause turns come at rank 6. OR mode ranks it 1st via the word "incident". |
| q5 | **Vocabulary gap** | "reduce the damage" vs "killed the long running queries" / "bumped the connection pool limit": no shared words, and the small embedder doesn't connect them. |
| q13 | **Vocabulary gap** | "cancel if not adopted" vs "scale down or exit at the next quarterly checkpoint without penalty". |

The remaining misses are embedding-quality limits. They point to a larger embedder or a cross-encoder reranker (production track), not to fusion settings.

## 8. Earlier stage checks

**Embedding model choice** (rec01 only, 14 paraphrased queries, 1 labeled chunk each; [`eval/compare_embeddings.py`](../eval/compare_embeddings.py)):

| Model | Hit@1 | Hit@3 | MRR |
|---|---|---|---|
| **all-MiniLM-L6-v2** (chosen) | 0.29 | 0.57 | 0.441 |
| multi-qa-MiniLM-L6-cos-v1 | 0.21 | 0.43 | 0.399 |

**ASR model choice** (rec01, 567.5 s of audio): `base` took 147 s and `small` 431 s. The transcripts agree on 98.4% of words; `small` fixed keyword-critical terms ("on-call", "lint", "write-up", "2.20").

**Diarization** (unsupervised, since there is no ground truth): silhouette 0.62–0.83 across the 6 recordings, within-speaker cosine 0.85–0.93 vs between-speaker 0.55–0.58. A manual check of 30 rec01 segments was 30/30 correct.

## 9. Caveats

- **Small set:** 30 queries, so one query ≈ 3 percentage points of Hit@k. Differences of 1–2 queries are not significant.
- **Labels by one annotator** (the agent), from transcripts, with no inter-annotator agreement.
- **ASR output, not reference transcripts:** ASR errors are hidden from this eval. There's no WER or DER, because no ground truth exists.
- **Clean synthetic audio** (no overlap or noise), so real calls would be harder.
- **Binary relevance:** the +nb metrics partly compensate for the strict rule.

## 10. Status of improvement options

| ID | Issue | Status |
|---|---|---|
| O25 | Speaker filter crossed recordings | ✅ **Done:** a speaker filter requires a recording filter (`search()` raises `ValueError`; the CLI rejects `--speaker` without `--recording`) |
| O26 | Question-turn hits counted as misses | ✅ **Done:** secondary +nb metrics added; the strict rule is unchanged |
| O27 | Keyword arm empty on natural questions | ✅ **Tested:** OR mode is better for keyword-only search, worse for hybrid (paraphrase Hit@5 0.727 → 0.455). The default is still AND; **the choice is the user's** |
