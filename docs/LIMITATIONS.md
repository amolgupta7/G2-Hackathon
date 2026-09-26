# Limitations

Known limits of the prototype, and how each would be addressed in production.
This is a **time-boxed prototype** (6 recordings, CPU only). Most limits are deliberate trade-offs recorded in [AGENTS.md](../AGENTS.md) (decisions `L#`); actionable follow-ups are tracked in [ISSUES.md](ISSUES.md); numbers come from [EVALUATION.md](EVALUATION.md).

Status as of 2026-09-26. Some recent fixes (listed in ISSUES §0) are built but awaiting the full test run.

---

## 1. Data & scope

| Limitation | Impact | Production path |
|---|---|---|
| **6 recordings, ~52 min, 344 chunks** | Results say little about behavior at scale (index choice, latency, ranking noise with many near-duplicates) | Scale test at 100k–1M chunks (EVALUATION plan R8) |
| **Clean synthetic audio**: TTS-like voices, no background noise, no overlapping speech, distinct speakers | ASR and diarization quality are **optimistic**; real calls will have more errors | Evaluate on real call audio (ISSUES E4) |
| **English only** (ASR runs with `language="en"`, text search uses the `english` config) | Other languages are mis-transcribed and mis-stemmed | Language detection → per-language ASR and text-search config, multilingual embedder |
| **Exactly two speakers per recording**, mono audio | Calls with 3+ participants aren't supported | Estimate the speaker count (pyannote / NeMo); split channels for stereo call audio |

## 2. Speech-to-text (ASR)

| Limitation | Impact | Production path |
|---|---|---|
| faster-whisper **`small`** on CPU (chosen over `base` for domain terms, L22) | Slower than real time on this machine (~0.8–1.1× audio length); rare terms can still be misspelled | GPU workers; larger models (WhisperX / Parakeet); custom vocabulary / initial prompt |
| **Segment-level timestamps** (no word-level alignment) | Timestamps point to the start of a turn, not the exact word | Word timestamps / forced alignment (WhisperX) |
| **No reference transcripts**, so no WER | ASR errors are invisible to the retrieval eval (labels were made on ASR text) | Hand-correct a subset and measure WER (ISSUES E3) |

## 3. Diarization (who spoke)

| Limitation | Impact | Production path |
|---|---|---|
| Resemblyzer voice embedding **per ASR segment** + KMeans with **k = 2** fixed (L5) | A segment that contains a speaker change gets one label; overlapping speech isn't handled; a recording with < 2 segments is treated as single-speaker (all `A`, L60) | pyannote / NeMo Sortformer with overlap handling and speaker-count estimation |
| Labels `A`/`B` are **per recording**: "A" is simply the first voice heard, not a role or a global identity | "Speaker A" means a different person in each recording, so a **speaker filter requires a recording filter** (L32); "the agent" / "the customer" can't be queried directly | LLM role labeling (agent/customer, interviewer/guest) or voice-ID across recordings (ISSUES D3) |
| **No DER** (no ground truth); quality judged by silhouette 0.62–0.83 and a 30/30 manual spot check | Diarization errors on hard audio would go unmeasured | Labeled reference set, DER / word-level speaker error |

## 4. Chunking

| Limitation | Impact | Mitigation now | Production path |
|---|---|---|---|
| **One speaker turn per chunk** (merge same-speaker segments, gap ≤ 1 s, max 30 s; L6) | A question and its answer are separate chunks, so the question turn can outrank its answer | Every result includes the **previous and next turn**; neighbor-aware Hit@5 is 0.933 vs 0.867 strict | Multi-turn windows or parent–child chunks with a context header (ISSUES R4) |
| Chunk DB ids (`BIGSERIAL`) change on re-ingest | Stored chunk ids go stale | The API and eval use the stable `(recording_id, chunk_index)` pair | Keep that pair as the external key (ISSUES D1) |

## 5. Retrieval

| Limitation | Impact | Mitigation now | Production path |
|---|---|---|---|
| **Small embedder** (all-MiniLM-L6-v2, 384-d, chosen for CPU speed; L28) | Paraphrases with no shared words can be missed: q5 "reduce the damage" and q13 "cancel if not adopted" still miss the top 5 | A cross-encoder reranker (L48) moved q2/q4/q10 to rank 1 (MRR 0.724 → 0.827) | Larger embedder (BGE / e5), a stronger reranker on GPU (ISSUES R1) |
| **Keyword ranking is `ts_rank_cd`, not BM25** (no IDF or length normalization; L10) | Weaker term weighting on a large corpus | RRF uses ranks, not scores, so scale mismatch doesn't matter | True BM25 (ParadeDB / OpenSearch) or learned sparse |
| **Keyword arm uses AND** (every term must match; L34) | Natural-language questions get no keyword hits, so hybrid behaves like vector-only for them | Measured and accepted: OR mode made hybrid worse (paraphrase Hit@5 0.727 → 0.455) | BM25 / SPLADE (ISSUES R2) |
| **Fixed fusion weights** (RRF k = 60, keyword weight 1.5) tuned on 30 queries | Weights may not transfer to other data | Keyword weight 1.5 and rerank keep exact keyword matches at rank 1 (keyword-type Hit@1 1.000) | Tune on a larger golden set |
| **Similarity cut-off only for searches across all recordings** (0.3; 0.45 without letters; L41, L46, L51) | Gibberish/off-topic queries return nothing (TNR 1.0 on 10 negatives). Inside a selected recording there's no cut-off, so nonsense there still returns that recording's closest turns | Scoping keeps weakly-worded speaker questions answerable (no false "No matches") | Reranker-score or relative cut-off (ISSUES O23.3) |
| **`-exclude` is a word-prefix match** on the fused results (L39, L47) | `-invoice` removes "invoices" too; a query of only exclusions returns nothing (by design, D6) | Excluded words are also stripped from the embedder input | Lemma/stem-aware exclusion |
| No fuzzy/typo matching; English stemming only | Misspelled names or queries can miss | Vector arm tolerates some variation | `pg_trgm` / phonetic matching (ISSUES O9) |
| No query understanding (no filter extraction or rewriting) | Filters must be chosen in the UI; "last week's calls" isn't understood | UI filters for recording and speaker | LLM / rule-based filter extraction |
| **Search results only**, no generated answer | The user reads the turns to get the answer | Snippets with context and audio from the turn start | RAG answer with timestamp citations (ISSUES O8) |

## 6. Evaluation

| Limitation | Impact | Production path |
|---|---|---|
| **30 labeled queries + 10 negatives** | 1 query ≈ 3 percentage points; small differences aren't significant | 100+ queries, paired bootstrap tests (ISSUES E1) |
| **Single annotator** (the agent), labels from transcripts | Label bias; no agreement measure | Second annotator, Cohen's κ (ISSUES E2) |
| **Binary relevance** | No partial credit for near-misses (partly covered by the neighbor-aware metrics) | Graded relevance, nDCG with grades |
| **Offline metrics only** | No usage signals (zero-result rate in production, clicks, reformulations) | Event logging and online metrics (ISSUES E5) |

## 7. System & scale

| Limitation | Impact | Production path |
|---|---|---|
| **Exact vector scan**, no ANN index (L11) | Fine at 344 chunks, linear cost at scale | pgvector HNSW / halfvec above ~100k–1M chunks, or a dedicated vector store (ISSUES P3) |
| **CPU only, single API process** | Without rerank ~25–35 ms per query warm (20 concurrent: server p50 129 ms). **The cross-encoder rerank adds ~200 ms sequentially and is slow under concurrent load** (a top-50 measurement: client p50 2.3 s / p95 4.5 s; top 20 is now the default, but its load latency isn't reliably measured yet, ISSUES R1.2) | GPU or a dedicated reranker service, multiple API workers, batching (ISSUES P2) |
| **Batch, synchronous ingestion** (one recording at a time; no queue) | New audio is searchable only after the pipeline is re-run (a failing recording is logged and skipped; the run continues and exits 1, O22) | Job queue with `pending` state, worker pool |
| Stale-job recovery uses a **one-time `started_at`**, not a live heartbeat | With parallel workers, a very long job could be treated as crashed | Periodic heartbeat / lease renewal (ISSUES K1.1) |
| Intermediate outputs are **local files** (`data/processed/`) and audio is read from local disk | Single-machine setup | Object storage for audio and artifacts |

## 8. Security & operations

| Limitation | Impact | Production path |
|---|---|---|
| **Local-only deployment**: no authentication, CORS policy or rate limiting on the API | Must not be exposed beyond localhost | Auth, CORS, rate limits, TLS (ISSUES S2) |
| Dev DB password (credentials are in git-ignored `.env`, but the value is still the default) | Unsafe if the database is ever exposed | Rotate the password; secrets manager (ISSUES S1.1) |
| UI plays the **full ~22 MB WAV** per result (starting at the turn) | Heavy pages with many results | Serve clipped segments or stream ranges (ISSUES U1) |
| Windows single-machine setup (Docker Desktop, global Python) | Environment-specific quirks (e.g. Docker not on PATH in old terminals) | Containerized app + pinned dependencies |
| **8 GB RAM**: each Python process (API, eval, ingest) loads its own models, and Docker's VM takes ~1.2 GB | Running API + UI + eval together grew the Windows page file (on C) to 9.4 GB, leaving C nearly full (OOM risk) | **Operational rule:** run heavy jobs one at a time (README). Production: dedicated hosts, one model server shared by workers |
