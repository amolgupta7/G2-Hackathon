# Open Issues

What's still left. Fixed items are removed from this file; their decisions and results live in [AGENTS.md](../AGENTS.md) (L-numbers) and [EVALUATION.md](EVALUATION.md).

**Priority:** P1 = affects correctness or the demo · P2 = quality/performance improvement · P3 = production track / nice to have
**Conventions:** ⭐ = suggested option; **nothing here is decided until the user picks**. Follow-ups found while fixing an issue keep the parent's ID (`O29.1`, `O29.2`…). This file is the only issue list; AGENTS.md records decisions, not issues.

---

## 0. Built, verification pending

Verified and removed: O29, O29.1, O29.2, O30, O30.1, O23/O23.1, P1, P4, K1, K2, O22, S1, S3 (L50); O23.2, R1.1, R3 (L51).
The fixes below are **built and kept (user decision, L59)**; the verification run is deferred until the machine is restarted (low RAM). Run jobs one at a time (README memory note).

| ID | Pri | Status | What's left | Log to check |
|---|---|---|---|---|
| D5.1 | P2 | 🟡 fixed (L60): `diarize.py` with **< 2 segments** skips KMeans, labels all turns `A`, writes `single_speaker: true` with `null` metrics and logs a warning. Also guarded: silhouette / between-speaker cosine are `null` when undefined (e.g. exactly 2 segments), instead of raising | Silent / one-utterance WAV through ASR → diarize → chunk → embed → ingest completes; the existing 6 recordings keep identical labels | `<rec>: N speech segment(s) -> clustering skipped, single-speaker recording (all labeled A)` |
| R1.2 | **P1** | 🟡 `RERANK_TOP=20` set (L51). **Latency under load not measured yet:** the run after the change was confounded by paging (0.3 GB RAM free, API startup 17.6 s vs 6.7 s) and a harness bug (unread stdout/stderr pipes blocked the server). The A/B was interrupted by the user | A same-conditions A/B (top 20 / top 50 / off) with output to files, with the machine not paging | `rerank=on(N) … rerank=Xms` |
| X2 | P3 | 🟡 fixed (L51): `app` imported before `sentence_transformers` in `eval/compare_embeddings.py` | Run it once; confirm nothing is written to `~/.cache/huggingface` on C | — |
| D4 | P2 | 🟡 fixed (L52): re-ingesting a **same file name with different audio** hit a raw `UniqueViolation` on the `id` primary key. Now `ON CONFLICT DO NOTHING` (both keys) + a clean `id conflict` error; the existing recording is untouched; counts as failed (exit 1) | Copy a different WAV over an existing file name, run ingest → one clean error line, no traceback, the old row unchanged | `claim <rec>: id conflict, … existing recording left unchanged` |
| D10 | P3 | 🟡 fixed (L58): duplicated hand-rolled `sum(x)/len(x)` stats in `asr.py` / `chunk.py` → `statistics.fmean` (the D5 empty guards stay; no new helper module). No other `sum/len` averages left in `app/` | ASR/chunk summary log lines print the same values/format as before | `… avg_seg=…` / `… avg_dur=… avg_words=…` |
| D9 | P3 | 🟡 fixed (L57): `VECTOR(384)` in schema.sql was decoupled from `EMBED_MODEL`, so a model swap to another dimension would fail with an opaque pgvector error. Now `embedding_dim(conn)` reads the declared column size; **ingest** rejects mismatched embeddings per recording with a clear message; the **API refuses to start** on a model/column mismatch (skipped with a warning if the DB is down) | Normal start → `embedding dimension check: model=384 db=384`; ingest → `chunks.embedding is vector(384)`. A mismatch (e.g. `EMBED_MODEL` set to a 768-d model) → a clear error at API startup / per recording in ingest | `embedding dimension check: …` / `… embeddings are N-d … but chunks.embedding is vector(384)` |
| D8 | P2 | 🟡 fixed (L56): `recording=` (empty string) is falsy, so it skipped the O30 404 check and silently filtered on `recording_id = ''` (0 results; it also re-enabled the unscoped similarity floor). Now `Query(min_length=1)` → **422** before any query; 422s are logged | `/search?q=refund&recording=` → 422 (`string_too_short`); `recording=%20` → 404; valid / absent recording unchanged | `app.api 422 GET /search: [('query.recording', 'String should have at least 1 character')]` |
| D7 | P3 | 🟡 fixed (L55): `diarize.py` `mean_cos_within` averaged over **all** pairs including self-pairs (1.0), inflating the diagnostic. Now distinct pairs only, `(sum − trace)/(n(n−1))`; `null` for a cluster with < 2 segments. Labels, silhouette and between-speaker cosine are unaffected | Re-run `python -m app.pipeline.diarize` (labels must be byte-identical to the committed JSON; only `mean_cos_within` should drop). Then update the recomputed values in EVALUATION §8 and commit the refreshed diarization JSON | `app.pipeline.diarize <rec> {"… "mean_cos_within": [...] …}` |
| D6 | P2 | 🟡 fixed (L54): an **exclusion-only query** (`-invoice`) had no positive terms, so `websearch_to_tsquery` → `!invoice` matched nearly every chunk (~50 arbitrary results). Now `search()` returns an explicit empty result → API `count: 0`, UI "No matches" | `-invoice` / `-"account credit"` → 0 results (API + UI); `refund -invoice` unchanged | `no positive terms (only exclusions) -> 0 results` |
| D5 | P2 | 🟡 fixed (L53): `asr.py` / `chunk.py` end-of-run stats crashed with `ZeroDivisionError` on a silent recording (0 segments/chunks) **after** the output was written, which also stopped the loop. Now a warning replaces the stats and the loop continues | Run ASR + chunk on a silent WAV → a warning line, a valid (empty) JSON, the next file processed | `no speech segments detected -> …` / `no chunks (recording has no speech segments)` |

| ID | Pri | Open | Options |
|---|---|---|---|
| ↳ D5.2 | P3 | *(found while fixing D5)* `asr.py` `real_time_factor = elapsed / info.duration` → `ZeroDivisionError` for a **zero-length** audio file (caught and logged inside `transcribe`, but the run stops) | ⭐ guard `duration == 0` (RTF = null) · leave |
| ↳ D4.1 | P3 | *(found while fixing D4)* **Same audio under a new file name** after a *failed* first attempt: the retry path updates the old row (old id) but loads chunks under the new file stem → FK error. Only this failed-then-renamed case; completed ones are skipped correctly | **User decision (L59): leave documented, no code change** (rare, low impact). Fix if needed: use the stored id (`existing[0]`) in the retry path |
| X3 | P3 | Memory/disk pressure on this 8 GB machine: documented as an operational rule (README, LIMITATIONS: run heavy jobs one at a time). The system-level fixes remain | limit Docker/WSL memory (`.wslconfig`) · move the page file to D (user's call) |

## 1. Search & API

| ID | Pri | Issue | Impact | Options |
|---|---|---|---|---|
| O23.3 | P3 | *(from O23.1, fixed)* By design, recording-scoped searches skip the similarity floor, so gibberish **inside** a recording returns that recording's closest turns instead of "No matches" | Only when a recording is selected and the query is nonsense | ⭐ leave (a scoped search is a deliberate browse of one conversation) · a lower floor for scoped searches (e.g. 0.1) |

## 2. Retrieval quality

| ID | Pri | Issue | Evidence | Options |
|---|---|---|---|---|
| R1 | P3 | **Vocabulary gap, partly closed:** the cross-encoder rerank (L48) moved q2/q4/q10 to rank 1 (MRR 0.724 → 0.827), but q5 "reduce the damage" and q13 "cancel if not adopted" still miss the top 5 (no shared words with the answering turns) | EVALUATION §12 | Documented, left as-is for the prototype (user, P3). Production: larger embedder / stronger reranker on GPU |
| R2 | P3 | With AND mode, **hybrid = vector** on paraphrase/speaker queries (keyword arm empty) | EVALUATION §5 | Accepted (L34; OR tested and rejected). Revisit with BM25 / learned sparse |
| R4 | P3 | **Q/A split across chunks**, so the question turn can outrank its answer | q2; Hit@5+nb 0.933 vs strict 0.867 | Neighbor context already shown. Production: multi-turn windows / parent–child chunks |
| O9 | P3 | No fuzzy/typo matching for misspelled names | No failures in the current eval | `pg_trgm`, only if the eval shows misses |
| O21 | P3 | Embedder input is chunk text only (no context header) | Not measured | ⭐ keep · try a contextual header and re-evaluate |

## 3. Performance & scale

| ID | Pri | Issue | Current state | Options |
|---|---|---|---|---|
| P2 | P3 | Single uvicorn process, no workers or rate limiting | Fine for the demo | `--workers N` / gunicorn, rate limiting in production |
| P3 | P3 | Exact vector scan (no ANN index) | Instant at 344 chunks | pgvector HNSW above ~100k–1M chunks (L11) |
| O32 | P3 | Query embedding showed 47–150 ms right after startup | Not reproduced after warm-up (L37): 25–35 ms | ⭐ close · monitor in the full run |

## 4. Ingestion & data

| ID | Pri | Issue | Impact | Options |
|---|---|---|---|---|
| D1 | P2 | DB `chunks.id` is BIGSERIAL and **changes on re-ingest** | Anything storing chunk ids breaks (queries.json and the API already use `(recording_id, chunk_index)`) | ⭐ keep `(recording_id, chunk_index)` as the external key everywhere |
| K1.1 | P3 | *(from K1)* `started_at` is set once per claim, with no periodic heartbeat, so with parallel workers a job longer than the timeout could be marked stale | Only with parallel workers and very long jobs (jobs take < 1 s today) | ⭐ leave · refresh `started_at` during long steps |
| D2 | P3 | A segment containing a speaker change gets one label; overlap isn't handled | None seen on this clean audio | Production: pyannote / NeMo Sortformer, word-level alignment |
| D3 | P3 | `A`/`B` labels are per recording, not roles or identities | Speaker filter requires a recording (L32) | LLM role labeling or voice-ID across recordings |

## 5. Evaluation gaps

| ID | Pri | Issue | Options |
|---|---|---|---|
| E1 | P2 | Small query set (30), so 1 query ≈ 3 points of Hit@k | Grow to 100+ queries; paired bootstrap significance test |
| E2 | P3 | Single annotator; no agreement measure | Second annotator on a sample, Cohen's κ |
| E3 | P3 | No ASR/diarization ground truth, so no WER/DER | Hand-correct a subset of transcripts |
| E4 | P3 | Clean synthetic audio, so results are optimistic | Test on real call audio |
| E5 | P3 | No online/usage metrics (zero-result rate, CTR, reformulation) | Event logging in the UI/API |

## 6. Repo, ops & security

| ID | Pri | Issue | Options |
|---|---|---|---|
| S1.1 | P3 | *(from S1, fixed)* The DB password value is still `postgres` (the volume was initialized with it), and it's in the history of earlier commits | ⭐ before a public push: `ALTER USER postgres PASSWORD '…'` + update `.env` · leave for local dev |
| S2 | P3 | API has no auth, CORS config or rate limits | Needed only if exposed beyond localhost |
| U1 | P3 | Each result's audio player references the full ~22 MB WAV (user's choice) | ⭐ keep for the demo · clip the turn ±2 s if pages get heavy |
| X1 | P3 | Docker CLI isn't on PATH in terminals opened before the install | Open a new terminal, or prepend `%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin` |

## 7. Not started

| ID | Pri | Item | Notes |
|---|---|---|---|
| O8 | P3 | LLM answer with citations on top of search results (RAG) | The prototype returns search results only |
