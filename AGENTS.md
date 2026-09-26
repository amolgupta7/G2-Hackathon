# AGENTS.md — Hybrid Retrieval over 2-Speaker Audio Transcripts

> The single source of truth for this project. Every discussion, option, decision and change gets written here.
> **Rules for any agent working here:**
> 1. Read this file before doing anything.
> 2. Never lock a decision on your own. Add options to §5 and wait for the user to pick one.
> 3. After each session, add an entry to §9 (Discussion Log) and update §6 (Decision Register).
> 4. **Every prompt:** record what was decided, **why**, and **why the other options were not chosen**.
> 5. **Two tracks:** §6 decisions are for the **time-boxed working prototype**. Production-grade choices are listed separately (the "Production alternative" in each §6 entry) and are decided later. Never mix them up.
> 6. Undecided items are locked **during development**, when they come up, by asking the user, never by assuming.
> 7. **Don't break the working flow** (pipeline → DB → search → API → UI → eval). When fixing an issue: (a) make no unnecessary changes; (b) when reverting or changing code, check for and remove dead or unused code; (c) keep the fix simple, then **fix and test**: targeted tests plus a full-flow regression check (the eval compared against the previous run). If a requested fix causes a regression, report it and let the user choose.

---

## 1. Problem Statement (verbatim)

**Effective retrieval from audio transcripts.** You have audio recordings of many conversations, each with two speakers. How might you achieve effective hybrid search (keyword + semantic) across them? What diarization, database, embedding and indexing strategies might give "ideal" retrieval quality at scale? What metrics matter if this system goes to production, and what would an effective evaluation of the system look like?

## 2. Problem Analysis

### 2.1 What is really being asked
| # | Sub-question | What it really tests |
|---|---|---|
| Q1 | Hybrid search across transcripts | Combining exact lexical matching (names, product codes, numbers) with semantic matching (paraphrase, intent) |
| Q2 | Diarization strategy | Knowing *who* said *what*, so queries like "what did the **customer** say about pricing" work |
| Q3 | Database, embedding and indexing strategy | Chunking of conversational text, choice of model, ANN index, sparse index, fusion, scale |
| Q4 | Production metrics | Quality, latency, cost, freshness and reliability, measured per stage |
| Q5 | Evaluation | A reproducible, stage-wise and end-to-end evaluation with a labeled set and ablations |

### 2.2 Key challenges specific to *audio transcripts* (unlike plain document RAG)
1. **Errors cascade.** ASR errors (WER) → diarization errors (DER) → wrong chunk text or wrong speaker → retrieval miss. Keyword search is hit hardest (a misspelled name gets 0 BM25 hits), so phonetic/fuzzy matching or semantic backup matters.
2. **Conversational text is sparse in meaning.** Turns like "yeah", "uh-huh" or "right, so…" carry little. Chunking must span several turns and keep speaker context.
3. **Meaning lives across turns.** A question is in turn *n* and its answer in turn *n+1*. Chunks that split Q/A pairs hurt recall.
4. **Speaker-aware retrieval.** You need a speaker/role label on every chunk, filterable at query time.
5. **Timestamps are the "citation".** A result must link back to `conversation_id + start_ms–end_ms`, so the user can play the audio.
6. **Scale.** Thousands to millions of hours means tens to hundreds of millions of chunks. ANN index, quantization, sharding and ingestion throughput all become real concerns.
7. **Two speakers is a gift.** If the audio is stereo/dual-channel (as in call centers), diarization becomes trivial and near-perfect through channel separation. That is worth checking first.

### 2.3 Assumptions (to confirm with user)
- A1: Language is English (multilingual changes model choices).
- A2: Audio may be mono (worst case, so diarization is needed) or stereo (channel split).
- A3: The hackathon deliverable is a working prototype, an eval report and a design for production scale.
- A4: Hardware is a Windows machine. **GPU availability is unknown** and affects ASR, diarization and embedding choices.

### 2.4 Confirmed facts / constraints (2026-09-26)
- **Scale:** 6 recordings, a few hundred chunks.
- **Time:** the build must fit in a short window (about one hour), so favor simplicity and defensibility.
- **Environment (checked):** Windows 10, Python 3.12.5. **No Docker, no Postgres/psql, no NVIDIA GPU.** ASR, diarization and embeddings run on CPU (fine for 6 recordings), or through an API.

---

## 3. High-Level Architecture

### 3.1 Component diagram
```mermaid
flowchart LR
  subgraph INGEST["Offline / Async Ingestion Pipeline"]
    A[Audio files<br/>wav/mp3] --> B[Pre-processing<br/>resample 16k, VAD,<br/>channel check]
    B --> C[ASR<br/>word timestamps]
    B --> D[Diarization<br/>2 speakers]
    C --> E[Alignment & Merge<br/>word→speaker, turns]
    D --> E
    E --> F[Enrichment<br/>role labeling, NER,<br/>summary, topics]
    F --> G[Chunking<br/>speaker-turn windows<br/>+ contextual prefix]
    G --> H1[Dense Embedder]
    G --> H2[Sparse / BM25<br/>Tokenizer]
  end

  subgraph STORE["Storage & Indexes"]
    H1 --> I1[(Vector Index<br/>HNSW / quantized)]
    H2 --> I2[(Inverted Index<br/>BM25 / SPLADE)]
    E --> I3[(Metadata / Transcript Store<br/>conv, speaker, ts, raw words)]
    A --> I4[(Object Store<br/>audio)]
  end

  subgraph QUERY["Online Query Path"]
    Q[User query + filters] --> R[Query Understanding<br/>rewrite, filter extraction]
    R --> S1[Dense retrieve top-K]
    R --> S2[Keyword retrieve top-K]
    S1 --> T[Fusion<br/>RRF / weighted]
    S2 --> T
    T --> U[Cross-encoder Reranker<br/>top-N]
    U --> V[Context expansion<br/>neighbor turns]
    V --> W[Results: snippet, speaker,<br/>timestamp, audio link]
    W -.optional.-> X[LLM Answer w/ citations]
  end

  I1 --- S1
  I2 --- S2
  I3 --- V
  I4 --- W

  subgraph EVAL["Evaluation & Observability"]
    Y[Golden query set<br/>+ relevance labels] --> Z[Offline eval harness<br/>Recall@k, nDCG, MRR]
    W --> O[Online metrics<br/>latency, CTR, zero-result]
  end
```

### 3.2 Ingestion flow (per recording)
1. **Load & normalize:** 16 kHz mono (or keep 2 channels if stereo), loudness normalization, VAD to drop silence.
2. **Channel check:** if stereo with one speaker per channel, diarize by channel. Otherwise run a diarization model with `num_speakers=2`.
3. **ASR:** transcribe with word-level timestamps.
4. **Merge:** give each word the speaker whose diarization segment overlaps it most. Group consecutive same-speaker words into **turns** `{speaker, start, end, text}`.
5. **Enrich (optional):** an LLM or classifier maps `SPEAKER_00/01 → role` (agent/customer, interviewer/guest), and extracts entities, topics and a conversation summary.
6. **Chunk:** a sliding window over turns (e.g. 3–6 turns, ~200–400 tokens, 1-turn overlap), each turn prefixed with its speaker label. Optionally prepend a *contextual header* (conversation title/summary) before embedding.
7. **Index:** a dense vector plus a sparse/BM25 representation plus metadata (`conv_id, chunk_id, speakers, roles, start_ms, end_ms, date, turn_ids`).

### 3.3 Query flow
1. Parse the query and extract structured filters (speaker/role, date, conversation).
2. Run **dense** and **keyword** retrieval in parallel, top-K each (K≈50–100), with the same filters.
3. **Fuse** the lists (RRF, k=60, as the baseline).
4. **Rerank** the fused top ~50 with a cross-encoder, keeping the top N (≈10).
5. **Expand** each hit with its neighboring turns for readability. Return snippet, speaker, timestamp and audio deep-link.
6. (Optional) generate an LLM answer with timestamp citations.

---

## 4. Solution Approaches (end-to-end), with pros and cons

### Approach A — "Postgres-only" (pgvector + BM25 extension)
Stack: WhisperX → Postgres (`pgvector` HNSW + `pg_search`/ParadeDB BM25, or built-in `tsvector`) → RRF in SQL → reranker.
- **Pros:** one database for metadata, transcripts, vectors and text. SQL joins and filters (speaker, date) are trivial. It is transactional, easy to demo and easy to reason about. It is good up to roughly 10–50M vectors on a single node.
- **Cons:** built-in `tsvector` is not true BM25 (it needs ParadeDB/pg_search for that). ANN scale-out is harder than in dedicated engines. Tuning HNSW together with filters (post-filter recall loss) takes care.

### Approach B — Search-engine native hybrid (OpenSearch / Elasticsearch)
Stack: WhisperX → OpenSearch (BM25 + k-NN field, hybrid query + normalization pipeline) → reranker.
- **Pros:** the best keyword features: analyzers, **phonetic** and fuzzy matching (helps with ASR errors), synonyms, phrase and proximity queries, highlighting. Hybrid search is built in. It scales out horizontally and is proven in production.
- **Cons:** heavy to run (JVM, cluster tuning). Vector search is slower and uses more memory than dedicated vector DBs. It is heavier to set up on a Windows laptop (Docker needed).

### Approach C — Vector-DB native hybrid (Qdrant / Weaviate / Milvus)
Stack: WhisperX → Qdrant with named vectors: dense plus sparse (BM25-style, SPLADE, or BGE-M3 sparse). Built-in fusion (RRF/DBSF), payload filters and quantization.
- **Pros:** very fast ANN, with scalar/binary quantization for scale and native sparse vectors. Hybrid search and fusion come in one query API. Payload filters do not hurt recall (filterable HNSW). It is simple to run in Docker, and multi-vector (ColBERT) support is available.
- **Cons:** "keyword" means sparse vectors, not a full-text engine: there are no rich analyzers, phrase queries or phonetic matching. Transcripts and metadata often need a second store. Exact-match edge cases (IDs, numbers) depend on the tokenizer.

### Approach D — Fully managed / API-first
Stack: Deepgram / AssemblyAI (ASR + diarization) → OpenAI / Cohere / Voyage embeddings → Pinecone / managed Elastic → Cohere Rerank.
- **Pros:** the fastest way to high quality with no GPU needed. Strong diarization and ASR out of the box, elastic scaling and minimal ops.
- **Cons:** cost per audio hour and per query, data privacy (recordings leave your infrastructure) and vendor lock-in. It shows less engineering depth, which matters for a hackathon judged on design. Model internals are harder to ablate.

### Approach E — Late-interaction / multi-vector (ColBERT, BGE-M3 multi-vector, ColBERT in Vespa)
Stack: WhisperX → BGE-M3 (dense + sparse + multi-vector) → Vespa or Qdrant multivector → BM25 + dense first stage, ColBERT MaxSim rerank.
- **Pros:** top retrieval quality. Token-level matching is robust to paraphrase and still sensitive to exact terms. Vespa handles hybrid ranking natively at huge scale.
- **Cons:** storage grows 10–100× (one vector per token) and indexing is more complex. Vespa has a steep learning curve. It is overkill for a first prototype.

### Approach F — LLM-enriched hierarchical index (a layer on top of A–E)
Add per-conversation summaries, per-chunk "contextual retrieval" headers, extracted entities/topics as filterable fields, and query rewriting/HyDE. Search at two levels: conversation first, then chunks inside it.
- **Pros:** large recall gains on vague or topic-level queries ("calls where the customer was angry about billing"). Entities make keyword search more precise. Supports faceted UX.
- **Cons:** LLM cost and latency at ingest time, possible hallucinated metadata, and more pieces to evaluate.

### 4.1 Comparison matrix (1 = poor, 5 = excellent)
| Criterion | A: Postgres | B: OpenSearch | C: Qdrant | D: Managed | E: ColBERT/Vespa | +F: LLM enrich |
|---|---|---|---|---|---|---|
| Retrieval quality | 3 | 4 | 4 | 4 | 5 | +1 |
| Keyword richness (phrase, fuzzy, phonetic) | 3 | 5 | 2 | 3–4 | 4 | +0 |
| Scale (100M+ chunks) | 2 | 4 | 4 | 5 | 5 | 0 |
| Ops simplicity | 5 | 2 | 4 | 5 | 1 | −1 |
| Hackathon build speed | 5 | 3 | 4 | 5 | 2 | −1 |
| Cost | 5 | 3 | 5 | 2 | 3 | −1 |
| Privacy / self-hosted | 5 | 5 | 5 | 1 | 5 | depends |

> **LOCKED (2026-09-26): Approach A — Postgres only.** See §6. The rest of §4 is kept as design rationale and as the "production at scale" migration story.
>
> **How A scales (for the defense):** at a few hundred chunks, an exact (brute-force) vector scan gives 100% recall in under 1 ms. Past ~1M chunks, add a pgvector HNSW index with halfvec/int8. Past ~10–50M chunks or at high QPS, either shard with Citus or move the retrieval tier to C (Qdrant) or B (OpenSearch), keeping Postgres as the system of record. The schema, chunking, fusion and eval harness all carry over unchanged.

### 4.2 Originally suggested approach (superseded by the user's lock on A)
**C (Qdrant hybrid) + reranker + a light touch of F**, with the evaluation harness treated as a first-class deliverable:

`WhisperX (faster-whisper large-v3 + pyannote 3.1, num_speakers=2, or channel split if stereo)` → `speaker-turn window chunks + contextual header` → `BGE-M3 (dense + sparse from ONE model)` or `dense model + BM25` → `Qdrant (dense HNSW + sparse, int8 quantization, payload filters on speaker/role/conv/date)` → `RRF` → `bge-reranker-v2-m3` → neighbor-turn expansion → results with timestamps.

**Reasoning:**
1. **Quality:** hybrid plus a cross-encoder reranker is the most consistent quality win in the retrieval literature (BEIR/MTEB-style results). Dense covers paraphrase, sparse covers names and numbers, and the reranker fixes the fusion ordering.
2. **Scale story:** Qdrant supports quantization, sharding, on-disk vectors and filterable HNSW. That gives a credible path from a laptop to 100M+ chunks without changing the design.
3. **Simplicity:** BGE-M3 produces dense and sparse vectors in one forward pass. That is one model for both retrieval arms, with no separate BM25 engine to run.
4. **Handles audio-specific problems:** speaker-turn chunking keeps Q/A pairs together, speaker/role payloads allow speaker-scoped queries, and timestamps give playable citations.
5. **Evaluability:** each stage (ASR, diarization, dense, sparse, fusion, rerank) is swappable, so ablations are easy. That directly answers the "effective evaluation" part of the problem.
6. **Known weakness and mitigation:** no phonetic/fuzzy keyword search. Mitigate with an ASR custom vocabulary / initial prompt, or add Approach B's phonetic analyzer later if the eval shows keyword misses on names.

**Runner-up:** A (Postgres + ParadeDB). Pick it if simplicity and one database matter more than the scale story.

---

## 5. Open Decisions (options for the user to lock)

> Format: options; ⭐ = suggested default. **Nothing here is locked until it appears in §6.**
>
> **Status (2026-09-26):** Locked for the prototype: D1, D2, D3, D4, D5, D6, D7, D7.1, D7.2, D7.3, D8, D12, D13. Still open: D9 (moot for the prototype because of D7.3), D10, D11, plus the new sub-decisions in §6.2.

### D1. Input data source
- (a) Public 2-speaker corpus (e.g. CALLHOME/Switchboard-style telephone speech; check licensing, since LDC corpora are paid)
- (b) ⭐ **Synthetic conversations:** an LLM writes scripts and TTS gives two distinct voices. Ground truth for transcript, speaker and relevance comes *for free*, which is ideal for evaluation.
- (c) Public podcasts/interviews (realistic, but no ground truth)
- (d) Mix: (b) for eval ground truth + (c) for realism

### D2. ASR engine
- (a) ⭐ **WhisperX** (faster-whisper large-v3 + wav2vec2 word alignment). Good accuracy and precise word timestamps. GPU preferred.
- (b) faster-whisper alone (simpler, but less accurate word timestamps)
- (c) NVIDIA Parakeet/Canary (NeMo): very fast and accurate, but a heavier install on Windows
- (d) API: Deepgram / AssemblyAI (diarization included)

### D3. Diarization
- (a) **Channel split**, if the audio is stereo (near-perfect and free)
- (b) ⭐ **pyannote speaker-diarization-3.1** with `num_speakers=2` (the open-source standard)
- (c) NVIDIA NeMo (MSDD / Sortformer)
- (d) Bundled with the API choice in D2(d)
- Add-on: LLM **role labeling** (SPEAKER_00 → agent/customer), yes or no?

### D4. Chunking strategy
- (a) One chunk per utterance/turn (precise, but too little context)
- (b) ⭐ **Speaker-turn sliding window** (3–6 turns / ~300 tokens, 1-turn overlap), speaker labels inline
- (c) Q/A pair chunks (question turn + answer turn)
- (d) Semantic/topic segmentation (split at embedding-similarity drops)
- Add-on: **parent–child** (retrieve small chunks, return the larger window), yes or no?
- Add-on: **contextual header** (conversation summary prepended before embedding), yes or no?

### D5. Dense embedding model
- (a) ⭐ **BGE-M3** (open source, dense + sparse + multi-vector, 8k context, multilingual)
- (b) Qwen3-Embedding (0.6B / 4B): top of MTEB, open source
- (c) OpenAI text-embedding-3-large / Cohere embed-v4 / Voyage (API)
- (d) Small/fast: bge-small / nomic-embed / e5-small (CPU-friendly)

### D6. Keyword / sparse representation
- (a) Classic **BM25** (Lucene/tantivy/pg_search/Qdrant BM25 sparse)
- (b) ⭐ **BGE-M3 sparse** (learned lexical weights from the same model as D5a)
- (c) SPLADE v3 (strong learned sparse, extra model)
- Add-on: fuzzy/phonetic matching for ASR-misspelled names (needs B-style engine), yes or no?

### D7. Database / index — ✅ LOCKED: (a) Postgres (see §6)
- (a) **Postgres + pgvector + full-text search** ← LOCKED
- (b) OpenSearch / Elasticsearch
- (c) ⭐ **Qdrant** (dense + sparse named vectors, payload filters, quantization)
- (d) Weaviate / Milvus
- (e) Vespa (+ ColBERT)
- (f) LanceDB (embedded, zero-server, good for a laptop demo)

### D7.1 How to run Postgres (new; no Docker/Postgres installed)
- (a) ⭐ **`pgserver` (pip)**: embedded Postgres with pgvector bundled, started from Python with no admin rights and no installer. *Verify that the Windows wheel installs on Py 3.12 before locking.*
- (b) **Managed Postgres free tier** (Supabase / Neon): pgvector included, zero install, but needs internet and signup, and the data leaves the machine.
- (c) Native Windows Postgres installer (EDB) + **compile pgvector** with VS Build Tools (`nmake`): slow and fragile, risky in a one-hour window.
- (d) Install Docker Desktop + ParadeDB image: true BM25, but needs WSL2 and a reboot, which is too slow for this window.

### D7.2 Lexical (keyword) ranking inside Postgres
- (a) ⭐ **Built-in `tsvector` + GIN index + `ts_rank_cd`**: zero extensions, supports phrase (`<->`) and prefix queries. Not true BM25 (no IDF saturation), which is acceptable at a few hundred chunks.
- (b) ParadeDB `pg_search` (real BM25): only realistic with D7.1(d).
- (c) Hybrid: `tsvector` for candidate filtering, BM25 scores computed in Python (`rank_bm25`) over candidates.
- Add-on: `pg_trgm` trigram similarity for fuzzy matching of names ASR misspelled (ships with Postgres contrib), yes or no?

### D7.3 Vector index
- (a) ⭐ **No ANN index: exact scan** (`ORDER BY embedding <=> q`), which gives 100% recall and is instant at this scale. The HNSW DDL is documented as the scale path.
- (b) pgvector HNSW index anyway (to demo the production config; approximate recall is irrelevant at this size)

### D8. Fusion + reranking
- Fusion: (a) ⭐ **RRF** (k=60) · (b) weighted score fusion (min-max / z-score normalized, α tuned on dev set) · (c) Qdrant DBSF
- Reranker: (a) ⭐ **bge-reranker-v2-m3** (open) · (b) Cohere Rerank 3.5 (API) · (c) LLM-as-reranker (listwise) · (d) none (baseline)

### D9. ANN index and scale settings
- (a) ⭐ HNSW (m=16–32, ef_construct=100–200) + **int8 scalar quantization** with rescoring
- (b) HNSW + binary quantization (32× smaller; needs oversampling and rescoring)
- (c) IVF-PQ / DiskANN (billion scale, on disk)

### D10. Query understanding
- (a) ⭐ Rule/LLM filter extraction (speaker/role, date) + raw query
- (b) + query rewriting / multi-query expansion
- (c) + HyDE (hypothetical answer embedding)
- (d) None (plain query)

### D11. Output layer
- (a) Search results only (snippet + speaker + timestamp + audio player)
- (b) ⭐ Search results + optional LLM answer with timestamp citations (RAG)

### D12. App / serving stack
- Backend: (a) ⭐ Python FastAPI · (b) Python scripts + notebook only
- UI: (a) ⭐ Streamlit / Gradio (fast) · (b) Next.js/React (polished) · (c) CLI only

### D13. Compute
- (a) Local GPU (which one? how much VRAM?)
- (b) CPU-only local (use smaller models / API ASR)
- (c) Cloud GPU (Colab/Kaggle/RunPod) for batch ingestion, local for serving

---

## 6. Decision Register (LOCKED decisions only)

> **Scope: time-boxed working PROTOTYPE** (6 recordings, a few hundred chunks, CPU-only, about a one-hour build).
> The production track is listed per decision as "Production alternative". Those are **not locked**.

### 6.1 Locked decisions

**L1 — Overall approach: Approach A (Postgres only)** · 2026-09-26
- **Why:** one transactional DB handles metadata, lexical full-text search and vector search. At 6 recordings and a few hundred chunks, the system stays simple, fully understandable, fast to build and defensible in the time window, with zero infrastructure risk.
- **Why not the others:** B (OpenSearch) is heavy JVM ops for no gain at this scale. C (Qdrant) needs a second store for metadata and has weaker keyword search. D (managed APIs) costs money, sends data out, and shows less engineering. E (ColBERT/Vespa) is overkill with a steep learning curve. F (LLM enrichment) adds cost, latency and new failure modes.
- **Production alternative:** Postgres stays the system of record. Add HNSW at 1M+ chunks. Move retrieval to Qdrant/OpenSearch (or shard with Citus) at 10–50M+ chunks or at high QPS.

**L2 — D7: Postgres + pgvector + Postgres full-text search** · 2026-09-26
- **Why:** follows from L1.
- **Why not the others:** see L1.

**L3 — D1: Data = (b) synthetic: LLM-written scripts + TTS** · 2026-09-26
- **Why:** ground truth comes for free: exact transcript, speaker per turn, and which turn holds which fact. That enables real WER/DER and retrieval metrics without hand labeling. Fully controllable (topics, names, numbers for keyword tests) and no licensing issues.
- **Why not the others:** (a) public corpora (CALLHOME/Switchboard) are LDC-licensed/paid and slow to obtain. (c) podcasts have no ground truth, so no measurable evaluation. (d) mix: no time for it.
- **Known trade-off:** TTS audio is cleaner than real calls (no overlap, noise or accents), so WER/DER will be optimistic. State this in the report.
- **Production alternative:** real recordings plus a hand-corrected, labeled subset as the golden set.

**L4 — D2: ASR = (b) faster-whisper alone** · 2026-09-26
- **Why:** CTranslate2 int8 runs well on CPU, pip-installs cleanly on Windows, and gives segment timestamps (and word timestamps if needed). Accurate enough on clean TTS audio.
- **Why not the others:** (a) WhisperX adds a wav2vec2 alignment model and a heavier dependency tree, and its benefit (precise word timing) matters little when chunks are turn-level. (c) NeMo Parakeet/Canary is a painful install on Windows and GPU-oriented. (d) APIs (Deepgram/AssemblyAI) cost money and send data out.
- **Production alternative:** WhisperX or Parakeet on GPU, or a managed ASR API. Add custom vocabulary / initial prompt for domain names.

**L5 — D3: Diarization = Resemblyzer speaker embeddings + KMeans (k=2)** · 2026-09-26
- **Why:** no model-access tokens or accounts. Lightweight and CPU-friendly. With exactly two speakers, k=2 clustering of per-segment voice embeddings is simple, explainable and works on distinct TTS voices.
- **Why not the others:** pyannote was **rejected because of auth overhead** (Hugging Face gated model, accept terms, token). NeMo is a heavy install. The API option conflicts with self-hosting and cost. Channel split doesn't apply, because synthetic audio will be mixed mono.
- **Design note:** run faster-whisper first → Resemblyzer embedding per ASR segment → KMeans(k=2) → label segments SPEAKER_A/B. Segments that contain a speaker change can be mislabeled, which is acceptable on TTS audio with clean turn boundaries.
- **Production alternative:** pyannote 3.1 / NeMo Sortformer (handles overlap and speaker changes inside a segment), or channel split for stereo call audio.

**L6 — D4: Chunking = speaker-turn chunks (merge consecutive same-speaker segments, gap ≤ 1 s, max 30 s)** · 2026-09-26
- **Why:** a chunk is one natural speaker turn, so attribution is unambiguous (every chunk has exactly one speaker, which makes speaker filtering exact). The ≤ 1 s gap merges fragments of one turn, and the 30 s cap keeps chunks within the embedding model's input limit (MiniLM ~256 tokens ≈ 30 s of speech).
- **Why not the others:** (a) per-utterance is too fragmented for semantic embedding. (c) Q/A pairs mix two speakers in one chunk. (d) topic segmentation is more complex and fragile on a few hundred chunks.
- **Known trade-off:** a question and its answer land in different chunks. Mitigate at display time by returning neighboring turns.
- **Production alternative:** multi-turn sliding windows / parent–child chunks with a contextual header.

**L7 — D5: Embeddings = (d) small/fast MiniLM (sentence-transformers)** · 2026-09-26
- **Why:** 384 dimensions, ~80 MB, fast on CPU, one-line install, and good enough for a few hundred chunks.
- **Why not the others:** (a) BGE-M3 (~2.2 GB) and (b) Qwen3-Embedding are slow on CPU and heavy to download. (c) API models cost money and need keys and network access.
- **Production alternative:** BGE-M3 / Qwen3-Embedding / an API model, chosen by eval on real data.

**L8 — D6: Keyword = (a) classic lexical via Postgres `tsvector`** · 2026-09-26
- **Why:** built into Postgres (in line with L1), with no extra model or extension.
- **Why not the others:** (b) BGE-M3 sparse and (c) SPLADE need an extra heavy model and a sparse-vector store, which conflicts with L1 and L7.
- **Production alternative:** true BM25 (ParadeDB pg_search / OpenSearch) and/or learned sparse (SPLADE). Add `pg_trgm`/phonetic matching for ASR-misspelled names.

**L9 — D7.1: Postgres hosting = Docker, plain `pgvector/pgvector` image (no ParadeDB)** · 2026-09-26
- **Why:** a reproducible one-command setup, pgvector preinstalled, and the same image works everywhere.
- **Why not the others:** `pgserver` is less standard and Windows support is unverified. A managed free tier (Supabase/Neon) needs signup and sends data out. A native install needs pgvector compiled on Windows. ParadeDB is not needed, because of L8/L10.
- ⚠️ **Setup risk:** Docker is **not installed** on this machine (checked 2026-09-26). It needs Docker Desktop + WSL2 (possibly a reboot). Do this **before** the timed window.
- **Production alternative:** managed Postgres (RDS / Cloud SQL / Supabase) with pgvector.

**L10 — D7.2: Lexical ranking = `tsvector` + GIN index + `ts_rank_cd`** · 2026-09-26
- **Why:** zero extensions, supports phrase and prefix queries, and the GIN index is standard.
- **Why not the others:** (b) pg_search needs ParadeDB (rejected in L9). (c) Python-side BM25 splits ranking across two places.
- **Known trade-off:** `ts_rank_cd` is not BM25 (no IDF or length saturation), which is acceptable at this scale. Fusion (L12) uses **ranks, not scores**, so score calibration doesn't matter.
- **Production alternative:** true BM25.

**L11 — D7.3: Vector index = none (exact scan)** · 2026-09-26
- **Why:** at a few hundred chunks a brute-force `ORDER BY embedding <=> q` takes under a millisecond and gives **100% recall**, so there is nothing to tune.
- **Why not the others:** an HNSW index gives approximate recall and tuning work for no benefit at this size.
- **Production alternative:** pgvector HNSW (m=16, ef_construction=64–200, halfvec/int8) above ~100k–1M chunks.

**L12 — D8: Fusion = (a) RRF (k=60); reranker = (d) none** · 2026-09-26
- **Why:** RRF is parameter-free, needs no score normalization (important because `ts_rank_cd` and cosine scores are on different scales), is a standard strong baseline, and can be written in one SQL CTE or a few lines of Python.
- **Why not the others:** weighted fusion needs a tuned α and normalization, and there's no dev set yet. DBSF is Qdrant-specific. Rerankers (bge-reranker, Cohere, LLM) are slow on CPU or cost money, and the time box doesn't allow them.
- **Production alternative:** a cross-encoder reranker (bge-reranker-v2-m3 / Cohere Rerank) over the fused top 50, and α tuned on the golden set.

**L13 — D12: Backend = FastAPI; UI = Streamlit/Gradio** · 2026-09-26
- **Why:** all Python, so there's no context switch. FastAPI gives a clean `/search` API with automatic docs. Streamlit/Gradio gives a UI (with audio playback) in about 50 lines.
- **Why not the others:** scripts or a notebook alone isn't a demoable product. Next.js/React takes too long for the time box. A CLI alone is a weaker demo.
- **Sub-choice still open:** Streamlit vs Gradio (decide during development).
- **Production alternative:** FastAPI behind a gateway, with a React/Next.js frontend.

**L14 — D13: Compute = (b) CPU-only local** · 2026-09-26
- **Why:** matches the actual hardware (no NVIDIA GPU, checked). Six short recordings are cheap to process on CPU.
- **Why not the others:** a local GPU doesn't exist. Cloud GPU adds setup and data transfer overhead for no gain at this scale.
- **Consequence:** use smaller faster-whisper models (e.g. `small`/`base`, int8).
- **Production alternative:** GPU batch workers for ingestion, CPU for query serving.

**L15 — O5: UI = Streamlit** · 2026-09-26
- **Why:** page-style layout suits a search box plus a result list, and `st.audio(..., start_time=)` jumps playback to a result's timestamp. Simplest option for this UI.
- **Why not Gradio:** built around demo components; laying out a ranked list of results is less flexible.
- **Production alternative:** React/Next.js frontend (see L13).

**L16 — O2: No TTS package; the user supplies the 6 recordings** · 2026-09-26
- **Why:** the user will paste the audio files into `data/audio/`, so there's nothing to generate.
- **Why not edge-tts / pyttsx3:** unnecessary dependencies.
- **Note:** L3 ground truth (exact transcript, speaker per turn) is only available if the user also provides the scripts used to make the audio. Ask when evaluation starts.

**L17 — Python environment: global Python 3.12 (no venv)** · 2026-09-26
- **Why:** user's choice. Simplest option, with no activation step.
- **Why not a project `.venv`:** the user preferred global. Trade-off: possible version conflicts with other projects on this machine. `requirements.txt` keeps the setup reproducible.
- **Production alternative:** a pinned venv/lockfile, or a container image.

**L18 — Build strategy: S4 (pipeline order with file checkpoints, plus an early database smoke test once Docker is up)** · 2026-09-26
- **Order:** C1 ASR → C2 diarization → C3 turn merge/chunks → C4 embeddings → C5 database schema + load → C6 search (lexical, vector, RRF) → C7 API → C8 UI → C9 eval. Insert the database smoke test as soon as Docker runs. Build one small component at a time.
- **Why:** ASR is the slowest step and feeds everything else, and it's the only step not blocked right now. Caching each stage's output avoids re-running slow steps. The early smoke test catches Docker/pgvector problems early.
- **Why not the others:** S1 surfaces database problems late. S2 and S3 need Docker immediately (blocked). S3's fake data hides ASR and diarization problems, which are the main risk.

**L19 — O3 procedure: benchmark `base` vs `small` on ONE recording, then the user picks the size for all 6** · 2026-09-26
- **Why:** CPU speed on this machine is unknown, so the choice should rest on measured time and quality, not a guess.
- **Why not pick directly:** `small` blind could be too slow; `base` blind could miss names and numbers.

**L20 — Intermediate outputs: JSON files under `data/processed/<stage>/`** · 2026-09-26
- **Why:** easy to inspect and diff. Later stages re-run without redoing ASR. Not blocked on Docker.
- **Why not straight into Postgres:** blocked until Docker works, and harder to inspect while iterating.

**L21 — No ground-truth scripts for the 6 recordings** · 2026-09-26
- **Consequence:** WER/DER can't be computed automatically. Eval (C9) will use hand-checked samples plus LLM-generated queries with manual relevance labels. L3's "free ground truth" benefit does not apply.

**L22 — O3: Whisper model = `small` (int8, CPU)** · 2026-09-26
- **Why:** the benchmark on rec01 showed `small` fixes keyword-critical terms ("on-call", "lint", "write-up", "2.20") that `base` gets wrong, and keyword search (L8) can't recover from misspelled tokens. It's a one-time cost of ~40 min, cached to JSON (L20).
- **Why not `base`:** ~2.9× faster, but it misspells domain terms, so keyword queries return nothing for them.

**L23 — Schema decisions (O11–O19)** · 2026-09-26
| ID | Choice | Why | Why not the alternative |
|---|---|---|---|
| O11 | **2 tables:** `recordings` + `chunks` | Recording-level fields (hash, status, duration) are stored once; chunks reference them | A flat table repeats recording fields on every chunk and has nowhere natural for status |
| O12 | **`english` text-search config** | Stemming ("refunds" ↔ "refund") and stopword removal help conversational queries | `simple` is exact-only, so recall drops on word variants; two columns are over-engineering for now |
| O13 | **Raw speaker label only (`A`/`B`)** | Comes straight from diarization, with no extra model | `role` needs LLM role labeling, which isn't in scope |
| O14 | **No segment table; chunks only** | Raw ASR segments already live in JSON (L20) | A DB copy is duplication with no query use |
| O15 | **Status: `processing` / `completed` / `failed`** | Ingestion is synchronous, so there is no queued state | `pending` only matters with an async queue |
| O16 | **`error TEXT`** | Records why a recording failed | Without it a `failed` row isn't debuggable |
| O17 | **`created_at` / `updated_at`** (default `now()`; ingest code sets `updated_at` on every status change) | Shows timing and spots stuck `processing` rows | A trigger is extra machinery for a single writer |
| O18 | **Re-ingest rule by `content_sha256`:** completed → skip, failed → retry, processing → skip | Idempotent: the same audio is never processed twice, and failures are recoverable | "Always reprocess" wastes ASR time and churns rows |
| O19 | **`id` = file stem; `content_sha256` UNIQUE** | Readable IDs in results and logs; the hash still guarantees idempotency | A hash as the ID is unreadable in the UI, logs and joins |

- **Production alternative:** `pending` state plus a job queue, a status history table, and `processing` rows with a lease/timeout so a crashed worker's rows get retried.

**L24 — Ingestion error-handling rules (user-specified)** · 2026-09-26
1. **Truncate `error` text to 2000 chars** in the ingestion code before saving. **Why:** full stack traces would bloat the DB. **Why app-level, not a DB CHECK:** a CHECK would *reject* the write and lose the failure record, whereas truncation keeps it. *To implement in C5 (ingest code doesn't exist yet).*
2. **Same file while `processing` → skip** (restates the O18 rule in L23). **Why it's safe:** ingestion runs **sequentially, not concurrently**, so no two runs race on the same hash.
3. **A recording stuck in `processing` (crash mid-run) is NOT auto-recovered.** Documented as a known limitation (§6.3). Don't build auto-retry.
4. `updated_at` already existed (added with O17 in L23). Stuck rows are visible by age (see §6.3).

### 6.3 Known limitations (prototype)
| # | Limitation | Impact | Manual workaround | Production fix |
|---|---|---|---|---|
| K1 | A crash mid-ingest leaves the recording in `processing` forever. The O18 rule skips it, so it's never retried. | That recording stays unsearchable until someone intervenes | Find it: `SELECT id, updated_at FROM recordings WHERE status='processing' AND updated_at < now() - interval '30 minutes';` Then `DELETE` the row (chunks cascade) or set `status='failed'` and re-run ingestion. | Lease/heartbeat column + timeout → auto-mark `failed` → retry, or a job queue with visibility timeouts |
| K2 | Skipping on `processing` assumes **sequential** ingestion | Concurrent workers could race on the same hash (the UNIQUE constraint would reject the second insert, but not cleanly) | Don't run ingestion in parallel | `INSERT … ON CONFLICT` + `SELECT … FOR UPDATE SKIP LOCKED` job claiming |

**L25 — Project structure: layered packages inside `app/` (pipeline / db / search / api / ui), clearer stub names, and a README** · 2026-09-26
- **Why:** the folders follow the system flow (offline pipeline → DB → online search → API → UI), so anyone can find things. `config.py` stays at `app/config.py`, so there are **zero code changes** (all imports are `from app.config`).
- **Why not a `src/` layout:** an extra nesting level, only useful for installable packages. **Why not top-level folders without `app/`:** generic names (`db`, `search`) can collide with installed libraries.
- **Renames:** `db.py→db/connection.py`, `search.py→search/hybrid.py`, `api.py→api/main.py`, `ui.py→ui/streamlit_app.py`. Stage files for C3/C4/C9 are created only when built (no empty placeholders).

**L26 — O4: embed with BOTH `all-MiniLM-L6-v2` and `multi-qa-MiniLM-L6-cos-v1`; pick the winner by evaluation** · 2026-09-26
- **Why:** an empirical choice beats a guess. Both are 384-dim and ~1 s per recording on CPU, so running both costs almost nothing. Vectors are kept apart: `data/processed/embeddings/<model>/<rec>.npz` (`vectors`, `chunk_index`), L2-normalized so pgvector cosine (`<=>`) is correct.
- **Why not pick one now:** general vs question→passage training is exactly the kind of thing the C9 eval should decide.

**L27 — Logging in all code (user request)** · 2026-09-26
- **What:** `app/log.py` → `get_logger(name)` (stdlib `logging`). Every `app.*` logger writes to **console + `logs/app.log`** (on D, gitignored, rotating 5 MB × 3). Format: `time LEVEL [module] message`. Each per-recording step is wrapped in `try/except → log.exception(...recording...) → raise`, so a failure records **which recording** and the **full traceback**, while behavior stays the same (the run still stops on the first error). `print`s were replaced by `log.info`. The DB URL is never logged (it contains the password).
- **Why stdlib logging:** zero dependencies and standard. **Why not structlog/loguru:** an extra dependency for no gain at this size.
- **Verified:** re-running chunking produced byte-identical outputs; the log file is written.
- **Production alternative:** JSON logs → a central log store (Loki/ELK/CloudWatch), with a request/recording ID for correlation.

**L28 — Embedding model = `all-MiniLM-L6-v2` (supersedes L26)** · 2026-09-26
- **Why:** it won the rec01 comparison on all three metrics (Hit@1 0.29 vs 0.21, Hit@3 0.57 vs 0.43, MRR 0.441 vs 0.399), and it was already cached. The single DB column `embedding VECTOR(384)` is unchanged. Set as `EMBED_MODEL` in config.
- **Why not `multi-qa-MiniLM-L6-cos-v1`:** lower on every aggregate metric here, despite winning 5 individual queries. The margin is small, so it's worth re-checking on the full eval (C9). Swapping = re-run embed + ingest (seconds).
- **Production alternative:** choose by the full golden-set eval, and consider larger models (bge / e5 / Qwen3-Embedding).

**L29 — Git configuration** · 2026-09-26
- Identity: the user's existing global git config. **Branch `main`** (renamed from `master` before the first commit). **`.gitattributes`: `* text=auto eol=lf`, `*.wav`/`*.npz` binary**, plus local `core.autocrlf=false`, so the repo stores LF consistently across Windows/Mac/Linux/Docker. **Remote: `origin` = https://github.com/amolgupta7/G2-Hackathon.git** (reachable, empty); pushing only on the user's go-ahead.
- **Why not `master` / default line endings / local-only:** `main` is the current hosting default; default Windows autocrlf causes CRLF/LF diff noise; the user wants a GitHub remote.

**L30 — Commit policy (user rules)** · 2026-09-26
1. **Small commits, never everything at once.** 2. **Dependency order**, one layer per commit, so any single commit can be reverted independently. 3. **Only fully developed files.** Stubs (`app/search`, `app/api`, `app/ui`) are committed when built. `data/processed/` waits on G1.
- **Initial sequence:** setup → config + logging → DB → C1 asr → C2 diarize → C3 chunk → C4 embed → C5 ingest → eval → docs.
- **Why:** a failure or bad change in one layer (e.g. ingest) can be `git revert`-ed without touching the others; the history mirrors the pipeline (L18).

**L31 — C6 search details** · 2026-09-26
| Choice | Why | Why not the alternative |
|---|---|---|
| **Keyword = AND via `websearch_to_tsquery('english', q)`** (supports `"phrase"`, `or`, `-exclude`) | Precise keyword hits; users get search-engine syntax | OR-of-terms has higher recall but noisier. Trade-off: long natural-language questions may get 0 keyword hits, so RRF falls back to the vector arm alone |
| **Filters: optional `speaker` + `recording_id`, applied to both arms before fusion** | Speaker-scoped queries are a core requirement (Q2); per-recording search for the UI | Speaker-only / none lose that |
| **RRF in Python** (two SQL queries) | Per-arm ranks are kept and logged, which helps debugging and eval ablations (keyword-only vs vector-only vs hybrid) | Single SQL CTE is one round-trip, but per-arm ranks are harder to inspect |
| **Neighbor context: previous + next turn** returned with each hit | Mitigates the L6 trade-off (a question and its answer sit in adjacent chunks) | Hit-only hides the answer |
| Defaults (not user-locked, parameters): **K = 50 per arm, N = 10 results, RRF k = 60** | Standard values; K = 50 is ~15% of 344 chunks | — |

**L32 — O25: a speaker filter requires a recording filter** · 2026-09-26
- **What:** `search()` raises `ValueError` if `speaker` is set without `recording`; the CLI rejects `--speaker` without `--recording`. The 6 speaker queries in `dataset/queries.json` got a `recording` field (derived from their own labels; labels unchanged).
- **Why:** diarization labels `A`/`B` are assigned independently per recording, so "speaker A" only identifies a person within one recording. Run 1 showed cross-recording pollution (q20 #1 came from rec01).
- **Why not the alternatives:** LLM role labels (agent/customer) mean extra model cost and scope; leaving it as-is gives wrong results by design.
- **Effect (run 2):** speaker Hit@5 0.500 → 0.833; hybrid overall Hit@5 0.800 → 0.867, nDCG 0.635 → 0.665.

**L33 — O26: secondary "+nb" metrics (Hit@1+nb, Hit@5+nb, MRR@10+nb); the strict rule is unchanged** · 2026-09-26
- **Why:** the UI shows prev/next turns, so a hit next to the answer (e.g. the question turn) puts the answer on screen. Reporting it separately keeps the strict numbers honest while showing the user-visible quality (hybrid Hit@5+nb 0.933).
- **Why not relabel:** that would change the ground truth and make runs incomparable.

**L34 — O27: keyword operator stays AND (user decision)** · 2026-09-26 — the better overall results (below). OR remains available as `keyword_op="or"` for experiments.

**L35 — C7 API design (user decisions)** · 2026-09-26
| Choice | Why | Why not the alternative |
|---|---|---|
| **Endpoints: `GET /search`, `GET /recordings`, `GET /health`** | `/recordings` feeds the UI's recording dropdown (id, duration, status, chunk count); `/health` checks the DB | search-only makes the UI query the DB itself; audio streaming isn't needed (Streamlit reads local files) |
| **GET with query params** | cacheable, shareable URLs, easy to test in the browser and `/docs` | POST body isn't cacheable or shareable |
| **Fixed hybrid + AND; no `mode`/`keyword_op` params** | the API serves the chosen config; experiments stay in eval/CLI; smaller surface | exposing knobs invites misuse |

**L36 — O28: DB connection pool in the API (user decision)** · 2026-09-26
- **What:** `app/db/connection.create_pool()` → `psycopg_pool.ConnectionPool(min=1, max=5, configure=register_vector, open=False)`. `app/api/main.py` creates **one module-level pool**, opens it once in lifespan (`wait(timeout=10)`; if the DB is down the app still starts and the pool retries), and closes it on shutdown. A **`get_db()` dependency** borrows a connection per request (`with pool.connection()`); all 3 endpoints use it, and `/search` passes it to `hybrid.search(conn=...)`. `PoolTimeout` → 503, like `OperationalError`. `/health` also reports pool size/available. New dependency `psycopg-pool` 3.3.3 (installed `--no-cache-dir`; added to requirements.txt). Offline scripts (pipeline/eval) keep the plain `connect()`.
- **Why:** the user asked for engine/SessionLocal-style reuse, created once at startup, never per request (the psycopg equivalent of SQLAlchemy's engine is the pool).
- **Why not one shared connection:** it would serialize concurrent requests.
- **Measured:** DB time per request **50–83 ms → 3–5 ms** (keyword arm incl. connection acquire). 30 requests → **5 pooled DB connections** reused (`pg_stat_activity`). 20 concurrent requests all OK (1.18 s wall total). Sequential server time p50 ~60 ms, now dominated by query embedding (47–150 ms; O32). The first request after startup is cold (~1.5 s; O31).

**L37 — O31: warm up at FastAPI startup, not on the first request (user decision)** · 2026-09-26
- **Finding:** the model was already loaded in lifespan. The ~1.5 s cold first request (`embed=462ms keyword=845ms vector=133ms`) came from the **first `encode()`** (PyTorch kernel/thread init) and the **first SQL on a fresh pooled connection** (plans, pgvector types, cold cache), not from model loading.
- **Fix:** after loading the model, lifespan runs one real `hybrid.search("warm up")` through the pool. If the DB is down it falls back to an embed-only warm-up and logs a warning, so startup never fails because of it.
- **Measured:** startup (lifespan) 7.5 s; **first `/search` after startup 1467 ms → 26.7 ms**; next requests 25–35 ms server-side (the earlier 47–150 ms embed times weren't reproduced once warm → O32).
- **Why not lazy loading / warm on the first request:** the first real user pays the cost.

**L38 — C8 UI design (user decisions)** · 2026-09-26
| Choice | Why | Why not the alternative |
|---|---|---|
| **UI calls the FastAPI** (`API_URL`, default `http://127.0.0.1:8000`) | Thin client, matches the architecture, reuses the pool and warm model | Importing `search()` duplicates model loading and DB access and bypasses the API |
| **Audio: full WAV with `st.audio(start_time=turn start)`** | The listener can keep going past the turn for context | A clip (±2 s) is lighter but cuts context; tracked as U1 in ISSUES |
| **No debug info** (score, per-arm ranks) | Clean end-user view | Debug expander was declined |

**L39 — O29 fix: `-exclude` enforced on the fused results (user spec)** · 2026-09-26
- **What:** `excluded_terms(query)` parses websearch-style exclusions (`-word`, `-"phrase"`; the `-` must follow whitespace or the start, so `on-call`/`self-serve` are not exclusions). In `search()`, after RRF the **whole fused list** is filtered (drop any chunk whose lowercased text contains an excluded term, whichever arm it came from), **then** the top-N is taken, so N results are still returned when enough remain. The log records `excluded=[…] dropped=N`. The API/UI get it automatically.
- **Why after fusion, before top-N:** filtering the top-N afterwards would return fewer than N results; filtering only the vector arm would still let fusion change ranks.
- **Verified:** parser 7/7 edge cases; `refund` → 3/10 contain "invoice", `refund -invoice` → **0/10** (5 dropped, still 10 results); `-"account credit"` and `billing -refund` → 0 leaks; the eval's quality metrics are **identical to run 2** (no exclusions in the eval set; `eval/results.json` restored to the committed version); API end-to-end `refund -invoice` → 0 leaks.
- Follow-ups are tracked in ISSUES.md under O29 (O29.1, O29.2).

**L40 — O30 fix: unknown recording → 404 (user spec)** · 2026-09-26
- **What:** in `GET /search` (the only recording-filtered endpoint), after the speaker-needs-recording check (422) and **before running the query**, `SELECT 1 FROM recordings WHERE id = %s` on the request's pooled connection; missing → `HTTPException(404, detail="unknown recording")` (logged).
- **Why:** a typo'd recording id used to look like "no matches" (an empty 200).
- **Verified (API restarted):** `nope`, a typo with a speaker, and wrong case → 404 `unknown recording`; valid recording (± speaker) → 200 with 3 results; no recording → 200; speaker without recording → still 422. Follow-up tracked in ISSUES.md under O30 (O30.1).

**L41 — O23 fix: minimum vector similarity 0.3 before RRF; explicit empty results (user spec)** · 2026-09-26
- **What:** `VECTOR_SQL` adds `(embedding <=> qvec) <= 1 - min_vector_sim` (vectors are unit-length, so distance = 1 − cosine). `search(min_vector_sim=0.3)` (constant `MIN_VECTOR_SIM`), applied before fusion; the keyword arm is unchanged. If both arms are empty, the result is `[]` → API `count: 0` → UI "No matches" (no padding). Eval: `--min-vector-sim` (−1 = off) and `--out` flags; new `dataset/negative_queries.json` (5 gibberish + 5 off-topic); reports true-negative rate and zero-result rate.
- **Why before fusion:** a weak vector hit shouldn't earn any RRF credit.
- **Measured (hybrid, vs no threshold):** TNR **0.0 → 0.9** (off-topic 1.0, gibberish 0.8); Hit@1 0.600 → 0.600; **Hit@5 0.867 → 0.800; Recall@10 0.733 → 0.660**; MRR 0.724 → 0.699; nDCG 0.665 → 0.620; zero-result rate on labeled queries 0 → 0.033 (q20). The regression is **entirely in speaker queries** (Hit@5 0.833 → 0.500): role-word questions score 0.12–0.27 against their correct turns. Paraphrase is unchanged. Sweep 0.2–0.4 in EVALUATION §11.3. The baseline run (−1) reproduces run 2 exactly. API: gibberish/off-topic → `count: 0`; `refund` now 9 results (the weak 10th removed). UI (AppTest): gibberish → "0 results" + "No matches" message, no audio.
- **Open:** the threshold choice/scope → ISSUES O23.1 (⭐ unscoped-only); O23.2 digit strings.

**L42 — P1 + P4: torch threads + pool size from env (user spec)** · 2026-09-26 · 🟡 built, **tests pending** (the user deferred testing to one full run after all fixes)
- **What:** config `TORCH_NUM_THREADS` (default 1), `DB_POOL_SIZE` (1), `DB_POOL_MAX_OVERFLOW` (4). `hybrid._model()` calls `torch.set_num_threads()` before loading the query embedder and logs it (offline `pipeline/embed.py` is untouched). `create_pool()` now takes no args and sizes `min = DB_POOL_SIZE`, `max = DB_POOL_SIZE + DB_POOL_MAX_OVERFLOW` (the old hard-coded args were removed from `create_pool` and `main.py`; the defaults reproduce the old 1/5). API startup logs the pool size.
- **Baseline to compare against (before the change, 3 bursts × 20 concurrent):** client p50 497 / p95 897 ms; server `took_ms` p50 144 / p95 241 ms; torch default 2 threads on 4 logical cores. Load test script: scratchpad `load_test.py`.

**L43 — K1 + K2: stale-processing recovery + SKIP LOCKED claiming (user spec)** · 2026-09-26 · 🟡 built, **tests pending**
- **What:** schema `recordings.started_at TIMESTAMPTZ` (+ `ALTER TABLE … ADD COLUMN IF NOT EXISTS`; migration applied, data intact). Ingest startup `recover_stale()`: `processing` rows with `COALESCE(started_at, updated_at)` older than `INGEST_STALE_MINUTES` (default 30) → `failed` with an explanatory error → retried by the existing rule. Claim: `SELECT … FOR UPDATE SKIP LOCKED` → completed/processing = skip; failed = retry (`started_at = now()`); none = `INSERT … ON CONFLICT (content_sha256) DO NOTHING RETURNING id` (no row returned = another worker has it → skip). Every claim path commits immediately; the chunk-load phase is unchanged.
- **Logs per flow step:** `ingest run … stale_timeout`, `stale recovery: none | marked failed for retry: [...]`, `claim <rec>: new | retry | skip, same content already … | skip, claimed by another worker`, then the existing `loaded …` / `ingest failed …`.
- **Why these choices:** the smallest change that keeps the existing skip/retry rules (L23/L24). `COALESCE` covers rows from before `started_at` existed.
- Follow-up in ISSUES under K1 (K1.1: no periodic heartbeat refresh).

**L44 — S1 + S3: credentials in `.env`, Streamlit on localhost (user spec)** · 2026-09-26
- **S1:** `.env` (already git-ignored) holds `POSTGRES_USER/PASSWORD/DB/HOST/PORT`, with **values unchanged** (the existing volume was initialized with them). `docker-compose.yml` uses `${POSTGRES_*:?set … in .env}` (fails loudly if missing). `app/config.py` has a minimal built-in `.env` reader (no new dependency; real env vars win) and builds `DATABASE_URL` from the vars (URL-escaped); an explicit `DATABASE_URL` still overrides; missing vars → a clear `RuntimeError`. `.env.example` (placeholders) is committed; README setup has `cp .env.example .env`. **Logs:** `DB pool target user@host:port/db` at API startup and the target in connect-failure logs; the password is never logged.
- **S3:** `.streamlit/config.toml` → `[server] address = "127.0.0.1"`, auto-loaded when Streamlit runs from the project root (no command change).
- **Quick check only (per the user, full tests later):** config loads from `.env` and queries the DB (344 chunks); `docker compose config` resolves the vars; `git check-ignore` confirms `.env` is ignored; no hard-coded password left in tracked code. Follow-up: ISSUES S1.1 (rotate the password before a public push).

**L45 — G1: commit processed JSON, ignore `.npz` (user decision)** · 2026-09-26
- **What:** `.gitignore` gets `*.npz`. Committed `f322434`: `.gitignore` + 19 JSON files (7 ASR incl. the rec01 `base` benchmark, 6 diarization, 6 chunks). Only these were staged; all other uncommitted work stays uncommitted (the user holds those commits).
- **Why:** the DB can be rebuilt without ~40 min of CPU ASR; the `.npz` embeddings are binary and regenerated in seconds.

**L46 — O23.1 fix: similarity floor only for unscoped searches (user spec)** · 2026-09-26
- **What:** in `search()`, `threshold = None if recording else min_vector_sim`; `None` → `max_dist = 2.0` (the full cosine-distance range, i.e. no filter), otherwise `1 − threshold`. The log shows `vector_hits(threshold=0.30)` or `vector_hits(threshold=off:recording-scoped)`. No other code changed.
- **Why:** role-word speaker questions have low absolute similarity (0.12–0.27) to correct turns, and speaker queries always carry a recording (L32). All gibberish/off-topic negatives are unscoped.
- **Eval run 4 (targets met):** hybrid Hit@5 **0.867** (= run 2), zero-result rate **0.000**, TNR **0.9** (off-topic 1.0, gibberish 0.8, only `1234 5678 9012` = O23.2). Speaker type is identical to run 2 (Hit@5 0.833, R@10 0.778). Remaining cost: Recall@10 0.733 → 0.727 and nDCG 0.665 → 0.660 (one unscoped keyword query loses an on-topic chunk below 0.3). EVALUATION §11.4; `eval/results.json` = run 4. Follow-up ISSUES O23.3 (gibberish inside a recording isn't filtered, by design).

**L47 — O23.2, O29.1, O29.2, O30.1 (user spec)** · 2026-09-26 · 🟡 built, **untested** (the user deferred tests)
- **O23.2:** `MIN_VECTOR_SIM_NO_LETTERS = 0.4`; if the positive query text has no letters, the unscoped floor is `max(min_vector_sim, 0.4)` (scoped searches remain off, L46). The log shows `threshold=0.40(no-letters)`. Note: the earlier sweep showed `1234 5678 9012` still had 1 hit at 0.4.
- **O29.1:** `positive_text(query)` removes `-word` / `-"phrase"`; only that text is embedded (the keyword arm and the post-filter still use the full query). An exclusion-only query → no embedding, vector arm skipped. Logged as `embedded=…`.
- **O29.2:** the post-filter uses `re.compile(rf"\b{re.escape(term)}")` on the lowercased text (word-boundary prefix) instead of a substring.
- **O30.1:** UI `fetch_search`: a 404 → `fetch_recordings.clear()`, log warning, show the API `detail` + "recording list has been refreshed" (via the existing warning path).

**L48 — R1 reranker + R3 configurable RRF (user spec)** · 2026-09-26 · 🟡 built, **untested**
- **R1:** the user chose `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB; `bge-reranker-v2-m3` was rejected for its ~2.2 GB size, RAM and CPU latency) and **on by default**. `search(rerank=RERANK)` (env `RERANK`, default 1): after fusion and exclusion, the top `RERANK_TOP` (50) are scored by `CrossEncoder.predict((positive query, turn text))` and reordered; the tail is kept; it only reorders and never drops; skipped for exclusion-only queries. `_reranker()` is lazy and cached; the API warm-up search loads it at startup (the model downloads to `.local/huggingface` on first start). Log: `rerank=on(N)` + `rerank=Xms`.
- **R3:** `rrf(ranked_lists, k, weights)`; `search(rrf_k=RRF_K, keyword_weight=RRF_KEYWORD_WEIGHT)` from env (defaults 60 / 1.0 = unchanged); hybrid weights = [keyword_weight, 1.0]. The local `RRF_K = 60` constant in hybrid.py was removed (now from config). Eval flags `--rerank on|off`, `--rrf-k`, `--kw-weight` for the sweep in the full run.
- **Interaction:** with rerank on, the final order is decided by the cross-encoder, so R3 weights only affect which candidates make the top 50 (and the order when rerank is off).

**L49 — O22: ingest log-and-continue (user spec)** · 2026-09-26 · 🟡 built, **untested**
- **What:** in `ingest.main()`, each `ingest(conn, f)` is wrapped in `try/except`: on an exception → `conn.rollback()` (clean transaction for the next file), the stem is added to `failed`, `log.warning("continuing after failure: …")`, and the loop continues. `ingest()` itself still logs the traceback and marks the row `failed` (L24). Summary: `done: loaded=… skipped=… failed=N [stems]; chunks in DB=…`; **exit code 1** if any failed, so scripts/CI still see the failure.
- **Why:** one bad recording shouldn't block the rest (supersedes the earlier "stop at first failure" behavior, O22).
- **Note:** a pre-claim failure (e.g. a missing chunks file) doesn't create a DB row (unchanged behavior); it's only in the log and the summary.

**L50 — Full verification pass (user request)** · 2026-09-26 — every §0 item tested with its log line.
| Item | Result | Evidence (log / measurement) |
|---|---|---|
| O29 | ✅ | `refund -invoice` → 0/6 contain "invoice" (plain `refund`: 3/10); `excluded=['invoice'] dropped=3`. 6 results, not 10, because the O23 floor leaves only 9 vector candidates (expected interaction) |
| O29.1 | ✅ | `embedded='refund'`; `-invoice` alone → `embedded=''`, vector arm skipped, 10 keyword results, 0 leaks |
| O29.2 | ✅ | `start -art` keeps all 10 "start" turns, `dropped=0`; `-invoices` correctly leaves singular "invoice" (prefix semantics) |
| O30 | ✅ | `recording=nope` → 404 `unknown recording`; log `GET /search unknown recording='nope' -> 404` |
| O30.1 | ✅ | UI warning "unknown recording: the recording list has been refreshed"; log `recordings cache cleared`; the deleted recording is gone from the dropdown on the next run |
| O23 / O23.1 | ✅ | gibberish/off-topic unscoped → 0 (`threshold=0.30 … -> 0 results`); speaker query in rec04 → 10 results (`threshold=off:recording-scoped`), q20 relevant at rank 4 |
| O23.2 | ❌ | `1234 5678 9012` → 1 hit (`threshold=0.40(no-letters)`); the `447129` turn scores 0.403 |
| P1 | ✅ | 3×20 load, rerank off: client p50 497 → **295** / p95 897 → **685** ms; server p50 144 → **129** / p95 241 → **194** ms; log `torch threads=1` |
| P4 | ✅ | defaults `pool min=1 max=5`; env `DB_POOL_SIZE=2 DB_POOL_MAX_OVERFLOW=3` → `pool min=2 max=5`, `/health` pool size 2 |
| K1 | ✅ | `stale recovery: marked failed for retry: ['rec06_climate_policy']` → `claim … retry` → `loaded … 49 chunks` |
| K2 | ✅ | two parallel workers: one `claim … retry`, the other `skip, claimed by another worker`; loaded once (64 rows) |
| O22 | ✅ | `could not start ingest for rec02` → `continuing after failure: rec02_hr_burnout` → `done: … failed=1` exit 1; after the restore, `retry` → loaded |
| S1 | ✅ | `DB pool target postgres@localhost:5432/transcripts (credentials from env/.env)` |
| S3 | ✅ | Streamlit prints only `URL: http://127.0.0.1:8501`; socket bound to 127.0.0.1 |
| R1 | ✅ built / ⚠ latency | Hit@1 0.600 → 0.733, Hit@5 0.867 → 0.933, MRR 0.724 → 0.823, nDCG 0.660 → 0.691; sequential p50 24 → 210 ms, p95 29 → 1080 ms; load p50 2253 / p95 4521 ms (client) |
| R3 | ✅ | kw_weight 1.5 → keyword Hit@1 0.846 → 0.923 (fixes q16), no paraphrase/speaker change; k 10/30 no effect |
- **Findings (ISSUES §0):** O23.2 fail; R1.1 warm-up doesn't load the reranker; R1.2 rerank latency under load; R3 weight choice; X2 `compare_embeddings.py` caches to C; X3 RAM/page-file pressure (C had 0.1 GB free; removed a ~174 MB duplicate model cache from C, D copies verified first).
- **End state:** all 6 recordings `completed`, 344 chunks; the rec02 chunks file restored; temp recording row deleted; `eval/results.json` = run 5 (rerank on); API and UI restarted with defaults.

**L51 — O23.2, R1.1, R1.2, R3, X2, X3 (user spec)** · 2026-09-26
- **Changes:** `MIN_VECTOR_SIM_NO_LETTERS` 0.4 → **0.45**; API lifespan calls `hybrid._reranker().predict(...)` when `RERANK` is on (the ready log shows `reranker=on(top N)`); `RERANK_TOP` default 50 → **20**; `RRF_KEYWORD_WEIGHT` default 1.0 → **1.5**; `eval/compare_embeddings.py` imports `app` before `sentence_transformers` (HF cache stays on D); README + LIMITATIONS: run heavy jobs one at a time (no code change).
- **Eval (defaults: rerank on/top 20, kw 1.5), vs run 5:** hybrid Hit@1 **0.733** (=), Hit@5 **0.933** (=), MRR **0.827** (0.823), nDCG 0.695 (0.691), Recall@10 0.741 (=); **TNR 1.0** (all 10 negatives → 0; digit query `threshold=0.45(no-letters) … -> 0 results`); misses@5 only q5, q13. O23.2 ✅, R3 ✅.
- **R1.1 ✅:** `reranker: … (top 20)` logged at 16:06:13, **before** `API ready` at 16:06:21 (`reranker=on(top 20)`).
- **R1.2 not measured yet:** the median rerank candidate count is 10 in both runs (the O23 floor keeps lists small), so the top-20 cap mainly affects recording-scoped queries (the p95 cases). The load test after the change (client p50 7183 / p95 10337 ms) is **confounded**: 0.3 GB RAM free, API startup 17.6 s. A same-conditions A/B then failed on a **test-harness bug** (unread stdout/stderr pipes blocked uvicorn; all variants, even rerank off, timed out). The harness was fixed (output to files); **the A/B was interrupted by the user and is still pending**.

**L52 — D4: same file name + different audio → clean id-conflict error (user report)** · 2026-09-26 · 🟡 built, **untested**
- **Bug:** the new-recording INSERT used `ON CONFLICT (content_sha256) DO NOTHING`, which didn't cover the `id` primary key, so a re-upload under an existing file name with new content raised a raw `UniqueViolation` (since O22 it was caught, but with a noisy traceback and an unclear message).
- **Fix:** `ON CONFLICT DO NOTHING` (no target → both unique keys, race-safe). If nothing was inserted, look up the row by `id`: a different stored hash → `log.error("claim <rec>: id conflict … existing recording left unchanged. Rename the file or delete the old recording first")` + `raise RecordingIdConflict` (re-raised without the generic traceback handler; counted as failed by the O22 loop → exit 1). Otherwise → `skip, claimed by another worker` (as before).
- **Why this behavior:** clean error + keep the existing data (the user asked for a clean error, not a silent overwrite). Follow-up found: ISSUES D4.1 (retry path after a rename).

**L53 — D5: no-speech recordings crash end-of-run stats (user report)** · 2026-09-26 · 🟡 built, **untested**
- **Bug:** `asr.py` (`sum(lengths)/len(lengths)`, `max(lengths)`) and `chunk.py` (`durs`, `words`) computed stats after writing the output; with 0 segments/chunks → `ZeroDivisionError` outside the try block → the run aborted and the remaining files were skipped.
- **Fix (stats only; processing and outputs untouched):** if the list is empty → `log.warning(... "no speech segments detected" / "no chunks (recording has no speech segments)")` and `continue` to the next file.
- Follow-ups found (not fixed, rule 7): ISSUES D5.1 (diarize k=2 with < 2 segments), D5.2 (zero-length audio → RTF division).

**L54 — D6: exclusion-only query returns empty (user report)** · 2026-09-26 · 🟡 built, **untested**
- **Bug:** `-invoice` → `websearch_to_tsquery` = `!invoice` → the keyword arm matched every chunk without "invoice" (~50 arbitrary results). L50 had recorded this behavior ("10 keyword results") as a pass; it's now treated as a bug.
- **Fix:** in `search()`, after the existing validation (mode, speaker needs recording), `positive_text(query)` empty → log `no positive terms (only exclusions) -> 0 results` and `return []` (API `count: 0`, UI "No matches").
- **Dead code removed (rule 7):** the O29.1 special case that skipped the vector arm for an empty `embed_q` (`… if embed_q else []`), the conditional query embedding (`… if embed_q else None`), and `bool(embed_q)` in the rerank condition. All are unreachable after the early return.

**L55 — D7: `mean_cos_within` excluded self-pairs (user report)** · 2026-09-26 · 🟡 built, **not re-run**
- **Bug:** `np.mean(E_c @ E_c.T)` included the diagonal (each segment vs itself = 1.0), inflating within-speaker cosine (most for small clusters). The values recorded earlier in this log and in EVALUATION §8 (0.845–0.928) are **overstated**.
- **Fix:** `_mean_pairwise_cos(e)` = `(sum − trace)/(n(n−1))` over distinct pairs; `None` (JSON `null`) if the cluster has < 2 segments. `mean_cos_between` (no self-pairs), silhouette (sklearn excludes self-distances) and **KMeans labels are untouched** (the diagnostic isn't used for clustering).
- **Docs:** EVALUATION §8 flags the old within values as inflated; the true values come from the next diarization run (ISSUES D7).

**L56 — D8: empty `recording=` → 422 instead of a silent 0 (user report)** · 2026-09-26 · 🟡 built, **untested**
- **Bug:** `recording=""` is falsy → `if recording and …` skipped the O30 existence check → `hybrid.search()` filtered `recording_id = ''` → 0 results with HTTP 200 (and the threshold scope logic treated it as unscoped).
- **Fix:** `recording: str | None = Query(None, min_length=1)` → FastAPI rejects it with a 422 before any DB work (same approach as `q`). Whitespace-only `" "` still reaches the O30 check → 404.
- **Logging:** new `RequestValidationError` handler logs `422 <method> <path>: [(field, msg)]` (no values) and returns FastAPI's **default** response via `request_validation_exception_handler` (response body/status unchanged for every existing 422).
- **Why 422, not "treat empty as no filter":** a blank filter is almost always a client bug; failing loudly beats silently widening the search.

**L57 — D9: embedding dimension checked against the DB column (user report, low)** · 2026-09-26 · 🟡 built, **untested**
- **Bug:** `VECTOR(384)` is hard-coded in schema.sql; switching `EMBED_MODEL` to a different dimension would fail deep inside pgvector with an opaque error.
- **Fix (fail fast, no schema templating):** `app/db/connection.embedding_dim(conn)` parses `format_type(...)` of `chunks.embedding` (→ 384). **Ingest:** logs `chunks.embedding is vector(N)` at the start; `load_rows(recording, db_dim)` raises a clear `ValueError` if a recording's `.npz` width ≠ N (pre-claim failure → logged, counted as failed by the O22 loop). **API lifespan:** compares `model.get_sentence_embedding_dimension()` with the column; mismatch → `log.error` + `RuntimeError` (the API doesn't start); DB unreachable → a warning and startup continues (consistent with the warm-up). schema.sql: a comment on the column.
- **Why not generate the schema from config:** `CREATE TABLE IF NOT EXISTS` wouldn't alter an existing table, so a runtime check is needed either way; a real dimension change needs a migration + re-ingest.

**L58 — D10: duplicated unsafe averaging → `statistics.fmean` (user report, low)** · 2026-09-26 · 🟡 built, **untested**
- **What:** `asr.py` (`avg_seg`) and `chunk.py` (`avg_dur`, `avg_words`) use `fmean(...)` instead of `sum(x)/len(x)`; the D5 empty-list guards (warning + `continue`) stay. The log format is unchanged. A grep confirms no other `sum(...)/len(...)` in `app/`.
- **Why not a shared helper:** two call sites, and the stdlib already provides it; a new module would be premature abstraction.

**L59 — User decisions before the first push** · 2026-09-26
- **Keep** the D6/D7/D9 fixes (the user's earlier P3 summary had listed them as "left as-is"; they were already built and are kept).
- **D4.1:** leave documented in ISSUES, no code change (rare, low impact).
- **Full verification run deferred** until the laptop is restarted (RAM exhausted); pending items are listed in ISSUES §0.
- **The repo is public:** removed the email and local user paths from the docs being committed (the git author metadata still carries the user's name/email, as with any commit). Dev DB password history → ISSUES S1.1.
- **Commit and push** after the D5.1 fix + docs update (user go-ahead).

**L60 — D5.1: diarize handles < 2 segments (user spec)** · 2026-09-26 · 🟡 built, **untested**
- **What:** if `len(segments) < 2` → skip embedding + KMeans, label all `A`, diagnostics `single_speaker: true`, `null` silhouette/within/between, `log.warning("<rec>: N speech segment(s) -> clustering skipped, single-speaker recording (all labeled A)")`. Also guarded (needed for the same goal): silhouette only if 2 ≤ clusters ≤ n−1, between-speaker cosine only if both clusters have members → otherwise `null` (exactly 2 segments previously made `silhouette_score` raise). Normal recordings: same labels/metrics, plus the new `single_speaker: false` key on re-run.

**L61 — Docs for the first public push (user request)** · 2026-09-26
- **ARCHITECTURE.md** rewritten to match the code (query path: parse/exclusions → keyword AND → vector with scoped floor → weighted RRF → exclusion filter → rerank top 20; ingest state machine with stale recovery, SKIP LOCKED, id conflict, dimension check; serving/validation; the config table; tech choices incl. the reranker). **README.md**: results at a glance (numbers checked against `eval/results.json`), docs index, config table, run commands. **LIMITATIONS.md**: rows updated for the fixes (exclusion, fusion weights, threshold, rerank latency, ingest). **ISSUES.md**: §0 rebuilt (fixes pending verification incl. D5.1), R1 re-listed as partly closed (q5/q13 still miss). **New `docs/AGENT_DISCLOSURE.md`**: roles, working agreement, workflow diagram, the 14 phases, human-vs-agent decision table, agent mistakes and how they were caught.
- **Public-repo hygiene:** email and local user paths removed from AGENTS.md/ISSUES.md. The untracked presentation `.pptx` is not committed (not requested). Static checks before commit: `py_compile` on all changed modules OK; the unused-import/definition scan is clean.

**L62 — Test scaffold (user request: scaffold only, no test cases)** · 2026-09-26
- **Files:** `tests/__init__.py`, `tests/conftest.py`, `pytest.ini` (`testpaths = tests`, `asyncio_mode = auto`, a `db` marker); `requirements.txt` + `pytest`, `pytest-asyncio` (**not installed yet**; `httpx` for a future FastAPI TestClient is already present).
- **Fixtures:** `test_db_url` (session: resolves `TEST_DATABASE_URL`, or the app DB name + `_test` on the same server; creates the DB if missing via an autocommit connection to `postgres`; applies `schema.sql`; **skips** DB tests if there are no settings or the DB is unreachable); `db_conn` (per test, pgvector registered, **rolled back** after); `clean_db` (for code that commits: `TRUNCATE recordings, chunks RESTART IDENTITY CASCADE` before and after).
- **Safety:** a guard refuses any database whose name doesn't end in `_test` (`pytest.UsageError`), so the real `transcripts` DB can't be written to or truncated.
- **Why a real DB, not SQLite/mocks:** tsvector, pgvector `<=>` and `FOR UPDATE SKIP LOCKED` (the behavior worth testing) don't exist in SQLite; the Docker Postgres already runs, so a separate `_test` DB is cheap. conftest imports `app.config` lazily (it needs `.env`) and never imports the model-loading modules.
- Static check: `py_compile` OK. No tests run (per the user).

**L63 — `tests/test_rrf.py` (user request)** · 2026-09-26 · written, **not run** (pytest not installed yet)
- 8 unit tests for `hybrid.rrf()`: both arms agree (order + exact score 2/61); keyword-only / vector-only hits (arm order kept, not id order); no hits → `{}`; disagreement where a doc in both arms outranks single-arm top hits (`[1,3,2]`); **symmetric disagreement → exact tie → lower id first** (mirrors `search()`'s `(-score, id)` key via a `ranking()` helper); keyword weight 1.5 breaks the tie toward the keyword top hit; smaller `k` widens the rank gap. Expected values checked by hand; `py_compile` OK.
- **Note:** importing `hybrid` pulls in torch/sentence-transformers and `app.config` (needs `.env`), with no model loaded. Moving `rrf()` to a tiny module would make it a pure, fast unit test; not done (a production code change that wasn't requested).

**L64 — `tests/test_search_filters.py` (user request)** · 2026-09-26 · written, **not run** (pytest not installed)
- **No DB, no model:** a `FakeConn` answers `search()`'s SQL from an in-memory chunk table (keyword arm = preset ids; the vector arm applies the same rule as `VECTOR_SQL`: keep if `1 − sim ≤ max_dist`, plus recording/speaker filters; text lookups; result rows) and records every call. An autouse fixture monkeypatches `hybrid._model`/`hybrid.embed` (records the embedded text); `rerank=False`. No production code changed.
- **O29 tests (6):** parsing (hyphenated words aren't exclusions; `positive_text`); exclusion drops matches from **both** arms + the excluded word never reaches the embedder (O29.1); word-prefix (`-art` keeps "start", drops "art"/"artwork", O29.2); phrase exclusion; backfill up to `n`; exclusion-only → `[]` with **zero SQL calls and no embedding** (D6).
- **O23 tests (6):** unscoped below the floor → explicit `[]`, `max_dist = 0.7`; boundary `sim = 0.30` kept, `0.29` dropped; recording-scoped keeps low-sim turns (`max_dist = 2.0`) and filters other recordings (O23.1); no-letters query → floor 0.45 (`max_dist = 0.55`, `0.403` dropped, O23.2); a keyword hit survives an empty vector arm; speaker without recording → `ValueError`.
- Expected values traced by hand through `search()`; `py_compile` OK.

**L65 — `tests/test_api.py` (user request)** · 2026-09-26 · written, **not run**
- **Integration** (marker `integration`, registered in pytest.ini): `TestClient(app)` as a context manager runs the real lifespan (models, pool, dimension check, warm-up) against the **ingested DB, read-only** (`/search` never writes; the `_test` DB has no chunks). Module fixture skips if there are no DB settings, the DB is unreachable, or `rec04_support_billing` isn't `completed`. Heavy (loads the models): run alone with `-m integration`.
- **Tests (5):** valid `refund` → 200, `0 < count ≤ n`, top hit from rec04, some hit contains "refund" (the reranker may rank an answer turn without the word first), response fields present; scoped recording + speaker → all hits match; unknown recording → **404** `unknown recording` (O30); **empty `recording=` → 422** with `loc == ["query","recording"]`; gibberish → 200, `count 0`, `results []` (O23).
- **Spec mismatch flagged to the user:** the request said the empty recording should return **404**; the code (D8, L56) returns **422** (validation rejects it before the O30 lookup). The test asserts the actual behavior; switching to 404 would be a small code change and is the user's call.

**L66 — `tests/test_ingestion.py` (user request)** · 2026-09-26 · written, **not run**
- **Runs on the `_test` DB** (marker `db`, `clean_db` fixture: truncated before/after; the `_test` name guard). The real `ingest()` / `recover_stale()` SQL runs unchanged; stubs only for inputs: `load_rows` → 3 synthetic 384-d chunks (no pipeline outputs / git-ignored `.npz` needed), audio = bytes in `tmp_path` (its SHA-256 is the key), `ingest.ROOT` → `tmp_path` (for `file_path`). No models loaded.
- **Tests (5):** re-running on the same file → `loaded` then `skipped` ×2, exactly 1 recording + 3 chunks; the same bytes under another name → `skipped` (the hash is the key); **K1:** a 2-hour-old `processing` row is skipped before recovery, `recover_stale(30)` returns it and marks it `failed` with a "stale" error, the next ingest retries → `completed`, no duplicates; a 5-minute-old `processing` row is **not** recovered; **D4:** same file name + new bytes → `RecordingIdConflict` (not `UniqueViolation`), log contains "id conflict … existing recording left unchanged", and the stored row keeps its status/hash/chunks.
- `py_compile` OK.

**L67 — `tests/test_pipeline_edge_cases.py` (user request)** · 2026-09-26 · written, **not run**
- Pure unit tests (no models/DB/audio; Whisper, `transcribe`, the voice encoder and `librosa.load` stubbed; I/O in `tmp_path`). **D5:** `asr.main()` over [silent, speech] → no `ZeroDivisionError`, empty JSON for silent, the "no speech segments detected" warning, **the loop continues** (the speech file is written, `segments=2` logged); `chunk.chunk()` with 0 segments → `[]`; `chunk.main()` over [silent, speech] → the empty chunk file + "no chunks …" warning, speech → 2 chunks. **D5.1:** `diarize()` with 0 and 1 segments (parametrized) → `single_speaker: true`, all `A`, null metrics, warning, and the encoder is **never called**; exactly 2 segments → KMeans runs, silhouette `null` (previously raised), within `[null, null]`, between 0.0 (orthogonal stub voices).
- `py_compile` OK.

**L68 — Empty `recording=` stays 422 (user decision)** · 2026-09-26
- Keep the D8 behavior (validation 422 before the O30 lookup); tests follow the code (`tests/test_api.py` asserts 422 with `loc == ["query","recording"]`). Resolves the mismatch flagged in L65.

**L69 — Pre-restart cleanup (user request)** · 2026-09-26
- **Deleted (regenerable, no reinstall):** `%LOCALAPPDATA%\Temp` leftovers (the VS Code update installer 222 MB, the Docker `DiagOutputDir` 111 MB, and 34 other Temp items older than 2 days; 7 in use were skipped, and the Claude session scratchpad with the load-test scripts was kept); the old `D:\pip_install.log`, `D:\docker_install.log`. **C: 0.98 → 1.32 GB free.** The pip cache and `~/.cache/huggingface` were already empty.
- **Kept on purpose:** `.local/docker` (the DB), the models in `.local/huggingface` (whisper small, all-MiniLM, ms-marco reranker; whisper base and multi-qa are only used by benchmarks, ~150 MB, D isn't constrained), `logs/app.log`, `data/`. `__pycache__` removal in the project was blocked by the tool's path protection; skipped (a few KB, regenerated automatically).
- The Windows page file (9.4 GB on C) resets its usage on restart.

**L70 — Complete verification pass after restart (user request)** · 2026-09-26
- **Environment:** after the restart, C 5.1 GB free, RAM 2.0 GB free, page file 4.9 GB; Docker Desktop started, `db` container up. Jobs run one at a time.
- **V1 data:** 6 recordings `completed`, 344 chunks.
- **V2 regression eval (run 7, current defaults) vs committed run 6: 0 quality-metric differences** (all 5 configs × all types, TNR/zero-result rates) and **no per-query rank changes** (programmatic diff). Hybrid: Hit@1 0.733, Hit@5 0.933, R@10 0.741, MRR 0.827, nDCG 0.695, TNR 1.0.
- **V3 diarization re-run (D7, D5.1 normal path) + chunk re-run (D10):** all 6 recordings have **identical speaker labels**; silhouette, between-speaker cosine and switches unchanged; within-speaker cosine corrected (e.g. rec01 0.856/0.899 → 0.853/0.897; range now 0.843–0.926); `single_speaker: false` everywhere. Chunking → **0 changed files**, same stats lines. The refreshed diarization JSON (corrected within values + `single_speaker` key) is to be committed.
- **V4:** D6 `-invoice` / `-"account credit"` → 0 results + the `no positive terms` log; D9 `load_rows(…, 768)` → a clear `ValueError`, ingest logs `chunks.embedding is vector(384)`; ingest re-run `loaded=0 skipped=6 failed=0` exit 0; X2 `compare_embeddings` (MRR 0.441 / 0.399 unchanged) writes nothing to `~/.cache/huggingface` on C.
- **V5 API:** startup order: DB pool target → embedder (torch threads=1) → **`embedding dimension check: model=384 db=384`** → reranker (top 20) → `API ready … reranker=on(top 20)`; `recording=` → **422** `string_too_short` + log `422 GET /search: [('query.recording', …)]`; `nope` → 404; `-invoice` → 200 count 0; valid → 200.
- **R1.2 latency A/B (same machine state, 3×20 concurrent):** top 20 server p50 817 / p95 1572 ms (client 2367 / 6395, first burst after start 7.2 s); top 50 server 878 / 2756 (client 2576 / 4612); rerank off server 118 / 136 (client 383 / 569). Top 20 cuts the server p95 by ~43%; the reranker dominates under concurrency. Decision left to the user (ISSUES R1.2).
- **Docs updated:** EVALUATION §8 (corrected within values) + new §12.4 (regression + latency); ARCHITECTURE status; LIMITATIONS latency row; README latency line; ISSUES §0 (D6/D7/D8/D9/X2/D10-chunk removed; D4/D5/D5.1/D10-ASR mapped to tests; R1.2 → a decision).

**O27 A/B result:** `keyword_op` = `and` (default) | `or` added to `search()` / CLI `--keyword-op`. OR helps keyword-only search (MRR 0.433 → 0.660) but hurts hybrid (Hit@5 0.867 → 0.767, nDCG 0.665 → 0.616; paraphrase Hit@5 0.727 → 0.455), because common words match unrelated turns. Details: [docs/EVALUATION.md](docs/EVALUATION.md) §5.

### 6.2 Open — to be locked during development

> **2026-09-26: all open issues are consolidated in [docs/ISSUES.md](docs/ISSUES.md)** (user request: finish development first; latency/concurrency later). New issues go there. The table below is kept for history.
> **Rule (user):** issues are tracked **only in ISSUES.md**, never in AGENTS.md. Follow-ups found while fixing an issue go **under the original issue** as sub-items (`O29.1`, `O29.2`), not as new top-level issues.
| ID | Question | Options (⭐ suggestion) |
|---|---|---|
| O1 | LLM for writing the scripts (L3) | Probably moot: the user supplies the audio (see L16) |
| ~~O2~~ | ~~TTS engine~~ | **Closed → L16** |
| ~~O3~~ | ~~Whisper size~~ | **Closed → L22 (`small`)** |
| ~~O4~~ | ~~MiniLM checkpoint~~ | **Closed → L26 (both; eval decides)** |
| O20 | How both embeddings live in the DB (L26) | ⭐ two columns `embedding_general` + `embedding_qa` (one load; search picks a column; easy A/B) · one `embedding` column, re-load per experiment · separate table per model |
| O21 | Text fed to the embedder | ⭐ chunk text only (current) · prefix speaker (`"A: …"`); the label carries no meaning for the model, so probably noise |
| G1 | Commit `data/processed/`? | ⭐ JSON (asr/diarization/chunks, 0.33 MB; ASR takes ~40 min to redo) yes, `.npz` embeddings no (regenerated in seconds) · all · none |
| G2 | Keep the rec01 `base` benchmark transcript? | ⭐ keep (evidence for L22) · delete |
| O23 | Relevance cutoff for weak matches (C6) | ⭐ none for now; decide after C9 eval shows whether tails hurt · min cosine similarity on the vector arm (e.g. 0.3) · only return vector-only hits above a threshold |
| ~~O25~~ | ~~Speaker filter scope~~ | **Closed → L32** |
| ~~O26~~ | ~~Neighbor metric~~ | **Closed → L33** |
| ~~O27~~ | ~~Keyword operator~~ | **Closed → L34 (AND)** |
| ~~O28~~ | ~~DB connection per request~~ | **Closed → L36 (pool)** |
| ~~O31~~ | ~~Cold first request~~ | **Closed → L37** |
| O32 | API: query embedding 47–150 ms right after startup | **Not reproduced after L37:** warm sequential requests are 25–35 ms server-side. ⭐ close · keep watching |
| O29 | `-exclude` not applied to the vector arm | ⭐ leave + document (semantic arm can't "exclude words" cleanly) · post-filter fused results whose text contains an excluded term |
| O30 | Unknown `recording` returns an empty 200 | ⭐ 404 "unknown recording" (one extra lookup) · keep an empty 200 |
| O24 | queries.json key | ⭐ keep DB `chunk_id` (user's format), but never re-ingest without regenerating · switch to stable `(recording_id, chunk_index)` pairs |
| O22 | On a per-recording failure | ⭐ stop the run (current behavior, logged) · log and continue with the next recording, with a failure summary at the end |
| ~~O5~~ | ~~Streamlit vs Gradio~~ | **Closed → L15** |
| ~~O6~~ | ~~Resemblyzer install~~ | **Resolved:** `webrtcvad-wheels` + `resemblyzer --no-deps` import fine |
| O7 | D10 query understanding | ⭐ simple speaker filter via UI dropdown · none · LLM filter extraction |
| O8 | D11 output layer | ⭐ search results only (snippet + speaker + timestamp + audio) · + LLM answer |
| O9 | Fuzzy matching add-on (`pg_trgm`) | yes · ⭐ no (only if the eval shows name misses) |
| ~~O11–O19~~ | ~~Schema details~~ | **Closed → L23** |
| O10 | Docker install failed: how to get Postgres | ⭐ user enables WSL2 (`wsl --install` in an admin PowerShell, then reboot) and reinstalls Docker Desktop manually, approving the admin prompt · fall back to `pgserver` (pip, no admin; changes L9) · Supabase/Neon free tier (changes L9) |

---

## 7. Metrics for Production

### 7.1 Stage-wise quality metrics (offline)
| Stage | Metric | Why it matters |
|---|---|---|
| ASR | **WER**, entity/keyword WER (on names, numbers, products) | Keyword recall depends on correctly transcribed rare terms |
| Diarization | **DER** (miss + false alarm + confusion), **JER**, **cpWER / WDER** (word-level speaker errors) | Speaker-scoped queries fail if words are attributed to the wrong speaker |
| Role labeling | Accuracy / F1 | Needed for "customer said…" filters |
| Retrieval | **Recall@k** (k=10, 50, 100: first-stage ceiling), **nDCG@10**, **MRR@10**, **Hit@k**, Precision@k | Core ranking quality; Recall@100 bounds what the reranker can fix |
| Localization | **Timestamp accuracy** (IoU of returned span vs. true span; "hit within ±N s") | The user must land at the right moment in the audio |
| Speaker-scoped | Recall@k on speaker-filtered queries | Checks diarization and retrieval together |
| RAG (if D11b) | Faithfulness/groundedness, answer relevance, citation precision/recall | Hallucination control |

### 7.2 System / operational metrics (online)
- **Latency:** p50/p95/p99 per stage (embed query, ANN, BM25, fusion, rerank), end to end. Target e.g. p95 < 300 ms without the LLM.
- **Throughput:** QPS at target latency; ingestion **real-time factor** (audio hours processed per wall-clock hour, per GPU).
- **Freshness:** time from upload until the recording is searchable.
- **Cost:** $ per audio hour ingested, $ per 1k queries, storage GB per 1M chunks (with and without quantization).
- **Index health:** ANN recall versus exact (brute-force) search on a sample; drift after re-embedding.
- **Reliability:** error rate, ingestion failure rate, availability.

### 7.3 User / business metrics (online)
- Zero-result rate, query reformulation rate, click-through at rank 1/3/10, time-to-first-useful-result, "played audio at returned timestamp" rate, explicit thumbs up/down.
- A/B tests and **interleaving** (team-draft) to compare rankers on live traffic.

---

## 8. Evaluation Plan

### 8.1 Build a golden set
1. **Ground truth:** if D1(b) synthetic, you know the exact transcript, speaker per turn and the "fact → turn" mapping. For real audio, hand-correct transcripts for a subset.
2. **Queries (~200–500),** stratified by type:
   - *Exact keyword/entity:* "order number 4471", "Dr. Mehta"
   - *Semantic/paraphrase:* "customer wanted their money back" (transcript says "refund")
   - *Speaker-scoped:* "what did the agent promise about delivery"
   - *Cross-turn Q/A:* the answer spans a question turn and the reply
   - *Conversation-level/topic:* "calls about switching providers"
   - *Negative/no-answer:* the system should return nothing relevant
   - *ASR-stress:* rare names, numbers, accents, overlapping speech
3. **Relevance labels:** graded (0/1/2) at chunk or turn level, and mapped to time spans so labels survive chunking changes. Generate candidates with an LLM, then validate with an LLM judge plus a human spot-check (measure judge–human agreement, Cohen's κ).

### 8.2 Experiments (ablation ladder)
| Run | Configuration |
|---|---|
| R0 | BM25 only |
| R1 | Dense only |
| R2 | Hybrid (RRF) |
| R3 | Hybrid + reranker |
| R4 | R3 + contextual headers / enrichment |
| R5 | R3 on **ground-truth transcripts** vs. ASR transcripts (isolates ASR/diarization error propagation) |
| R6 | Chunking variants (D4 a/b/c/d) |
| R7 | Quantization / HNSW param sweep: recall vs. latency vs. memory |
| R8 | Scale test: 1k → 100k → 1M+ chunks (synthetic duplication/noise): quality and latency curves |

- Report metrics **per query type** (hybrid should win on the mix; BM25 on exact terms; dense on paraphrase).
- Statistical significance: paired bootstrap or a permutation test over queries.
- Error analysis: sample failures and bucket them by cause (ASR error, speaker error, chunk boundary, ranking).

### 8.3 Deliverables of eval
- An `eval/` harness that runs any config and writes a metrics table plus plots.
- A short report: the ablation table, latency/cost numbers and the error-analysis buckets.

---

## 9. Discussion Log

### 2026-09-26 — Session 1: Problem analysis
- Analyzed the problem statement. Wrote the architecture (§3), six end-to-end approaches (§4), open decisions D1–D13 (§5), metrics (§7) and the evaluation plan (§8).
- Suggested (not locked) optimal approach: **C (Qdrant hybrid) + reranker + light F enrichment**, with Postgres (A) as runner-up.
- **Pending from user:** lock decisions D1–D13; confirm assumptions A1–A4 (especially GPU availability and whether audio is stereo).

### 2026-09-26 — Session 1b: Approach A locked
- The user locked **Approach A (Postgres only)**. Rationale: 6 recordings / a few hundred chunks, a short build window, simplicity and defensibility, zero infrastructure risk. Recorded as L1 and L2 in §6.
- Environment check: Python 3.12.5; **no Docker, no Postgres, no NVIDIA GPU** (§2.4). This means CPU inference and an embedded or managed Postgres.
- Added sub-decisions D7.1 (Postgres hosting), D7.2 (lexical ranking) and D7.3 (vector index), plus the "how A scales" note in §4.
- **Pending from user:** D7.1–D7.3, D1–D6, D8, D10–D12. Is the audio stereo? Where are the 6 recordings (path/format)?

### 2026-09-26 — Session 1c: Prototype stack locked
- The user locked D1 (synthetic), D2 (faster-whisper), D3 (Resemblyzer + KMeans; pyannote rejected because of auth overhead), D4 (speaker-turn merge, gap ≤ 1 s, max 30 s), D5 (MiniLM), D6/D7.2 (tsvector + GIN + ts_rank_cd), D7.1 (Docker, plain pgvector image), D7.3 (exact scan), D8 (RRF k=60, no reranker), D12 (FastAPI + Streamlit/Gradio) and D13 (CPU only). Recorded as L3–L14, each with why, why not the alternatives, and a production alternative.
- The user said: these choices are for the **time-boxed working prototype**; production gets its own decisions later. Added as rule 5. The user also asked that every prompt record what was decided and why the others weren't. Added as rule 4.
- The user will create the project folder themselves.
- Flagged risks: Docker not installed (L9), and Resemblyzer's `webrtcvad` build on Windows (O6).
- Open items O1–O9 (§6.2) are to be locked during development.

### 2026-09-26 — Session 2: Skeleton + dependencies
- The user locked: O5 → Streamlit (L15), O2 → no TTS, the user supplies the audio (L16), global Python (L17).
- Created a minimal skeleton only (§10). Modules are one-line stubs, no logic yet.
- Pre-install check: none of the required Python packages were installed. Git is present. **Docker and ffmpeg are not installed.** ffmpeg is not needed, because faster-whisper decodes audio through PyAV.
- Install approach: `webrtcvad-wheels` (prebuilt) plus `pip install --no-deps resemblyzer`, which avoids needing C++ build tools for `webrtcvad` (resolves O6).
- **Suggestions awaiting the user's decision:** install Docker Desktop (needed for L9); `git init` the repo.
- **Audio received:** 6 WAVs in `data/audio/` (rec01_standup_incident, rec02_hr_burnout, rec03_sales_pricing, rec04_support_billing, rec05_friends_cooking_trip, rec06_climate_policy). All **mono, 22.05 kHz, 16-bit, ~8–9.5 min each (~53 min total)**. Mono rules out channel-split diarization, which confirms L5.
- **Build strategy options (NOT locked; the user will choose):** S1 pipeline order with file checkpoints · S2 thin end-to-end slice first · S3 search/UI first with fake chunks · S4 = S1 plus an early database smoke test once Docker is up (suggested). Also asked the user: O3 Whisper size (time on one file first), where intermediate results live (JSON in `data/processed/` suggested), and whether they have the original scripts for ground truth.
- The user asked to build in **small components, one at a time**, never a full system in one go.
- **Docker Desktop install FAILED** (winget: installer exit code 1, twice). Likely causes: the admin prompt was declined or timed out, or WSL2 is missing. **Open issue O10**, which blocks only the database smoke test and C5.
- The pip install finished; all 12 packages import (faster-whisper 1.2.1, torch 2.14 CPU, sentence-transformers 6.1, psycopg 3.3, streamlit 1.64, fastapi 0.141, resemblyzer 0.1.4, …).
- **Benchmark (rec01, 567.5 s of audio):** `base` took 147 s (RTF 0.26); `small` took 431 s (RTF 0.76, ~2.9× slower). Both give 113 segments (avg 5.0 s, max ~13 s). The transcripts agree on 98.4% of words (21 diff blocks). `small` fixes domain terms that matter for keyword search: "uncall"→"on-call" (×2), "lent"→"lint", "right up"→"write-up", "yet"→"yeah", "220"→"2.20". Many diffs are formatting only (two/2, check out/checkout). Estimated time for all 6 (~3110 s of audio): `base` ≈ 13 min, `small` ≈ 40 min. The user picked `small` (L22).
- **User locked O3 and O11–O19** (L22, L23). The schema was updated with `error`, `created_at` and `updated_at`. `small` ASR is running in the background on rec02–rec06 (rec01 is already done).
- **The user flagged error-handling gaps** → L24 + §6.3. Findings: `updated_at` and "skip while processing" were already in place (O17/O18), so there's no schema change. Error truncation to 2000 chars is recorded as a C5 implementation requirement. Stuck `processing` is documented as known limitation K1 (no auto-retry, per the user). Also added K2 (the sequential-only assumption).
- **Plan:** Docker needs a reboot, which would kill the running ASR. So: let ASR finish → reboot → DB setup. Meanwhile, build C2.
- **Built C2 `app/diarize.py`:** one Resemblyzer embedding per ASR segment (audio loaded with librosa at 16 kHz, raw slice, no silence trimming, so timestamps stay intact) → KMeans(k=2, n_init=10, seed 0) → labels `A`/`B`, where **A is always the first voice heard**, so labels are readable. Writes `data/processed/diarization/<rec>.json` with the segments plus a speaker, and diagnostics. Added `PROCESSED_DIR` and `ASR_MODEL="small"` to config.
- **C2 result on rec01:** 113 segments (A=49, B=64), 63 speaker switches, **silhouette 0.724**, mean cosine within clusters 0.856 / 0.899 vs between 0.559, so the two voices separate clearly. A manual read of the first 30 segments shows every label correct (A = the lead asking questions, B = the on-call engineer), including one-word turns ("Triple.", "Pretty much.").
- **Unsupervised diarization checks, without ground truth:** silhouette, within/between cosine, cluster balance, and a manual read. A true DER needs labeled references (L21).
- **The user flagged the flat `app/` as unclear** → proposed a layered structure plus alternatives. The user chose all ⭐ (L25). Moved the files; verified that imports and paths resolve, and that **re-running diarization on rec01 from `app.pipeline.diarize` produces a byte-identical output file**. The running ASR job was unaffected (the module was already loaded). Added README.md.
- **C2 on rec02/rec03:** silhouette 0.624 / 0.772, within-cluster cosine 0.85/0.845 and 0.916/0.89 vs between 0.581 / 0.57. Clean separation, balanced clusters (71/70, 49/45).
- **Built C3 `app/pipeline/chunk.py`** (L6 rule as locked: same speaker, gap ≤ 1 s, max 30 s) → `data/processed/chunks/<rec>.json` (chunk_index, speaker, start, end, text, segment_ids). Results: rec01 64 chunks, rec02 57, rec03 55. **Chunk count = speaker switches + 1 in every file**, so the gap and cap rules never split a turn. Avg ~9–10 s / 21–25 words, max 17.9 s / 41 words (well under MiniLM's 256-token limit). Only 4/176 chunks (2%) have ≤ 5 words, and those still carry meaning. No change needed.
- **DB is up:** Docker 29.8.0. `docker compose up -d` → `g2-hackathon-db-1` running on 5432. Built `app/db/connection.py` (`connect()` registers pgvector; `init_schema()` applies schema.sql; `python -m app.db.connection` = apply + report). Schema applied: tables `recordings`, `chunks`; **pgvector 0.8.6**.
- **DB smoke test PASSED** (throwaway script, rolled back, 0 rows left): keyword `'refund'` matched "refunds" (the `english` stemming from O12 works, `ts_rank_cd` = 0.1); vector exact scan self-match distance 0.0; a duplicate `content_sha256` was **rejected (UniqueViolation)**, so idempotency is enforced by the DB. Note: `english` splits "on-call" into `'on-cal'` + `'call'`, so hyphenated terms are still findable.
- **All 6 recordings processed through C1–C3:** ASR `small` on CPU ran at RTF 0.95–1.06 while Docker was up (vs 0.76 before). Diarization silhouettes: rec04 0.747, rec05 0.751, rec06 0.825 (all 6: 0.62–0.83). **344 chunks** in total; chunk count = speaker switches + 1 in every recording.
- **The user chose to try both MiniLM models** (L26) and asked for **logging everywhere** (L27). Built C4 `app/pipeline/embed.py` and embedded all 344 chunks with both models (~1 s per recording each). New open items: O20–O22.
- **The user revised L26:** do NOT store both embeddings. Compare them on one recording, then pick one. The DB keeps a single `embedding` column (O20 is moot).
- **Embedding comparison (rec01, 14 paraphrased queries, 1 labeled chunk each)** via `eval/compare_embeddings.py` + `eval/queries_rec01.json`: **all-MiniLM-L6-v2: Hit@1 0.29, Hit@3 0.57, MRR 0.441** vs **multi-qa-MiniLM-L6-cos-v1: Hit@1 0.21, Hit@3 0.43, MRR 0.399**. all-MiniLM leads on all three, but the gap is 1–2 queries (not significant). multi-qa wins 5 individual queries. The scores are pessimistic by design (no lexical overlap, a single relevant chunk, near-misses such as the summary chunk 48 count as misses). Caveats: one recording, queries written by the agent. **Awaiting the user's pick.**
- **The user picked `all-MiniLM-L6-v2`** (L28).
- **Built C5 `app/pipeline/ingest.py`:** SHA-256 of the audio bytes → lookup by `content_sha256` → `completed`/`processing` = skip, `failed` = retry, new = INSERT `processing` → delete + insert the chunks (text, speaker, times, all-MiniLM vector) → `completed`. On error: rollback → `failed` with `traceback[:2000]` (L24) → log → raise. It checks that the embedding `chunk_index` matches the chunks before loading. `updated_at = now()` on every status change. Reads only the processed JSON/npz (no re-ASR).
- **C5 verified:** 1st run loaded 6 recordings / **344 chunks**; 2nd run skipped 6 (idempotent); a simulated `failed` on rec06 → "retry" → `completed`, error cleared, `updated_at` advanced, still 344 chunks (no duplicates).
- O21 (embedder input = chunk text only) and O22 (stop on first failure) were **not answered**; the code keeps the existing behavior for both.
- **Git (L29, L30):** branch `main`, `.gitattributes` LF, remote `origin` = github.com/amolgupta7/G2-Hackathon. First commits made in dependency order (setup → config/logging → db → C1…C5 → eval → docs); stubs and `data/processed/` not committed yet. **Not pushed** (waiting for the user's go-ahead).
- **Docs:** the user asked for README / ARCHITECTURE / EVALUATION / LIMITATIONS, then said to write only what's needed at this stage. `docs/ARCHITECTURE.md` was written (built parts ✅, query path marked planned). **EVALUATION.md and LIMITATIONS.md are deferred** until C6–C9 exist; README stays as it is. Not committed yet.
- **Built C6 `app/search/hybrid.py`** (L31): `search(query, speaker, recording, top_k=50, n=10)` → keyword ids (websearch AND, `ts_rank_cd`) + vector ids (exact `<=>`) → RRF(k=60) in Python → rows with prev/next turn, per-arm ranks, score. Logs per-arm hit counts + latency. CLI: `python -m app.search.hybrid "q" --speaker B --recording rec03_sales_pricing -n 5`.
- **C6 tests:** `refund` (kw 7 hits; top 3 = rec04 refund turns, the arms agree), `how long were customers affected by the outage` (kw 0 hits → vector-only, #1 = "How long was the total customer impact window?"), `connection pool` (kw 3, both arms), `burnout` (kw & vec both rank the rec02 burnout turn #1), `burnout --speaker B` (filter works; kw 0 → vector-only, with a weak irrelevant tail), `discount --recording rec03` (filter works, all on topic). SQL latency ~3–290 ms (first query ~1.4 s cold). CLI "embed" time 7–15 s = model load per process; the API will load it once.
- **Observations for the user (not built):** (1) no relevance threshold, so the vector arm always fills K and irrelevant tails can appear when nothing matches (possible min-similarity cutoff, open item O23); (2) phrase syntax untested from the CLI (PowerShell strips quotes); test via the API.
- **C6 committed** (`270349c`, user request).
- **Built `dataset/queries.json` for C9 (user request):** 30 queries, 5 per recording (13 keyword / 11 paraphrase / 6 speaker-specific), including the user's 4 tested queries (refund, connection pool, burnout, how long were customers affected). Format `{query_id, query, relevant_chunk_ids}` plus optional `type` and `speaker` (added for per-type reporting and the speaker filter; removable). **Relevance rule:** a chunk is relevant if its text contains information answering or matching the query; pure question turns and passing mentions are excluded. Every label was taken from reading the transcripts (DB dump), not from search output. **Validated against the DB:** all ids exist, one recording per query, speaker queries only reference that speaker's chunks, 0 problems. Keyword queries deliberately include on-topic chunks without the literal word (burnout 1/5, carbon tax 1/6, custard tarts 2/5), plus tricky tokens (`4,420`, `r-8812`, `Marcus`).
- ⚠️ **Chunk ids are DB-serial:** re-ingesting a recording assigns new ids (rec06 is already 346–394 after the retry test), which invalidates queries.json. A stable alternative is `(recording_id, chunk_index)`, open item O24.
- **The user locked:** O24 → `(recording_id, chunk_index)` pairs (BIGSERIAL ids change on re-ingest); keep `type` + `speaker` (type drives failure bucketing); commit queries.json separately. Converted from the DB mapping (82 relevant pairs), committed `3a0f5d0`. Order: **C9 before C7.**
- **C9 built:** `app/search/hybrid.py` got `mode ∈ {hybrid, keyword, vector}` (default hybrid, so behavior is unchanged; also `--mode` in the CLI). `eval/run_eval.py` runs 30 queries × 3 modes (top 10, speaker filter applied when set) → Hit@1, Hit@5, Recall@5/10, MRR@10, nDCG@10 (binary), per type, latency p50/p95, hybrid misses@5 → `eval/results.json`.
- **C9 results → [docs/EVALUATION.md](docs/EVALUATION.md)** (full tables, per-type, per-query ranks, failure analysis, caveats). Headline: hybrid MRR@10 0.708 / nDCG@10 0.635 / Hit@5 0.80 vs vector 0.699 / 0.615 / 0.80 vs keyword 0.433 / 0.335 / 0.433; p50 37 ms. The hybrid gain comes only from keyword-type queries (AND keyword arm is empty for paraphrase/speaker). 6 misses@5, bucketed: question-turn label artifact, per-recording speaker labels, small-embedder vocabulary gap, no lexical rescue. Options O25–O27.
- The user asked for eval results in a separate file → `docs/EVALUATION.md` (numbers generated from `eval/results.json`, not hand-copied).
- **The user locked O27 = AND (L34)** and asked to commit: `53d49c2` search modes/operator/speaker guard · `8c22998` dataset recording field · `a7eeda5` C9 eval harness + results.json · `552d4d9` docs (ARCHITECTURE status updated: C6/C9 built).
- **Built C7 `app/api/main.py`** (L35): FastAPI with lifespan preload of the embedding model (~42 s cold start incl. imports); pydantic response models (`SearchResponse` → hits with speaker, times, text, score, per-arm ranks, prev/next turn; the unstable `chunk_id` isn't exposed); validation: `q` 1–500 chars, `speaker ∈ {A,B}` **and requires `recording`** (422), `n` 1–50; `psycopg.OperationalError` → 503 "database unavailable"; request logging. Run: `python -m uvicorn app.api.main:app --port 8000` (docs at `/docs`).
- **C7 tests:** `/health` ok; `/recordings` → 6 recordings, all `completed`, chunk counts 64/57/55/55/64/49; `/search?q=refund` → rec04 refund turns with prev/next; quoted phrase `"connection pool"` → #1 contains the exact phrase; `refund -invoice` → #1 has no "invoice"; speaker+recording works. Errors: speaker without recording → 422 with a clear message; speaker=C → 422; empty q → 422; n=500 → 422; unknown recording → 200 with 0 results.
- **Observations (not changed, the user decides):** (1) **latency 77–121 ms per request** vs ~23 ms in eval, because each request opens a new DB connection (~50–80 ms of the "keyword" time); a connection pool would fix it (O28). (2) `-exclude` applies only to the keyword arm, so vector hits may still contain the excluded word (O29). (3) Unknown recording → empty 200 rather than 404 (O30).
- **The user asked to list all issues in a separate file and finish development first** (latency/concurrency later) → `docs/ISSUES.md`: API behavior (O29, O30, O23), retrieval quality (R1–R4, O9, O21), performance/concurrency (O32, P1–P4), ingestion/data (K1, K2, O22, D1–D3), eval gaps (E1–E5), repo/ops/security (G1, G2, S1, S2, X1), features not started (O8, F1), with priority P1–P3 and ⭐ options. Next: C8 UI.
- **The user said: don't commit C7 yet; proceed to C8.**
- **Built C8 `app/ui/streamlit_app.py`** (L38): sidebar filters (recording dropdown from `/recordings`, cached 60 s; **speaker dropdown disabled until a recording is picked**, enforcing L32; result count slider 1–20); a search form (submit button, no rerun per keystroke); results show `recording · Speaker X · mm:ss–mm:ss`, the previous turn (caption), the hit (quote), the next turn (caption), and an audio player starting at the turn. API down → clear error with the start command; 422 → warning with the API's message; 0 results → hint. Logs each search (`app.ui`). Run: `python -m streamlit run app/ui/streamlit_app.py` (API must be running).
- **C8 tests (Streamlit `AppTest`, headless):** page loads without exceptions, 6 recordings listed, speaker filter disabled → enabled after picking a recording; `refund` → 3 rec04 hits with headers, context and 3 audio players; speaker A @ rec04 → all A; gibberish query in rec05 → still 3 results (O23 visible in the UI, P1 in ISSUES). The real server is up at http://localhost:8501 (`/_stcore/health` ok). **Not verified: actual in-browser audio playback** (needs a person to click play).
- New issues in ISSUES.md: S3 (Streamlit binds all interfaces), U1 (full-file audio weight); O23 raised to P1.
- **The user asked to commit C7 and C8** → 3 self-consistent commits: `20a5a93` config (env credentials + runtime settings: config.py, docker-compose.yml, .env.example; a prerequisite of the API pool) · `54e07ac` C7 API (app/api, db/connection.py pool, requirements psycopg-pool) · `c09141f` C8 UI (app/ui, .streamlit/config.toml, README). **Still uncommitted:** hybrid.py (O29/O23/P1), ingest.py + schema.sql (K1/K2), eval/run_eval.py + results.json + dataset/negative_queries.json (O23 eval), docs (EVALUATION, ISSUES), AGENTS.md.
- **ISSUES.md cleaned (user request):** only open items remain; fixed rows were removed (their records stay here as L39–L45). §0 lists the built-but-unverified fixes with the log lines to check.
- **Created `docs/LIMITATIONS.md` (user request):** the prototype's limits by area (data, ASR, diarization, chunking, retrieval, evaluation, system, security), each with its impact, current mitigation and production path, linked to decisions (L#) and ISSUES IDs. F1 removed from ISSUES.
- **Docker CLI isn't on PATH in already-open terminals.** Open a new terminal, or prepend `%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin`.
- Noted for O4: `all-MiniLM-L6-v2` is **already in the local HF cache** (87 MB, no download). `multi-qa-MiniLM-L6-cos-v1` would need an ~90 MB download.
- The user chose to work on the **database schema** while Docker installs. Draft `app/schema.sql` (NOT applied, NOT locked); open points are O11–O14.
- **The user asked for:** (1) an audio processing status (processing/completed/failed), and (2) hashing for idempotency. Clarified the terminology: "consistent hashing" is a sharding/distribution technique; idempotency needs a **content hash** (SHA-256 of the audio bytes, UNIQUE), so the same audio is never ingested twice, even under a different file name. Added `recordings.content_sha256 TEXT NOT NULL UNIQUE` and `recordings.status` with a CHECK constraint. Chunk-level idempotency already exists through `UNIQUE (recording_id, chunk_index)` plus upsert. Follow-ups are O15–O19.
- O10: **the user chose to install Docker manually** (enable WSL2, install Docker Desktop, then `docker compose up -d`). Steps were given in chat. L9 is unchanged.
- Built C1 `app/asr.py` (faster-whisper, CPU int8, `language="en"` per assumption A1, default segmentation, no VAD filter). Started the base-vs-small benchmark on rec01.
- **User approved both.** `git init` done (no commit yet; the user decides when). Docker Desktop is installing via `winget` (needs UAC approval). WSL2 appears to be missing (`wsl --status` exit 50); Docker Desktop on Win10 Home needs it, so a reboot may follow.

---

## 10. Project Structure
```
G2-Hackathon/
├── AGENTS.md            # this file: decisions + discussion log
├── CLAUDE.md            # imports AGENTS.md for Claude Code sessions
├── requirements.txt     # Python deps (+ resemblyzer via --no-deps)
├── docker-compose.yml   # Postgres + pgvector (pgvector/pgvector:pg17), L9
├── .gitignore
├── README.md            # purpose, folder map, run commands (L25)
├── app/
│   ├── __init__.py      # sets HF_HOME → .local/huggingface (model cache inside the project)
│   ├── config.py        # paths, DATABASE_URL, ASR_MODEL, EMBED_MODEL
│   ├── log.py           # get_logger → console + logs/app.log (L27)
│   ├── pipeline/        # OFFLINE: audio → searchable chunks
│   │   ├── asr.py       # C1 ✅  python -m app.pipeline.asr --model small
│   │   ├── diarize.py   # C2 ✅  python -m app.pipeline.diarize
│   │   ├── chunk.py     # C3 ✅  python -m app.pipeline.chunk
│   │   ├── embed.py     # C4 ✅  python -m app.pipeline.embed --model all-MiniLM-L6-v2
│   │   └── ingest.py    # C5 ✅  python -m app.pipeline.ingest
│   ├── db/
│   │   ├── schema.sql   # L23
│   │   └── connection.py  # ✅ python -m app.db.connection (apply schema)
│   ├── search/hybrid.py # C6 keyword + vector + RRF (stub)
│   ├── api/main.py      # C7 FastAPI (stub)
│   └── ui/streamlit_app.py  # C8 Streamlit (stub)
├── eval/
│   ├── compare_embeddings.py  # model comparison (L26/L28)
│   └── queries_rec01.json     # 14 labeled queries for rec01
├── data/
│   ├── audio/           # 6 input recordings (gitignored)
│   └── processed/{asr,diarization,chunks,embeddings}/   # per-stage outputs (L20)
├── logs/                # app.log (gitignored)
└── .local/              # model cache + Docker data (gitignored)
```
Every sub-folder has an `__init__.py`. Run all commands from the project root.
