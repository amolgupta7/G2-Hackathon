# Hybrid Search over 2-Speaker Audio Transcripts

Search across recorded two-person conversations using **keyword + semantic (hybrid) search**. Every result is linked to the **speaker**, the **timestamp** and the surrounding turns, and the audio plays from that moment.

Built as a time-boxed hackathon prototype: 6 recordings (~52 min), CPU only, one Postgres database.

## Results at a glance

On 30 hand-labeled queries + 10 negative (gibberish/off-topic) queries ([docs/EVALUATION.md](docs/EVALUATION.md), §12.3):

| Hybrid (current defaults) | Hit@1 | Hit@5 | Recall@10 | MRR@10 | True-negative rate |
|---|---|---|---|---|---|
| keyword + vector + RRF + rerank | **0.733** | **0.933** | 0.741 | **0.827** | **1.0** (all 10 negatives → no results) |

For comparison (MRR@10): keyword-only 0.433 · vector-only (reranked) 0.794 · hybrid without rerank 0.741.

Latency (CPU-only laptop): ~200 ms per query with rerank, ~25 ms without. Under 20 concurrent requests the reranker dominates (server p50 ~0.8 s vs ~0.12 s with `RERANK=0`); see EVALUATION §12.4.

## How it works

```
OFFLINE (app/pipeline)                                  ONLINE (app/search → app/api → app/ui)
audio → speech-to-text (faster-whisper)                  query → keyword arm (Postgres full-text, AND)
      → who spoke (Resemblyzer + KMeans k=2)                   + vector arm (MiniLM, similarity floor)
      → speaker-turn chunks                                    → weighted RRF → -exclusion filter
      → MiniLM embeddings                                      → cross-encoder rerank (top 20)
      → Postgres + pgvector (idempotent ingest)                → results + neighbor turns + audio
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Documentation

| Doc | What's inside |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Components, data model, query path, serving, technology choices, scaling path |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Query sets, metrics, all eval runs (A/B tests: AND vs OR, threshold, rerank, RRF sweep), failure analysis |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | Known limits of the prototype and the production path for each |
| [docs/ISSUES.md](docs/ISSUES.md) | What's still open (fixes pending verification, deferred decisions) |
| [docs/AGENT_DISCLOSURE.md](docs/AGENT_DISCLOSURE.md) | How an AI coding agent was used to build this, and how decisions stayed with the human |
| [AGENTS.md](AGENTS.md) | Full decision log (L1–L60): every option considered, what was chosen and why |

## Folder map

| Path | What's inside |
|---|---|
| `app/config.py` | Paths and settings (reads `.env`) |
| `app/pipeline/` | Offline stages: `asr.py`, `diarize.py`, `chunk.py`, `embed.py`, `ingest.py` |
| `app/db/` | `schema.sql` (tables) and `connection.py` (connections, pool, schema apply) |
| `app/search/` | `hybrid.py`: keyword + vector search, RRF, exclusions, rerank |
| `app/api/` | `main.py`: FastAPI `/search`, `/recordings`, `/health` |
| `app/ui/` | `streamlit_app.py`: search page |
| `eval/` | `run_eval.py` (retrieval eval), `compare_embeddings.py`, `results.json` |
| `dataset/` | `queries.json` (30 labeled queries), `negative_queries.json` (10 negatives) |
| `data/audio/` | Input recordings (`.wav`, not committed) |
| `data/processed/` | Output of each pipeline stage (JSON committed; `.npz` embeddings git-ignored) |

## Setup

```bash
pip install -r requirements.txt
pip install --no-deps resemblyzer   # its webrtcvad dependency is provided by webrtcvad-wheels
cp .env.example .env                # set DB credentials (git-ignored; read by docker compose and app/config.py)
docker compose up -d                # Postgres 17 + pgvector
python -m app.db.connection         # apply the schema
```

## Run (from the project root)

```bash
python -m app.pipeline.asr --model small              # 1. speech-to-text   -> data/processed/asr/
python -m app.pipeline.diarize                        # 2. who spoke        -> data/processed/diarization/
python -m app.pipeline.chunk                          # 3. speaker turns    -> data/processed/chunks/
python -m app.pipeline.embed --model all-MiniLM-L6-v2 # 4. embeddings       -> data/processed/embeddings/<model>/
python -m app.pipeline.ingest                         # 5. load into Postgres (idempotent; exit 1 if any recording failed)
python -m app.search.hybrid "refund -invoice" -n 5    # 6. search from the CLI (--speaker needs --recording)
python -m uvicorn app.api.main:app --port 8000        # 7. API (interactive docs at /docs)
python -m streamlit run app/ui/streamlit_app.py       # 8. UI at http://127.0.0.1:8501 (needs the API)
python -m eval.run_eval                               # 9. evaluation -> eval/results.json
```

Tests (`tests/`, pytest): `python -m pytest`. DB tests use a separate `<db>_test` database on the same Postgres (created automatically; override with `TEST_DATABASE_URL`), and are skipped if it's unreachable.

Search syntax: plain words (all must match in the keyword arm), `"exact phrase"`, `-exclude`; natural-language questions work through the vector arm.

> **Memory note (8 GB machines):** run the heavy jobs **one at a time**: the API + UI, *or* the eval / ingest scripts, not all at once. Each Python process loads its own embedding and reranker models, and Docker's VM also needs RAM.

## Configuration (`.env` or environment)

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` / `POSTGRES_HOST` / `POSTGRES_PORT` | see `.env.example` | DB credentials (or set `DATABASE_URL`) |
| `DB_POOL_SIZE` / `DB_POOL_MAX_OVERFLOW` | 1 / 4 | API connection pool (kept open / extra under load) |
| `TORCH_NUM_THREADS` | 1 | Threads per query-embedding call (less CPU contention under concurrency) |
| `RERANK` / `RERANK_MODEL` / `RERANK_TOP` | 1 / `cross-encoder/ms-marco-MiniLM-L-6-v2` / 20 | Cross-encoder rerank of the top fused candidates |
| `RRF_K` / `RRF_KEYWORD_WEIGHT` | 60 / 1.5 | Fusion constant and keyword-arm weight |
| `INGEST_STALE_MINUTES` | 30 | A recording stuck in `processing` longer than this is retried |

Logs: console + `logs/app.log` (every stage and fix has a traceable log line). Model files are cached in `.local/huggingface` inside the project.
