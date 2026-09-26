# Hybrid Search over 2-Speaker Audio Transcripts

Search across recorded two-person conversations using **keyword + semantic (hybrid) search**, with results linked to the speaker and timestamp.

Decisions, reasoning and progress are tracked in [AGENTS.md](AGENTS.md).

## How it works

```
OFFLINE (app/pipeline)                                   ONLINE (app/search, api, ui)
audio → speech-to-text → who-spoke → speaker turns       query → keyword + vector search
      → embeddings → Postgres (app/db)                         → RRF fusion → results
```

## Folder map

| Path | What's inside |
|---|---|
| `app/config.py` | Paths and settings |
| `app/pipeline/` | Offline processing: `asr.py` (speech-to-text), `diarize.py` (who spoke), `ingest.py` (load into DB) |
| `app/db/` | `schema.sql` (tables) and `connection.py` |
| `app/search/` | `hybrid.py`: keyword + vector search fused with RRF |
| `app/api/` | `main.py`: FastAPI `/search` endpoint |
| `app/ui/` | `streamlit_app.py`: search page |
| `data/audio/` | Input recordings (`.wav`) |
| `data/processed/` | Output of each pipeline step (JSON) |

## Setup

```bash
pip install -r requirements.txt
pip install --no-deps resemblyzer
docker compose up -d          # Postgres + pgvector
```

## Run (from the project root)

```bash
python -m app.pipeline.asr --model small      # 1. speech-to-text  -> data/processed/asr/
python -m app.pipeline.diarize                # 2. who spoke       -> data/processed/diarization/
python -m app.pipeline.chunk                  # 3. speaker turns   -> data/processed/chunks/
python -m app.pipeline.embed --model all-MiniLM-L6-v2           # 4. embeddings -> data/processed/embeddings/<model>/
python -m app.pipeline.embed --model multi-qa-MiniLM-L6-cos-v1
python -m app.db.connection                   # apply DB schema (needs `docker compose up -d`)
python -m app.pipeline.ingest                 # 5. load chunks + embeddings into Postgres (idempotent)
```

Logs: console + `logs/app.log`. Model files are cached in `.local/huggingface` (D drive).

More steps will be added as components are built.
