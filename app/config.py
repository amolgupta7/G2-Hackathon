import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
PROCESSED_DIR = ROOT / "data" / "processed"
ASR_MODEL = "small"
EMBED_MODEL = "all-MiniLM-L6-v2"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/transcripts")
