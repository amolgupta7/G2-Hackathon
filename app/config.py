import os
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines); real environment variables take precedence."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _database_url() -> str:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    missing = [k for k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing DB settings {missing}: copy .env.example to .env and fill it in")
    return (f"postgresql://{quote(os.environ['POSTGRES_USER'])}:{quote(os.environ['POSTGRES_PASSWORD'])}"
            f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.environ['POSTGRES_DB']}")


_load_dotenv(ROOT / ".env")
AUDIO_DIR = ROOT / "data" / "audio"
PROCESSED_DIR = ROOT / "data" / "processed"
ASR_MODEL = "small"
EMBED_MODEL = "all-MiniLM-L6-v2"
DATABASE_URL = _database_url()
# API connection pool: DB_POOL_SIZE connections kept open, up to DB_POOL_MAX_OVERFLOW extra under load.
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "1"))
DB_POOL_MAX_OVERFLOW = int(os.getenv("DB_POOL_MAX_OVERFLOW", "4"))
# Ingest: a recording in 'processing' longer than this is treated as crashed (marked failed, then retried).
INGEST_STALE_MINUTES = int(os.getenv("INGEST_STALE_MINUTES", "30"))
# Search ranking: RRF constant and keyword-arm weight (R3: 1.5 keeps exact keyword hits on top), cross-encoder
# rerank of the top RERANK_TOP fused candidates (R1; 20 keeps most of the gain at lower latency).
RRF_K = int(os.getenv("RRF_K", "60"))
RRF_KEYWORD_WEIGHT = float(os.getenv("RRF_KEYWORD_WEIGHT", "1.5"))
RERANK = os.getenv("RERANK", "1") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP = int(os.getenv("RERANK_TOP", "20"))
# Threads per query-embedding call; 1 avoids CPU oversubscription when many /search requests run at once.
TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "1"))
