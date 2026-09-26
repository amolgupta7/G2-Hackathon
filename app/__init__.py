import os
from pathlib import Path

# Keep model downloads/caches on D (C drive is full). Must run before huggingface_hub is imported.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".local" / "huggingface"))
