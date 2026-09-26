import logging
from logging.handlers import RotatingFileHandler

from app.config import ROOT

LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"


def get_logger(name: str) -> logging.Logger:
    """Loggers under "app" write to console + logs/app.log (rotating, 5 MB x 3)."""
    base = logging.getLogger("app")
    if not base.handlers:
        LOG_DIR.mkdir(exist_ok=True)
        base.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        for handler in (
            logging.StreamHandler(),
            RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"),
        ):
            handler.setFormatter(fmt)
            base.addHandler(handler)
    return logging.getLogger(name)
