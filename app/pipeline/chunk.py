"""C3: merge diarized segments into speaker-turn chunks -> data/processed/chunks/<recording>.json (L6, L20)."""
import argparse
import json

from app.config import PROCESSED_DIR
from app.log import get_logger

DIAR_DIR = PROCESSED_DIR / "diarization"
OUT_DIR = PROCESSED_DIR / "chunks"
log = get_logger("app.pipeline.chunk")
MAX_GAP_S = 1.0
MAX_CHUNK_S = 30.0


def chunk(recording: str) -> dict:
    diar = json.loads((DIAR_DIR / f"{recording}.json").read_text(encoding="utf-8"))
    chunks = []
    for seg in diar["segments"]:
        cur = chunks[-1] if chunks else None
        if (
            cur is not None
            and seg["speaker"] == cur["speaker"]
            and seg["start"] - cur["end"] <= MAX_GAP_S
            and seg["end"] - cur["start"] <= MAX_CHUNK_S
        ):
            cur["end"] = seg["end"]
            cur["text"] += " " + seg["text"]
            cur["segment_ids"].append(seg["id"])
        else:
            chunks.append({
                "chunk_index": len(chunks),
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "segment_ids": [seg["id"]],
            })
    return {"recording": recording, "chunks": chunks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", nargs="*", help="recording stems (default: all diarized)")
    args = parser.parse_args()

    recordings = args.recordings or sorted(p.stem for p in DIAR_DIR.glob("*.json"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("chunk run: recordings=%d gap<=%ss max=%ss", len(recordings), MAX_GAP_S, MAX_CHUNK_S)
    for rec in recordings:
        try:
            result = chunk(rec)
            (OUT_DIR / f"{rec}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        except Exception:
            log.exception("chunking failed for %s", rec)
            raise
        cs = result["chunks"]
        durs = [c["end"] - c["start"] for c in cs]
        words = [len(c["text"].split()) for c in cs]
        log.info(
            "%s: chunks=%d avg_dur=%.1fs max_dur=%.1fs avg_words=%.0f max_words=%d",
            rec, len(cs), sum(durs) / len(durs), max(durs), sum(words) / len(words), max(words),
        )


if __name__ == "__main__":
    main()
