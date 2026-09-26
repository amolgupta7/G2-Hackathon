"""C1: faster-whisper ASR -> data/processed/asr/<recording>.<model>.json (L4, L19, L20)."""
import argparse
import json
import time
from pathlib import Path

from faster_whisper import WhisperModel

from app.config import AUDIO_DIR, ROOT
from app.log import get_logger

OUT_DIR = ROOT / "data" / "processed" / "asr"
log = get_logger("app.pipeline.asr")


def transcribe(audio_path: Path, model: WhisperModel, model_size: str) -> dict:
    t0 = time.perf_counter()
    segments, info = model.transcribe(str(audio_path), language="en")
    segs = [
        {"id": s.id, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments
    ]
    elapsed = time.perf_counter() - t0
    return {
        "recording": audio_path.stem,
        "model": model_size,
        "audio_duration_s": round(info.duration, 1),
        "asr_seconds": round(elapsed, 1),
        "real_time_factor": round(elapsed / info.duration, 3),
        "segments": segs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small")
    parser.add_argument("files", nargs="*", help="audio file names in data/audio (default: all .wav)")
    args = parser.parse_args()

    files = [AUDIO_DIR / f for f in args.files] or sorted(AUDIO_DIR.glob("*.wav"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("ASR run: model=%s files=%d", args.model, len(files))
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    for f in files:
        log.info("transcribing %s", f.name)
        try:
            result = transcribe(f, model, args.model)
            out = OUT_DIR / f"{f.stem}.{args.model}.json"
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        except Exception:
            log.exception("ASR failed for %s", f.name)
            raise
        lengths = [s["end"] - s["start"] for s in result["segments"]]
        log.info(
            "%s [%s] audio=%ss asr=%ss RTF=%s segments=%d avg_seg=%.1fs max_seg=%.1fs -> %s",
            f.name, args.model, result["audio_duration_s"], result["asr_seconds"],
            result["real_time_factor"], len(lengths), sum(lengths) / len(lengths), max(lengths), out.name,
        )


if __name__ == "__main__":
    main()
