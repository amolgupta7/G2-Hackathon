"""C2: Resemblyzer embedding per ASR segment + KMeans(k=2) -> data/processed/diarization/<recording>.json (L5, L20)."""
import argparse
import json

import librosa
import numpy as np
from resemblyzer import VoiceEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.config import ASR_MODEL, AUDIO_DIR, PROCESSED_DIR
from app.log import get_logger

SR = 16000
ASR_DIR = PROCESSED_DIR / "asr"
OUT_DIR = PROCESSED_DIR / "diarization"
log = get_logger("app.pipeline.diarize")


def diarize(recording: str, encoder: VoiceEncoder) -> dict:
    asr = json.loads((ASR_DIR / f"{recording}.{ASR_MODEL}.json").read_text(encoding="utf-8"))
    wav, _ = librosa.load(AUDIO_DIR / f"{recording}.wav", sr=SR, mono=True)

    segments = asr["segments"]
    embeddings = np.array([
        encoder.embed_utterance(wav[int(s["start"] * SR):int(s["end"] * SR)]) for s in segments
    ])
    clusters = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(embeddings)

    # Name speakers by order of appearance so the first voice heard is always "A".
    first = clusters[0]
    labels = ["A" if c == first else "B" for c in clusters]

    within = [
        float(np.mean(embeddings[clusters == c] @ embeddings[clusters == c].T)) for c in (0, 1)
    ]
    between = float(np.mean(embeddings[clusters == 0] @ embeddings[clusters == 1].T))
    diagnostics = {
        "segments": len(segments),
        "segments_per_speaker": {"A": labels.count("A"), "B": labels.count("B")},
        "speaker_switches": sum(labels[i] != labels[i - 1] for i in range(1, len(labels))),
        "silhouette": round(float(silhouette_score(embeddings, clusters, metric="cosine")), 3),
        "mean_cos_within": [round(w, 3) for w in within],
        "mean_cos_between": round(between, 3),
    }
    return {
        "recording": recording,
        "asr_model": ASR_MODEL,
        "diagnostics": diagnostics,
        "segments": [{**s, "speaker": lab} for s, lab in zip(segments, labels)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", nargs="*", help="recording stems (default: all with an ASR file)")
    args = parser.parse_args()

    recordings = args.recordings or sorted(
        p.name.removesuffix(f".{ASR_MODEL}.json") for p in ASR_DIR.glob(f"*.{ASR_MODEL}.json")
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("diarization run: recordings=%d asr_model=%s", len(recordings), ASR_MODEL)
    encoder = VoiceEncoder(device="cpu")

    for rec in recordings:
        log.info("diarizing %s", rec)
        try:
            result = diarize(rec, encoder)
            (OUT_DIR / f"{rec}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        except Exception:
            log.exception("diarization failed for %s", rec)
            raise
        log.info("%s %s", rec, json.dumps(result["diagnostics"]))


if __name__ == "__main__":
    main()
