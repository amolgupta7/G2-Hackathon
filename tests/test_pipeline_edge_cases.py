"""Edge cases for silent / near-silent recordings (D5, D5.1). No models, DB or real audio:
Whisper, transcribe() and the voice encoder are stubbed; all file I/O goes to tmp_path.
"""
import json
import sys

import numpy as np
import pytest

from app.pipeline import asr, chunk, diarize


# ---------------------------------------------------------------- D5: ASR stats on a no-speech recording

def test_asr_silent_recording_logs_warning_and_continues(monkeypatch, tmp_path, caplog):
    results = {
        "silent": [],
        "speech": [{"id": 1, "start": 0.0, "end": 2.0, "text": "hello"},
                   {"id": 2, "start": 2.5, "end": 4.0, "text": "hi there"}],
    }
    monkeypatch.setattr(asr, "WhisperModel", lambda *a, **kw: object())
    monkeypatch.setattr(asr, "transcribe", lambda f, model, size: {
        "recording": f.stem, "model": size, "audio_duration_s": 5.0, "asr_seconds": 1.0,
        "real_time_factor": 0.2, "segments": results[f.stem]})
    monkeypatch.setattr(asr, "AUDIO_DIR", tmp_path)
    monkeypatch.setattr(asr, "OUT_DIR", tmp_path / "asr")
    monkeypatch.setattr(sys, "argv", ["asr", "--model", "small", "silent.wav", "speech.wav"])

    asr.main()  # previously: ZeroDivisionError in the stats line, and speech.wav never processed

    assert json.loads((tmp_path / "asr" / "silent.small.json").read_text())["segments"] == []
    assert (tmp_path / "asr" / "speech.small.json").exists()  # loop continued past the silent file
    assert "no speech segments detected" in caplog.text
    assert "segments=2" in caplog.text  # normal stats still logged for the next file


# ---------------------------------------------------------------- D5: chunk stats on zero segments

def test_chunk_function_with_no_segments_returns_no_chunks(monkeypatch, tmp_path):
    (tmp_path / "rec_silent.json").write_text(json.dumps({"segments": []}))
    monkeypatch.setattr(chunk, "DIAR_DIR", tmp_path)

    assert chunk.chunk("rec_silent") == {"recording": "rec_silent", "chunks": []}


def test_chunk_main_silent_recording_logs_warning_and_continues(monkeypatch, tmp_path, caplog):
    diar = tmp_path / "diar"
    diar.mkdir()
    (diar / "rec_silent.json").write_text(json.dumps({"segments": []}))
    (diar / "rec_speech.json").write_text(json.dumps({"segments": [
        {"id": 1, "start": 0.0, "end": 2.0, "text": "hello", "speaker": "A"},
        {"id": 2, "start": 2.2, "end": 3.0, "text": "again", "speaker": "A"},
        {"id": 3, "start": 3.5, "end": 5.0, "text": "hi", "speaker": "B"}]}))
    monkeypatch.setattr(chunk, "DIAR_DIR", diar)
    monkeypatch.setattr(chunk, "OUT_DIR", tmp_path / "chunks")
    monkeypatch.setattr(sys, "argv", ["chunk", "rec_silent", "rec_speech"])

    chunk.main()

    assert json.loads((tmp_path / "chunks" / "rec_silent.json").read_text())["chunks"] == []
    assert len(json.loads((tmp_path / "chunks" / "rec_speech.json").read_text())["chunks"]) == 2
    assert "no chunks (recording has no speech segments)" in caplog.text


# ---------------------------------------------------------------- D5.1: diarization with < 2 segments

class NeverCalledEncoder:
    def embed_utterance(self, _wav):
        raise AssertionError("encoder must not run when clustering is skipped")


class OrthogonalEncoder:
    """Returns a different unit vector per call, so two segments form two clean clusters."""

    def __init__(self):
        self.calls = 0

    def embed_utterance(self, _wav):
        v = np.zeros(256, np.float32)
        v[self.calls] = 1.0
        self.calls += 1
        return v


@pytest.fixture
def asr_file(monkeypatch, tmp_path):
    monkeypatch.setattr(diarize, "ASR_DIR", tmp_path)
    monkeypatch.setattr(diarize.librosa, "load", lambda *a, **kw: (np.zeros(16000 * 10, np.float32), 16000))

    def write(recording, segments):
        (tmp_path / f"{recording}.{diarize.ASR_MODEL}.json").write_text(json.dumps({"segments": segments}))
    return write


@pytest.mark.parametrize("n_segments", [0, 1])
def test_diarize_fewer_than_two_segments_is_single_speaker(asr_file, caplog, n_segments):
    segments = [{"id": i, "start": float(i), "end": i + 1.0, "text": f"t{i}"} for i in range(n_segments)]
    asr_file("rec_short", segments)

    result = diarize.diarize("rec_short", NeverCalledEncoder())  # previously: KMeans raised

    d = result["diagnostics"]
    assert d["single_speaker"] is True and d["segments"] == n_segments
    assert d["silhouette"] is None and d["mean_cos_between"] is None and d["mean_cos_within"] == [None, None]
    assert [s["speaker"] for s in result["segments"]] == ["A"] * n_segments
    assert "clustering skipped, single-speaker recording" in caplog.text


def test_diarize_exactly_two_segments_reports_undefined_metrics_as_null(asr_file):
    asr_file("rec_two", [{"id": 1, "start": 0.0, "end": 1.0, "text": "a"},
                         {"id": 2, "start": 1.5, "end": 2.5, "text": "b"}])

    d = diarize.diarize("rec_two", OrthogonalEncoder())["diagnostics"]

    assert d["single_speaker"] is False and d["segments_per_speaker"] == {"A": 1, "B": 1}
    assert d["silhouette"] is None  # needs 2 <= clusters <= n-1; silhouette_score used to raise here
    assert d["mean_cos_within"] == [None, None]  # one member per cluster: no distinct pairs
    assert d["mean_cos_between"] == pytest.approx(0.0)  # orthogonal voices
