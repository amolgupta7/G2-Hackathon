"""Integration tests for GET /search, through FastAPI's TestClient (the real app + its startup).

The app starts exactly as in production (embedding model, reranker, DB pool, warm-up) and queries the already
ingested database READ-ONLY: /search never writes. Skipped if that DB is unreachable or has no recordings.
Heavy (loads the models): run separately on small machines, e.g. `python -m pytest -m integration`.
"""
import psycopg
import pytest

pytestmark = pytest.mark.integration

RECORDING = "rec04_support_billing"


@pytest.fixture(scope="module")
def client():
    try:
        from app.config import DATABASE_URL
    except RuntimeError as exc:
        pytest.skip(f"no DB settings: {exc}")
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn:
            ok = conn.execute("SELECT 1 FROM recordings WHERE id = %s AND status = 'completed'",
                              (RECORDING,)).fetchone()
    except psycopg.OperationalError as exc:
        pytest.skip(f"database not reachable ({exc.__class__.__name__}); run `docker compose up -d`")
    if not ok:
        pytest.skip(f"{RECORDING} not ingested; run the pipeline first")

    from fastapi.testclient import TestClient

    from app.api.main import app
    with TestClient(app) as c:  # the context manager runs the lifespan (pool, models, checks, warm-up)
        yield c


def test_valid_query_returns_200_with_results(client):
    r = client.get("/search", params={"q": "refund", "n": 5})

    assert r.status_code == 200
    body = r.json()
    assert 0 < body["count"] <= 5 and body["count"] == len(body["results"])
    hit = body["results"][0]
    assert hit["recording_id"] == RECORDING
    # The reranker may put an answer turn without the literal word first ("three to five business days").
    assert any("refund" in h["text"].lower() for h in body["results"])
    assert {"speaker", "start_s", "end_s", "prev", "next"} <= hit.keys()


def test_valid_recording_scoped_query(client):
    r = client.get("/search", params={"q": "refund", "recording": RECORDING, "speaker": "A", "n": 3})

    assert r.status_code == 200
    assert all(h["recording_id"] == RECORDING and h["speaker"] == "A" for h in r.json()["results"])


def test_unknown_recording_returns_404(client):
    r = client.get("/search", params={"q": "refund", "recording": "nope"})

    assert r.status_code == 404  # O30
    assert r.json()["detail"] == "unknown recording"


def test_empty_recording_is_rejected_not_silently_empty(client):
    # D8 (L56): recording="" is rejected by validation (min_length=1) before the O30 lookup, so it's a 422,
    # not a 404, and never the old silent "200 with 0 results".
    r = client.get("/search", params={"q": "refund", "recording": ""})

    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["query", "recording"]


def test_gibberish_query_returns_empty_results(client):
    r = client.get("/search", params={"q": "zzzqqq nonexistent gibberish term"})

    assert r.status_code == 200  # O23: explicit empty result, not an error and not padded results
    assert r.json()["count"] == 0 and r.json()["results"] == []
