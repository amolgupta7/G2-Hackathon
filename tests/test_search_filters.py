"""Unit tests for search() post-processing rules, without a real DB or model.

- O29 (+O29.1/O29.2/D6): -exclusions are enforced on the fused results of BOTH arms (word-prefix match),
  stripped from the embedder input, and an exclusion-only query returns nothing.
- O23/O23.1 (+O23.2): the vector similarity floor (0.3; 0.45 without letters) applies only to unscoped
  searches; recording-scoped searches skip it; nothing left -> explicit empty result.

A FakeConn answers search()'s SQL from an in-memory chunk table. Its vector query applies the same rule as
VECTOR_SQL: keep a chunk only if cosine distance (1 - sim) <= params["max_dist"].
"""
import numpy as np
import pytest

from app.search import hybrid


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConn:
    """chunks: {id: dict(rec, idx, speaker, text, sim)}; keyword_ids: what the keyword arm 'matches'."""

    def __init__(self, chunks, keyword_ids=()):
        self.chunks, self.keyword_ids, self.calls = chunks, list(keyword_ids), []

    def _match_filters(self, c, params):
        return (params["rec"] is None or c["rec"] == params["rec"]) and \
               (params["speaker"] is None or c["speaker"] == params["speaker"])

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if sql in hybrid.KEYWORD_SQL.values():
            return _Result([(i,) for i in self.keyword_ids if self._match_filters(self.chunks[i], params)])
        if sql == hybrid.VECTOR_SQL:
            hits = [(1 - c["sim"], i) for i, c in self.chunks.items()
                    if self._match_filters(c, params) and 1 - c["sim"] <= params["max_dist"]]
            return _Result([(i,) for _, i in sorted(hits)][: params["k"]])
        if sql.startswith("SELECT id, lower(text)"):
            return _Result([(i, self.chunks[i]["text"].lower()) for i in params[0]])
        if sql == hybrid.RESULT_SQL:
            return _Result([(i, c["rec"], c["idx"], c["speaker"], 0.0, 1.0, c["text"], None, None, None, None)
                            for i, c in self.chunks.items() if i in params["ids"]])
        raise AssertionError(f"unexpected SQL in test: {sql[:60]}")

    def vector_params(self):
        return next(p for sql, p in self.calls if sql == hybrid.VECTOR_SQL)


def chunk(text, sim, rec="rec_a", speaker="A", idx=0):
    return {"rec": rec, "idx": idx, "speaker": speaker, "text": text, "sim": sim}


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """Record what would be embedded instead of loading MiniLM."""
    embedded = []
    monkeypatch.setattr(hybrid, "_model", lambda: None)
    monkeypatch.setattr(hybrid, "embed", lambda _m, texts: embedded.extend(texts) or [np.zeros(384, np.float32)])
    return embedded


def run(conn, query, **kw):
    return [h["chunk_id"] for h in hybrid.search(query, conn=conn, rerank=False, **kw)]


# ---------------------------------------------------------------- O29: -exclude post-filter

def test_exclusion_parsing_ignores_hyphenated_words():
    assert hybrid.excluded_terms('refund -invoice -"Account Credit"') == ["invoice", "account credit"]
    assert hybrid.excluded_terms("on-call self-serve") == []
    assert hybrid.positive_text('refund -invoice -"account credit"') == "refund"


def test_exclusion_drops_matches_from_both_arms(no_model):
    conn = FakeConn({
        1: chunk("refund issued today", 0.9),
        2: chunk("refund for invoice 032", 0.8),   # keyword arm hit containing the excluded word
        3: chunk("new invoice email", 0.85),       # vector-arm-only hit containing it
        4: chunk("refund timing is five days", 0.7),
    }, keyword_ids=[1, 2])

    ids = run(conn, "refund -invoice")

    assert set(ids) == {1, 4}
    assert no_model == ["refund"]  # O29.1: the excluded word never reaches the embedder


def test_exclusion_is_a_word_prefix_match_not_a_substring(no_model):
    conn = FakeConn({
        1: chunk("we start the art class now", 0.9),
        2: chunk("let's start the migration", 0.8),
        3: chunk("artwork deadline", 0.7),
    })

    ids = run(conn, "start -art")

    assert ids == [2]  # "start" survives; "art" and the prefix "artwork" are dropped


def test_phrase_exclusion(no_model):
    conn = FakeConn({1: chunk("add an account credit", 0.9), 2: chunk("credit card account", 0.8)})

    assert run(conn, 'credit -"account credit"') == [2]


def test_excluded_results_are_backfilled_up_to_n(no_model):
    conn = FakeConn({i: chunk("invoice copy" if i == 1 else f"refund step {i}", 0.9 - i / 100) for i in range(1, 6)})

    ids = run(conn, "refund -invoice", n=3)

    assert len(ids) == 3 and 1 not in ids


def test_exclusion_only_query_returns_empty_without_db_or_model(no_model):
    conn = FakeConn({1: chunk("anything", 0.99)}, keyword_ids=[1])

    assert run(conn, "-invoice") == []
    assert conn.calls == [] and no_model == []  # D6: short-circuits before any SQL or embedding


# ---------------------------------------------------------------- O23 / O23.1: similarity floor

def test_unscoped_query_below_floor_returns_explicit_empty():
    conn = FakeConn({1: chunk("unrelated turn", 0.25), 2: chunk("another turn", 0.10)})

    assert run(conn, "zzzqqq nonexistent gibberish term") == []
    assert conn.vector_params()["max_dist"] == pytest.approx(0.7)  # floor 0.3


def test_unscoped_floor_keeps_only_hits_at_or_above_0_3():
    conn = FakeConn({1: chunk("close match", 0.62), 2: chunk("borderline", 0.30), 3: chunk("weak", 0.29)})

    assert set(run(conn, "close match")) == {1, 2}


def test_recording_scoped_search_skips_the_floor():
    conn = FakeConn({
        1: chunk("I can add a small credit to your account", 0.18, rec="rec04"),
        2: chunk("the refund is submitted", 0.12, rec="rec04"),
        3: chunk("other recording turn", 0.90, rec="rec01"),
    })

    ids = run(conn, "what compensation did the agent offer", recording="rec04", speaker="A")

    assert ids == [1, 2]  # low-similarity but correct turns still returned; other recordings filtered out
    assert conn.vector_params()["max_dist"] == pytest.approx(2.0)  # full cosine range = no floor


def test_query_without_letters_uses_stricter_floor():
    conn = FakeConn({1: chunk("my account is 447129", 0.403), 2: chunk("reference r-8812", 0.47)})

    assert run(conn, "1234 5678 9012") == [2]
    assert conn.vector_params()["max_dist"] == pytest.approx(0.55)  # floor 0.45 (O23.2)


def test_keyword_hit_survives_even_when_vector_arm_is_empty():
    conn = FakeConn({1: chunk("r-8812 is your reference", 0.05)}, keyword_ids=[1])

    assert run(conn, "r-8812") == [1]  # the floor only filters the vector arm


def test_speaker_filter_without_recording_is_rejected():
    with pytest.raises(ValueError, match="requires a recording"):
        hybrid.search("refund", speaker="A", conn=FakeConn({}), rerank=False)
