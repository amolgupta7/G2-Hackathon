"""Unit tests for Reciprocal Rank Fusion (app.search.hybrid.rrf).

rrf() returns scores only; search() orders them with key (-score, chunk_id), so ties go to the lower id.
`ranking()` below mirrors that key so these tests check the order search() actually returns.
"""
import pytest

from app.search.hybrid import rrf


def ranking(scores: dict[int, float]) -> list[int]:
    return sorted(scores, key=lambda i: (-scores[i], i))


def test_both_arms_agree_keeps_shared_order():
    scores = rrf([[10, 20, 30], [10, 20, 30]], k=60)

    assert ranking(scores) == [10, 20, 30]
    assert scores[10] == pytest.approx(2 / 61)  # rank 1 in both arms
    assert scores[30] == pytest.approx(2 / 63)


def test_only_keyword_arm_has_hits():
    scores = rrf([[5, 3, 9], []], k=60)

    assert ranking(scores) == [5, 3, 9]
    assert scores[5] == pytest.approx(1 / 61)


def test_only_vector_arm_has_hits():
    scores = rrf([[], [7, 1, 4]], k=60)

    assert ranking(scores) == [7, 1, 4]  # arm order preserved, not id order


def test_no_hits_in_either_arm_gives_empty_result():
    assert rrf([[], []]) == {}


def test_disagreement_doc_in_both_arms_beats_single_arm_top_hits():
    # 1 is 1st in keyword and 2nd in vector; 2 and 3 appear in only one arm each.
    scores = rrf([[1, 2], [3, 1]], k=60)

    assert ranking(scores) == [1, 3, 2]
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)


def test_symmetric_disagreement_ties_break_by_lower_id():
    # Mirror-image ranks give identical scores; search() then puts the lower chunk id first.
    scores = rrf([[20, 10], [10, 20]], k=60)

    assert scores[10] == pytest.approx(scores[20])
    assert ranking(scores) == [10, 20]


def test_keyword_weight_breaks_the_tie_toward_the_keyword_top_hit():
    # Same mirror-image ranks, but the keyword arm weighs 1.5 (the default in search()).
    scores = rrf([[20, 10], [10, 20]], k=60, weights=[1.5, 1.0])

    assert ranking(scores) == [20, 10]


def test_smaller_k_widens_the_gap_between_ranks():
    wide = rrf([[1, 2]], k=1)
    narrow = rrf([[1, 2]], k=60)

    assert wide[1] - wide[2] > narrow[1] - narrow[2]
