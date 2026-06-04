"""Tests for recommendation metrics: worked examples, edge cases, properties."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ds_llm_eval import recommendation as rec

# --- worked numeric examples ------------------------------------------------


def test_hit_rate_worked_example():
    # user1 top-2 [a,b] hits {a}; user2 top-2 [x,y] misses {q} -> mean 0.5
    res = rec.hit_rate_at_k([["a", "b", "c"], ["x", "y", "z"]], [{"a"}, {"q"}], k=2)
    assert res.value == 0.5


def test_ndcg_worked_example():
    idcg = 1.0 + 1.0 / math.log2(3)  # 2 relevant items -> ideal at ranks 1,2
    res = rec.ndcg_at_k([["a", "b", "c"]], [{"a", "c"}], k=3)
    assert res.value == pytest.approx(1.5 / idcg)


def test_catalog_coverage_worked_example():
    # shown across users (top-2) = {a,b,c}; catalog has 4 -> 0.75
    res = rec.catalog_coverage_at_k([["a", "b"], ["b", "c"]], {"a", "b", "c", "d"}, k=2)
    assert res.value == 0.75


def test_novelty_worked_example():
    # popularity a=1,b=1,c=2 (total 4); recs [a,c] -> [-log2(.25), -log2(.5)] = [2,1]
    res = rec.novelty_at_k([["a", "c"]], {"a": 1, "b": 1, "c": 2}, k=2)
    assert res.value == pytest.approx(1.5)


# --- edge cases -------------------------------------------------------------


def test_hit_rate_skips_users_without_holdout():
    res = rec.hit_rate_at_k([["a"], ["b"]], [{"a"}, set()], k=1)
    assert res.value == 1.0 and res.n == 1


def test_empty_recommended_raises():
    with pytest.raises(ValueError):
        rec.hit_rate_at_k([], [], k=1)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        rec.ndcg_at_k([["a"]], [{"a"}, {"b"}], k=1)


def test_nonpositive_popularity_total_raises():
    with pytest.raises(ValueError):
        rec.novelty_at_k([["a"]], {"a": 0}, k=1)


# --- property tests ---------------------------------------------------------

_recs = st.lists(st.integers(0, 9), min_size=1, max_size=10, unique=True)
_truth = st.sets(st.integers(0, 9), max_size=10)


@given(recs=_recs, truth=_truth, k=st.integers(1, 12))
def test_hit_rate_bounded_unit_interval(recs, truth, k):
    assert 0.0 <= rec.hit_rate_at_k([recs], [truth], k=k).value <= 1.0
