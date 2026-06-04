"""Tests for IR/search metrics: worked examples, edge cases, properties."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ds_llm_eval import search

# --- worked numeric examples (checked by hand) ------------------------------


def test_precision_at_k_worked_example():
    # top-2 of [a,b,c,d] is [a,b]; relevant={a,c} -> 1 hit / 2 = 0.5
    assert search.precision_at_k([["a", "b", "c", "d"]], [{"a", "c"}], k=2).value == 0.5


def test_recall_at_k_worked_example():
    # top-4 contains both relevant items {a,c} -> recall 1.0
    assert search.recall_at_k([["a", "b", "c", "d"]], [{"a", "c"}], k=4).value == 1.0


def test_f1_at_k_worked_example():
    # p=0.5, r=0.5 -> f1=0.5
    assert search.f1_at_k([["a", "b", "c", "d"]], [{"a", "c"}], k=2).value == pytest.approx(0.5)


def test_mrr_worked_example():
    # q1 first relevant at rank 1 (1.0); q2 first relevant at rank 3 (1/3); mean=0.6667
    res = search.mrr([["a", "x"], ["x", "y", "z"]], [{"a"}, {"z"}])
    assert res.value == pytest.approx((1.0 + 1 / 3) / 2)


def test_average_precision_worked_example():
    # ranks [a,b,c], relevant {a,c}: AP = (1/1 + 2/3) / 2 = 0.8333
    res = search.average_precision([["a", "b", "c"]], [{"a", "c"}])
    assert res.value == pytest.approx((1.0 + 2 / 3) / 2)


def test_ndcg_at_k_binary_worked_example():
    # gains [1,0,1]; dcg=1+0.5=1.5; idcg=1+1/log2(3); ndcg=1.5/idcg
    idcg = 1.0 + 1.0 / math.log2(3)
    res = search.ndcg_at_k([["a", "b", "c"]], [{"a", "c"}], k=3)
    assert res.value == pytest.approx(1.5 / idcg)


def test_ndcg_at_k_graded_worked_example():
    # graded gains a=3,c=2; ranked [a,b,c]: dcg=3+0+1=4; idcg=3+2/log2(3)
    idcg = 3.0 + 2.0 / math.log2(3)
    res = search.ndcg_at_k([["a", "b", "c"]], [{"a": 3, "c": 2, "b": 0}], k=3)
    assert res.value == pytest.approx(4.0 / idcg)


# --- edge cases -------------------------------------------------------------


def test_recall_skips_queries_with_no_relevant():
    # the second query has no relevant items and must not drag the mean to 0
    res = search.recall_at_k([["a"], ["x"]], [{"a"}, set()], k=1)
    assert res.value == 1.0
    assert res.n == 1


def test_k_larger_than_list_is_fine():
    assert search.precision_at_k([["a"]], [{"a"}], k=10).value == pytest.approx(1 / 10)


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        search.precision_at_k([], [], k=1)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        search.mrr([["a"]], [{"a"}, {"b"}])


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        search.ndcg_at_k([["a"]], [{"a"}], k=0)


# --- property tests ---------------------------------------------------------

_items = st.lists(st.integers(0, 9), min_size=1, max_size=10, unique=True)
_rel = st.sets(st.integers(0, 9), max_size=10)


@given(ranked=_items, relevant=_rel, k=st.integers(1, 12))
def test_precision_bounded_unit_interval(ranked, relevant, k):
    assert 0.0 <= search.precision_at_k([ranked], [relevant], k=k).value <= 1.0


@given(ranked=_items, relevant=_rel, k=st.integers(1, 12))
def test_ndcg_bounded_unit_interval(ranked, relevant, k):
    assert 0.0 <= search.ndcg_at_k([ranked], [relevant], k=k).value <= 1.0
