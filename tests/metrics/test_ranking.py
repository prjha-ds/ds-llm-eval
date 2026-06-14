"""Tests for RankingMetrics: worked examples, edge cases, properties."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ds_llm_eval import ranking
from ds_llm_eval.metrics import RankingMetrics

# --- worked numeric examples (checked by hand) ------------------------------


def test_precision_at_k_worked_example():
    # top-2 of [a,b,c,d] is [a,b]; relevant={a,c} -> 1 hit / 2 = 0.5
    assert ranking.precision_at_k([["a", "b", "c", "d"]], [{"a", "c"}], k=2).value == 0.5


def test_recall_at_k_worked_example():
    assert ranking.recall_at_k([["a", "b", "c", "d"]], [{"a", "c"}], k=4).value == 1.0


def test_f1_at_k_worked_example():
    assert ranking.f1_at_k([["a", "b", "c", "d"]], [{"a", "c"}], k=2).value == pytest.approx(0.5)


def test_mrr_worked_example():
    res = ranking.mrr([["a", "x"], ["x", "y", "z"]], [{"a"}, {"z"}])
    assert res.value == pytest.approx((1.0 + 1 / 3) / 2)


def test_average_precision_worked_example():
    res = ranking.average_precision([["a", "b", "c"]], [{"a", "c"}])
    assert res.value == pytest.approx((1.0 + 2 / 3) / 2)
    assert res.name == "ranking.map"


def test_ndcg_at_k_binary_worked_example():
    idcg = 1.0 + 1.0 / math.log2(3)
    res = ranking.ndcg_at_k([["a", "b", "c"]], [{"a", "c"}], k=3)
    assert res.value == pytest.approx(1.5 / idcg)


def test_ndcg_at_k_graded_worked_example():
    idcg = 3.0 + 2.0 / math.log2(3)
    res = ranking.ndcg_at_k([["a", "b", "c"]], [{"a": 3, "c": 2, "b": 0}], k=3)
    assert res.value == pytest.approx(4.0 / idcg)


def test_hit_rate_worked_example():
    # row1 top-2 [a,b] hits {a}; row2 top-2 [x,y] misses {q} -> 0.5
    res = ranking.hit_rate_at_k([["a", "b", "c"], ["x", "y", "z"]], [{"a"}, {"q"}], k=2)
    assert res.value == 0.5


def test_result_names_are_ranking_namespaced():
    assert ranking.ndcg_at_k([["a"]], [{"a"}], k=1).name == "ranking.ndcg@1"
    assert ranking.mrr([["a"]], [{"a"}]).name == "ranking.mrr"


# --- edge cases -------------------------------------------------------------


def test_recall_skips_rows_with_no_relevant():
    res = ranking.recall_at_k([["a"], ["x"]], [{"a"}, set()], k=1)
    assert res.value == 1.0 and res.n == 1


def test_k_larger_than_list_is_fine():
    assert ranking.precision_at_k([["a"]], [{"a"}], k=10).value == pytest.approx(1 / 10)


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        ranking.precision_at_k([], [], k=1)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        ranking.mrr([["a"]], [{"a"}, {"b"}])


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        ranking.ndcg_at_k([["a"]], [{"a"}], k=0)


def test_singleton_and_class_agree():
    a = ranking.precision_at_k([["a", "b"]], [{"a"}], k=2).value
    b = RankingMetrics().precision_at_k([["a", "b"]], [{"a"}], k=2).value
    assert a == b


# --- property tests ---------------------------------------------------------

_items = st.lists(st.integers(0, 9), min_size=1, max_size=10, unique=True)
_rel = st.sets(st.integers(0, 9), max_size=10)


@given(ranked=_items, relevant=_rel, k=st.integers(1, 12))
def test_precision_bounded_unit_interval(ranked, relevant, k):
    assert 0.0 <= ranking.precision_at_k([ranked], [relevant], k=k).value <= 1.0


@given(ranked=_items, relevant=_rel, k=st.integers(1, 12))
def test_ndcg_bounded_unit_interval(ranked, relevant, k):
    assert 0.0 <= ranking.ndcg_at_k([ranked], [relevant], k=k).value <= 1.0


# --- Milestone 2 additions: r_precision, bpref, rbp -------------------------


def test_r_precision_worked_example():
    # R=2, top-2 of [a,b,c,d] = [a,b], 1 relevant -> 0.5
    assert ranking.r_precision([["a", "b", "c", "d"]], [{"a", "c"}]).value == 0.5


def test_bpref_worked_example():
    # ranks [a,x,c], rel {a,c} (R=2): a -> 1-0/2=1; c after 1 nonrel -> 1-0.5=0.5; mean 0.75
    assert ranking.bpref([["a", "x", "c"]], [{"a", "c"}]).value == pytest.approx(0.75)


def test_rbp_worked_example():
    # p=0.5; relevant at rank 2 -> gain p^1=0.5; RBP=(1-0.5)*0.5=0.25
    assert ranking.rbp([["x", "a"]], [{"a"}], p=0.5).value == pytest.approx(0.25)


def test_rbp_rejects_bad_p():
    with pytest.raises(ValueError):
        ranking.rbp([["a"]], [{"a"}], p=1.0)


def test_r_precision_and_bpref_skip_empty_relevant():
    assert ranking.r_precision([["a"], ["b"]], [{"a"}, set()]).n == 1
    assert ranking.bpref([["a"], ["b"]], [{"a"}, set()]).n == 1


# --- polymorphic dict inputs ------------------------------------------------


def test_dict_keyed_inputs_align_by_key():
    res = ranking.precision_at_k({"q1": ["a", "b"], "q2": ["c"]}, {"q1": {"a"}, "q2": {"c"}}, k=2)
    # q1: 1/2; q2: 1/2 (top-2 of [c] is [c], 1 hit / k=2) -> mean 0.5
    assert res.value == pytest.approx(0.5)
    assert res.n == 2


def test_dict_inputs_missing_key_raises():
    with pytest.raises(ValueError):
        ranking.mrr({"q1": ["a"]}, {"q2": {"a"}})


def test_mixed_mapping_and_sequence_raises():
    with pytest.raises(ValueError):
        ranking.precision_at_k({"q1": ["a"]}, [{"a"}], k=1)
