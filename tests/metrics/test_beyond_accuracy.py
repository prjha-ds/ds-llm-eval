"""Tests for BeyondAccuracyMetrics: coverage and novelty."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ds_llm_eval import beyond_accuracy as ba


def test_catalog_coverage_worked_example():
    # shown across rows (top-2) = {a,b,c}; catalog has 4 -> 0.75
    res = ba.catalog_coverage_at_k([["a", "b"], ["b", "c"]], {"a", "b", "c", "d"}, k=2)
    assert res.value == 0.75
    assert res.name == "beyond_accuracy.catalog_coverage@2"


def test_novelty_worked_example():
    # popularity a=1,b=1,c=2 (total 4); recs [a,c] -> [-log2(.25), -log2(.5)] = [2,1]
    res = ba.novelty_at_k([["a", "c"]], {"a": 1, "b": 1, "c": 2}, k=2)
    assert res.value == pytest.approx(1.5)


def test_empty_ranked_raises():
    with pytest.raises(ValueError):
        ba.catalog_coverage_at_k([], {"a"}, k=1)


def test_empty_catalog_raises():
    with pytest.raises(ValueError):
        ba.catalog_coverage_at_k([["a"]], set(), k=1)


def test_nonpositive_popularity_total_raises():
    with pytest.raises(ValueError):
        ba.novelty_at_k([["a"]], {"a": 0}, k=1)


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        ba.novelty_at_k([["a"]], {"a": 1}, k=0)


@given(
    rows=st.lists(st.lists(st.integers(0, 9), max_size=8), min_size=1, max_size=6),
    k=st.integers(1, 12),
)
def test_coverage_bounded_unit_interval(rows, k):
    assert 0.0 <= ba.catalog_coverage_at_k(rows, set(range(10)), k=k).value <= 1.0


# --- Milestone 2 additions --------------------------------------------------


def test_intra_list_diversity_all_dissimilar_is_one():
    res = ba.intra_list_diversity([[1, 2, 3]], lambda a, b: 0.0)
    assert res.value == 1.0


def test_intra_list_diversity_all_similar_is_zero():
    assert ba.intra_list_diversity([[1, 2, 3]], lambda a, b: 1.0).value == 0.0


def test_intra_list_diversity_skips_singletons():
    assert ba.intra_list_diversity([[1], [1, 2]], lambda a, b: 0.0).n == 1


def test_personalization_identical_users_is_zero():
    assert ba.personalization([["a", "b"], ["a", "b"]]).value == pytest.approx(0.0)


def test_personalization_disjoint_users_is_one():
    assert ba.personalization([["a", "b"], ["c", "d"]]).value == pytest.approx(1.0)


def test_personalization_needs_two_users():
    with pytest.raises(ValueError):
        ba.personalization([["a"]])


def test_gini_index_worked_example():
    # counts a=2,b=1,c=1 -> props sorted [.25,.25,.5], n=3 -> G=0.25
    assert ba.gini_index([["a", "b"], ["a", "c"]]).value == pytest.approx(0.25)


def test_gini_index_uniform_is_zero():
    assert ba.gini_index([["a", "b"]]).value == pytest.approx(0.0)


def test_gini_needs_two_items():
    with pytest.raises(ValueError):
        ba.gini_index([["a", "a"]])


def test_shannon_entropy_uniform_two_items_is_one_bit():
    assert ba.shannon_entropy([["a", "b"]]).value == pytest.approx(1.0)


def test_shannon_entropy_single_item_is_zero():
    assert ba.shannon_entropy([["a", "a"]]).value == pytest.approx(0.0)


def test_beyond_accuracy_accepts_dict_input():
    assert ba.gini_index({"u1": ["a", "b"], "u2": ["a", "c"]}).value == pytest.approx(0.25)
