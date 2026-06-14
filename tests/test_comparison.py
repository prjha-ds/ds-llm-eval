"""Tests for RunComparison and its pure-Python significance tests."""

import pytest

from ds_llm_eval import RunComparison
from ds_llm_eval.comparison import _paired_ttest_p, _t_two_sided_p, _wilcoxon_p

# --- statistics correctness (vs. known reference values) --------------------


def test_t_two_sided_p_matches_reference():
    # scipy.stats.t.sf(2.0, 10) * 2 == 0.07339...
    assert _t_two_sided_p(2.0, 10) == pytest.approx(0.07339, abs=1e-4)


def test_t_two_sided_p_zero_statistic_is_one():
    assert _t_two_sided_p(0.0, 10) == pytest.approx(1.0)


def test_paired_ttest_all_zero_diffs_is_one():
    assert _paired_ttest_p([0.0, 0.0, 0.0]) == 1.0


def test_paired_ttest_constant_positive_shift_is_significant():
    # identical nonzero diffs -> zero variance, mean != 0 -> p = 0.0
    assert _paired_ttest_p([0.2, 0.2, 0.2, 0.2]) == 0.0


def test_wilcoxon_all_zero_is_one():
    assert _wilcoxon_p([0.0, 0.0]) == 1.0


def test_wilcoxon_tie_correction_lowers_pvalue():
    # All-positive tied diffs: tie correction shrinks variance -> smaller (or equal) p.
    diffs = [0.2, 0.2, 0.2, 0.2, 0.2]
    p = _wilcoxon_p(diffs)
    assert 0.0 <= p <= 1.0
    # with 5 equal positive diffs the signed-rank sum is maximal -> clearly small p
    assert p < 0.1


# --- RunComparison ----------------------------------------------------------


def _runs():
    # 4 queries; run "good" ranks the relevant doc first, "bad" ranks it last
    good = [["r", "x", "y"]] * 4
    bad = [["x", "y", "r"]] * 4
    relevant = [{"r"}] * 4
    return {"good": good, "bad": bad}, relevant


def test_compare_means_and_structure():
    runs, relevant = _runs()
    result = RunComparison("ranking.mrr").compare(runs, relevant, baseline="bad")
    assert result.baseline == "bad"
    assert result.n_queries == 4
    assert result.means["good"] == pytest.approx(1.0)  # relevant at rank 1
    assert result.means["bad"] == pytest.approx(1 / 3)  # relevant at rank 3
    comp = result.comparisons["good"]
    assert comp["mean_diff"] == pytest.approx(1.0 - 1 / 3)
    assert comp["significant"] is True
    assert "good" in result.significant_runs()


def test_compare_identical_runs_not_significant():
    same = [["r", "x"]] * 5
    result = RunComparison("ranking.ndcg_at_k", params={"k": 2}).compare(
        {"a": same, "b": same}, [{"r"}] * 5
    )
    assert result.comparisons["b"]["p_value"] == pytest.approx(1.0)
    assert result.comparisons["b"]["significant"] is False


def test_wilcoxon_test_runs():
    runs, relevant = _runs()
    result = RunComparison("ranking.mrr", test="wilcoxon").compare(runs, relevant)
    assert result.test == "wilcoxon"
    assert "good" in result.comparisons or "bad" in result.comparisons


def test_needs_two_runs():
    with pytest.raises(ValueError):
        RunComparison("ranking.mrr").compare({"only": [["a"]]}, [{"a"}])


def test_unknown_baseline_raises():
    runs, relevant = _runs()
    with pytest.raises(ValueError):
        RunComparison("ranking.mrr").compare(runs, relevant, baseline="ghost")


def test_invalid_test_raises():
    with pytest.raises(ValueError):
        RunComparison("ranking.mrr", test="bogus")  # type: ignore[arg-type]
