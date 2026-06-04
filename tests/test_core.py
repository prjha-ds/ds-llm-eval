"""Tests for the shared core contracts and registry."""

import pytest
from pydantic import ValidationError

from ds_llm_eval import EvalReport, EvalResult, get_metric, list_metrics
from ds_llm_eval.core.base import check_k, check_non_empty


def test_eval_result_is_floatable_and_frozen():
    r = EvalResult(name="ndcg@10", value=0.5, n=3, params={"k": 10})
    assert float(r) == 0.5
    with pytest.raises(ValidationError):  # frozen pydantic model rejects mutation
        r.value = 0.9  # type: ignore[misc]


def test_eval_report_flattens_to_dict():
    report = EvalReport(results=[EvalResult(name="a", value=1.0), EvalResult(name="b", value=0.0)])
    assert report.as_dict() == {"a": 1.0, "b": 0.0}


def test_registry_lookup_and_listing():
    names = list_metrics()
    assert "search.ndcg_at_k" in names
    assert "rec.hit_rate_at_k" in names
    assert "llm.token_f1" in names
    assert callable(get_metric("search.precision_at_k"))


def test_get_unknown_metric_raises():
    with pytest.raises(KeyError):
        get_metric("does.not.exist")


@pytest.mark.parametrize("bad", [0, -1, True, 2.5])
def test_check_k_rejects_invalid(bad):
    with pytest.raises(ValueError):
        check_k(bad)


def test_check_non_empty_rejects_empty():
    with pytest.raises(ValueError):
        check_non_empty([], name="x")
