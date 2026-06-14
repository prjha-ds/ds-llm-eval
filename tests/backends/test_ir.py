"""Tests for IR backend delegation — via injected fake engines + missing-dep paths."""

import sys
import types

import pytest

from ds_llm_eval.backends import (
    PyTrecEvalBackend,
    RanxBackend,
    evaluate_with_backend,
    get_ir_backend,
)
from ds_llm_eval.backends.ir import _qrels_dict, _run_dict


def test_qrels_and_run_conversion():
    assert _qrels_dict([{"a"}, {"b": 2, "c": 0}]) == {"0": {"a": 1}, "1": {"b": 2}}
    run = _run_dict([["a", "b"]])
    assert run == {"0": {"a": 2.0, "b": 1.0}}


def test_qrels_fractional_gain_is_at_least_grade_one():
    # 0 < gain < 1 must NOT truncate to a zero-relevance entry (regression).
    assert _qrels_dict([{"a": 0.5, "c": 2.0, "b": 0}]) == {"0": {"a": 1, "c": 2}}


def test_ranx_backend_with_injected_engine(monkeypatch):
    fake = types.ModuleType("ranx")
    captured: dict = {}

    class Qrels:
        def __init__(self, d):
            captured["qrels"] = d

    class Run:
        def __init__(self, d):
            captured["run"] = d

    def evaluate(qrels, run, metrics):
        captured["metrics"] = metrics
        return {m: 0.5 for m in metrics}

    fake.Qrels, fake.Run, fake.evaluate = Qrels, Run, evaluate
    monkeypatch.setitem(sys.modules, "ranx", fake)

    out = RanxBackend().evaluate([["a", "b"]], [{"a"}], ["ndcg@2", "map"])
    assert out == {"ndcg@2": 0.5, "map": 0.5}
    assert captured["qrels"] == {"0": {"a": 1}}
    assert captured["run"] == {"0": {"a": 2.0, "b": 1.0}}


def test_ranx_backend_single_metric_float(monkeypatch):
    fake = types.ModuleType("ranx")
    fake.Qrels = lambda d: None
    fake.Run = lambda d: None
    fake.evaluate = lambda q, r, m: 0.42
    monkeypatch.setitem(sys.modules, "ranx", fake)
    assert RanxBackend().evaluate([["a"]], [{"a"}], ["map"]) == {"map": 0.42}


def test_pytrec_eval_backend_averages_per_query(monkeypatch):
    fake = types.ModuleType("pytrec_eval")

    class RelevanceEvaluator:
        def __init__(self, qrels, measures):
            pass

        def evaluate(self, run):
            return {"0": {"map": 0.4, "ndcg": 0.6}, "1": {"map": 0.6, "ndcg": 0.8}}

    fake.RelevanceEvaluator = RelevanceEvaluator
    monkeypatch.setitem(sys.modules, "pytrec_eval", fake)

    out = PyTrecEvalBackend().evaluate([["a"], ["b"]], [{"a"}, {"b"}], ["map", "ndcg"])
    assert out == {"map": pytest.approx(0.5), "ndcg": pytest.approx(0.7)}


def test_missing_engine_raises_clear_error():
    # ranx / pytrec_eval are not installed in the test env
    with pytest.raises(ImportError, match=r"ds-llm-eval\[backends\]"):
        RanxBackend().evaluate([["a"]], [{"a"}], ["map"])
    with pytest.raises(ImportError, match=r"ds-llm-eval\[backends\]"):
        evaluate_with_backend([["a"]], [{"a"}], ["map"], backend="pytrec_eval")


def test_get_ir_backend_unknown_raises():
    with pytest.raises(KeyError):
        get_ir_backend("does_not_exist")
