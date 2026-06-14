"""Tests for the Evaluator convenience runner."""

import io

import pytest

from ds_llm_eval import EvalReport, Evaluator, MetricSpec
from ds_llm_eval.logging import ConsoleLogger


def test_runs_metric_set_and_returns_report():
    ev = Evaluator(
        [
            MetricSpec("ranking.ndcg_at_k", {"k": 3}),
            ("ranking.precision_at_k", {"k": 3}),
            "ranking.mrr",
        ]
    )
    report = ev.run([["a", "b", "c"]], [{"a", "c"}])
    assert isinstance(report, EvalReport)
    assert set(report.as_dict()) == {"ranking.ndcg@3", "ranking.precision@3", "ranking.mrr"}
    assert ev.metric_names == ["ranking.ndcg_at_k", "ranking.precision_at_k", "ranking.mrr"]


def test_logs_each_result_when_logger_given():
    stream = io.StringIO()
    ev = Evaluator(["ranking.mrr"], logger=ConsoleLogger(stream=stream, run_id="r"))
    ev.run([["a"]], [{"a"}])
    assert "ranking.mrr = 1.0000" in stream.getvalue()


def test_empty_metrics_raises():
    with pytest.raises(ValueError):
        Evaluator([])


def test_bad_spec_raises():
    with pytest.raises(ValueError):
        Evaluator([123])  # type: ignore[list-item]
