"""Tests for the Langfuse sink — fully offline via an injected fake client."""

import builtins

import pytest

from ds_llm_eval import EvalResult
from ds_llm_eval.integrations import log_results_to_langfuse, run_langfuse_experiment


class _FakeDataset:
    def __init__(self):
        self.calls: list[dict] = []

    def run_experiment(self, **kwargs):
        self.calls.append(kwargs)
        return {"run": kwargs.get("name")}


class _FakeLangfuse:
    def __init__(self):
        self.calls: list[dict] = []
        self.dataset = _FakeDataset()

    def create_score(self, **kwargs):
        self.calls.append(kwargs)

    def get_dataset(self, name):
        self.requested_dataset = name
        return self.dataset


def test_logs_each_result_as_a_score():
    client = _FakeLangfuse()
    results = [
        EvalResult(name="ndcg@10", value=0.5, n=3, params={"k": 10}),
        EvalResult(name="mrr", value=0.8, n=3),
    ]
    count = log_results_to_langfuse(results, trace_id="trace-1", client=client)
    assert count == 2
    assert [c["name"] for c in client.calls] == ["ndcg@10", "mrr"]
    assert client.calls[0]["trace_id"] == "trace-1"
    assert client.calls[0]["value"] == 0.5


def test_missing_langfuse_extra_raises_clear_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("langfuse"):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"ds-llm-eval\[llm\]"):
        log_results_to_langfuse([EvalResult(name="x", value=1.0)], trace_id="t")


def test_run_langfuse_experiment_with_injected_client():
    client = _FakeLangfuse()

    def task(item):
        return item

    def evaluator(**kwargs):
        return None

    out = run_langfuse_experiment(
        "my-dataset", task, [evaluator], run_name="exp-1", client=client, metadata={"x": 1}
    )
    assert client.requested_dataset == "my-dataset"
    assert out == {"run": "exp-1"}
    call = client.dataset.calls[0]
    assert call["name"] == "exp-1"
    assert call["task"] is task
    assert call["evaluators"] == [evaluator]
    assert call["metadata"] == {"x": 1}
