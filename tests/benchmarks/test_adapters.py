"""Tests for benchmark runner adapters (offline loader + mocked harnesses)."""

import sys
import types

import pytest

from ds_llm_eval import ranking
from ds_llm_eval.benchmarks import (
    LmEvalAdapter,
    LocalRetrievalBenchmark,
    SweBenchAdapter,
    get_benchmark,
)


def test_local_retrieval_benchmark_runs_offline():
    ranked = [["a", "b", "c"]]
    relevant = [{"a", "c"}]
    report = LocalRetrievalBenchmark(k=3).run(ranked, relevant)
    [result] = report.results
    assert result.name == "bench.beir.ndcg@3"
    assert result.value == pytest.approx(ranking.ndcg_at_k(ranked, relevant, k=3).value)
    assert result.metadata["benchmark"] == "beir"


def test_lm_eval_adapter_with_injected_harness(monkeypatch):
    fake = types.ModuleType("lm_eval")

    def simple_evaluate(*, model, tasks, **kwargs):
        return {"results": {t: {"acc,none": 0.7} for t in tasks}}

    fake.simple_evaluate = simple_evaluate
    monkeypatch.setitem(sys.modules, "lm_eval", fake)

    report = LmEvalAdapter(get_benchmark("mmlu")).run(model="hf", tasks=["mmlu"])
    [result] = report.results
    assert result.name == "bench.mmlu.accuracy"
    assert result.value == pytest.approx(0.7)


def test_lm_eval_adapter_missing_harness_raises():
    with pytest.raises(ImportError, match=r"benchmarks-knowledge"):
        LmEvalAdapter().run(model="hf", tasks=["mmlu"])


def test_swebench_preflight_missing_package_raises():
    with pytest.raises(ImportError, match=r"benchmarks-swe"):
        SweBenchAdapter().preflight()


def test_swebench_preflight_missing_docker_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "swebench", types.ModuleType("swebench"))
    monkeypatch.setattr("ds_llm_eval.benchmarks.adapters.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="Docker"):
        SweBenchAdapter().preflight()
