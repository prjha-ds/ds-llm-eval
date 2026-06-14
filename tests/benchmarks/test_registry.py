"""Tests for the benchmark catalog/registry."""

import pytest

from ds_llm_eval import list_benchmarks
from ds_llm_eval.benchmarks import BenchmarkSpec, get_benchmark


def test_known_benchmarks_present():
    ids = list_benchmarks()
    for expected in ["swebench_verified", "humaneval", "mmlu", "gpqa_diamond", "beir", "tau_bench"]:
        assert expected in ids


def test_filter_by_domain():
    search_ids = list_benchmarks(domain="search")
    assert set(search_ids) == {"beir", "mteb_retrieval"}
    assert "swebench_verified" in list_benchmarks(domain="llm")
    assert list_benchmarks(domain="recommendation") == []  # none catalogued (yet)


def test_get_benchmark_returns_populated_spec():
    spec = get_benchmark("swebench_verified")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.domain == "llm"
    assert spec.metric == "resolved"
    assert spec.execution == "execution"
    assert spec.license == "MIT"
    assert spec.url.startswith("https://")


def test_get_unknown_benchmark_raises():
    with pytest.raises(KeyError):
        get_benchmark("not_a_benchmark")


def test_every_spec_is_well_formed():
    for bid in list_benchmarks():
        spec = get_benchmark(bid)
        assert spec.id == bid
        assert spec.task and spec.metric and spec.license
        assert spec.url.startswith("https://")
        # an installable benchmark should name the extra that provides its harness
        if spec.harness is not None and spec.execution != "gated":
            assert spec.extra is not None


def test_spec_maps_to_namespaced_eval_result():
    spec = get_benchmark("humaneval")
    result = spec.to_eval_result(0.82, n=164, k=1)
    assert result.name == "bench.humaneval.pass@1"
    assert result.value == 0.82
    assert result.n == 164
    assert result.params == {"k": 1}
    assert result.metadata["benchmark"] == "humaneval"
    assert result.metadata["license"] == "MIT"
