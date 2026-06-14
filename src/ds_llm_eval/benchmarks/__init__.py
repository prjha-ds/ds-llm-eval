"""Benchmark-based evaluation: a catalog of standard ML/LLM/SWE benchmarks.

Today this exposes a metadata registry (``list_benchmarks``/``get_benchmark``) and
a mapping from a benchmark's headline score onto :class:`~ds_llm_eval.core.EvalResult`
(``BenchmarkSpec.to_eval_result``). Runner *adapters* that delegate to official
harnesses (swebench, lm_eval, mteb, ...) are on the roadmap — see ``docs/plan.md``.
"""

from .adapters import (
    Benchmark,
    LmEvalAdapter,
    LocalRetrievalBenchmark,
    SweBenchAdapter,
)
from .registry import BenchmarkSpec, get_benchmark, list_benchmarks

__all__ = [
    "Benchmark",
    "BenchmarkSpec",
    "LmEvalAdapter",
    "LocalRetrievalBenchmark",
    "SweBenchAdapter",
    "get_benchmark",
    "list_benchmarks",
]
