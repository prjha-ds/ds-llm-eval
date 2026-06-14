"""Benchmark runner adapters: normalize benchmark outputs into EvalReports.

Two flavours, matching the survey in ``docs/research.md`` §6:

* **Loaders** that reuse our own metrics — :class:`LocalRetrievalBenchmark` runs
  a BEIR/MTEB-style retrieval benchmark from user-provided qrels/run through the
  existing ``ranking`` metrics, fully offline.
* **Harness adapters** — :class:`LmEvalAdapter`, :class:`SweBenchAdapter` delegate
  to the official packages (`lm_eval`, `swebench`), imported lazily behind their
  ``benchmarks-*`` extras. They **preflight** for the harness (and Docker, for
  SWE-bench) and raise a clear, actionable error rather than degrading silently.

Every adapter maps results onto :class:`~ds_llm_eval.core.EvalResult` named
``bench.<id>.<metric>`` via :meth:`BenchmarkSpec.to_eval_result`.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Sequence
from typing import Any

from ..core import EvalReport
from .registry import BenchmarkSpec, get_benchmark

Relevance = Collection[Hashable]


class Benchmark(ABC):
    """A runnable benchmark that yields an :class:`EvalReport`."""

    spec: BenchmarkSpec

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> EvalReport:
        """Execute the benchmark and return its results as an EvalReport."""


class LocalRetrievalBenchmark(Benchmark):
    """Score a retrieval benchmark from local qrels/run via the ``ranking`` metrics.

    Offline and dependency-free: feed already-loaded ranked lists + relevance
    judgments (e.g. a BEIR dataset you fetched yourself) and get the benchmark's
    headline metric (nDCG@k) as a ``bench.<id>.ndcg@k`` result.
    """

    def __init__(self, spec: BenchmarkSpec | None = None, *, k: int = 10) -> None:
        self.spec = spec if spec is not None else get_benchmark("beir")
        self.k = k

    def run(
        self,
        ranked: Sequence[Sequence[Hashable]],
        relevant: Sequence[Relevance],
    ) -> EvalReport:
        """Compute the benchmark's nDCG@k over ``(ranked, relevant)``."""
        from ..core import EvalResult
        from ..metrics import ranking

        ndcg = ranking.ndcg_at_k(ranked, relevant, k=self.k)
        # Name reflects the actual cutoff used (not the spec's fixed label).
        result = EvalResult(
            name=f"bench.{self.spec.id}.ndcg@{self.k}",
            value=ndcg.value,
            n=ndcg.n,
            params={"k": self.k},
            metadata={"benchmark": self.spec.id, "license": self.spec.license},
        )
        return EvalReport(results=[result])


def _pick_accuracy(task_scores: dict[str, Any]) -> float:
    """Pull the primary accuracy out of an lm-eval per-task score dict."""
    for key, value in task_scores.items():
        if key.split(",")[0] in {"acc", "exact_match", "acc_norm"}:
            return float(value)
    raise ValueError(f"no accuracy-like metric found in {sorted(task_scores)}")


class LmEvalAdapter(Benchmark):
    """Delegate to EleutherAI ``lm-evaluation-harness`` (MMLU/GPQA/MBPP, …)."""

    def __init__(self, spec: BenchmarkSpec | None = None) -> None:
        self.spec = spec if spec is not None else get_benchmark("mmlu")

    def run(self, *, model: str, tasks: Sequence[str], **kwargs: Any) -> EvalReport:
        """Run ``lm_eval.simple_evaluate`` and map each task's accuracy to a result."""
        try:
            from lm_eval import simple_evaluate
        except ImportError as exc:  # pragma: no cover - exercised via mocks
            raise ImportError(
                'LmEvalAdapter requires: pip install "ds-llm-eval[benchmarks-knowledge]"'
            ) from exc
        raw = simple_evaluate(model=model, tasks=list(tasks), **kwargs)
        results = raw["results"]
        return EvalReport(
            results=[
                self.spec.to_eval_result(_pick_accuracy(results[task]), task=task, model=model)
                for task in tasks
            ]
        )


class SweBenchAdapter(Benchmark):
    """Delegate to the ``swebench`` harness (% resolved). Needs Docker."""

    def __init__(self, spec: BenchmarkSpec | None = None) -> None:
        self.spec = spec if spec is not None else get_benchmark("swebench_verified")

    def preflight(self) -> None:
        """Verify the harness and Docker are available; raise a clear error if not."""
        try:
            import swebench  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised via mocks
            raise ImportError(
                'SweBenchAdapter requires: pip install "ds-llm-eval[benchmarks-swe]"'
            ) from exc
        if shutil.which("docker") is None:
            raise RuntimeError(
                "SWE-bench evaluation requires Docker on PATH (≈120GB disk). Install/start Docker."
            )

    def run(self, *, predictions_path: str, run_id: str, **kwargs: Any) -> EvalReport:
        """Preflight, then run the SWE-bench harness over a predictions file."""
        self.preflight()
        from swebench.harness.run_evaluation import (
            run_instances,
        )

        report = run_instances(predictions_path=predictions_path, run_id=run_id, **kwargs)
        resolved = float(report["resolved"]) if isinstance(report, dict) else float(report)
        return EvalReport(results=[self.spec.to_eval_result(resolved, run_id=run_id)])
