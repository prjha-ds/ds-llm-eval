"""Optional delegation of IR metrics to validated external engines.

The pure-Python ``ranking`` group is the default. For parity-checking against
reference implementations (or speed on large runs), these backends convert our
``(ranked, relevant)`` batches into the engine's Qrels/Run format and delegate to
``ranx`` or ``pytrec_eval``. Engines are imported lazily (optional ``backends``
extra); a missing engine raises a clear, actionable error — never a silent fallback.

References
----------
.. [1] ranx — https://github.com/AmenRa/ranx
.. [2] pytrec_eval — https://github.com/cvangysel/pytrec_eval
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable, Mapping, Sequence

Relevance = Collection[Hashable] | Mapping[Hashable, float]


def _qrels_dict(relevant: Sequence[Relevance]) -> dict[str, dict[str, int]]:
    """Convert per-row relevance into a trec-style ``{qid: {doc: gain}}`` mapping."""
    out: dict[str, dict[str, int]] = {}
    for i, rel in enumerate(relevant):
        if isinstance(rel, Mapping):
            # Any positive gain is at least minimally relevant (grade >= 1); never
            # emit a zero-relevance entry by truncating a fractional gain.
            graded = {str(d): max(1, round(g)) for d, g in rel.items() if g > 0}
        else:
            graded = {str(d): 1 for d in rel}
        if graded:
            out[str(i)] = graded
    return out


def _run_dict(ranked: Sequence[Sequence[Hashable]]) -> dict[str, dict[str, float]]:
    """Convert ranked lists into a ``{qid: {doc: score}}`` run (rank → score)."""
    out: dict[str, dict[str, float]] = {}
    for i, ranks in enumerate(ranked):
        n = len(ranks)
        out[str(i)] = {str(d): float(n - j) for j, d in enumerate(ranks)}
    return out


class IRBackend(ABC):
    """Adapter that computes IR metrics via an external engine."""

    name: str

    @abstractmethod
    def evaluate(
        self,
        ranked: Sequence[Sequence[Hashable]],
        relevant: Sequence[Relevance],
        metrics: Sequence[str],
    ) -> dict[str, float]:
        """Return ``{metric: score}`` for the engine's metric names."""


class RanxBackend(IRBackend):
    """Delegate to `ranx <https://github.com/AmenRa/ranx>`_ (metric names like ``ndcg@10``)."""

    name = "ranx"

    def evaluate(
        self,
        ranked: Sequence[Sequence[Hashable]],
        relevant: Sequence[Relevance],
        metrics: Sequence[str],
    ) -> dict[str, float]:
        try:
            from ranx import Qrels, Run
            from ranx import evaluate as ranx_evaluate
        except ImportError as exc:  # pragma: no cover - exercised via mocks
            raise ImportError(
                "RanxBackend requires the optional 'backends' extra. "
                'Install it with: pip install "ds-llm-eval[backends]"'
            ) from exc
        qrels = Qrels(_qrels_dict(relevant))
        run = Run(_run_dict(ranked))
        scores = ranx_evaluate(qrels, run, list(metrics))
        if isinstance(scores, Mapping):
            return {m: float(scores[m]) for m in metrics}
        return {metrics[0]: float(scores)}


class PyTrecEvalBackend(IRBackend):
    """Delegate to `pytrec_eval <https://github.com/cvangysel/pytrec_eval>`_ (TREC measures)."""

    name = "pytrec_eval"

    def evaluate(
        self,
        ranked: Sequence[Sequence[Hashable]],
        relevant: Sequence[Relevance],
        metrics: Sequence[str],
    ) -> dict[str, float]:
        try:
            import pytrec_eval
        except ImportError as exc:  # pragma: no cover - exercised via mocks
            raise ImportError(
                "PyTrecEvalBackend requires the optional 'backends' extra. "
                'Install it with: pip install "ds-llm-eval[backends]"'
            ) from exc
        evaluator = pytrec_eval.RelevanceEvaluator(_qrels_dict(relevant), set(metrics))
        per_query = evaluator.evaluate(_run_dict(ranked))
        out: dict[str, float] = {}
        for metric in metrics:
            vals = [q[metric] for q in per_query.values() if metric in q]
            out[metric] = float(sum(vals) / len(vals)) if vals else 0.0
        return out


_BACKENDS: dict[str, type[IRBackend]] = {
    "ranx": RanxBackend,
    "pytrec_eval": PyTrecEvalBackend,
}


def get_ir_backend(name: str) -> IRBackend:
    """Construct an :class:`IRBackend` by name (``"ranx"`` or ``"pytrec_eval"``)."""
    key = name.lower()
    if key not in _BACKENDS:
        raise KeyError(f"Unknown IR backend {name!r}. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[key]()


def evaluate_with_backend(
    ranked: Sequence[Sequence[Hashable]],
    relevant: Sequence[Relevance],
    metrics: Sequence[str],
    *,
    backend: str = "ranx",
) -> dict[str, float]:
    """Compute IR ``metrics`` for ``(ranked, relevant)`` via an external backend."""
    return get_ir_backend(backend).evaluate(ranked, relevant, metrics)


__all__ = [
    "IRBackend",
    "PyTrecEvalBackend",
    "RanxBackend",
    "evaluate_with_backend",
    "get_ir_backend",
]
