"""Metric protocol, a lightweight registry, and shared validation helpers.

The registry lets callers discover and look up metrics by name across all
domains (``ds_llm_eval.list_metrics()``), which is also what the Langfuse
integration uses to attach metric scores to traces.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from .types import EvalResult


@runtime_checkable
class Metric(Protocol):
    """Callable contract every metric satisfies.

    Implementations are pure functions: deterministic and side-effect free.
    """

    def __call__(self, *args: object, **kwargs: object) -> EvalResult: ...


F = TypeVar("F", bound=Callable[..., EvalResult])

_REGISTRY: dict[str, Metric] = {}


def register(name: str) -> Callable[[F], F]:
    """Decorator registering a metric under ``name`` (e.g. ``search.ndcg``).

    Returns the function unchanged so the decorated symbol keeps its own signature.
    """

    def _wrap(fn: F) -> F:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Metric {name!r} is already registered.")
        _REGISTRY[key] = fn
        return fn

    return _wrap


def get_metric(name: str) -> Metric:
    """Look up a registered metric by name, raising ``KeyError`` if missing."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown metric {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def list_metrics() -> list[str]:
    """Return the sorted names of all registered metrics."""
    return sorted(_REGISTRY)


# --- validation helpers (used at every public boundary) ---------------------


def check_k(k: int, *, name: str = "k") -> int:
    """Validate a cutoff ``k`` is a positive integer."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError(f"{name} must be an int, got {type(k).__name__}.")
    if k < 1:
        raise ValueError(f"{name} must be >= 1, got {k}.")
    return k


def check_non_empty(seq: Sequence[object], *, name: str) -> None:
    """Reject empty input rather than returning a misleading 0.0."""
    if len(seq) == 0:
        raise ValueError(f"{name} must be non-empty.")
