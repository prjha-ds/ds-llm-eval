"""Shared data carriers used across all evaluation domains."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvalResult(BaseModel):
    """Result of a single metric computation.

    A small, serializable record that every metric returns so that results from
    search, recommendation, and LLM evaluations share one shape and can be logged
    to sinks like Langfuse uniformly.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Metric name, e.g. 'ndcg@10'.")
    value: float = Field(description="Scalar metric value (typically aggregated over samples).")
    n: int = Field(default=0, ge=0, description="Number of samples/queries aggregated.")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters used (e.g. k).")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Free-form extra context.")

    def __float__(self) -> float:
        return float(self.value)


class EvalReport(BaseModel):
    """A named collection of :class:`EvalResult` values for one evaluation run."""

    model_config = ConfigDict(frozen=True)

    results: list[EvalResult] = Field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        """Flatten to a ``{metric_name: value}`` mapping."""
        return {r.name: r.value for r in self.results}
