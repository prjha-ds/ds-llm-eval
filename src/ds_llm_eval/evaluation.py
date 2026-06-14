"""Run a set of registered metrics over one dataset and collect an EvalReport.

:class:`Evaluator` is the convenience entry point: give it the metrics to run
(by registry name, with per-metric params) and optionally a logger; call
:meth:`Evaluator.run` with the data and get back an :class:`EvalReport`, with
every result logged through the logger if one was supplied.

All metrics passed to one Evaluator must share an input signature (the same
positional data is forwarded to each), so group metrics by family — e.g. all
``ranking.*`` together.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .core import EvalReport, EvalResult, get_metric
from .logging import EvalLogger


@dataclass(frozen=True)
class MetricSpec:
    """A registered metric name plus the keyword params to call it with."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


MetricLike = str | MetricSpec | tuple[str, dict[str, Any]]


def _coerce_spec(metric: MetricLike) -> MetricSpec:
    if isinstance(metric, MetricSpec):
        return metric
    if isinstance(metric, str):
        return MetricSpec(metric)
    if isinstance(metric, tuple) and len(metric) == 2 and isinstance(metric[0], str):
        return MetricSpec(metric[0], dict(metric[1]))
    raise ValueError(f"Unsupported metric spec: {metric!r}")


class Evaluator:
    """Run a fixed set of metrics over a dataset and return an :class:`EvalReport`.

    Parameters
    ----------
    metrics : iterable
        Metrics to run, each a registry name (``"ranking.ndcg_at_k"``), a
        ``(name, params)`` tuple, or a :class:`MetricSpec`.
    logger : EvalLogger, optional
        If given, every produced :class:`EvalResult` is logged on each run.
    """

    def __init__(self, metrics: Iterable[MetricLike], *, logger: EvalLogger | None = None) -> None:
        specs = [_coerce_spec(m) for m in metrics]
        if not specs:
            raise ValueError("Evaluator needs at least one metric.")
        self._specs = specs
        self._logger = logger

    def run(self, *data: object) -> EvalReport:
        """Run every metric with ``data`` (forwarded positionally) + its params.

        Returns
        -------
        EvalReport
            One :class:`EvalResult` per configured metric, in order.
        """
        results: list[EvalResult] = [
            get_metric(spec.name)(*data, **spec.params) for spec in self._specs
        ]
        report = EvalReport(results=results)
        if self._logger is not None:
            self._logger.log_report(report)
        return report

    @property
    def metric_names(self) -> Sequence[str]:
        """The registry names this evaluator will run, in order."""
        return [s.name for s in self._specs]
