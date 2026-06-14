"""Abstract base for evaluation loggers.

A logger turns :class:`~ds_llm_eval.core.EvalResult` values into timestamped,
run-scoped records and emits them to a sink (console, JSONL file, Langfuse, ...).
Loggers are classes following a template-method pattern: the base builds the
record and calls the subclass :meth:`EvalLogger.emit`. They are usable as context
managers so file/remote handles are flushed and closed deterministically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from typing import Any

from ..core import EvalReport, EvalResult


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class EvalLogger(ABC):
    """Base class for all evaluation loggers.

    Parameters
    ----------
    run_id:
        Optional identifier grouping the metrics of one evaluation run.
    metadata:
        Static key/values attached to every record (e.g. model name, dataset).
    clock:
        Callable returning the current time; injectable for deterministic tests.
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.run_id = run_id
        self.metadata = dict(metadata or {})
        self._clock = clock

    def _record(self, result: EvalResult) -> dict[str, Any]:
        """Build the serializable record for one result."""
        record: dict[str, Any] = {
            "timestamp": self._clock().isoformat(),
            "metric": result.name,
            "value": result.value,
            "n": result.n,
            "params": result.params,
        }
        if self.run_id is not None:
            record["run_id"] = self.run_id
        merged = {**self.metadata, **result.metadata}
        if merged:
            record["metadata"] = merged
        return record

    @abstractmethod
    def emit(self, record: dict[str, Any], result: EvalResult) -> None:
        """Write one record to the sink. ``result`` is the source for sinks that
        need the original object (e.g. Langfuse)."""

    def log_result(self, result: EvalResult) -> dict[str, Any]:
        """Log a single result and return the record that was emitted."""
        record = self._record(result)
        self.emit(record, result)
        return record

    def log_many(self, results: Iterable[EvalResult]) -> list[dict[str, Any]]:
        """Log an iterable of results."""
        return [self.log_result(r) for r in results]

    def log_report(self, report: EvalReport) -> list[dict[str, Any]]:
        """Log every result in an :class:`EvalReport`."""
        return self.log_many(report.results)

    def close(self) -> None:  # noqa: B027  intentional no-op hook; not all sinks need teardown
        """Flush/close any underlying resource. No-op by default."""

    def __enter__(self) -> EvalLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
