"""Concrete evaluation loggers: console, JSONL file, Langfuse, and a fan-out.

Note: this module is ``ds_llm_eval.logging`` but imports the standard-library
``json``/``sys`` normally — Python 3 absolute imports mean the package name does
not shadow stdlib modules.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any

from ..core import EvalResult
from .base import EvalLogger


class ConsoleLogger(EvalLogger):
    """Human-readable one-line-per-metric logger (defaults to stdout)."""

    def __init__(self, *, stream: IO[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stream = stream if stream is not None else sys.stdout

    def emit(self, record: dict[str, Any], result: EvalResult) -> None:
        prefix = f"[{record['timestamp']}] "
        run = f"({record['run_id']}) " if "run_id" in record else ""
        self._stream.write(
            f"{prefix}{run}{record['metric']} = {record['value']:.4f} (n={record['n']})\n"
        )


class JSONLLogger(EvalLogger):
    """Append one JSON object per metric to a ``.jsonl`` file.

    Opens the file (creating parent dirs) on construction in append mode; use as a
    context manager or call :meth:`close` to flush and release the handle.
    """

    def __init__(self, path: str | Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] | None = self.path.open("a", encoding="utf-8")

    def emit(self, record: dict[str, Any], result: EvalResult) -> None:
        if self._fh is None:
            raise ValueError("JSONLLogger is closed.")
        self._fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class LangfuseLogger(EvalLogger):
    """Log each result as a Langfuse score on a trace (SDK v3+ ``create_score``).

    The Langfuse client is resolved once on construction (lazily imported); pass a
    client in tests. Requires the ``[llm]`` extra when no client is supplied.
    """

    def __init__(self, *, trace_id: str, client: object | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from ..integrations.langfuse import resolve_langfuse_client

        self.trace_id = trace_id
        self._client: Any = resolve_langfuse_client(client)

    def emit(self, record: dict[str, Any], result: EvalResult) -> None:
        self._client.create_score(
            trace_id=self.trace_id,
            name=record["metric"],
            value=record["value"],
            data_type="NUMERIC",
            comment=f"n={record['n']}, params={record['params']}",
        )


class MultiLogger(EvalLogger):
    """Fan out every result to several child loggers (each formats independently)."""

    def __init__(self, loggers: Iterable[EvalLogger], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._loggers = list(loggers)

    def emit(self, record: dict[str, Any], result: EvalResult) -> None:  # pragma: no cover
        # Not used: fan-out happens in log_result so each child applies its own record.
        for logger in self._loggers:
            logger.emit(record, result)

    def log_result(self, result: EvalResult) -> dict[str, Any]:
        records = [logger.log_result(result) for logger in self._loggers]
        return records[0] if records else self._record(result)

    def close(self) -> None:
        for logger in self._loggers:
            logger.close()
