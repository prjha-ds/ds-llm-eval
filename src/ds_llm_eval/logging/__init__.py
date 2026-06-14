"""Logging mechanism for evaluation runs.

Turn :class:`~ds_llm_eval.core.EvalResult` values into timestamped, run-scoped
records and emit them to one or more sinks::

    from ds_llm_eval.logging import ConsoleLogger, JSONLLogger, MultiLogger

    with MultiLogger([ConsoleLogger(), JSONLLogger("runs/eval.jsonl")],
                     run_id="exp-1", metadata={"model": "v2"}) as log:
        log.log_result(search.ndcg_at_k(ranked, relevant, k=10))
        log.log_report(report)
"""

from .base import EvalLogger
from .loggers import ConsoleLogger, JSONLLogger, LangfuseLogger, MultiLogger

__all__ = [
    "ConsoleLogger",
    "EvalLogger",
    "JSONLLogger",
    "LangfuseLogger",
    "MultiLogger",
]
