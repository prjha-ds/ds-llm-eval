"""ds-llm-eval — unified evaluation for search, recommendation, and LLM pipelines.

Importing this package is intentionally lightweight: every domain's core metrics
depend only on the standard library plus numpy/pandas, and heavy/optional
backends (ragas, langfuse) are imported lazily at call time.

Quickstart
----------
>>> from ds_llm_eval import search
>>> search.precision_at_k(ranked=[["a", "b", "c"]], relevant=[{"a", "c"}], k=2).value
0.5
"""

from __future__ import annotations

from . import integrations, llm, recommendation, search
from .core import (
    EvalReport,
    EvalResult,
    Metric,
    get_metric,
    list_metrics,
    register,
)

__version__ = "0.0.1"

__all__ = [
    "EvalReport",
    "EvalResult",
    "Metric",
    "__version__",
    "get_metric",
    "integrations",
    "list_metrics",
    "llm",
    "recommendation",
    "register",
    "search",
]
