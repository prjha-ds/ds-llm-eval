"""ds-llm-eval — unified evaluation for search, recommendation, and LLM pipelines.

Metrics are grouped by *family* (ranking, text, beyond-accuracy, LLM-judge) and
used irrespective of task: the same ``ranking`` metrics evaluate a search ranker,
a recommender, or RAG retrieval. Importing this package is intentionally
lightweight — core groups depend only on the standard library plus numpy/pandas,
and heavy/optional backends (ragas, langfuse) are imported lazily at call time.

Quickstart
----------
>>> from ds_llm_eval import ranking
>>> ranking.precision_at_k(ranked=[["a", "b", "c"]], relevant=[{"a", "c"}], k=2).value
0.5
"""

from __future__ import annotations

from . import backends, benchmarks, ingestion, integrations, logging, metrics
from .benchmarks import list_benchmarks
from .comparison import ComparisonResult, RunComparison
from .config import ExperimentConfig, ExperimentRunner
from .core import (
    EvalReport,
    EvalResult,
    Metric,
    get_metric,
    list_metrics,
    register,
)
from .evaluation import Evaluator, MetricSpec
from .ingestion import ClickLogIngestor, RankingDataset, click_log
from .logging import ConsoleLogger, EvalLogger, JSONLLogger, MultiLogger
from .metrics import agentic, beyond_accuracy, llm_judge, ranking, text

__version__ = "0.1.0"

__all__ = [
    "ClickLogIngestor",
    "ComparisonResult",
    "ConsoleLogger",
    "EvalLogger",
    "EvalReport",
    "EvalResult",
    "Evaluator",
    "ExperimentConfig",
    "ExperimentRunner",
    "JSONLLogger",
    "Metric",
    "MetricSpec",
    "MultiLogger",
    "RankingDataset",
    "RunComparison",
    "__version__",
    "agentic",
    "backends",
    "benchmarks",
    "beyond_accuracy",
    "click_log",
    "get_metric",
    "ingestion",
    "integrations",
    "list_benchmarks",
    "list_metrics",
    "llm_judge",
    "logging",
    "metrics",
    "ranking",
    "register",
    "text",
]
