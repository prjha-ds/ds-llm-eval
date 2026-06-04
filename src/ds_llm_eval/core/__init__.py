"""Core contracts shared by every evaluation domain."""

from .base import Metric, check_k, check_non_empty, get_metric, list_metrics, register
from .types import EvalReport, EvalResult

__all__ = [
    "EvalReport",
    "EvalResult",
    "Metric",
    "check_k",
    "check_non_empty",
    "get_metric",
    "list_metrics",
    "register",
]
