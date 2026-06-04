"""Optional integrations with observability / tracing suites (e.g. Langfuse)."""

from .langfuse import log_results_to_langfuse

__all__ = ["log_results_to_langfuse"]
