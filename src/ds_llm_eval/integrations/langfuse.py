"""Langfuse integration: push evaluation results as scores onto traces.

Langfuse is an open-source (MIT) LLM-engineering/observability platform; attaching
our :class:`EvalResult` values as Langfuse *scores* lets teams track offline eval
metrics next to production traces. A score can be ingested for a ``trace_id`` even
before that trace exists — Langfuse links them when the trace arrives — which makes
this an ideal sink for a batch/offline eval job. The ``langfuse`` SDK is imported
lazily so the core package has no hard dependency on it.

Targets the Langfuse Python SDK v3+ ``create_score(...)`` method. (The SDK has
notable churn across v2/v3/v4; pin a major version in your environment.)

Reference: Langfuse scores via SDK.
https://langfuse.com/docs/evaluation/evaluation-methods/scores-via-sdk
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..core import EvalResult


def log_results_to_langfuse(
    results: Iterable[EvalResult],
    *,
    trace_id: str,
    client: object | None = None,
) -> int:
    """Log each :class:`EvalResult` as a Langfuse score on ``trace_id``.

    Parameters
    ----------
    results:
        Metric results to log.
    trace_id:
        The Langfuse trace to attach scores to.
    client:
        An existing Langfuse client. If ``None``, one is constructed lazily from
        environment variables (requires ``pip install "ds-llm-eval[llm]"`` and
        ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``).

    Returns
    -------
    int
        The number of scores logged.
    """
    lf: Any = client
    if lf is None:
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via mocks
            raise ImportError(
                "Langfuse integration requires the optional 'llm' extra. "
                'Install it with: pip install "ds-llm-eval[llm]"'
            ) from exc
        lf = Langfuse()

    count = 0
    for result in results:
        lf.create_score(
            trace_id=trace_id,
            name=result.name,
            value=result.value,
            data_type="NUMERIC",
            comment=f"n={result.n}, params={result.params}",
        )
        count += 1
    return count
