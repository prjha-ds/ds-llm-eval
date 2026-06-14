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

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ..core import EvalResult


def resolve_langfuse_client(client: object | None = None) -> Any:
    """Return ``client`` if given, else lazily construct a Langfuse client.

    Constructing from the environment requires ``pip install "ds-llm-eval[llm]"``
    plus ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``. Raises a clear
    ``ImportError`` naming the extra when the SDK is missing.
    """
    if client is not None:
        return client
    try:
        from langfuse import Langfuse
    except ImportError as exc:  # pragma: no cover - exercised via mocks
        raise ImportError(
            "Langfuse integration requires the optional 'llm' extra. "
            'Install it with: pip install "ds-llm-eval[llm]"'
        ) from exc
    return Langfuse()


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
    lf: Any = resolve_langfuse_client(client)

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


def run_langfuse_experiment(
    dataset_name: str,
    task: Callable[..., Any],
    evaluators: Sequence[Callable[..., Any]],
    *,
    run_name: str,
    client: object | None = None,
    **kwargs: Any,
) -> Any:
    """Run a Langfuse dataset experiment (SDK v4+ ``dataset.run_experiment``).

    Fetches the dataset by name and runs ``task`` over each item, applying
    ``evaluators`` to score the outputs; a UI-visible dataset run is created for
    cross-run comparison.

    Parameters
    ----------
    dataset_name : str
        Name of the Langfuse dataset to run against.
    task : callable
        Processes each item (receives the item; returns the output).
    evaluators : sequence of callable
        Item-level evaluators returning Langfuse ``Evaluation`` objects.
    run_name : str
        Name for this experiment run.
    client : object, optional
        Existing Langfuse client; constructed lazily from the env if ``None``.
    **kwargs
        Forwarded to ``run_experiment`` (e.g. ``run_evaluators``, ``metadata``).

    Returns
    -------
    Any
        Whatever ``dataset.run_experiment`` returns (the run result object).
    """
    lf: Any = resolve_langfuse_client(client)
    dataset = lf.get_dataset(dataset_name)
    return dataset.run_experiment(name=run_name, task=task, evaluators=list(evaluators), **kwargs)
