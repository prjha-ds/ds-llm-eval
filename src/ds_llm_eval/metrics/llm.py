"""Model-graded LLM/agentic metrics (LLM-as-judge), behind the ``[llm]`` extra.

Unlike the deterministic groups (ranking/text/beyond_accuracy), these delegate to
an LLM judge and are therefore non-deterministic, cost-bearing, and opt-in. The
backend (``ragas``) is imported lazily so the core package stays light and offline.
Reference-free RAG metrics measure *consistency with provided context*, not
external factual correctness.

References
----------
.. [1] S. Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented
   Generation", EACL 2024 demo. https://aclanthology.org/2024.eacl-demo.16/
.. [2] RAGAS metrics documentation. https://docs.ragas.io/en/stable/concepts/metrics/
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class LLMJudgeMetrics:
    """LLM-as-judge metrics; stateless adapter. Shared singleton :data:`llm_judge`.

    Not registered in the deterministic metric registry: these require an LLM
    backend and return multiple scores rather than a single :class:`EvalResult`.
    """

    def ragas_evaluate(
        self,
        questions: Sequence[str],
        answers: Sequence[str],
        contexts: Sequence[Sequence[str]],
        ground_truths: Sequence[str] | None = None,
        metrics: Sequence[Any] | None = None,
    ) -> dict[str, float]:
        """Evaluate RAG/agent outputs with RAGAS (reference-free where possible).

        Parameters
        ----------
        questions : sequence of str
            User inputs / queries.
        answers : sequence of str
            Generated answers, aligned with ``questions``.
        contexts : sequence of sequence of str
            Retrieved context passages per question.
        ground_truths : sequence of str, optional
            Reference answers; enables context-recall when provided.
        metrics : sequence, optional
            RAGAS metric objects; defaults to faithfulness, answer relevancy,
            context precision (+ context recall when ``ground_truths`` given).

        Returns
        -------
        dict of str to float
            ``{metric_name: score}``.

        Raises
        ------
        ImportError
            If the optional ``[llm]`` extra (ragas/datasets) is not installed.
        """
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError as exc:  # pragma: no cover - exercised via mocks
            raise ImportError(
                "ragas_evaluate requires the optional 'llm' extra. "
                'Install it with: pip install "ds-llm-eval[llm]"'
            ) from exc

        if metrics is None:
            metrics = [faithfulness, answer_relevancy, context_precision]
            if ground_truths is not None:
                metrics = [*metrics, context_recall]

        payload: dict[str, Any] = {
            "question": list(questions),
            "answer": list(answers),
            "contexts": [list(c) for c in contexts],
        }
        if ground_truths is not None:
            payload["ground_truth"] = list(ground_truths)

        dataset = Dataset.from_dict(payload)
        result = evaluate(dataset, metrics=list(metrics))
        return {k: float(v) for k, v in dict(result).items()}


llm_judge = LLMJudgeMetrics()
"""Shared :class:`LLMJudgeMetrics` singleton."""
