"""LLM & agentic pipeline evaluation.

Two tiers:

* **Deterministic, offline reference metrics** (``exact_match``, ``token_f1``)
  for tasks with gold answers — fast, free, and unit-testable without network.
* **Model-graded metrics** via a lazily-imported RAGAS adapter
  (``ragas_evaluate``) for reference-free RAG/agent quality (faithfulness,
  answer relevancy, context precision/recall). Requires ``pip install
  "ds-llm-eval[llm]"`` and an LLM provider; never imported at module top level.

References
----------
- P. Rajpurkar et al., "SQuAD: 100,000+ Questions...", EMNLP 2016 (token-level F1).
  https://aclanthology.org/D16-1264/
- S. Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
  EACL 2024 demo. https://aclanthology.org/2024.eacl-demo.16/
- RAGAS metrics documentation. https://docs.ragas.io/en/stable/concepts/metrics/
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from typing import Any

from ..core import EvalResult, check_non_empty, register

_PUNCT = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _normalize(text: str) -> list[str]:
    """SQuAD-style normalization: lowercase, strip punctuation & articles, split."""
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return text.split()


def _validate(predictions: Sequence[str], references: Sequence[str]) -> None:
    check_non_empty(predictions, name="predictions")
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) length mismatch."
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@register("llm.exact_match")
def exact_match(predictions: Sequence[str], references: Sequence[str]) -> EvalResult:
    """Mean exact match after normalization (1.0 if normalized strings are equal)."""
    _validate(predictions, references)
    scores = [
        1.0 if _normalize(p) == _normalize(r) else 0.0
        for p, r in zip(predictions, references, strict=True)
    ]
    return EvalResult(name="llm.exact_match", value=_mean(scores), n=len(predictions))


@register("llm.token_f1")
def token_f1(predictions: Sequence[str], references: Sequence[str]) -> EvalResult:
    """Mean SQuAD-style token-overlap F1 between prediction and reference."""
    _validate(predictions, references)
    scores = []
    for p, r in zip(predictions, references, strict=True):
        pred_toks, ref_toks = _normalize(p), _normalize(r)
        if not pred_toks and not ref_toks:
            scores.append(1.0)
            continue
        if not pred_toks or not ref_toks:
            scores.append(0.0)
            continue
        common = sum((Counter(pred_toks) & Counter(ref_toks)).values())
        if common == 0:
            scores.append(0.0)
            continue
        precision = common / len(pred_toks)
        recall = common / len(ref_toks)
        scores.append(2 * precision * recall / (precision + recall))
    return EvalResult(name="llm.token_f1", value=_mean(scores), n=len(predictions))


def ragas_evaluate(
    questions: Sequence[str],
    answers: Sequence[str],
    contexts: Sequence[Sequence[str]],
    ground_truths: Sequence[str] | None = None,
    metrics: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Adapter around RAGAS for reference-free RAG/agent evaluation.

    Lazily imports ``ragas``/``datasets`` so the core package stays network- and
    dependency-light. Raises a clear error if the optional ``[llm]`` extra is not
    installed. Returns a ``{metric_name: score}`` mapping.
    """
    try:
        from datasets import Dataset  # type: ignore[import-not-found]
        from ragas import evaluate  # type: ignore[import-not-found]
        from ragas.metrics import (  # type: ignore[import-not-found]
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
