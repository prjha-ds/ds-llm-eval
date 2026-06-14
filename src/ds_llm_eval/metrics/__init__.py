"""Evaluation metrics, grouped by *metric family* (not by task).

Each group is a class of pure methods plus a shared singleton:

* :class:`RankingMetrics` / :data:`ranking` — precision/recall/F1@k, MRR, MAP,
  NDCG, hit_rate@k (search, recommendation, RAG retrieval).
* :class:`BeyondAccuracyMetrics` / :data:`beyond_accuracy` — catalog coverage, novelty.
* :class:`TextMetrics` / :data:`text` — exact match, token-F1 (reference-based).
* :class:`LLMJudgeMetrics` / :data:`llm_judge` — lazy RAGAS adapter (opt-in, model-graded).

Deterministic group methods are auto-registered in the global metric registry as
``<group>.<method>`` (e.g. ``ranking.ndcg_at_k``); discover them with
:func:`ds_llm_eval.list_metrics`.
"""

from ..core import register
from .agentic import AgenticMetrics, agentic
from .beyond_accuracy import BeyondAccuracyMetrics, beyond_accuracy
from .llm import LLMJudgeMetrics, llm_judge
from .ranking import RankingMetrics, ranking
from .text import TextMetrics, text


def _register_group(prefix: str, instance: object) -> None:
    """Register every public method of ``instance`` as ``<prefix>.<method>``."""
    for name in sorted(vars(type(instance))):
        if name.startswith("_"):
            continue
        attr = getattr(instance, name)
        if callable(attr):
            register(f"{prefix}.{name}")(attr)


# Deterministic groups are discoverable/loggable via the registry; the model-graded
# llm_judge group is intentionally excluded (requires an LLM backend, multi-score).
_register_group("ranking", ranking)
_register_group("beyond_accuracy", beyond_accuracy)
_register_group("text", text)
_register_group("agentic", agentic)

__all__ = [
    "AgenticMetrics",
    "BeyondAccuracyMetrics",
    "LLMJudgeMetrics",
    "RankingMetrics",
    "TextMetrics",
    "agentic",
    "beyond_accuracy",
    "llm_judge",
    "ranking",
    "text",
]
