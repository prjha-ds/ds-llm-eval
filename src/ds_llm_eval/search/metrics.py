"""Information-retrieval metrics for search pipelines.

All metrics consume *ranked predictions* plus *relevance judgments* (qrels) that
you can derive from editorial labels or click logs (e.g. treat a clicked doc as
relevant). Inputs are batched: ``ranked[i]`` is the ranked list of item ids for
query ``i`` and ``relevant[i]`` holds that query's relevant items — a set/list
for binary relevance, or a ``{item: gain}`` mapping for graded relevance (NDCG).

References
----------
- C. Manning, P. Raghavan, H. Schütze, *Introduction to Information Retrieval*,
  ch. 8 (Evaluation in IR). https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html
- K. Järvelin, J. Kekäläinen, "Cumulated gain-based evaluation of IR techniques",
  ACM TOIS 2002 (NDCG). https://doi.org/10.1145/582415.582418
"""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Mapping, Sequence

from ..core import EvalResult, check_k, check_non_empty, register

Relevance = Collection[Hashable] | Mapping[Hashable, float]


def _binary_set(rel: Relevance) -> set[Hashable]:
    """Relevant items as a set; a graded mapping counts gain > 0 as relevant."""
    if isinstance(rel, Mapping):
        return {item for item, gain in rel.items() if gain > 0}
    return set(rel)


def _validate(ranked: Sequence[Sequence[Hashable]], relevant: Sequence[Relevance]) -> None:
    check_non_empty(ranked, name="ranked")
    if len(ranked) != len(relevant):
        raise ValueError(f"ranked ({len(ranked)}) and relevant ({len(relevant)}) length mismatch.")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@register("search.precision_at_k")
def precision_at_k(
    ranked: Sequence[Sequence[Hashable]], relevant: Sequence[Relevance], k: int
) -> EvalResult:
    """Mean Precision@k: fraction of the top-``k`` results that are relevant."""
    check_k(k)
    _validate(ranked, relevant)
    scores = []
    for ranks, rel in zip(ranked, relevant, strict=True):
        rel_set = _binary_set(rel)
        top = ranks[:k]
        hits = sum(1 for item in top if item in rel_set)
        scores.append(hits / k)
    return EvalResult(
        name=f"search.precision@{k}", value=_mean(scores), n=len(ranked), params={"k": k}
    )


@register("search.recall_at_k")
def recall_at_k(
    ranked: Sequence[Sequence[Hashable]], relevant: Sequence[Relevance], k: int
) -> EvalResult:
    """Mean Recall@k: fraction of all relevant items found in the top ``k``."""
    check_k(k)
    _validate(ranked, relevant)
    scores = []
    for ranks, rel in zip(ranked, relevant, strict=True):
        rel_set = _binary_set(rel)
        if not rel_set:
            continue  # recall undefined with no relevant items; skip this query
        hits = sum(1 for item in ranks[:k] if item in rel_set)
        scores.append(hits / len(rel_set))
    return EvalResult(
        name=f"search.recall@{k}", value=_mean(scores), n=len(scores), params={"k": k}
    )


@register("search.f1_at_k")
def f1_at_k(
    ranked: Sequence[Sequence[Hashable]], relevant: Sequence[Relevance], k: int
) -> EvalResult:
    """Mean F1@k: harmonic mean of Precision@k and Recall@k, per query then averaged."""
    check_k(k)
    _validate(ranked, relevant)
    scores = []
    for ranks, rel in zip(ranked, relevant, strict=True):
        rel_set = _binary_set(rel)
        if not rel_set:
            continue
        top = ranks[:k]
        hits = sum(1 for item in top if item in rel_set)
        p = hits / k
        r = hits / len(rel_set)
        scores.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
    return EvalResult(name=f"search.f1@{k}", value=_mean(scores), n=len(scores), params={"k": k})


@register("search.mrr")
def mrr(ranked: Sequence[Sequence[Hashable]], relevant: Sequence[Relevance]) -> EvalResult:
    """Mean Reciprocal Rank: average of ``1/rank`` of the first relevant result."""
    _validate(ranked, relevant)
    scores = []
    for ranks, rel in zip(ranked, relevant, strict=True):
        rel_set = _binary_set(rel)
        rr = 0.0
        for pos, item in enumerate(ranks, start=1):
            if item in rel_set:
                rr = 1.0 / pos
                break
        scores.append(rr)
    return EvalResult(name="search.mrr", value=_mean(scores), n=len(ranked))


@register("search.average_precision")
def average_precision(
    ranked: Sequence[Sequence[Hashable]], relevant: Sequence[Relevance], k: int | None = None
) -> EvalResult:
    """Mean Average Precision (MAP): mean over queries of average precision."""
    _validate(ranked, relevant)
    scores = []
    for ranks, rel in zip(ranked, relevant, strict=True):
        rel_set = _binary_set(rel)
        if not rel_set:
            continue
        cut = ranks[:k] if k else ranks
        hits = 0
        precisions = []
        for pos, item in enumerate(cut, start=1):
            if item in rel_set:
                hits += 1
                precisions.append(hits / pos)
        scores.append(sum(precisions) / len(rel_set) if precisions else 0.0)
    name = f"search.map@{k}" if k else "search.map"
    return EvalResult(name=name, value=_mean(scores), n=len(scores), params={"k": k})


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


@register("search.ndcg_at_k")
def ndcg_at_k(
    ranked: Sequence[Sequence[Hashable]], relevant: Sequence[Relevance], k: int
) -> EvalResult:
    """Mean NDCG@k. Uses graded gains when ``relevant`` is a mapping, else binary."""
    check_k(k)
    _validate(ranked, relevant)
    scores = []
    for ranks, rel in zip(ranked, relevant, strict=True):
        gain_of: Mapping[Hashable, float]
        gain_of = rel if isinstance(rel, Mapping) else {item: 1.0 for item in rel}
        if not gain_of:
            continue
        dcg = _dcg([float(gain_of.get(item, 0.0)) for item in ranks[:k]])
        ideal = _dcg(sorted((float(g) for g in gain_of.values()), reverse=True)[:k])
        scores.append(dcg / ideal if ideal > 0 else 0.0)
    return EvalResult(name=f"search.ndcg@{k}", value=_mean(scores), n=len(scores), params={"k": k})
