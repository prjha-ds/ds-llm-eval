"""Recommendation-system evaluation metrics.

Inputs are batched per user: ``recommended[u]`` is the ranked item list shown to
user ``u`` and ``holdout[u]`` is the set of items the user actually engaged with
(the ground-truth relevant set). Beyond accuracy, recommender quality also depends
on *beyond-accuracy* aspects — coverage and novelty — included here.

References
----------
- M. Deshpande, G. Karypis, "Item-based top-N recommendation algorithms",
  ACM TOIS 2004 (hit rate / ARHR). https://doi.org/10.1145/963770.963776
- P. Castells, N. Hurley, S. Vargas, "Novelty and Diversity in Recommender Systems",
  Recommender Systems Handbook, 2015. https://doi.org/10.1007/978-1-4899-7637-6_26
"""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Mapping, Sequence

from ..core import EvalResult, check_k, check_non_empty, register


def _validate(
    recommended: Sequence[Sequence[Hashable]], holdout: Sequence[Collection[Hashable]]
) -> None:
    check_non_empty(recommended, name="recommended")
    if len(recommended) != len(holdout):
        raise ValueError(
            f"recommended ({len(recommended)}) and holdout ({len(holdout)}) length mismatch."
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@register("rec.hit_rate_at_k")
def hit_rate_at_k(
    recommended: Sequence[Sequence[Hashable]], holdout: Sequence[Collection[Hashable]], k: int
) -> EvalResult:
    """Hit Rate@k: fraction of users with at least one held-out item in the top ``k``."""
    check_k(k)
    _validate(recommended, holdout)
    scores = []
    for recs, truth in zip(recommended, holdout, strict=True):
        truth_set = set(truth)
        if not truth_set:
            continue
        scores.append(1.0 if any(item in truth_set for item in recs[:k]) else 0.0)
    return EvalResult(name=f"rec.hit_rate@{k}", value=_mean(scores), n=len(scores), params={"k": k})


@register("rec.ndcg_at_k")
def ndcg_at_k(
    recommended: Sequence[Sequence[Hashable]], holdout: Sequence[Collection[Hashable]], k: int
) -> EvalResult:
    """NDCG@k for implicit feedback (binary relevance from the held-out set)."""
    check_k(k)
    _validate(recommended, holdout)
    scores = []
    for recs, truth in zip(recommended, holdout, strict=True):
        truth_set = set(truth)
        if not truth_set:
            continue
        dcg = sum(
            1.0 / math.log2(i + 1) for i, item in enumerate(recs[:k], start=1) if item in truth_set
        )
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(truth_set)) + 1))
        scores.append(dcg / ideal if ideal > 0 else 0.0)
    return EvalResult(name=f"rec.ndcg@{k}", value=_mean(scores), n=len(scores), params={"k": k})


@register("rec.catalog_coverage_at_k")
def catalog_coverage_at_k(
    recommended: Sequence[Sequence[Hashable]],
    catalog: Collection[Hashable],
    k: int,
) -> EvalResult:
    """Catalog coverage@k: share of the catalog recommended to at least one user."""
    check_k(k)
    check_non_empty(recommended, name="recommended")
    catalog_set = set(catalog)
    check_non_empty(list(catalog_set), name="catalog")
    shown: set[Hashable] = set()
    for recs in recommended:
        shown.update(recs[:k])
    covered = len(shown & catalog_set) / len(catalog_set)
    return EvalResult(
        name=f"rec.catalog_coverage@{k}", value=covered, n=len(recommended), params={"k": k}
    )


@register("rec.novelty_at_k")
def novelty_at_k(
    recommended: Sequence[Sequence[Hashable]],
    popularity: Mapping[Hashable, float],
    k: int,
) -> EvalResult:
    """Mean self-information novelty@k: ``-log2(p(item))`` averaged over recs.

    ``popularity`` maps item -> observation count or probability; higher novelty
    means rarer (less popular) items are being recommended.
    """
    check_k(k)
    check_non_empty(recommended, name="recommended")
    total = sum(popularity.values())
    if total <= 0:
        raise ValueError("popularity values must sum to a positive number.")
    scores = []
    for recs in recommended:
        infos = [
            -math.log2(popularity[item] / total) for item in recs[:k] if popularity.get(item, 0) > 0
        ]
        if infos:
            scores.append(sum(infos) / len(infos))
    return EvalResult(name=f"rec.novelty@{k}", value=_mean(scores), n=len(scores), params={"k": k})
