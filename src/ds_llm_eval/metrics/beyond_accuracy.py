"""Beyond-accuracy metrics: coverage and novelty of ranked outputs.

These judge *what* a system surfaces beyond raw relevance — how much of the
catalog it exposes and how non-obvious its items are. They apply to any ranked
output (recommendation slates, search results), so they are grouped by family
rather than by task.

References
----------
.. [1] P. Castells, N. Hurley, S. Vargas, "Novelty and Diversity in Recommender
   Systems", Recommender Systems Handbook, 2015. https://doi.org/10.1007/978-1-4899-7637-6_26
.. [2] G. Shani, A. Gunawardana, "Evaluating Recommendation Systems", 2011
   (Gini index, Shannon entropy). https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/EvaluationMetrics.TR_.pdf
.. [3] recmetrics (intra-list similarity, personalization).
   https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Collection, Hashable, Mapping, Sequence

import numpy as np

from ..core import EvalResult, check_k, check_non_empty

RankedInput = Sequence[Sequence[Hashable]] | Mapping[Hashable, Sequence[Hashable]]


def _as_rows(ranked: RankedInput) -> list[Sequence[Hashable]]:
    """Coerce a dict keyed by user/query id, or a sequence, into a list of rows."""
    rows = list(ranked.values()) if isinstance(ranked, Mapping) else list(ranked)
    check_non_empty(rows, name="ranked")
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class BeyondAccuracyMetrics:
    """Coverage/novelty metrics; stateless. Shared singleton :data:`beyond_accuracy`."""

    def catalog_coverage_at_k(
        self,
        ranked: RankedInput,
        catalog: Collection[Hashable],
        k: int,
    ) -> EvalResult:
        """Catalog coverage@k: share of the catalog recommended to ≥1 row.

        Parameters
        ----------
        ranked : sequence of sequence of hashable
            Ranked item ids per row (e.g. per user).
        catalog : collection of hashable
            The full set of available items.
        k : int
            Positive rank cutoff considered per row.

        Returns
        -------
        EvalResult
            ``beyond_accuracy.catalog_coverage@k``.
        """
        check_k(k)
        rows = _as_rows(ranked)
        catalog_set = set(catalog)
        check_non_empty(list(catalog_set), name="catalog")
        shown: set[Hashable] = set()
        for ranks in rows:
            shown.update(ranks[:k])
        covered = len(shown & catalog_set) / len(catalog_set)
        return EvalResult(
            name=f"beyond_accuracy.catalog_coverage@{k}",
            value=covered,
            n=len(rows),
            params={"k": k},
        )

    def novelty_at_k(
        self,
        ranked: RankedInput,
        popularity: Mapping[Hashable, float],
        k: int,
    ) -> EvalResult:
        """Mean self-information novelty@k: ``-log2(p(item))`` averaged over items.

        Parameters
        ----------
        ranked : sequence of sequence of hashable
            Ranked item ids per row.
        popularity : mapping of hashable to float
            Item -> observation count or probability; rarer items score higher.
        k : int
            Positive rank cutoff considered per row.

        Returns
        -------
        EvalResult
            ``beyond_accuracy.novelty@k``.
        """
        check_k(k)
        rows = _as_rows(ranked)
        total = sum(popularity.values())
        if total <= 0:
            raise ValueError("popularity values must sum to a positive number.")
        scores = []
        for ranks in rows:
            infos = [
                -math.log2(popularity[item] / total)
                for item in ranks[:k]
                if popularity.get(item, 0) > 0
            ]
            if infos:
                scores.append(sum(infos) / len(infos))
        return EvalResult(
            name=f"beyond_accuracy.novelty@{k}", value=_mean(scores), n=len(scores), params={"k": k}
        )

    def intra_list_diversity(
        self,
        ranked: RankedInput,
        similarity: Callable[[Hashable, Hashable], float],
        *,
        k: int | None = None,
    ) -> EvalResult:
        """Mean intra-list diversity = 1 − average pairwise similarity within a list.

        Parameters
        ----------
        ranked : sequence or mapping of ranked lists
            Recommended item ids per row.
        similarity : callable
            ``similarity(a, b) -> float`` in [0, 1] between two items.
        k : int, optional
            Cap each list to its top ``k`` before measuring.

        Returns
        -------
        EvalResult
            ``beyond_accuracy.intra_list_diversity``. Lists with < 2 items are
            skipped (no pairs).

        References
        ----------
        recmetrics ``intra_list_similarity`` (diversity = 1 − ILS).
        """
        rows = _as_rows(ranked)
        scores = []
        for ranks in rows:
            items = list(ranks[:k]) if k is not None else list(ranks)
            if len(items) < 2:
                continue
            sims = [
                similarity(items[i], items[j])
                for i in range(len(items))
                for j in range(i + 1, len(items))
            ]
            scores.append(1.0 - sum(sims) / len(sims))
        return EvalResult(
            name="beyond_accuracy.intra_list_diversity", value=_mean(scores), n=len(scores)
        )

    def personalization(self, ranked: RankedInput, *, k: int | None = None) -> EvalResult:
        """Personalization = 1 − mean pairwise cosine similarity across users' lists.

        Each row becomes a binary indicator vector over the recommended-item
        universe; higher means users get more distinct recommendations.

        Parameters
        ----------
        ranked : sequence or mapping of ranked lists
            Recommended item ids per user.
        k : int, optional
            Cap each list to its top ``k``.

        Returns
        -------
        EvalResult
            ``beyond_accuracy.personalization`` (needs ≥ 2 users).

        References
        ----------
        recmetrics ``personalization``.
        """
        rows = [list(r[:k]) if k is not None else list(r) for r in _as_rows(ranked)]
        m = len(rows)
        if m < 2:
            raise ValueError("personalization needs at least 2 users.")
        universe = sorted({item for row in rows for item in row}, key=repr)
        index = {item: i for i, item in enumerate(universe)}
        matrix = np.zeros((m, len(universe)), dtype=float)
        for u, row in enumerate(rows):
            for item in row:
                matrix[u, index[item]] = 1.0
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0  # zero vectors -> zero similarity, avoid 0/0
        normed = matrix / norms[:, None]
        sim = normed @ normed.T
        off_diagonal = (sim.sum() - np.trace(sim)) / (m * (m - 1))
        return EvalResult(
            name="beyond_accuracy.personalization", value=float(1.0 - off_diagonal), n=m
        )

    def _item_proportions(self, rows: list[Sequence[Hashable]], k: int | None) -> list[float]:
        counts: Counter[Hashable] = Counter()
        for ranks in rows:
            counts.update(ranks[:k] if k is not None else ranks)
        total = sum(counts.values())
        if total == 0:
            raise ValueError("no recommended items to measure.")
        return [c / total for c in counts.values()]

    def gini_index(self, ranked: RankedInput, *, k: int | None = None) -> EvalResult:
        """Gini index of the item-recommendation frequency distribution.

        ``G = (1/(n-1)) Σ_j (2j − n − 1) p(i_j)`` with items sorted by increasing
        proportion ``p``. 0 = every item recommended equally often; → 1 = highly
        concentrated on a few items.

        Returns
        -------
        EvalResult
            ``beyond_accuracy.gini_index`` (needs ≥ 2 distinct items).

        References
        ----------
        Shani & Gunawardana (2011), Eq. 19.
        """
        rows = _as_rows(ranked)
        proportions = sorted(self._item_proportions(rows, k))
        n = len(proportions)
        if n < 2:
            raise ValueError("gini_index needs at least 2 distinct recommended items.")
        g = sum((2 * (j + 1) - n - 1) * p for j, p in enumerate(proportions)) / (n - 1)
        return EvalResult(name="beyond_accuracy.gini_index", value=float(g), n=n)

    def shannon_entropy(self, ranked: RankedInput, *, k: int | None = None) -> EvalResult:
        """Shannon entropy (base 2) of the item-recommendation distribution.

        ``H = − Σ_i p(i) log2 p(i)``. 0 when one item dominates; ``log2 n`` when all
        ``n`` items are recommended equally often.

        Returns
        -------
        EvalResult
            ``beyond_accuracy.shannon_entropy`` (nats-free; base-2 bits).

        References
        ----------
        Shani & Gunawardana (2011), Eq. 20.
        """
        rows = _as_rows(ranked)
        proportions = self._item_proportions(rows, k)
        h = -sum(p * math.log2(p) for p in proportions if p > 0)
        return EvalResult(
            name="beyond_accuracy.shannon_entropy", value=float(h), n=len(proportions)
        )


beyond_accuracy = BeyondAccuracyMetrics()
"""Shared :class:`BeyondAccuracyMetrics` singleton."""
