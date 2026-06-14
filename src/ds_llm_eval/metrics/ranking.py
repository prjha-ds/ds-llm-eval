"""Relevance-ranking metrics, grouped by *metric family* rather than by task.

The same ranking metrics apply wherever a system returns a *ranked list* judged
against *relevance labels*: search/IR (qrels or click logs), recommendation
(held-out interactions), and RAG retrieval. They are therefore exposed once, on
:class:`RankingMetrics`, and used irrespective of the task.

Conventions
-----------
Inputs are batched: ``ranked[i]`` is the ranked list of item ids for query/user
``i`` and ``relevant[i]`` holds that row's relevant items — a set/list for binary
relevance, or a ``{item: gain}`` mapping for graded relevance (NDCG). Each metric
returns an :class:`~ds_llm_eval.core.EvalResult` aggregated (mean) over rows.

References
----------
.. [1] C. Manning, P. Raghavan, H. Schütze, *Introduction to Information
   Retrieval*, ch. 8. https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html
.. [2] K. Järvelin, J. Kekäläinen, "Cumulated gain-based evaluation of IR
   techniques", ACM TOIS 2002 (NDCG). https://doi.org/10.1145/582415.582418
.. [3] M. Deshpande, G. Karypis, "Item-based top-N recommendation algorithms",
   ACM TOIS 2004 (hit rate). https://doi.org/10.1145/963770.963776
"""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Mapping, Sequence

from ..core import EvalResult, check_k, check_non_empty

Relevance = Collection[Hashable] | Mapping[Hashable, float]
# Polymorphic inputs: batched sequences, or dicts keyed by query/user id.
RankedInput = Sequence[Sequence[Hashable]] | Mapping[Hashable, Sequence[Hashable]]
RelevanceInput = Sequence[Relevance] | Mapping[Hashable, Relevance]


def _binary_set(rel: Relevance) -> set[Hashable]:
    """Relevant items as a set; a graded mapping counts gain > 0 as relevant."""
    if isinstance(rel, Mapping):
        return {item for item, gain in rel.items() if gain > 0}
    return set(rel)


def _prepare(
    ranked: RankedInput, relevant: RelevanceInput
) -> tuple[list[Sequence[Hashable]], list[Relevance]]:
    """Coerce dict-keyed or sequence inputs into aligned lists, then validate.

    Accepts both as plain batched sequences, or both as mappings keyed by query/
    user id (aligned by ``ranked``'s key order).
    """
    if isinstance(ranked, Mapping) or isinstance(relevant, Mapping):
        if not (isinstance(ranked, Mapping) and isinstance(relevant, Mapping)):
            raise ValueError("ranked and relevant must both be mappings or both sequences.")
        keys = list(ranked.keys())
        missing = [key for key in keys if key not in relevant]
        if missing:
            raise ValueError(f"relevant is missing keys present in ranked: {missing}.")
        ranked_rows: list[Sequence[Hashable]] = [ranked[key] for key in keys]
        relevant_rows: list[Relevance] = [relevant[key] for key in keys]
    else:
        ranked_rows = list(ranked)
        relevant_rows = list(relevant)
    check_non_empty(ranked_rows, name="ranked")
    if len(ranked_rows) != len(relevant_rows):
        raise ValueError(
            f"ranked ({len(ranked_rows)}) and relevant ({len(relevant_rows)}) length mismatch."
        )
    return ranked_rows, relevant_rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


class RankingMetrics:
    """Ranking-quality metrics for any ranked-retrieval task.

    A stateless group of pure methods; the module exposes a shared singleton
    :data:`ranking`. Use either ``RankingMetrics().ndcg_at_k(...)`` or the
    singleton ``ranking.ndcg_at_k(...)``.
    """

    def precision_at_k(self, ranked: RankedInput, relevant: RelevanceInput, k: int) -> EvalResult:
        """Mean Precision@k.

        Parameters
        ----------
        ranked : sequence of sequence of hashable
            ``ranked[i]`` is the ranked item ids returned for row ``i``.
        relevant : sequence of (collection or mapping)
            Relevant items per row (binary set/list, or ``{item: gain}``).
        k : int
            Positive rank cutoff.

        Returns
        -------
        EvalResult
            ``ranking.precision@k`` = mean over rows of (relevant in top-k) / k.
        """
        check_k(k)
        ranked, relevant = _prepare(ranked, relevant)
        scores = []
        for ranks, rel in zip(ranked, relevant, strict=True):
            rel_set = _binary_set(rel)
            hits = sum(1 for item in ranks[:k] if item in rel_set)
            scores.append(hits / k)
        return EvalResult(
            name=f"ranking.precision@{k}", value=_mean(scores), n=len(ranked), params={"k": k}
        )

    def recall_at_k(self, ranked: RankedInput, relevant: RelevanceInput, k: int) -> EvalResult:
        """Mean Recall@k: fraction of relevant items found in the top ``k``.

        Rows with no relevant items are skipped (recall undefined), so ``n`` may
        be smaller than ``len(ranked)``.

        Parameters
        ----------
        ranked, relevant, k
            See :meth:`precision_at_k`.

        Returns
        -------
        EvalResult
            ``ranking.recall@k``.
        """
        check_k(k)
        ranked, relevant = _prepare(ranked, relevant)
        scores = []
        for ranks, rel in zip(ranked, relevant, strict=True):
            rel_set = _binary_set(rel)
            if not rel_set:
                continue
            hits = sum(1 for item in ranks[:k] if item in rel_set)
            scores.append(hits / len(rel_set))
        return EvalResult(
            name=f"ranking.recall@{k}", value=_mean(scores), n=len(scores), params={"k": k}
        )

    def f1_at_k(self, ranked: RankedInput, relevant: RelevanceInput, k: int) -> EvalResult:
        """Mean F1@k: per-row harmonic mean of Precision@k and Recall@k.

        Parameters
        ----------
        ranked, relevant, k
            See :meth:`precision_at_k`.

        Returns
        -------
        EvalResult
            ``ranking.f1@k`` (rows with no relevant items skipped).
        """
        check_k(k)
        ranked, relevant = _prepare(ranked, relevant)
        scores = []
        for ranks, rel in zip(ranked, relevant, strict=True):
            rel_set = _binary_set(rel)
            if not rel_set:
                continue
            hits = sum(1 for item in ranks[:k] if item in rel_set)
            p = hits / k
            r = hits / len(rel_set)
            scores.append(0.0 if (p + r) == 0 else 2 * p * r / (p + r))
        return EvalResult(
            name=f"ranking.f1@{k}", value=_mean(scores), n=len(scores), params={"k": k}
        )

    def mrr(self, ranked: RankedInput, relevant: RelevanceInput) -> EvalResult:
        """Mean Reciprocal Rank: mean of ``1/rank`` of the first relevant result.

        Parameters
        ----------
        ranked, relevant
            See :meth:`precision_at_k`.

        Returns
        -------
        EvalResult
            ``ranking.mrr`` (0 for rows with no relevant result in ``ranked``).
        """
        ranked, relevant = _prepare(ranked, relevant)
        scores = []
        for ranks, rel in zip(ranked, relevant, strict=True):
            rel_set = _binary_set(rel)
            rr = 0.0
            for pos, item in enumerate(ranks, start=1):
                if item in rel_set:
                    rr = 1.0 / pos
                    break
            scores.append(rr)
        return EvalResult(name="ranking.mrr", value=_mean(scores), n=len(ranked))

    def average_precision(
        self,
        ranked: Sequence[Sequence[Hashable]],
        relevant: Sequence[Relevance],
        k: int | None = None,
    ) -> EvalResult:
        """Mean Average Precision (MAP).

        Parameters
        ----------
        ranked, relevant
            See :meth:`precision_at_k`.
        k : int, optional
            Optional rank cutoff; if ``None`` the full ranking is used.

        Returns
        -------
        EvalResult
            ``ranking.map`` (or ``ranking.map@k``); rows with no relevant items skipped.
        """
        ranked, relevant = _prepare(ranked, relevant)
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
        name = f"ranking.map@{k}" if k else "ranking.map"
        return EvalResult(name=name, value=_mean(scores), n=len(scores), params={"k": k})

    def ndcg_at_k(self, ranked: RankedInput, relevant: RelevanceInput, k: int) -> EvalResult:
        """Mean NDCG@k. Uses graded gains when ``relevant`` is a mapping, else binary.

        Parameters
        ----------
        ranked, relevant, k
            See :meth:`precision_at_k`. A ``{item: gain}`` mapping enables graded
            relevance; a plain collection is treated as binary (gain 1).

        Returns
        -------
        EvalResult
            ``ranking.ndcg@k``, normalized by the ideal DCG per row.
        """
        check_k(k)
        ranked, relevant = _prepare(ranked, relevant)
        scores = []
        for ranks, rel in zip(ranked, relevant, strict=True):
            gain_of: Mapping[Hashable, float]
            gain_of = rel if isinstance(rel, Mapping) else {item: 1.0 for item in rel}
            if not gain_of:
                continue
            dcg = _dcg([float(gain_of.get(item, 0.0)) for item in ranks[:k]])
            ideal = _dcg(sorted((float(g) for g in gain_of.values()), reverse=True)[:k])
            scores.append(dcg / ideal if ideal > 0 else 0.0)
        return EvalResult(
            name=f"ranking.ndcg@{k}", value=_mean(scores), n=len(scores), params={"k": k}
        )

    def hit_rate_at_k(self, ranked: RankedInput, relevant: RelevanceInput, k: int) -> EvalResult:
        """Mean Hit Rate@k: fraction of rows with ≥1 relevant item in the top ``k``.

        Parameters
        ----------
        ranked, relevant, k
            See :meth:`precision_at_k`.

        Returns
        -------
        EvalResult
            ``ranking.hit_rate@k`` (rows with no relevant items skipped).

        Notes
        -----
        Identical to *Success@k* in TREC terminology (1 iff any relevant in top k).
        """
        check_k(k)
        ranked, relevant = _prepare(ranked, relevant)
        scores = []
        for ranks, rel in zip(ranked, relevant, strict=True):
            rel_set = _binary_set(rel)
            if not rel_set:
                continue
            scores.append(1.0 if any(item in rel_set for item in ranks[:k]) else 0.0)
        return EvalResult(
            name=f"ranking.hit_rate@{k}", value=_mean(scores), n=len(scores), params={"k": k}
        )

    def r_precision(self, ranked: RankedInput, relevant: RelevanceInput) -> EvalResult:
        """Mean R-Precision: precision at rank R, where R = #relevant for the row.

        At cutoff R precision equals recall (the break-even point). Rows with no
        relevant items are undefined and skipped.

        Parameters
        ----------
        ranked, relevant
            See :meth:`precision_at_k`.

        Returns
        -------
        EvalResult
            ``ranking.r_precision``.

        References
        ----------
        TREC common evaluation measures.
        https://trec.nist.gov/pubs/trec15/appendices/CE.MEASURES06.pdf
        """
        ranked, relevant = _prepare(ranked, relevant)
        scores = []
        for ranks, rel in zip(ranked, relevant, strict=True):
            rel_set = _binary_set(rel)
            r = len(rel_set)
            if r == 0:
                continue
            hits = sum(1 for item in ranks[:r] if item in rel_set)
            scores.append(hits / r)
        return EvalResult(name="ranking.r_precision", value=_mean(scores), n=len(scores))

    def bpref(self, ranked: RankedInput, relevant: RelevanceInput) -> EvalResult:
        """Mean binary preference (bpref) — robust to incomplete judgments.

        ``bpref = (1/R) Σ_r (1 - min(|nonrel ranked above r|, R) / R)`` over the R
        relevant docs (non-retrieved relevant docs contribute 0). Here every
        non-relevant item present in ``ranked`` is treated as *judged* non-relevant;
        for true unjudged-aware bpref use the ``pytrec_eval`` backend.

        Parameters
        ----------
        ranked, relevant
            See :meth:`precision_at_k`.

        Returns
        -------
        EvalResult
            ``ranking.bpref`` (rows with no relevant items skipped).

        References
        ----------
        C. Buckley, E. Voorhees, "Retrieval evaluation with incomplete
        information", SIGIR 2004. https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=150469
        """
        ranked, relevant = _prepare(ranked, relevant)
        scores = []
        for ranks, rel in zip(ranked, relevant, strict=True):
            rel_set = _binary_set(rel)
            r = len(rel_set)
            if r == 0:
                continue
            nonrel_above = 0
            total = 0.0
            for item in ranks:
                if item in rel_set:
                    total += 1.0 - min(nonrel_above, r) / r
                else:
                    nonrel_above += 1
            scores.append(total / r)
        return EvalResult(name="ranking.bpref", value=_mean(scores), n=len(scores))

    def rbp(self, ranked: RankedInput, relevant: RelevanceInput, *, p: float = 0.8) -> EvalResult:
        """Mean Rank-Biased Precision: ``(1-p) Σ_i rel_i · p^(i-1)``.

        Models a user who advances to the next result with persistence ``p`` and
        stops with probability ``1-p``. Well-defined (0) when no relevant docs.

        Parameters
        ----------
        ranked, relevant
            See :meth:`precision_at_k`.
        p : float
            Persistence in [0, 1) (default 0.8).

        Returns
        -------
        EvalResult
            ``ranking.rbp@p``.

        References
        ----------
        A. Moffat, J. Zobel, "Rank-biased precision for measurement of retrieval
        effectiveness", ACM TOIS 2008. https://people.eng.unimelb.edu.au/jzobel/fulltext/acmtois08.pdf
        """
        if not 0.0 <= p < 1.0:
            raise ValueError(f"p must be in [0, 1), got {p}.")
        ranked, relevant = _prepare(ranked, relevant)
        scores = []
        for ranks, rel in zip(ranked, relevant, strict=True):
            rel_set = _binary_set(rel)
            gain = sum((p ** (i - 1)) for i, item in enumerate(ranks, start=1) if item in rel_set)
            scores.append((1.0 - p) * gain)
        return EvalResult(
            name=f"ranking.rbp@{p}", value=_mean(scores), n=len(ranked), params={"p": p}
        )


ranking = RankingMetrics()
"""Shared :class:`RankingMetrics` singleton for ergonomic access."""
