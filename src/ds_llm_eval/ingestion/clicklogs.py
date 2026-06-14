"""Turn raw interaction logs into ranked lists + relevance judgments (qrels/runs).

Search and recommendation evaluation needs, per query/user, a *ranked list* of
item ids and a *relevant set*. Production systems instead emit flat event logs
(``query, doc, clicked, rank``). :class:`ClickLogIngestor` reshapes such a log
into a :class:`RankingDataset` ready for the ``ranking`` metrics.

Position-bias caveat
--------------------
Treating a click as a relevance label conflates *relevance* with *examination*:
users click top-ranked results far more often regardless of true relevance, so
click-derived qrels are biased toward the logging policy. Use click-based labels
as a noisy proxy, and prefer position-bias-corrected estimators (e.g. inverse
propensity weighting) for unbiased offline evaluation.

References
----------
.. [1] T. Joachims et al., "Accurately Interpreting Clickthrough Data as Implicit
   Feedback", SIGIR 2005. https://www.cs.cornell.edu/people/tj/publications/joachims_etal_05a.pdf
.. [2] T. Joachims, A. Swaminathan, T. Schnabel, "Unbiased Learning-to-Rank with
   Biased Feedback", WSDM 2017. https://arxiv.org/abs/1608.04468
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class RankingDataset:
    """Aligned ranked lists + relevant sets, ready for the ``ranking`` metrics.

    Attributes
    ----------
    ranked : list of list of hashable
        ``ranked[i]`` is the ranked item ids for ``query_ids[i]``.
    relevant : list of set of hashable
        ``relevant[i]`` is the relevant item ids for ``query_ids[i]``.
    query_ids : list of hashable
        Query/user identifiers, aligned with ``ranked``/``relevant``.
    """

    ranked: list[list[Hashable]]
    relevant: list[set[Hashable]]
    query_ids: list[Hashable] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ranked)


class ClickLogIngestor:
    """Reshape a flat interaction log (a DataFrame) into a :class:`RankingDataset`.

    Stateless; the module exposes a shared singleton :data:`click_log`.
    """

    def from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        query_col: str,
        doc_col: str,
        clicked_col: str | None = None,
        rank_col: str | None = None,
        score_col: str | None = None,
    ) -> RankingDataset:
        """Build a :class:`RankingDataset` from an interaction-log DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            One row per (query, doc) impression/event.
        query_col, doc_col : str
            Column names for the query/user id and the item/doc id.
        clicked_col : str, optional
            Truthy-valued column marking relevance (e.g. clicked). If omitted,
            every listed doc is treated as relevant (rarely what you want).
        rank_col : str, optional
            If given, docs within a query are ordered by ascending rank.
        score_col : str, optional
            If given (and ``rank_col`` is not), docs are ordered by descending
            score. With neither, the existing row order is preserved.

        Returns
        -------
        RankingDataset
            Aligned ranked lists, relevant sets, and query ids.

        Raises
        ------
        ValueError
            If the DataFrame is empty or a named column is missing.
        """
        if df.empty:
            raise ValueError("click log DataFrame is empty.")
        needed = [query_col, doc_col]
        for col in (clicked_col, rank_col, score_col):
            if col is not None:
                needed.append(col)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}. Available: {list(df.columns)}")

        ranked: list[list[Hashable]] = []
        relevant: list[set[Hashable]] = []
        query_ids: list[Hashable] = []

        for qid, group in df.groupby(query_col, sort=False):
            if rank_col is not None:
                group = group.sort_values(rank_col, ascending=True)
            elif score_col is not None:
                group = group.sort_values(score_col, ascending=False)
            docs = list(group[doc_col])
            if clicked_col is not None:
                rel = {
                    doc
                    for doc, clicked in zip(group[doc_col], group[clicked_col], strict=True)
                    if bool(clicked)
                }
            else:
                rel = set(docs)
            ranked.append(docs)
            relevant.append(rel)
            query_ids.append(qid)

        return RankingDataset(ranked=ranked, relevant=relevant, query_ids=query_ids)


click_log = ClickLogIngestor()
"""Shared :class:`ClickLogIngestor` singleton."""
