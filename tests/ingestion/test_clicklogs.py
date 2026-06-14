"""Tests for click-log ingestion into a RankingDataset."""

import pandas as pd
import pytest

from ds_llm_eval import click_log
from ds_llm_eval.ingestion import ClickLogIngestor, RankingDataset


def _log() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "q": ["q1", "q1", "q1", "q2", "q2"],
            "doc": ["d2", "d1", "d3", "d9", "d4"],
            "clicked": [0, 1, 1, 0, 1],
            "rank": [2, 1, 3, 1, 2],
        }
    )


def test_from_dataframe_orders_by_rank_and_collects_clicks():
    ds = click_log.from_dataframe(
        _log(), query_col="q", doc_col="doc", clicked_col="clicked", rank_col="rank"
    )
    assert isinstance(ds, RankingDataset)
    assert len(ds) == 2
    assert ds.query_ids == ["q1", "q2"]
    # q1 ordered by rank: d1(1), d2(2), d3(3); clicked d1,d3 relevant
    assert ds.ranked[0] == ["d1", "d2", "d3"]
    assert ds.relevant[0] == {"d1", "d3"}
    assert ds.relevant[1] == {"d4"}


def test_feeds_ranking_metrics():
    from ds_llm_eval import ranking

    ds = click_log.from_dataframe(
        _log(), query_col="q", doc_col="doc", clicked_col="clicked", rank_col="rank"
    )
    res = ranking.mrr(ds.ranked, ds.relevant)
    # q1: first relevant d1 at rank 1 -> 1.0; q2: d4 at rank 2 -> 0.5; mean 0.75
    assert res.value == pytest.approx(0.75)


def test_orders_by_score_descending_when_no_rank():
    df = pd.DataFrame({"q": ["a", "a"], "doc": ["x", "y"], "clicked": [1, 0], "score": [0.2, 0.9]})
    ds = ClickLogIngestor().from_dataframe(
        df, query_col="q", doc_col="doc", clicked_col="clicked", score_col="score"
    )
    assert ds.ranked[0] == ["y", "x"]  # y has higher score


def test_missing_column_raises():
    with pytest.raises(ValueError):
        click_log.from_dataframe(_log(), query_col="q", doc_col="nope")


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        click_log.from_dataframe(pd.DataFrame({"q": [], "doc": []}), query_col="q", doc_col="doc")
