"""Data ingestion: reshape raw logs into evaluation-ready datasets."""

from .clicklogs import ClickLogIngestor, RankingDataset, click_log

__all__ = ["ClickLogIngestor", "RankingDataset", "click_log"]
