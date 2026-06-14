"""Declarative, reproducible evaluation runs from a single config (YAML or dict).

Wire a whole run — which metrics (with params), where to read the data, and where
to log results — into one declarative spec, so an experiment is reproducible and
diffable. Mirrors the Elliot YAML-driven pattern (see ``docs/research.md`` §2).

Example
-------
.. code-block:: yaml

    run_id: exp-1
    metadata: {model: ranker-v2}
    dataset:
      path: data/click_log.csv
      query_col: q
      doc_col: doc
      clicked_col: clicked
      rank_col: rank
    metrics:
      - {name: ranking.ndcg_at_k, params: {k: 10}}
      - ranking.mrr
    logging:
      - {type: console}
      - {type: jsonl, path: runs/exp-1.jsonl}

.. code-block:: python

    from ds_llm_eval.config import ExperimentRunner
    report = ExperimentRunner.from_yaml("experiment.yaml").run()
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .core import EvalReport
from .evaluation import Evaluator, MetricSpec
from .ingestion import ClickLogIngestor
from .logging import ConsoleLogger, EvalLogger, JSONLLogger, LangfuseLogger, MultiLogger


@dataclass(frozen=True)
class DatasetConfig:
    """A click-log CSV source to ingest into ranked lists + relevance judgments."""

    path: str
    query_col: str
    doc_col: str
    clicked_col: str | None = None
    rank_col: str | None = None
    score_col: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """A fully declarative evaluation run."""

    metrics: list[MetricSpec]
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    loggers: list[dict[str, Any]] = field(default_factory=list)
    dataset: DatasetConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Build from a plain dict (e.g. parsed YAML/JSON)."""
        raw_metrics = data.get("metrics")
        if not raw_metrics:
            raise ValueError("config must list at least one metric under 'metrics'.")
        metrics: list[MetricSpec] = []
        for m in raw_metrics:
            if isinstance(m, str):
                metrics.append(MetricSpec(m))
            elif isinstance(m, Mapping):
                if "name" not in m:
                    raise ValueError(f"metric entry must have a 'name': {m!r}")
                params = m.get("params", {})
                if not isinstance(params, Mapping):
                    raise ValueError(
                        f"metric 'params' must be a mapping, got {type(params).__name__}."
                    )
                metrics.append(MetricSpec(str(m["name"]), dict(params)))
            else:
                raise ValueError(
                    f"metric entry must be a string or mapping, got {type(m).__name__}."
                )
        dataset = DatasetConfig(**data["dataset"]) if data.get("dataset") else None
        return cls(
            metrics=metrics,
            run_id=data.get("run_id"),
            metadata=dict(data.get("metadata") or {}),
            loggers=list(data.get("logging") or data.get("loggers") or []),
            dataset=dataset,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Build from a YAML file."""
        with Path(path).open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a YAML mapping.")
        return cls.from_dict(data)


class ExperimentRunner:
    """Run an :class:`ExperimentConfig` end-to-end and return an :class:`EvalReport`."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentRunner:
        return cls(ExperimentConfig.from_yaml(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRunner:
        return cls(ExperimentConfig.from_dict(data))

    def _one_logger(self, spec: dict[str, Any]) -> EvalLogger:
        kind = spec.get("type")
        if kind is None:
            raise ValueError("logger spec must have a 'type' (console/jsonl/langfuse).")
        run_id, metadata = self.config.run_id, self.config.metadata
        if kind == "console":
            return ConsoleLogger(run_id=run_id, metadata=metadata)
        if kind == "jsonl":
            if "path" not in spec:
                raise ValueError("jsonl logger requires a 'path'.")
            return JSONLLogger(spec["path"], run_id=run_id, metadata=metadata)
        if kind == "langfuse":
            if "trace_id" not in spec:
                raise ValueError("langfuse logger requires a 'trace_id'.")
            return LangfuseLogger(trace_id=spec["trace_id"], run_id=run_id, metadata=metadata)
        raise ValueError(f"unknown logger type {kind!r} (use console/jsonl/langfuse).")

    def _build_logger(self) -> EvalLogger | None:
        built = [self._one_logger(s) for s in self.config.loggers]
        if not built:
            return None
        return built[0] if len(built) == 1 else MultiLogger(built)

    def _load_dataset(self) -> tuple[list[Any], list[Any]]:
        ds_cfg = self.config.dataset
        assert ds_cfg is not None
        frame = pd.read_csv(ds_cfg.path)
        dataset = ClickLogIngestor().from_dataframe(
            frame,
            query_col=ds_cfg.query_col,
            doc_col=ds_cfg.doc_col,
            clicked_col=ds_cfg.clicked_col,
            rank_col=ds_cfg.rank_col,
            score_col=ds_cfg.score_col,
        )
        return dataset.ranked, dataset.relevant

    def run(self, *data: object) -> EvalReport:
        """Run the configured metrics, reading the configured dataset if ``data`` is empty.

        Parameters
        ----------
        *data
            Positional data forwarded to each metric. If omitted, the config's
            ``dataset`` is ingested (a click-log CSV) into ``(ranked, relevant)``.

        Returns
        -------
        EvalReport
        """
        if not data:
            if self.config.dataset is None:
                raise ValueError("run() needs data, or a 'dataset' section in the config.")
            data = tuple(self._load_dataset())
        logger = self._build_logger()
        try:
            return Evaluator(self.config.metrics, logger=logger).run(*data)
        finally:
            if logger is not None:
                logger.close()
