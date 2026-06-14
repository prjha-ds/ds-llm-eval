"""Tests for the declarative YAML/dict experiment runner."""

import json

import pandas as pd
import pytest

from ds_llm_eval import ExperimentConfig, ExperimentRunner


def test_from_dict_runs_metrics_with_jsonl_logging(tmp_path):
    log_path = tmp_path / "out.jsonl"
    cfg = {
        "run_id": "exp1",
        "metadata": {"model": "v2"},
        "metrics": [{"name": "ranking.ndcg_at_k", "params": {"k": 3}}, "ranking.mrr"],
        "logging": [{"type": "jsonl", "path": str(log_path)}],
    }
    report = ExperimentRunner.from_dict(cfg).run([["a", "b", "c"]], [{"a", "c"}])
    assert set(report.as_dict()) == {"ranking.ndcg@3", "ranking.mrr"}
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["run_id"] == "exp1"
    assert rec["metadata"] == {"model": "v2"}


def test_from_yaml(tmp_path):
    yaml_path = tmp_path / "exp.yaml"
    yaml_path.write_text("metrics:\n  - ranking.mrr\n", encoding="utf-8")
    report = ExperimentRunner.from_yaml(yaml_path).run([["a"]], [{"a"}])
    assert report.as_dict()["ranking.mrr"] == 1.0


def test_dataset_section_ingests_csv(tmp_path):
    csv = tmp_path / "log.csv"
    pd.DataFrame(
        {"q": ["q1", "q1"], "doc": ["d1", "d2"], "clicked": [1, 0], "rank": [1, 2]}
    ).to_csv(csv, index=False)
    cfg = {
        "metrics": ["ranking.mrr"],
        "dataset": {
            "path": str(csv),
            "query_col": "q",
            "doc_col": "doc",
            "clicked_col": "clicked",
            "rank_col": "rank",
        },
    }
    report = ExperimentRunner.from_dict(cfg).run()  # no inline data -> ingests CSV
    assert report.as_dict()["ranking.mrr"] == 1.0  # d1 clicked at rank 1


def test_multiple_loggers_fan_out(tmp_path, capsys):
    log_path = tmp_path / "m.jsonl"
    cfg = {
        "metrics": ["ranking.mrr"],
        "logging": [{"type": "console"}, {"type": "jsonl", "path": str(log_path)}],
    }
    ExperimentRunner.from_dict(cfg).run([["a"]], [{"a"}])
    assert "ranking.mrr = 1.0000" in capsys.readouterr().out
    assert len(log_path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_unknown_logger_type_raises():
    cfg = {"metrics": ["ranking.mrr"], "logging": [{"type": "bogus"}]}
    with pytest.raises(ValueError, match="unknown logger type"):
        ExperimentRunner.from_dict(cfg).run([["a"]], [{"a"}])


def test_no_data_and_no_dataset_raises():
    with pytest.raises(ValueError):
        ExperimentRunner.from_dict({"metrics": ["ranking.mrr"]}).run()


def test_empty_metrics_raises():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"metrics": []})


def test_metric_entry_missing_name_raises():
    with pytest.raises(ValueError, match="must have a 'name'"):
        ExperimentConfig.from_dict({"metrics": [{"params": {"k": 3}}]})


def test_metric_params_not_mapping_raises():
    with pytest.raises(ValueError, match="params' must be a mapping"):
        ExperimentConfig.from_dict({"metrics": [{"name": "ranking.mrr", "params": [1, 2]}]})


def test_logger_missing_type_raises():
    cfg = {"metrics": ["ranking.mrr"], "logging": [{"path": "x.jsonl"}]}
    with pytest.raises(ValueError, match="must have a 'type'"):
        ExperimentRunner.from_dict(cfg).run([["a"]], [{"a"}])
