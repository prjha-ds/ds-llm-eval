"""Tests for the evaluation logging mechanism — fully offline and deterministic."""

import builtins
import io
import json
from datetime import datetime, timezone

import pytest

from ds_llm_eval import EvalReport, EvalResult
from ds_llm_eval.logging import ConsoleLogger, JSONLLogger, LangfuseLogger, MultiLogger


def _fixed_clock():
    return datetime(2026, 6, 4, 12, 0, 0, tzinfo=timezone.utc)


class _FakeLangfuse:
    def __init__(self):
        self.calls: list[dict] = []

    def create_score(self, **kwargs):
        self.calls.append(kwargs)


# --- ConsoleLogger ----------------------------------------------------------


def test_console_logger_formats_line_and_returns_record():
    stream = io.StringIO()
    log = ConsoleLogger(stream=stream, run_id="exp-1", clock=_fixed_clock)
    record = log.log_result(EvalResult(name="search.ndcg@10", value=0.5, n=3, params={"k": 10}))

    out = stream.getvalue()
    assert "search.ndcg@10 = 0.5000 (n=3)" in out
    assert "(exp-1)" in out
    assert out.startswith("[2026-06-04T12:00:00+00:00]")
    assert record["metric"] == "search.ndcg@10"
    assert record["run_id"] == "exp-1"
    assert record["params"] == {"k": 10}


def test_console_logger_omits_run_id_when_unset():
    stream = io.StringIO()
    ConsoleLogger(stream=stream, clock=_fixed_clock).log_result(EvalResult(name="m", value=1.0))
    assert "()" not in stream.getvalue()


# --- JSONLLogger ------------------------------------------------------------


def test_jsonl_logger_writes_one_object_per_line_and_creates_dirs(tmp_path):
    path = tmp_path / "runs" / "eval.jsonl"  # parent dir must be created
    report = EvalReport(
        results=[EvalResult(name="a", value=1.0, n=2), EvalResult(name="b", value=0.0, n=2)]
    )
    with JSONLLogger(path, metadata={"model": "v2"}, clock=_fixed_clock) as log:
        records = log.log_report(report)

    assert len(records) == 2
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {
        "timestamp": "2026-06-04T12:00:00+00:00",
        "metric": "a",
        "value": 1.0,
        "n": 2,
        "params": {},
        "metadata": {"model": "v2"},
    }


def test_jsonl_logger_merges_result_metadata_over_logger_metadata(tmp_path):
    path = tmp_path / "eval.jsonl"
    result = EvalResult(name="m", value=0.5, metadata={"split": "test", "model": "override"})
    with JSONLLogger(path, metadata={"model": "base"}, clock=_fixed_clock) as log:
        log.log_result(result)
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["metadata"] == {"model": "override", "split": "test"}


def test_jsonl_logger_raises_after_close(tmp_path):
    log = JSONLLogger(tmp_path / "eval.jsonl", clock=_fixed_clock)
    log.close()
    with pytest.raises(ValueError):
        log.log_result(EvalResult(name="m", value=1.0))


# --- LangfuseLogger ---------------------------------------------------------


def test_langfuse_logger_emits_create_score():
    client = _FakeLangfuse()
    log = LangfuseLogger(trace_id="trace-9", client=client, clock=_fixed_clock)
    log.log_result(EvalResult(name="search.mrr", value=0.75, n=2))

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["trace_id"] == "trace-9"
    assert call["name"] == "search.mrr"
    assert call["value"] == 0.75
    assert call["data_type"] == "NUMERIC"


def test_langfuse_logger_requires_extra_without_client(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("langfuse"):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"ds-llm-eval\[llm\]"):
        LangfuseLogger(trace_id="t")


# --- MultiLogger ------------------------------------------------------------


def test_multilogger_fans_out_to_all_children(tmp_path):
    stream = io.StringIO()
    path = tmp_path / "multi.jsonl"
    console = ConsoleLogger(stream=stream, clock=_fixed_clock)
    jsonl = JSONLLogger(path, clock=_fixed_clock)

    with MultiLogger([console, jsonl]) as log:
        log.log_result(EvalResult(name="x", value=0.25, n=1))

    assert "x = 0.2500 (n=1)" in stream.getvalue()
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 1
