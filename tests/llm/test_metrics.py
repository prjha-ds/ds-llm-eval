"""Tests for LLM/agentic metrics and the (mocked) RAGAS adapter."""

import builtins

import pytest

from ds_llm_eval import llm

# --- deterministic reference metrics ----------------------------------------


def test_exact_match_worked_example():
    # "Paris." normalizes to match "paris"; "London" != "paris" -> mean 0.5
    res = llm.exact_match(["Paris.", "London"], ["paris", "paris"])
    assert res.value == 0.5


def test_token_f1_worked_example():
    # pred tokens {quick,brown,fox} vs ref {brown,fox}: p=2/3, r=1 -> f1=0.8
    res = llm.token_f1(["the quick brown fox"], ["the brown fox"])
    assert res.value == pytest.approx(0.8)


def test_token_f1_both_empty_after_normalization_is_one():
    # only articles -> both normalize to [] -> defined as perfect match
    assert llm.token_f1(["the the"], ["a an"]).value == 1.0


def test_token_f1_one_empty_is_zero():
    assert llm.token_f1(["the"], ["hello"]).value == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        llm.exact_match(["a"], ["a", "b"])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        llm.token_f1([], [])


# --- RAGAS adapter: clear error when the optional extra is missing ----------


def test_ragas_evaluate_requires_llm_extra(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets" or name.startswith("ragas"):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"ds-llm-eval\[llm\]"):
        llm.ragas_evaluate(["q"], ["a"], [["ctx"]])
