"""Tests for TextMetrics and the (mocked) LLM-judge adapter."""

import builtins

import pytest

from ds_llm_eval import llm_judge, text

# --- deterministic text metrics ---------------------------------------------


def test_exact_match_worked_example():
    # "Paris." normalizes to match "paris"; "London" != "paris" -> mean 0.5
    res = text.exact_match(["Paris.", "London"], ["paris", "paris"])
    assert res.value == 0.5
    assert res.name == "text.exact_match"


def test_token_f1_worked_example():
    # pred {quick,brown,fox} vs ref {brown,fox}: p=2/3, r=1 -> f1=0.8
    res = text.token_f1(["the quick brown fox"], ["the brown fox"])
    assert res.value == pytest.approx(0.8)


def test_token_f1_both_empty_after_normalization_is_one():
    assert text.token_f1(["the the"], ["a an"]).value == 1.0


def test_token_f1_one_empty_is_zero():
    assert text.token_f1(["the"], ["hello"]).value == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        text.exact_match(["a"], ["a", "b"])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        text.token_f1([], [])


# --- LLM-judge adapter: clear error when the optional extra is missing ------


def test_ragas_evaluate_requires_llm_extra(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets" or name.startswith("ragas"):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"ds-llm-eval\[llm\]"):
        llm_judge.ragas_evaluate(["q"], ["a"], [["ctx"]])


# --- Milestone 2: BLEU / ROUGE-L / JSON correctness -------------------------


def test_bleu_identical_is_one():
    assert text.bleu(["the cat sat"], ["the cat sat"]).value == pytest.approx(1.0)


def test_bleu_no_overlap_is_zero():
    assert text.bleu(["dog"], ["cat"]).value == 0.0


def test_bleu_smoothing_avoids_zero_on_partial_match():
    # shares unigrams/bigrams but no 4-gram; smoothing keeps it > 0
    score = text.bleu(["the quick brown fox"], ["the quick brown dog"], smoothing=True).value
    assert 0.0 < score < 1.0


def test_rouge_l_identical_is_one():
    assert text.rouge_l(["a b c"], ["a b c"]).value == pytest.approx(1.0)


def test_rouge_l_worked_example_beta1():
    # LCS=2; P=2/2=1, R=2/4=0.5; F1=2PR/(P+R)=0.6667
    assert text.rouge_l(["a b"], ["a b c d"], beta=1.0).value == pytest.approx(2 / 3)


def test_rouge_l_empty_cases():
    assert text.rouge_l([""], [""]).value == 1.0
    assert text.rouge_l(["a"], [""]).value == 0.0


def test_json_correctness_valid_rate():
    res = text.json_correctness(['{"a": 1}', "not json", "[1, 2]"])
    assert res.value == pytest.approx(2 / 3)


def test_json_correctness_schema_requires_jsonschema(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("jsonschema"):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="jsonschema"):
        text.json_correctness(['{"a": 1}'], schema={"type": "object"})
