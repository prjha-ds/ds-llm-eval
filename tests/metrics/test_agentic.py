"""Tests for deterministic agentic metrics: tool calls, trajectories, goals."""

import pytest

from ds_llm_eval import agentic

# --- tool_call_accuracy -----------------------------------------------------


def test_tool_call_accuracy_exact_match():
    pred = [[("search", {"q": "x"})]]
    ref = [[("search", {"q": "x"})]]
    assert agentic.tool_call_accuracy(pred, ref).value == 1.0


def test_tool_call_accuracy_arg_mismatch_scores_zero():
    pred = [[("search", {"q": "y"})]]
    ref = [[("search", {"q": "x"})]]
    assert agentic.tool_call_accuracy(pred, ref).value == 0.0


def test_tool_call_accuracy_name_mismatch_not_aligned():
    pred = [[("foo", {})]]
    ref = [[("search", {})]]
    assert agentic.tool_call_accuracy(pred, ref).value == 0.0


def test_tool_call_accuracy_dict_form_and_unordered():
    pred = [[{"name": "b", "args": {}}, {"name": "a", "args": {}}]]
    ref = [[{"name": "a", "args": {}}, {"name": "b", "args": {}}]]
    assert agentic.tool_call_accuracy(pred, ref, strict_order=False).value == 1.0
    assert agentic.tool_call_accuracy(pred, ref, strict_order=True).value == 0.0


# --- tool_call_f1 -----------------------------------------------------------


def test_tool_call_f1_partial_overlap():
    pred = [[("a", {}), ("b", {})]]
    ref = [[("a", {}), ("c", {})]]
    # matched {a}=1; precision 1/2, recall 1/2 -> F1 0.5
    assert agentic.tool_call_f1(pred, ref).value == pytest.approx(0.5)


def test_tool_call_f1_both_empty_is_one():
    assert agentic.tool_call_f1([[]], [[]]).value == 1.0


# --- trajectory_match -------------------------------------------------------


def test_trajectory_match_modes():
    assert agentic.trajectory_match([["a", "b"]], [["a", "b"]], mode="strict").value == 1.0
    assert agentic.trajectory_match([["b", "a"]], [["a", "b"]], mode="strict").value == 0.0
    assert agentic.trajectory_match([["b", "a"]], [["a", "b"]], mode="unordered").value == 1.0
    assert agentic.trajectory_match([["a"]], [["a", "b"]], mode="subset").value == 1.0
    assert agentic.trajectory_match([["a", "b"]], [["a"]], mode="superset").value == 1.0


def test_trajectory_match_bad_mode_raises():
    with pytest.raises(ValueError):
        agentic.trajectory_match([["a"]], [["a"]], mode="nope")  # type: ignore[arg-type]


# --- goal_accuracy ----------------------------------------------------------


def test_goal_accuracy_worked_example():
    assert agentic.goal_accuracy(["s1", "s2"], ["s1", "x"]).value == 0.5


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        agentic.goal_accuracy(["a"], ["a", "b"])


def test_empty_input_raises():
    with pytest.raises(ValueError):
        agentic.tool_call_f1([], [])
