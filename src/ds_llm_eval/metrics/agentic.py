"""Deterministic agentic-pipeline metrics: tool calls, trajectories, goals.

These score an agent's *path* against ground truth without an LLM judge — pure
comparison, so they are deterministic, offline, and default-installable. (Model-
graded agentic metrics live in :mod:`ds_llm_eval.metrics.llm`.)

A *tool call* is given as ``(name, args)`` or ``{"name": ..., "args": {...}}``;
args are canonicalized (JSON, sorted keys) for order-independent comparison.
Inputs are batched per row (one agent run per row).

References
----------
.. [1] RAGAS agent/tool metrics (ToolCallAccuracy, ToolCallF1).
   https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/
.. [2] LangChain AgentEvals trajectory match (strict/unordered/subset/superset).
   https://github.com/langchain-ai/agentevals
.. [3] τ-bench — final database-state comparison to an annotated goal state.
   https://arxiv.org/abs/2406.12045
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal

from ..core import EvalResult, check_non_empty

ToolCall = tuple[str, Mapping[str, Any]] | Mapping[str, Any] | str
TrajectoryMode = Literal["strict", "unordered", "subset", "superset"]


def _norm_call(call: ToolCall) -> tuple[str, str]:
    """Canonicalize a tool call to ``(name, json-of-args)`` for comparison."""
    if isinstance(call, Mapping):
        name = str(call.get("name", call.get("tool", "")))
        args: Any = call.get("args", call.get("arguments", {}))
    elif isinstance(call, tuple) and len(call) == 2:
        name, args = str(call[0]), call[1]
    else:
        name, args = str(call), {}
    return name, json.dumps(args, sort_keys=True, default=str)


def _validate(predicted: Sequence[object], reference: Sequence[object]) -> None:
    check_non_empty(predicted, name="predicted")
    if len(predicted) != len(reference):
        raise ValueError(
            f"predicted ({len(predicted)}) and reference ({len(reference)}) length mismatch."
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class AgenticMetrics:
    """Deterministic tool-call / trajectory / goal metrics. Singleton :data:`agentic`."""

    def tool_call_accuracy(
        self,
        predicted: Sequence[Sequence[ToolCall]],
        reference: Sequence[Sequence[ToolCall]],
        *,
        strict_order: bool = True,
    ) -> EvalResult:
        """RAGAS-style tool-call accuracy: argument accuracy gated by sequence alignment.

        Per row: if the predicted tool-name sequence aligns with the reference
        (same order when ``strict_order``, else same multiset), the score is the
        fraction of reference calls whose name **and** args match; otherwise 0.

        Parameters
        ----------
        predicted, reference : sequence of sequence of tool call
            Per-row tool-call sequences (agent output vs. ground truth).
        strict_order : bool
            Require the same call order (default True).

        Returns
        -------
        EvalResult
            ``agentic.tool_call_accuracy`` in [0, 1].
        """
        _validate(predicted, reference)
        scores = []
        for pred_row, ref_row in zip(predicted, reference, strict=True):
            pred = [_norm_call(c) for c in pred_row]
            ref = [_norm_call(c) for c in ref_row]
            if not ref:
                scores.append(1.0 if not pred else 0.0)
                continue
            pred_names = [c[0] for c in pred]
            ref_names = [c[0] for c in ref]
            if strict_order:
                if pred_names != ref_names:
                    scores.append(0.0)
                    continue
                matched = sum(1 for i, call in enumerate(ref) if i < len(pred) and pred[i] == call)
            else:
                if sorted(pred_names) != sorted(ref_names):
                    scores.append(0.0)
                    continue
                matched = sum((Counter(pred) & Counter(ref)).values())
            scores.append(matched / len(ref))
        return EvalResult(
            name="agentic.tool_call_accuracy",
            value=_mean(scores),
            n=len(predicted),
            params={"strict_order": strict_order},
        )

    def tool_call_f1(
        self,
        predicted: Sequence[Sequence[ToolCall]],
        reference: Sequence[Sequence[ToolCall]],
    ) -> EvalResult:
        """RAGAS-style tool-call F1: unordered precision/recall over (name, args) calls.

        Precision = matched / (matched + extra); Recall = matched / (matched +
        missed); F1 the harmonic mean. Order-independent (multiset overlap).

        Returns
        -------
        EvalResult
            ``agentic.tool_call_f1`` in [0, 1]; both rows empty → 1.0.
        """
        _validate(predicted, reference)
        scores = []
        for pred_row, ref_row in zip(predicted, reference, strict=True):
            pred = Counter([_norm_call(c) for c in pred_row])
            ref = Counter([_norm_call(c) for c in ref_row])
            matched = sum((pred & ref).values())
            n_pred = sum(pred.values())
            n_ref = sum(ref.values())
            if n_pred == 0 and n_ref == 0:
                scores.append(1.0)
                continue
            if matched == 0:
                scores.append(0.0)
                continue
            precision = matched / n_pred
            recall = matched / n_ref
            scores.append(2 * precision * recall / (precision + recall))
        return EvalResult(name="agentic.tool_call_f1", value=_mean(scores), n=len(predicted))

    def trajectory_match(
        self,
        predicted: Sequence[Sequence[ToolCall]],
        reference: Sequence[Sequence[ToolCall]],
        *,
        mode: TrajectoryMode = "strict",
    ) -> EvalResult:
        """Fraction of rows whose trajectory matches the reference under ``mode``.

        Modes (AgentEvals): ``strict`` (identical ordered steps), ``unordered``
        (same step multiset), ``subset`` (predicted ⊆ reference), ``superset``
        (predicted ⊇ reference).

        Returns
        -------
        EvalResult
            ``agentic.trajectory_match`` in [0, 1].
        """
        if mode not in ("strict", "unordered", "subset", "superset"):
            raise ValueError(f"unknown trajectory mode {mode!r}.")
        _validate(predicted, reference)
        scores = []
        for pred_row, ref_row in zip(predicted, reference, strict=True):
            pred = [_norm_call(c) for c in pred_row]
            ref = [_norm_call(c) for c in ref_row]
            if mode == "strict":
                ok = pred == ref
            elif mode == "unordered":
                ok = Counter(pred) == Counter(ref)
            elif mode == "subset":
                cp, cr = Counter(pred), Counter(ref)
                ok = all(cp[s] <= cr[s] for s in cp)
            else:  # superset
                cp, cr = Counter(pred), Counter(ref)
                ok = all(cr[s] <= cp[s] for s in cr)
            scores.append(1.0 if ok else 0.0)
        return EvalResult(
            name="agentic.trajectory_match",
            value=_mean(scores),
            n=len(predicted),
            params={"mode": mode},
        )

    def goal_accuracy(
        self,
        predicted: Sequence[Hashable],
        reference: Sequence[Hashable],
    ) -> EvalResult:
        """Final-state goal accuracy (τ-bench style): exact match of end states.

        Parameters
        ----------
        predicted, reference : sequence of hashable
            Per-row achieved vs. goal final state (e.g. a frozen DB snapshot).

        Returns
        -------
        EvalResult
            ``agentic.goal_accuracy`` = fraction of rows whose states are equal.
        """
        _validate(predicted, reference)
        scores = [1.0 if p == r else 0.0 for p, r in zip(predicted, reference, strict=True)]
        return EvalResult(name="agentic.goal_accuracy", value=_mean(scores), n=len(predicted))


agentic = AgenticMetrics()
"""Shared :class:`AgenticMetrics` singleton."""
