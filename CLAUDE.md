# CLAUDE.md — ds-llm-eval

Pip-installable toolkit to evaluate **search**, **recommendation**, and **LLM/agentic** pipelines
behind one consistent API. Import name `ds_llm_eval`; repo `ds-llm-eval` (branch `master`).

## Architecture
- `src/ds_llm_eval/core/` — shared contracts: `EvalResult`, `Metric` protocol, registry, validation.
- `src/ds_llm_eval/search/` — IR metrics from qrels/click logs (precision@k, recall@k, F1, MRR, MAP, NDCG).
- `src/ds_llm_eval/recommendation/` — recsys metrics (HitRate, coverage, novelty, diversity, NDCG@k).
- `src/ds_llm_eval/llm/` — LLM/agentic metrics (faithfulness, relevancy, correctness) + RAGAS adapters.
- `src/ds_llm_eval/integrations/` — Langfuse and other tracing/observability sinks (optional deps).
- Heavy/optional deps (ragas, langfuse, sklearn) live under `[project.optional-dependencies]` and are
  imported lazily inside functions — **never** at module top level. Core import must stay light.

## Development patterns
- Every metric is a pure function `metric(predictions, ground_truth, **params) -> float | EvalResult`.
  Deterministic, side-effect free, vectorized with numpy/pandas where possible.
- One public concept per module; expose the public surface only via package `__init__.py`.
- Type-hint everything; data carriers are pydantic v2 models or dataclasses. `mypy --strict` must pass.
- Validate inputs at the boundary (shapes, empty sets, k bounds) and raise `ValueError` with a clear msg.
- Follow existing naming/idioms; prefer adding to an existing module over creating a near-duplicate.
- New metric ⇒ register in the registry, export it, add a docstring with the formula + a citation.

## Testing patterns
- `pytest` + `pytest-cov`; tests mirror `src/` layout under `tests/`. Target ≥90% line coverage on `core`.
- For each metric: (1) a worked numeric example checked against a known reference value, (2) edge cases
  (empty input, k>len, all-relevant, none-relevant, ties), (3) a `hypothesis` property test for
  invariants (bounded in [0,1], monotonicity, permutation behavior).
- Mock all network/LLM/Langfuse calls — unit tests must run fully offline and be deterministic.
- `pytest` must be green and coverage must not drop before any commit.

## Guardrails (hard constraints)
- **No secrets in code or git.** API keys via env vars only; never commit `.env` (only `.env.example`).
- **No silent failures.** Surface errors; never return a fake/placeholder metric value on bad input.
- **No network in core or in unit tests.** Live integrations are opt-in, mocked in tests, marked `@pytest.mark.integration`.
- **Citations required** for every metric formula and design choice (see `docs/research.md`).
- **Backward compatibility:** public API in `__init__.py` is stable; deprecate before removing.
- **Scope discipline:** stick to the requested change; do not refactor unrelated code or bump deps uninvited.
- **Commit/push only when asked.** Branch off `master`; never commit secrets, data, or generated artifacts.
- Run `ruff check`, `ruff format`, `mypy`, and `pytest` before declaring work done; report failures honestly.

## Commands
- Setup: `pip install -e ".[dev,all]"` · Lint: `ruff check . && ruff format --check .`
- Types: `mypy src` · Test: `pytest` · Build: `python -m build`

## AI tooling
See `.claude/` — `agents.md`, `skills.md`, `tools.md`, `mcp.md`, `memory.md` describe the
subagents, skills, allowed tools, MCP servers, and durable project context used in this repo.
