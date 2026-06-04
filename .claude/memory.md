# Project memory

Durable context for `ds-llm-eval` that is not obvious from the code or git history. Keep it short;
update it when a decision changes. (This is project-level memory; it is distinct from the
assistant's personal cross-session memory.)

## Identity
- Goal: one pip-installable library to evaluate **search**, **recommendation**, and **LLM/agentic**
  pipelines behind a unified API, with optional Langfuse/observability integration.
- Repo: `https://github.com/prjha-ds/ds-llm-eval`, primary branch `master`. Import name `ds_llm_eval`.
- License: Apache-2.0. Python ≥ 3.10. Status: `0.0.1` alpha.

## Key decisions (and why)
- **Unified `EvalResult`/registry across domains** — so a search ranker, a recommender, and a RAG
  agent share one result shape and one logging path (e.g. into Langfuse). Most peers cover one domain.
- **Pure-function metrics** — deterministic, testable, side-effect free; easy to compose into harnesses.
- **Light core, lazy optional deps** — ragas/langfuse/sklearn live behind extras and import inside
  functions, so `import ds_llm_eval` stays fast and offline-safe.
- **Citations required** for every formula — rationale and sources live in `docs/research.md`.
- **`src/` layout + strict gates** — `ruff`, `mypy --strict`, `pytest` (≥90% core coverage) before commits.

## Current state
- Implemented: search (precision/recall/F1@k, MRR, MAP, NDCG), recommendation (hit rate, NDCG,
  catalog coverage, novelty), llm (exact match, token-F1, RAGAS adapter), Langfuse sink.
- See `docs/plan.md` for the roadmap and `docs/research.md` for the cited landscape survey.

## Conventions pointer
- Dev/test/guardrail rules: `CLAUDE.md`. AI workflow: `.claude/{agents,skills,tools,mcp}.md`.
