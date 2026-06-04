# Plan & roadmap

The roadmap for `ds-llm-eval`, grounded in the landscape survey ([`research.md`](research.md)) and
the architecture ([`architecture.md`](architecture.md)). Goal: **one pip-installable library with a
unified evaluation API across search, recommendation, and LLM/agentic pipelines**, plus an
observability sink (Langfuse).

## Guiding principles
- **Unify what is already shared.** Search and recsys share the `@k` metric vocabulary; one
  `EvalResult` and one registry serve both, and the LLM surface plugs into the same shape.
- **Light core, optional extras.** `import ds_llm_eval` pulls only numpy/pandas/pydantic. RAGAS,
  Langfuse, and sklearn live behind extras and are imported lazily.
- **Deterministic by default, model-graded by opt-in.** Reference metrics are free, offline, and
  reproducible; LLM-as-judge metrics are explicit and documented as non-deterministic.
- **Correctness is tested, formulas are cited.** Every metric has a worked-example test and a source.

## Status legend
✅ done (in this scaffold) · 🟡 in progress · ⬜ planned

## Milestone 0 — Foundation (✅ shipped in `0.0.1`)
- ✅ `src/` layout, packaging (`hatchling`), Apache-2.0, CI matrix (3.10–3.12), pre-commit.
- ✅ Core contracts: `EvalResult`/`EvalReport`, `Metric` protocol, metric registry, validators.
- ✅ Search: precision/recall/F1@k, MRR, MAP, NDCG (binary + graded).
- ✅ Recommendation: hit rate, NDCG, catalog coverage, novelty.
- ✅ LLM: exact match, token-F1; lazy RAGAS adapter; Langfuse score sink.
- ✅ Quality gates green: `ruff`, `mypy --strict`, `pytest` (41 tests, ~93% coverage).

## Milestone 1 — Ergonomics & data ingestion (🟡 next)
- ⬜ **Click-log ingestion** for search: turn raw `(query, doc, clicked)` logs into qrels/runs
  (DataFrame in, batched lists out), with position-bias caveats documented.
- ⬜ **`compare()` + significance testing** (paired t-test / Wilcoxon) across runs — pattern from
  ranx/Elliot ([research §1, §5](research.md)).
- ⬜ Polymorphic inputs (dict / DataFrame / arrays) on the public metric functions.
- ⬜ `evaluate(predictions, ground_truth, metrics=[...])` convenience that runs a metric set and
  returns an `EvalReport`.

## Milestone 2 — Breadth of metrics (⬜)
- ⬜ Recsys: precision/recall@k, MAP@k, diversity (intra-list), personalization, Gini/Shannon
  (beyond-accuracy set per RecBole/recmetrics).
- ⬜ Search: R-precision, success@k, bpref, graded MAP.
- ⬜ LLM: ROUGE/BLEU reference metrics; structured-output (JSON) correctness; agentic tool-call
  accuracy and goal accuracy (deterministic where possible) ([research §3](research.md)).

## Milestone 3 — Backends & integrations (⬜)
- ⬜ Optional **delegation to validated IR backends** (`pytrec_eval` / `ranx`) behind a stable
  façade, defaulting to our pure-Python implementations — the ir_measures lesson ([research §5](research.md)).
- ⬜ Langfuse **`dataset.run_experiment`** runner (not just the score sink); pin a Langfuse major.
- ⬜ RAGAS adapter hardening + a TruLens/Phoenix bridge investigation.

## Milestone 4 — DX & release (⬜)
- ⬜ Declarative config entry point (Elliot-style YAML) for reproducible experiment runs.
- ⬜ Docs site (mkdocs) + metric reference auto-generated from the registry.
- ⬜ Benchmarks vs. TREC Eval / ranx for correctness parity; publish `0.1.0` to PyPI.

## Explicitly out of scope (for now)
- Training/serving models (we evaluate, not train).
- Online/counterfactual estimators (IPS, position-bias models) — revisit after Milestone 2.
- A hosted UI — we integrate with Langfuse/Phoenix rather than build our own.

## Risks
- **LLM-judge non-determinism & cost** → keep them opt-in, mock in tests ([research §4](research.md)).
- **Heavy/abandoned deps** (RecBole DL stack, dormant recmetrics, Elastic-licensed Phoenix) → extras
  only; prefer permissive + maintained backends.
- **Langfuse SDK churn** (v2/v3/v4) → pin a major version; target `create_score` (v3+).
