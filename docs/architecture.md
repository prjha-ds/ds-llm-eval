# Architecture

How `ds-llm-eval` is organized and why. For the development/testing/guardrail rules see
[`CLAUDE.md`](https://github.com/prjha-ds/ds-llm-eval/blob/master/CLAUDE.md); for the rationale
behind these choices see [`research.md`](research.md).

## Design goals
1. **One result shape across domains.** Search, recsys, and LLM metrics all return `EvalResult`, so
   a single harness can score any pipeline and log to one place (e.g. Langfuse).
2. **Light, offline core.** `import ds_llm_eval` must be fast and network-free; heavy/optional
   backends are extras imported lazily at call time.
3. **Pure, testable metrics.** Each metric is a deterministic function — easy to verify against a
   hand-computed value and to compose.

## Layering

```
                        ┌─────────────────────────────────────────┐
   public API  ───────▶ │ ds_llm_eval  (re-exports + list_metrics) │
                        └─────────────────────────────────────────┘
                                          │
                        ┌─────────────────┴───────────────────────┐
                        ▼                                          ▼
   ┌────────────────────────────────────────────┐      ┌──────────────────┐
   │                metrics/                      │      │  integrations/   │
   │ ranking · beyond_accuracy · text · llm_judge │      │ Langfuse sink    │
   │   (group classes of pure methods)            │      │ (lazy, optional) │
   └───────────────────────┬──────────────────────┘      └──────────────────┘
                           │
                           ▼
                  ┌──────────────┐
                  │    core/     │
                  │ EvalResult,  │
                  │ registry,    │
                  │ validators   │
                  └──────────────┘
```

- **`core/`** — `EvalResult`/`EvalReport` (frozen pydantic models), the `Metric` protocol, the
  `register`/`get_metric`/`list_metrics` registry, and boundary validators (`check_k`,
  `check_non_empty`). Depends on nothing but pydantic. ~100% test coverage target.
- **`metrics/`** — metrics grouped by **family** (not task): `RankingMetrics`/`ranking`,
  `BeyondAccuracyMetrics`/`beyond_accuracy`, `TextMetrics`/`text`, `LLMJudgeMetrics`/`llm_judge`.
  Each metric is a pure **method** `(self, *data, **params) -> EvalResult`; deterministic group
  methods are auto-registered as `<group>.<method>` by `_register_group`. Result `name`s are
  group-namespaced (`ranking.ndcg@10`, `text.token_f1`) so they never collide in an `EvalReport`.
  The same `ranking` metrics serve search, recommendation, and RAG retrieval — no duplication.
- **`integrations/`** — sinks to external systems (Langfuse today). Lazy imports; never required to
  compute a metric.
- **`logging/`** — the logging mechanism: an `EvalLogger` ABC (template-method `emit`) with
  `ConsoleLogger`, `JSONLLogger`, `LangfuseLogger`, and a `MultiLogger` fan-out. Turns each
  `EvalResult` into a timestamped, run-scoped record. (Class name does not shadow stdlib `logging` —
  Python 3 absolute imports.)
- **`benchmarks/`** — `BenchmarkSpec` registry + runner adapters: offline `LocalRetrievalBenchmark`
  (reuses `ranking`), and lazy `LmEvalAdapter` / `SweBenchAdapter` that delegate to official harnesses
  with a sandbox/auth preflight.
- **`ingestion/`** — `ClickLogIngestor`: a `(query, doc, clicked)` DataFrame → `RankingDataset`.
- **`evaluation.py`** — `Evaluator`: run a set of registry metrics over a dataset → `EvalReport`.
- **`comparison.py`** — `RunComparison`: per-query paired significance testing (t-test / Wilcoxon),
  with the p-value math implemented in pure Python (no SciPy).
- **`backends/`** — optional IR delegation to `ranx` / `pytrec_eval` behind a stable façade; the
  pure-Python `ranking` group remains the default.
- **`config.py`** — `ExperimentRunner`/`ExperimentConfig`: a declarative YAML/dict run that wires a
  dataset (click-log CSV) + a metric set + loggers into one reproducible call returning an `EvalReport`.
- **`reference.py`** — `MetricReferenceGenerator`: renders the metric reference from the live
  registry (names, signatures, docstrings) for the mkdocs site (`scripts/gen_reference.py`).

## Key conventions
- **Batched inputs.** Metrics take per-query/per-user lists and aggregate (mean), returning the
  sample count in `EvalResult.n`. Queries with no relevant items are skipped, not counted as 0.
- **Relevance representation.** Binary (a set/collection) or graded (`{item: gain}` mapping); NDCG
  uses graded gains when given a mapping, else binary.
- **Optional dependencies are lazy.** ragas/langfuse/sklearn import *inside* the function that needs
  them and raise a clear `ImportError` naming the right extra (e.g. `ds-llm-eval[llm]`).
- **No silent failures.** Bad input raises `ValueError` at the boundary; metrics never fabricate a value.

## Extension points
- **Add a metric:** add a method to the right group class in `metrics/<group>.py` (auto-registered) →
  test triad. The `.claude/skills/add-metric` skill and `metric-implementer` agent encode this flow.
- **Add a group:** new `metrics/<group>.py` with a group class + singleton; register it in
  `metrics/__init__.py` via `_register_group` and export both.
- **Add a sink:** new module under `integrations/` with a lazy import and an `EvalResult`-consuming
  function, mirroring `langfuse.log_results_to_langfuse`.
- **Delegate to a backend (future):** wrap a validated engine (pytrec_eval/ranx) behind the same
  method signature, defaulting to the pure-Python path — the ir_measures pattern
  ([research §5](research.md)).

## Data flow (typical run)
```
raw data (qrels / click logs / interactions / LLM I/O)
      │  (user-prepared into batched lists)
      ▼
metric method  ──▶  EvalResult(name, value, n, params)   # e.g. ranking.ndcg_at_k(...)
      │
      ├─▶  EvalReport.as_dict()        # {metric: value} for printing
      ├─▶  logging.MultiLogger([...]).log_report(report)   # console / JSONL / Langfuse
      └─▶  integrations.log_results_to_langfuse(results, trace_id=...)   # one-shot Langfuse
```

Benchmark scores follow the same path: `BenchmarkSpec.to_eval_result(value, ...)` produces a
`bench.<id>.<metric>` `EvalResult` that logs through the same loggers.
