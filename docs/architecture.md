# Architecture

How `ds-llm-eval` is organized and why. For the development/testing/guardrail rules see
[`../CLAUDE.md`](../CLAUDE.md); for the rationale behind these choices see [`research.md`](research.md).

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
                              │            │             │
            ┌─────────────────┘     ┌──────┘      └──────────────┐
            ▼                       ▼                            ▼
   ┌──────────────┐        ┌──────────────────┐         ┌───────────────┐
   │   search/    │        │ recommendation/  │         │     llm/      │
   │  IR metrics  │        │ recsys metrics   │         │ ref + RAGAS   │
   └──────┬───────┘        └────────┬─────────┘         └───────┬───────┘
          │                         │                           │
          └──────────────┬──────────┴───────────────┬───────────┘
                         ▼                           ▼
                  ┌──────────────┐          ┌──────────────────┐
                  │    core/     │          │  integrations/   │
                  │ EvalResult,  │          │ Langfuse sink    │
                  │ registry,    │          │ (lazy, optional) │
                  │ validators   │          └──────────────────┘
                  └──────────────┘
```

- **`core/`** — `EvalResult`/`EvalReport` (frozen pydantic models), the `Metric` protocol, the
  `register`/`get_metric`/`list_metrics` registry, and boundary validators (`check_k`,
  `check_non_empty`). Depends on nothing but pydantic. ~100% test coverage target.
- **`search/` · `recommendation/` · `llm/`** — domain metrics. Each is a pure function
  `(*data, **params) -> EvalResult`, decorated `@register("<domain>.<name>")`, exported via the
  package `__init__`. Metric `name`s are domain-namespaced (`search.ndcg@10`, `rec.ndcg@10`) so
  results never collide in an `EvalReport`.
- **`integrations/`** — sinks to external systems (Langfuse today). Lazy imports; never required to
  compute a metric.

## Key conventions
- **Batched inputs.** Metrics take per-query/per-user lists and aggregate (mean), returning the
  sample count in `EvalResult.n`. Queries with no relevant items are skipped, not counted as 0.
- **Relevance representation.** Binary (a set/collection) or graded (`{item: gain}` mapping); NDCG
  uses graded gains when given a mapping, else binary.
- **Optional dependencies are lazy.** ragas/langfuse/sklearn import *inside* the function that needs
  them and raise a clear `ImportError` naming the right extra (e.g. `ds-llm-eval[llm]`).
- **No silent failures.** Bad input raises `ValueError` at the boundary; metrics never fabricate a value.

## Extension points
- **Add a metric:** implement → `@register` → export → test triad. The `.claude/skills/add-metric`
  skill and `metric-implementer` agent encode this flow.
- **Add a sink:** new module under `integrations/` with a lazy import and an `EvalResult`-consuming
  function, mirroring `langfuse.log_results_to_langfuse`.
- **Delegate to a backend (future):** wrap a validated engine (pytrec_eval/ranx) behind the same
  function signature, defaulting to the pure-Python path — the ir_measures pattern
  ([research §5](research.md)).

## Data flow (typical run)
```
raw data (qrels / click logs / interactions / LLM I/O)
      │  (user-prepared into batched lists)
      ▼
metric fn  ──▶  EvalResult(name, value, n, params)
      │
      ├─▶  EvalReport.as_dict()        # {metric: value} for logging/printing
      └─▶  integrations.log_results_to_langfuse(results, trace_id=...)   # optional
```
