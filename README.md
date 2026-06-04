# ds-llm-eval

Unified, pip-installable evaluation toolkit for **search**, **recommendation**, and
**LLM / agentic** pipelines — one consistent API and result type across all three, with
optional integrations into observability suites like [Langfuse](https://langfuse.com).

> Status: `0.0.1` — early alpha. APIs may change before `0.1.0`.

## Why

Most teams stitch together a different evaluation library per surface: `pytrec_eval`/`ranx`
for search, RecBole/recmetrics for recommenders, RAGAS/DeepEval for LLMs. `ds-llm-eval`
gives you **one import, one `EvalResult` shape, one registry** so the same harness can score
a retrieval ranker, a recommender, and a RAG agent — and ship those scores to the same
dashboard. See [`docs/research.md`](docs/research.md) for the landscape survey and citations,
and [`docs/plan.md`](docs/plan.md) for the roadmap.

## Install

```bash
pip install ds-llm-eval                 # core (search + recommendation, light deps)
pip install "ds-llm-eval[llm]"          # + RAGAS & Langfuse for LLM/agentic eval
pip install "ds-llm-eval[all,dev]"      # everything + test/lint tooling
```

## Quickstart

```python
from ds_llm_eval import search, recommendation as rec, llm

# --- Search / IR: ranked predictions vs. relevance judgments (qrels/click logs) ---
ranked   = [["d1", "d2", "d3"], ["d9", "d4"]]
relevant = [{"d1", "d3"},        {"d4"}]
print(search.ndcg_at_k(ranked, relevant, k=3).value)
print(search.mrr(ranked, relevant).value)

# --- Recommendation: ranked items per user vs. held-out interactions ---
recs    = [["i1", "i2", "i3"], ["i7", "i8"]]
holdout = [{"i2"},             {"i9"}]
print(rec.hit_rate_at_k(recs, holdout, k=2).value)

# --- LLM: deterministic reference metrics (offline, no API key) ---
print(llm.token_f1(["the quick brown fox"], ["the brown fox"]).value)
```

Discover everything registered:

```python
import ds_llm_eval
ds_llm_eval.list_metrics()   # ['llm.exact_match', 'rec.hit_rate_at_k', 'search.ndcg_at_k', ...]
```

### Model-graded LLM eval (RAGAS) + Langfuse

```python
from ds_llm_eval.llm import ragas_evaluate
from ds_llm_eval.integrations import log_results_to_langfuse

scores = ragas_evaluate(questions, answers, contexts, ground_truths)   # needs [llm] extra
log_results_to_langfuse(results, trace_id="...")                       # needs [llm] extra
```

## Layout

| Path | What lives there |
|------|------------------|
| `src/ds_llm_eval/core/` | Shared `EvalResult`/`EvalReport`, metric registry, input validation |
| `src/ds_llm_eval/search/` | IR metrics: precision/recall/F1@k, MRR, MAP, NDCG |
| `src/ds_llm_eval/recommendation/` | Hit rate, NDCG, catalog coverage, novelty |
| `src/ds_llm_eval/llm/` | Exact match, token-F1, RAGAS adapter |
| `src/ds_llm_eval/integrations/` | Langfuse score sink (lazy, optional) |
| `docs/` | `plan.md` (roadmap), `research.md` (cited landscape survey) |

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy src
pytest
```

See [`CLAUDE.md`](CLAUDE.md) for development, testing, and guardrail conventions, and `.claude/`
for the AI-assisted workflow (agents, skills, tools, MCP, memory).

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
