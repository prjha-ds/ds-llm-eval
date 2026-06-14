# ds-llm-eval

Unified, pip-installable evaluation toolkit for **search**, **recommendation**, and
**LLM / agentic** pipelines — one consistent API and result type across all three, with
optional integrations into observability suites like [Langfuse](https://langfuse.com).

> Status: `0.1.0` — beta. Search, recommendation, LLM/agentic metric families; ingestion,
> comparison, backends, benchmarks, logging, and a declarative config runner are all in place.

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

Metrics are grouped by **family** (`ranking`, `beyond_accuracy`, `text`, `llm_judge`) and used
irrespective of task — the same `ranking` metrics score a search ranker, a recommender, or RAG
retrieval.

```python
from ds_llm_eval import ranking, beyond_accuracy, text

# --- Search / IR: ranking metrics on ranked predictions vs. qrels (e.g. click logs) ---
ranked   = [["d1", "d2", "d3"], ["d9", "d4"]]
relevant = [{"d1", "d3"},        {"d4"}]
print(ranking.ndcg_at_k(ranked, relevant, k=3).value)
print(ranking.mrr(ranked, relevant).value)

# --- Recommendation: the SAME ranking metrics + beyond-accuracy ---
recs    = [["i1", "i2", "i3"], ["i7", "i8"]]
holdout = [{"i2"},             {"i9"}]
print(ranking.hit_rate_at_k(recs, holdout, k=2).value)
print(beyond_accuracy.catalog_coverage_at_k(recs, {f"i{n}" for n in range(9)}, k=2).value)

# --- LLM: deterministic reference metrics (offline, no API key) ---
print(text.token_f1(["the quick brown fox"], ["the brown fox"]).value)
```

Discover everything registered:

```python
import ds_llm_eval
ds_llm_eval.list_metrics()   # ['ranking.ndcg_at_k', 'text.token_f1', 'beyond_accuracy.novelty_at_k', ...]
```

### Logging eval runs

Every metric returns an `EvalResult`; loggers turn those into timestamped, run-scoped records and
fan them out to one or more sinks (console, JSONL file, Langfuse):

```python
from ds_llm_eval.logging import ConsoleLogger, JSONLLogger, MultiLogger

with MultiLogger([ConsoleLogger(), JSONLLogger("runs/eval.jsonl")],
                 run_id="exp-1", metadata={"model": "ranker-v2"}) as log:
    log.log_result(ranking.ndcg_at_k(ranked, relevant, k=3))
    log.log_report(report)   # logs every EvalResult in an EvalReport
```

### Model-graded LLM eval (RAGAS) + Langfuse

```python
from ds_llm_eval import llm_judge
from ds_llm_eval.integrations import log_results_to_langfuse

scores = llm_judge.ragas_evaluate(questions, answers, contexts, ground_truths)  # needs [llm] extra
log_results_to_langfuse(results, trace_id="...")                                # needs [llm] extra
```

### Click logs → qrels, run a metric set, compare with significance

```python
from ds_llm_eval import click_log, Evaluator, RunComparison

# 1. Reshape a flat click log into ranked lists + relevance judgments
ds = click_log.from_dataframe(df, query_col="q", doc_col="doc", clicked_col="clicked", rank_col="rank")

# 2. Run several metrics at once -> EvalReport
report = Evaluator([("ranking.ndcg_at_k", {"k": 10}), "ranking.mrr"]).run(ds.ranked, ds.relevant)

# 3. Is run B significantly better than A? (paired t-test / Wilcoxon, pure-Python)
result = RunComparison("ranking.ndcg_at_k", params={"k": 10}).compare(
    {"A": run_a, "B": run_b}, ds.relevant, baseline="A")
print(result.means, result.significant_runs())
```

Agentic & benchmark evaluation:

```python
from ds_llm_eval import agentic
from ds_llm_eval.benchmarks import LocalRetrievalBenchmark

agentic.tool_call_f1(predicted_calls, reference_calls)          # deterministic, offline
LocalRetrievalBenchmark(k=10).run(ds.ranked, ds.relevant)      # BEIR/MTEB-style nDCG via ranking
```

### Reproducible runs from a YAML config

```yaml
# experiment.yaml
run_id: exp-1
dataset: {path: data/click_log.csv, query_col: q, doc_col: doc, clicked_col: clicked, rank_col: rank}
metrics:
  - {name: ranking.ndcg_at_k, params: {k: 10}}
  - ranking.mrr
logging:
  - {type: console}
  - {type: jsonl, path: runs/exp-1.jsonl}
```

```python
from ds_llm_eval import ExperimentRunner
report = ExperimentRunner.from_yaml("experiment.yaml").run()   # ingests CSV, runs metrics, logs
```

## Docs site

```bash
pip install -e ".[docs]"
python scripts/gen_reference.py    # regenerate the registry-driven metric reference
mkdocs serve                       # browse locally (mkdocs build --strict in CI)
```

## Layout

| Path | What lives there |
|------|------------------|
| `src/ds_llm_eval/core/` | Shared `EvalResult`/`EvalReport`, metric registry, input validation |
| `src/ds_llm_eval/metrics/ranking.py` | `RankingMetrics`: P/R/F1@k, MRR, MAP, NDCG, hit_rate, R-precision, bpref, RBP |
| `src/ds_llm_eval/metrics/beyond_accuracy.py` | catalog coverage, novelty, intra-list diversity, personalization, Gini, Shannon |
| `src/ds_llm_eval/metrics/text.py` | exact match, token-F1, BLEU, ROUGE-L, JSON correctness |
| `src/ds_llm_eval/metrics/agentic.py` | tool-call accuracy/F1, trajectory match, goal accuracy |
| `src/ds_llm_eval/metrics/llm.py` | `LLMJudgeMetrics`: lazy RAGAS adapter (opt-in) |
| `src/ds_llm_eval/ingestion/` | `ClickLogIngestor` → `RankingDataset` (qrels/runs from logs) |
| `src/ds_llm_eval/evaluation.py` · `comparison.py` | `Evaluator`; `RunComparison` + significance tests |
| `src/ds_llm_eval/config.py` | `ExperimentRunner` — declarative YAML/dict reproducible runs |
| `src/ds_llm_eval/reference.py` · `mkdocs.yml` | Registry-driven metric reference + docs site |
| `src/ds_llm_eval/backends/` | Optional IR delegation to `ranx` / `pytrec_eval` (lazy) |
| `src/ds_llm_eval/benchmarks/` | Benchmark catalog + runner adapters (BEIR, lm-eval, SWE-bench) |
| `src/ds_llm_eval/integrations/` · `logging/` | Langfuse sink + experiment runner; eval loggers |
| `docs/` | `plan.md` (roadmap), `research.md` (cited survey), `architecture.md` |

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
mypy src
pytest
```

See [`CLAUDE.md`](CLAUDE.md) for development, testing, and contribution conventions, and the
[`docs/`](docs/) folder for the roadmap, design, and the cited research survey.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
