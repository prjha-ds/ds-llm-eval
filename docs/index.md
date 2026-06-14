# ds-llm-eval

Unified, pip-installable evaluation for **search**, **recommendation**, and **LLM / agentic**
pipelines — one `EvalResult` shape and one metric registry across all three, with optional
integrations into observability suites like [Langfuse](https://langfuse.com).

## Why

Most teams stitch a different evaluation library per surface. `ds-llm-eval` groups metrics by
**family** (ranking, beyond-accuracy, text, agentic, LLM-judge) and lets you use the same metric
irrespective of task: the `ranking` metrics score a search ranker, a recommender, or RAG retrieval.

```python
from ds_llm_eval import ranking, text, agentic

ranking.ndcg_at_k(ranked, relevant, k=10)
text.rouge_l(predictions, references)
agentic.tool_call_f1(predicted_calls, reference_calls)
```

## Where to go next

- **[Metric reference](reference.md)** — every registered metric, generated from the registry.
- **[Architecture](architecture.md)** — how the package is organized and why.
- **[Roadmap](plan.md)** — milestones and status.
- **[Research survey](research.md)** — the cited landscape study behind the design.

## Install

```bash
pip install ds-llm-eval                 # core (search + recommendation, light deps)
pip install "ds-llm-eval[llm]"          # + RAGAS & Langfuse for LLM/agentic eval
pip install "ds-llm-eval[all]"          # + scikit-learn, IR backends (ranx/pytrec_eval)
```

See the project [README](https://github.com/prjha-ds/ds-llm-eval#readme) for full quickstarts.
