# Research: the evaluation-tooling landscape

> Survey of existing open-source evaluation libraries across search/IR, recommendation, and
> LLM/agentic pipelines, and the design lessons we draw for `ds-llm-eval`. Every factual claim is
> cited inline. Compiled June 2026 via a multi-source, adversarially-verified deep-research pass
> (search & recsys claims carry 3-0 verification votes) plus a focused follow-up on LLM tooling.
> Library versions and star counts are time-sensitive snapshots.

## TL;DR

The ecosystem is mature but **fragmented by domain**. Search/IR has battle-tested primitives
(`pytrec_eval`, `ir_measures`, `ranx`); recommendation has full frameworks (RecBole, Elliot) and
metric libraries (recmetrics); LLM/agentic has a fast-moving set of LLM-as-judge frameworks (RAGAS,
DeepEval, TruLens) and observability platforms (Langfuse, Phoenix). **No single tool spans all
three.** `ranx` is the strongest cross-domain precedent — one library for both IR *and* recsys —
which validates `ds-llm-eval`'s unified-API premise. Our differentiator is extending that unity to
the LLM/agentic surface and to one observability sink (Langfuse).

---

## 1. Search & information retrieval

| Tool | Role | License | Key facts |
|------|------|---------|-----------|
| **pytrec_eval** | Low-level primitive | wraps trec_eval (separate license) | Native C-extension exposing TREC's reference `trec_eval` implementations (not a reimplementation), ~10× faster than subprocessing it. |
| **ir_measures** | Unified interface | Apache-2.0 | Delegates to ~11 backends rather than implementing metric math; standardizes names & `@k`. |
| **ranx** | Cross-domain library | MIT | Numba-accelerated; 12 ranking metrics; Qrels/Run objects; `evaluate`/`compare`/`fuse`. |

- **pytrec_eval** is the canonical low-level IR primitive: a Python native extension that *exposes
  the reference implementations of `trec_eval`*, invoked as
  `RelevanceEvaluator(qrel, {"map","ndcg"}).evaluate(run)`, ~an order of magnitude faster than
  subprocess-invoking the C tool. ([SIGIR 2018 paper](https://arxiv.org/pdf/1805.01597),
  [repo](https://github.com/cvangysel/pytrec_eval))
- **ir_measures** is an Apache-2.0 *common interface* that does **not** implement its own metric
  math; it wraps ~11 providers (pytrec_eval, ranx, trectools, gdeval, cwl_eval, msmarco, pyndeval,
  judged, accuracy, compat, runtime), auto-dispatching each measure to the right backend and
  re-invoking a tool multiple times when one call cannot satisfy all requested metrics. It exposes a
  `calc_aggregate([AP, nDCG@10, RR, P(rel=2)@10], qrels, run)` API with `@k` cutoffs and graded
  relevance. ([ECIR 2022 demo](https://arxiv.org/pdf/2111.13466),
  [providers doc](https://ir-measur.es/en/latest/providers.html),
  [measures doc](https://ir-measur.es/en/latest/measures.html))
- **ranx** is MIT-licensed and Numba-accelerated (vectorized, JIT-compiled, parallelized), covering
  12 metrics — Hits, Hit Rate/Success, Precision, Recall, F1, R-Precision, Bpref, RBP, MRR, MAP,
  DCG, NDCG (with `@k` and DCG variants) — *validated against TREC Eval for correctness*. Its API is
  built on `Qrels` (judgments) and `Run` (rankings) objects: `evaluate(qrels, run, "ndcg@5")`
  returns a float, `evaluate(qrels, run, ["map@5","mrr"])` returns a dict, and `compare(...)` runs a
  two-sided paired Student's t-test. Crucially, **ranx targets both IR and recsys** — the strongest
  precedent for a unified library. ([repo](https://github.com/AmenRa/ranx),
  [evaluate doc](https://amenra.github.io/ranx/evaluate/))

---

## 2. Recommendation systems

| Tool | Role | License | Key facts |
|------|------|---------|-----------|
| **RecBole** | Full library (94 algos) | MIT | Top-k accuracy + beyond-accuracy metrics; `@K` cutoffs. |
| **Elliot** | Reproducible experiments | Apache-2.0 | Single-YAML end-to-end runs; 36 metrics; significance tests. |
| **Microsoft Recommenders** | Best-practice lifecycle | MIT | Notebook examples across the full recsys lifecycle. |
| **recmetrics** | Metric library | MIT | Beyond-accuracy formulas; **dormant** since Apr 2022. |

- **RecBole** spans four task categories with 94 algorithms / 44 datasets; its evaluator reports
  top-k accuracy (Recall, MRR, NDCG, Hit/HR, Precision, MAP, GAUC) at cutoff `K` (default top-k=10),
  and ships **beyond-accuracy** metrics as first-class evaluators: ItemCoverage, AveragePopularity,
  ShannonEntropy, GiniIndex, TailPercentage. This demonstrates the shared `@K` metric vocabulary
  between recsys and IR. ([repo](https://github.com/RUCAIBox/RecBole),
  [metrics doc](https://recbole.io/docs/recbole/recbole.evaluator.metrics.html))
- **Elliot** runs a whole experiment from a single YAML config (loading → splitting → HPO → training
  → metrics), computing 36 metrics across accuracy, beyond-accuracy, bias, and fairness, with
  Wilcoxon and paired-t significance testing. ([SIGIR 2021 paper](https://arxiv.org/pdf/2103.02590),
  [repo](https://github.com/sisinflab/elliot))
- **Microsoft Recommenders** (now `recommenders-team/` under the Linux Foundation, MIT) is a
  best-practices library of utilities + Jupyter notebooks covering data prep, modeling, offline
  evaluation, tuning, and Azure operationalization. ([repo](https://github.com/recommenders-team/recommenders))
- **recmetrics** (MIT) implements ranking metrics (Map@k, its own "Mar@k") plus beyond-accuracy
  metrics — `prediction_coverage`, `catalog_coverage`, `novelty` (self-information `-log2(pop)`),
  `personalization` (cosine dissimilarity across users), `intra_list_similarity`. It is a compact
  formula reference but **dormant** (no release since v0.1.5, Apr 2022) — a maintenance risk if used
  as a backend. ([repo](https://github.com/statisticianinstilettos/recmetrics))

---

## 3. LLM & agentic pipelines

### Eval frameworks (LLM-as-judge)

| Tool | Role | License | Flagship metrics |
|------|------|---------|------------------|
| **RAGAS** | RAG/agent eval | Apache-2.0 | Faithfulness, Response Relevancy, Context Precision/Recall; Tool-Call Accuracy, Agent Goal Accuracy. |
| **DeepEval** | Pytest-style eval | Apache-2.0 | G-Eval, Hallucination, Answer Relevancy, Task Completion, Tool Correctness. |
| **TruLens** | Instrumented eval | MIT | RAG triad: Context Relevance, Groundedness, Answer Relevance. |

- **RAGAS** evaluates RAG and agentic apps with both LLM-based and traditional metrics. RAG metrics
  include Faithfulness (= claims supported by context / total claims, via LLM claim extraction),
  Response Relevancy (reference-free: generates artificial questions from the answer, scores by
  embedding cosine similarity), Context Precision/Recall; agentic metrics include Tool Call
  Accuracy and Agent Goal Accuracy. Core API: `evaluate(dataset=EvaluationDataset.from_list(...),
  metrics=[Faithfulness(), ...], llm=LangchainLLMWrapper(...))`. It also offers a non-LLM
  faithfulness path (Vectara HHEM classifier).
  ([metrics doc](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/),
  [faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/),
  [repo](https://github.com/explodinggradients/ragas))
- **DeepEval** is a pytest-style framework: define `LLMTestCase(input, actual_output,
  expected_output, retrieval_context)` and `assert_test(test_case, [metrics])`, run via
  `deepeval test run`. It has a large catalog — G-Eval (CoT LLM-judge, normalized 0–1), Hallucination
  (= contradicted contexts / total), Answer Relevancy, Contextual Precision/Recall, plus agentic
  (Task Completion, Tool Correctness) and safety (Bias, Toxicity) metrics. Docs explicitly note
  G-Eval "is **NOT** deterministic" and steer to `DAGMetric` for deterministic scoring; the
  Hallucination metric assumes trusted `context` and is "not for live RAG."
  ([getting started](https://deepeval.com/docs/getting-started),
  [LLM evals](https://deepeval.com/docs/metrics-llm-evals),
  [hallucination](https://deepeval.com/docs/metrics-hallucination),
  [repo](https://github.com/confident-ai/deepeval))
- **TruLens** centers on the **RAG triad** — Context Relevance, Groundedness, Answer Relevance —
  framed as evidence an app is "hallucination free up to the limit of its knowledge base."
  Abstraction is the *feedback function* + selectors over OpenTelemetry spans; apps are wrapped with
  TruChain/TruLlama/TruCustom recorders. MIT, smaller ecosystem than RAGAS/DeepEval.
  ([RAG triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/),
  [repo](https://github.com/truera/trulens))

### Observability / experiment platforms

- **Langfuse** (MIT; `ee/` separate) is an OSS LLM-engineering/observability platform: tracing,
  prompt management, datasets, experiments, and **scores**. Eval results are stored as scores with
  four data types — NUMERIC, CATEGORICAL, BOOLEAN, TEXT. A score can be ingested for a `trace_id`
  **before** the trace exists (Langfuse links them on arrival), making it a clean sink for offline
  eval. Python SDK v3+ uses `create_score(name, value, trace_id, observation_id, data_type,
  comment)`; benchmark runs use `dataset.run_experiment(task=..., evaluators=[...])`. Note real SDK
  churn (v2 `score()` → v3 `create_score()`/`item.run()` → v4 removes `item.run()`).
  ([scores via SDK](https://langfuse.com/docs/evaluation/evaluation-methods/scores-via-sdk),
  [experiments](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk),
  [repo](https://github.com/langfuse/langfuse))
- **Arize Phoenix** is an OpenTelemetry/OpenInference-based observability + eval platform with
  LLM-as-judge evaluators (hallucination, QA correctness, toxicity, retrieval relevance). Licensed
  **Elastic-2.0** — *source-available, not OSI open source* (managed-service restriction).
  ([repo](https://github.com/Arize-ai/phoenix), [OpenInference](https://github.com/Arize-ai/openinference))
- **OpenAI Evals** (MIT) is an eval framework + benchmark registry: YAML + JSONL evals, templated
  graders (Match/Includes/FuzzyMatch) or custom Python, model-graded evals, CLI `oaieval`. Centered
  on the OpenAI API; standalone repo momentum has shifted to the hosted Dashboard.
  ([repo](https://github.com/openai/evals/blob/main/README.md?plain=1))
- **promptfoo** (MIT) is a CLI/library for declarative YAML test cases + assertions (negatable with
  `not-`, weightable), multi-provider comparison, CI integration, and red-teaming. TypeScript-first;
  Python is secondary. ([repo](https://github.com/promptfoo/promptfoo),
  [assertions](https://www.promptfoo.dev/docs/configuration/expected-outputs/))
- **MLflow LLM Evaluate** (Apache-2.0) exposes `mlflow.evaluate(model, data, model_type=...)` with
  metric bundles per type — e.g. `question-answering` (exact-match, toxicity, readability),
  `retriever` (**precision_at_k, recall_at_k, ndcg_at_k**), plus LLM-judge metrics
  (answer_correctness, faithfulness, relevance) and custom metrics via `make_metric` /
  `make_genai_metric`. The classic `mlflow.evaluate` LLM path is now *legacy* vs. `mlflow.genai`.
  ([LLM evaluate doc](https://www.mlflow.org/docs/2.21.3/llms/llm-evaluate/))

### Evaluation paradigms — when to use what
- **Reference-based** (BLEU/ROUGE): deterministic and cheap, but "relatively low correlation with
  human judgments" on creative/open-ended tasks. Use only where a fixed gold answer exists.
  ([arXiv 2303.16634](https://arxiv.org/abs/2303.16634))
- **Reference-free / LLM-as-judge**: applies where no gold reference exists. G-Eval (CoT +
  form-filling) reaches Spearman 0.514 with humans on summarization; GPT-4-as-judge hits ~85%
  agreement with human experts on MT-Bench (above the 81% human–human agreement) — but the judge's
  own model becomes the source of truth.
  ([G-Eval 2303.16634](https://arxiv.org/abs/2303.16634), [MT-Bench 2306.05685](https://arxiv.org/html/2306.05685))

### Agentic / tool-use / trajectory evaluation
The agentic dimension scores the *path* (tools, arguments, order, goal), not just the final string.
- **Tool-call accuracy.** RAGAS `ToolCallAccuracy` (sequence + argument correctness, 0–1) and
  `ToolCallF1` (unordered precision/recall); DeepEval `ToolCorrectness` is **deterministic**
  (name/arg/output/order matching), invoking an LLM judge only when `available_tools` are supplied.
  ([RAGAS agents](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/),
  [DeepEval ToolCorrectness](https://deepeval.com/docs/metrics-tool-correctness))
- **Trajectory matching.** LangChain AgentEvals provides deterministic strict/unordered/subset/
  superset matchers plus an optional-reference LLM judge.
  ([agentevals](https://github.com/langchain-ai/agentevals))
- **Goal / task completion.** τ-bench compares the **final DB state** to an annotated goal
  (programmatic); DeepEval `TaskCompletion` is LLM-judged on the outcome.
  ([τ-bench 2406.12045](https://arxiv.org/abs/2406.12045),
  [DeepEval TaskCompletion](https://deepeval.com/docs/metrics-task-completion))
- **Multi-turn.** DeepEval `KnowledgeRetention` / `ConversationCompleteness` / `RoleAdherence`
  (LLM-judged ratios); RAGAS `TopicAdherenceScore` (P/R/F1).
  ([DeepEval multi-turn](https://deepeval.com/guides/guides-multi-turn-evaluation-metrics))

### LLM-as-judge bias (and mitigations)
- **Position bias** — judges favor a position regardless of content; GPT-4 swap-consistency ~65%
  (Claude-v1 23.8%, GPT-3.5 46.2%); a 15-judge / 150k-instance study confirms it is systematic, not
  chance. ([MT-Bench 2306.05685](https://arxiv.org/html/2306.05685),
  [2406.07791](https://arxiv.org/abs/2406.07791))
- **Verbosity bias** — favor longer answers; on a repetition attack GPT-3.5/Claude-v1 failed 91.3%
  vs GPT-4's 8.7%. **Self-preference** — GPT-4 ~+10%, Claude-v1 ~+25%.
  ([2306.05685](https://arxiv.org/html/2306.05685), [2410.21819](https://arxiv.org/abs/2410.21819))
- **Mitigations:** position swapping (count a win only if consistent both ways); reference-guided
  judging (math-grading failures 70%→15%); few-shot judging (GPT-4 consistency 65%→77.5%); CoT
  (can backfire by replicating supplied errors). ([2306.05685](https://arxiv.org/html/2306.05685))

---

## 4. Pitfalls (carry into our design & docs)

- **LLM-as-judge is non-deterministic.** DeepEval marks G-Eval "NOT deterministic"; RAGAS/TruLens
  LLM metrics inherit the same variability. → Keep deterministic reference metrics as a first-class,
  default-installable tier; mark model-graded metrics as opt-in and document non-determinism.
  ([DeepEval](https://deepeval.com/docs/metrics-llm-evals))
- **Cost/latency scale with judge calls.** RAGAS: "LLM Based metrics might use one or more LLM calls
  to arrive at the score." → Never import or invoke LLM backends in the core path.
  ([RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/))
- **Reference-free ≠ correct.** Faithfulness/relevancy measure consistency with provided context,
  not external factual accuracy. → Label metric semantics precisely in docstrings.
  ([RAGAS faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/))
- **Offline ≠ online for recommenders.** Strong offline ranking metrics need not predict online
  engagement. → Document that our recsys metrics are offline proxies.
  ([Shaped.ai](https://www.shaped.ai/blog/evaluating-recommender-models-offline-vs-online-evaluation))
- **License diligence.** Phoenix is Elastic-2.0 (source-available); recmetrics is dormant. → Prefer
  permissive, maintained backends; keep heavy/optional ones behind extras.
- **LLM judges are biased, not just noisy** — position/verbosity/self-preference biases are
  systematic. → When model-graded metrics land, default bias mitigations on (position swapping,
  reference-guided prompts). ([2306.05685](https://arxiv.org/html/2306.05685))
- **Benchmarks need sandboxes/datasets pip can't provide.** SWE-bench needs Docker (~120GB);
  HumanEval ships with execution disabled; GAIA's test set is gated. → Adapters preflight-check for
  the sandbox/auth and raise a clear error (no silent degradation).
  ([SWE-bench eval](https://www.swebench.com/SWE-bench/guides/evaluation/))

---

## 5. Design learnings for ds-llm-eval

1. **Unify the search + recsys surfaces** — they already share the `@k` vocabulary
   (Precision/Recall/F1/MRR/MAP/NDCG/HitRate); `ranx` proves one library can serve both. *(done: a
   shared `EvalResult` + cross-domain metric registry.)*
2. **Adopt the Qrels/Run + `evaluate()`/`compare()` data-object pattern** (ranx/pytrec_eval):
   polymorphic inputs, float-for-single / dict-for-many returns. *(partially done; `compare()` is on
   the roadmap.)*
3. **Prefer orchestration over reimplementation where a validated backend exists** (ir_measures):
   long-term, delegate IR math to pytrec_eval/ranx behind a stable façade rather than maintaining our
   own. *(roadmap — see `docs/plan.md`.)*
4. **Two-tier LLM eval:** deterministic reference metrics (exact match, token-F1) in light core;
   model-graded metrics (RAGAS faithfulness/relevancy) behind the `[llm]` extra, lazily imported.
   *(done.)*
5. **One observability sink, done right:** push `EvalResult`s into Langfuse as scores **by
   `trace_id`** via v3+ `create_score`, and support `dataset.run_experiment` for benchmark runs;
   pin a Langfuse major version. *(score sink done; experiment runner on roadmap.)*
6. **First-class significance testing & beyond-accuracy metrics** (ranx/Elliot/RecBole): paired-t /
   Wilcoxon and coverage/novelty/diversity. *(novelty + coverage done; significance testing on roadmap.)*
7. **Validate correctness against references** (ranx vs. TREC Eval): test each metric against a
   hand-computed or reference value. *(done via the worked-example test triad.)*
8. **Deterministic agentic metrics first.** Tool-call/trajectory matching against ground truth
   (DeepEval `ToolCorrectness`, RAGAS `ToolCallAccuracy/F1`, AgentEvals matchers) is pure comparison
   → default tier; LLM-judged agentic metrics → opt-in. *(roadmap.)*
9. **Catalog benchmarks; delegate to official harnesses.** Don't reimplement pass@k/% resolved —
   normalize harness output into `EvalResult`. *(registry done; adapters on roadmap — see §6.)*
10. **A logging mechanism is part of the package.** Every `EvalResult` flows through pluggable
    loggers (console / JSONL / Langfuse / fan-out) with run-id + metadata. *(done — `ds_llm_eval.logging`.)*

---

## 6. Benchmark-based evaluation

Standardized benchmarks complement metric computation: they answer "how does my model/agent rank on
a shared task?". `ds-llm-eval` ships a **catalog** (`ds_llm_eval.benchmarks`, the `BenchmarkSpec`
registry) today; runner **adapters** that delegate to official harnesses are on the roadmap.

| Benchmark | Domain | Metric | Harness | Exec | License |
|-----------|--------|--------|---------|------|---------|
| [SWE-bench Verified](https://github.com/swe-bench/SWE-bench) | llm (code) | % resolved | `swebench` | Docker | MIT |
| [HumanEval](https://github.com/openai/human-eval) | llm (code) | pass@k | `human-eval` | sandbox | MIT |
| [BigCodeBench](https://github.com/bigcode-project/bigcodebench) | llm (code) | pass@1 | `bigcodebench` | sandbox | Apache-2.0 |
| [LiveCodeBench](https://arxiv.org/abs/2403.07974) | llm (code) | pass@1 | repo | sandbox | see repo |
| [MMLU](https://arxiv.org/abs/2009.03300) / [MMLU-Pro](https://arxiv.org/abs/2406.01574) | llm (knowledge) | accuracy | `lm_eval` | pure | MIT |
| [GPQA Diamond](https://arxiv.org/abs/2311.12022) | llm (reasoning) | accuracy | `lm_eval` | pure | MIT |
| [GAIA](https://arxiv.org/abs/2311.12983) | llm (agent) | accuracy | HF leaderboard | gated | gated |
| [τ-bench](https://arxiv.org/abs/2406.12045) | llm (agent) | pass^k | `tau-bench` | LLM+sim | MIT |
| [AgentBench](https://arxiv.org/abs/2308.03688) | llm (agent) | success | `AgentBench` | exec | Apache-2.0 |
| [BEIR](https://arxiv.org/abs/2104.08663) | search | nDCG@10 | `beir` | pure | Apache-2.0 |
| [MTEB](https://arxiv.org/abs/2210.07316) | search | nDCG@10 | `mteb` | pure | Apache-2.0 |

Runners devs actually use: **EleutherAI lm-evaluation-harness** (`lm_eval`, MIT, 60+ tasks incl.
MMLU/GPQA/MBPP — the HF Open-LLM-Leaderboard backend, [repo](https://github.com/EleutherAI/lm-evaluation-harness))
and **Stanford HELM** (`crfm-helm`, Apache-2.0, multi-metric incl. bias/toxicity/efficiency,
[repo](https://github.com/stanford-crfm/helm)).

### Designing the `benchmarks` module
1. **Map benchmarks to the existing domains; lead with search ones that reuse our metrics.** BEIR &
   MTEB-retrieval both report nDCG@10 — feed their qrels/runs through the existing `ranking.ndcg_at_k`.
2. **Delegate, don't reimplement.** Thin adapters shell out to `swebench`/`human-eval`/`lm_eval`/
   `mteb`/`tau-bench` and parse their JSON into `EvalResult`; the harness output is the source of truth.
3. **Standardize knowledge/MC on lm-evaluation-harness.** One `lm_eval` adapter (`simple_evaluate`)
   covers MMLU, MMLU-Pro, GPQA, MBPP at once.
4. **Heavy/sandbox deps → optional extras + preflight checks.** `benchmarks-{search,knowledge,code,
   swe,agents}` extras, lazy imports, and explicit "Docker/dataset/API key missing" errors.
5. **Map onto `EvalResult` with name fidelity.** `bench.<id>.<metric>` (e.g. `bench.humaneval.pass@1`),
   `n` = instances, `params` = harness knobs, `metadata` = harness version / model id / license /
   contamination cutoff. *(implemented in `BenchmarkSpec.to_eval_result`.)*
6. **Encode license & gating in metadata.** GAIA is gated (HF terms, private test set) — support only
   the public validation split locally and surface terms in `metadata`.

> Unverified: LiveCodeBench and GAIA license terms could not be confirmed from primary sources —
> treat as "see repo" / "gated" until checked directly.

## Open questions
- Unifying data model bridging labeled-relevance IR/recsys (Qrels/Run) with reference-free LLM eval.
- Dependency-footprint trade-offs of wrapping heavyweight backends (RecBole DL stack, Numba ranx,
  C-extension pytrec_eval) — almost certainly all optional extras.
- Online/counterfactual evaluation from click logs (IPS/position-bias correction) — future scope.
