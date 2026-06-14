"""Catalog of standard benchmark-based evaluations.

A `BenchmarkSpec` is *metadata* about a public benchmark (what it measures, its
headline metric, the official harness package, license, and which `ds-llm-eval`
domain it maps to). Concrete *adapters* that actually run these — delegating to
the official harnesses (`swebench`, `lm_eval`, `human-eval`, `mteb`, ...) and
normalizing their output into :class:`EvalResult` — are on the roadmap
(see ``docs/plan.md``); several require Docker/datasets/API keys and live behind
optional extras. This registry lets callers discover what is supported and map a
benchmark's headline number onto the library's result shape today.

See ``docs/research.md`` for the cited survey behind these entries.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..core import EvalResult

Domain = Literal["search", "recommendation", "llm"]
Execution = Literal["pure", "execution", "llm_judge", "gated"]


class BenchmarkSpec(BaseModel):
    """Metadata describing one public benchmark."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Stable benchmark id, e.g. 'swebench_verified'.")
    domain: Domain = Field(description="Which ds-llm-eval domain this maps to.")
    task: str = Field(description="One-line description of the capability evaluated.")
    metric: str = Field(description="Headline metric key, e.g. 'resolved', 'pass@1', 'ndcg@10'.")
    harness: str | None = Field(default=None, description="Official runner package, if any.")
    extra: str | None = Field(
        default=None, description="Optional-dependency extra that installs it."
    )
    execution: Execution = Field(
        description="Scoring mode: pure (offline), execution (sandbox/Docker), llm_judge, gated."
    )
    license: str = Field(description="SPDX id or short license note.")
    url: str = Field(description="Primary source URL.")

    def to_eval_result(self, value: float, *, n: int = 0, **params: Any) -> EvalResult:
        """Map a benchmark's headline score onto an :class:`EvalResult`.

        Names the result ``bench.<id>.<metric>`` so benchmark scores coexist with
        the search/rec/llm metric namespaces and log through the same sinks.
        """
        return EvalResult(
            name=f"bench.{self.id}.{self.metric}",
            value=value,
            n=n,
            params=params,
            metadata={"benchmark": self.id, "harness": self.harness, "license": self.license},
        )


def _spec(**kwargs: Any) -> BenchmarkSpec:
    return BenchmarkSpec(**kwargs)


# Catalog — entries are cited in docs/research.md ("Benchmark-based evaluation").
_BENCHMARKS: dict[str, BenchmarkSpec] = {
    spec.id: spec
    for spec in [
        # --- search / retrieval (reuse the existing search metrics) ---
        _spec(
            id="beir",
            domain="search",
            task="Zero-shot IR generalization (18 datasets)",
            metric="ndcg@10",
            harness="beir",
            extra="benchmarks-search",
            execution="pure",
            license="Apache-2.0",
            url="https://github.com/beir-cellar/beir",
        ),
        _spec(
            id="mteb_retrieval",
            domain="search",
            task="Embedding retrieval (MTEB retrieval split)",
            metric="ndcg@10",
            harness="mteb",
            extra="benchmarks-search",
            execution="pure",
            license="Apache-2.0",
            url="https://github.com/embeddings-benchmark/mteb",
        ),
        # --- LLM: coding / SWE agents (execution-based) ---
        _spec(
            id="swebench_verified",
            domain="llm",
            task="Resolve real GitHub issues (human-validated subset)",
            metric="resolved",
            harness="swebench",
            extra="benchmarks-swe",
            execution="execution",
            license="MIT",
            url="https://github.com/swe-bench/SWE-bench",
        ),
        _spec(
            id="humaneval",
            domain="llm",
            task="Function synthesis from docstring",
            metric="pass@1",
            harness="human-eval",
            extra="benchmarks-code",
            execution="execution",
            license="MIT",
            url="https://github.com/openai/human-eval",
        ),
        _spec(
            id="mbpp",
            domain="llm",
            task="Entry-level Python problems",
            metric="pass@1",
            harness="lm_eval",
            extra="benchmarks-knowledge",
            execution="execution",
            license="CC-BY-4.0",
            url="https://huggingface.co/datasets/google-research-datasets/mbpp",
        ),
        _spec(
            id="bigcodebench",
            domain="llm",
            task="Code gen with diverse library calls",
            metric="pass@1",
            harness="bigcodebench",
            extra="benchmarks-code",
            execution="execution",
            license="Apache-2.0",
            url="https://github.com/bigcode-project/bigcodebench",
        ),
        _spec(
            id="livecodebench",
            domain="llm",
            task="Contamination-free competitive coding",
            metric="pass@1",
            harness="livecodebench",
            extra="benchmarks-code",
            execution="execution",
            license="see-repo",
            url="https://github.com/LiveCodeBench/LiveCodeBench",
        ),
        # --- LLM: knowledge / reasoning (multiple-choice) ---
        _spec(
            id="mmlu",
            domain="llm",
            task="Multitask knowledge (57 subjects)",
            metric="accuracy",
            harness="lm_eval",
            extra="benchmarks-knowledge",
            execution="pure",
            license="MIT",
            url="https://github.com/hendrycks/test",
        ),
        _spec(
            id="mmlu_pro",
            domain="llm",
            task="Harder reasoning-augmented MC (10 options)",
            metric="accuracy",
            harness="lm_eval",
            extra="benchmarks-knowledge",
            execution="pure",
            license="MIT",
            url="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        ),
        _spec(
            id="gpqa_diamond",
            domain="llm",
            task="Graduate-level science QA (198 Q)",
            metric="accuracy",
            harness="lm_eval",
            extra="benchmarks-knowledge",
            execution="pure",
            license="MIT",
            url="https://github.com/idavidrein/gpqa",
        ),
        # --- LLM: agents / tool use ---
        _spec(
            id="gaia",
            domain="llm",
            task="General assistant (tools + web), validation split",
            metric="accuracy",
            harness=None,
            extra=None,
            execution="gated",
            license="gated",
            url="https://huggingface.co/datasets/gaia-benchmark/GAIA",
        ),
        _spec(
            id="tau_bench",
            domain="llm",
            task="Tool-agent + simulated user (retail/airline)",
            metric="pass^k",
            harness="tau-bench",
            extra="benchmarks-agents",
            execution="llm_judge",
            license="MIT",
            url="https://github.com/sierra-research/tau-bench",
        ),
        _spec(
            id="agentbench",
            domain="llm",
            task="Multi-environment agent eval (8 envs)",
            metric="success",
            harness="agentbench",
            extra="benchmarks-agents",
            execution="execution",
            license="Apache-2.0",
            url="https://github.com/THUDM/AgentBench",
        ),
    ]
}


def list_benchmarks(domain: Domain | None = None) -> list[str]:
    """Sorted benchmark ids, optionally filtered to one domain."""
    return sorted(b for b, s in _BENCHMARKS.items() if domain is None or s.domain == domain)


def get_benchmark(benchmark_id: str) -> BenchmarkSpec:
    """Look up a :class:`BenchmarkSpec` by id, raising ``KeyError`` if unknown."""
    if benchmark_id not in _BENCHMARKS:
        raise KeyError(f"Unknown benchmark {benchmark_id!r}. Available: {list_benchmarks()}")
    return _BENCHMARKS[benchmark_id]
