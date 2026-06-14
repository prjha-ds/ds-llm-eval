"""End-to-end quickstart: one metric API across search, recommendation, and LLM.

Metrics are grouped by family (``ranking``, ``beyond_accuracy``, ``text``) and used
irrespective of task. Run: ``python examples/quickstart.py`` (core install only).
"""

from __future__ import annotations

import ds_llm_eval
from ds_llm_eval import EvalReport, beyond_accuracy, list_benchmarks, ranking, text
from ds_llm_eval.logging import ConsoleLogger


def main() -> None:
    # --- Search / IR: ranking metrics on qrels (e.g. from click logs) ----
    ranked = [["d1", "d2", "d3", "d4"], ["d9", "d4", "d2"]]
    relevant = [{"d1", "d3"}, {"d4"}]
    search_results = [
        ranking.precision_at_k(ranked, relevant, k=3),
        ranking.recall_at_k(ranked, relevant, k=3),
        ranking.mrr(ranked, relevant),
        ranking.ndcg_at_k(ranked, relevant, k=3),
    ]

    # --- Recommendation: the SAME ranking metrics + beyond-accuracy ------
    recs = [["i1", "i2", "i3"], ["i7", "i8", "i9"]]
    holdout = [{"i2"}, {"i9"}]
    catalog = {f"i{n}" for n in range(1, 11)}
    rec_results = [
        ranking.hit_rate_at_k(recs, holdout, k=2),
        beyond_accuracy.catalog_coverage_at_k(recs, catalog, k=3),
    ]

    # --- LLM / agentic: deterministic reference metrics (offline) --------
    predictions = ["The capital of France is Paris.", "It is roughly 384,400 km."]
    references = ["Paris", "384,400 kilometers"]
    llm_results = [
        text.exact_match(predictions, references),
        text.token_f1(predictions, references),
    ]

    report = EvalReport(results=[*search_results, *rec_results, *llm_results])
    # Log the whole run through the logging mechanism.
    with ConsoleLogger(run_id="quickstart", metadata={"demo": True}) as log:
        log.log_report(report)

    print("\nRegistered metrics:")
    print("  " + ", ".join(ds_llm_eval.list_metrics()))

    print("\nCatalogued LLM benchmarks (see ds_llm_eval.benchmarks):")
    print("  " + ", ".join(list_benchmarks(domain="llm")))


if __name__ == "__main__":
    main()
