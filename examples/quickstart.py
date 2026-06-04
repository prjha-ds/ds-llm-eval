"""End-to-end quickstart across all three evaluation domains.

Run: ``python examples/quickstart.py`` (needs only the core install).
"""

from __future__ import annotations

import ds_llm_eval
from ds_llm_eval import EvalReport, llm, search
from ds_llm_eval import recommendation as rec


def main() -> None:
    # --- 1. Search / IR --------------------------------------------------
    # ranked[i] = documents returned for query i; relevant[i] = qrels (e.g. clicked docs).
    ranked = [["d1", "d2", "d3", "d4"], ["d9", "d4", "d2"]]
    relevant = [{"d1", "d3"}, {"d4"}]
    search_results = [
        search.precision_at_k(ranked, relevant, k=3),
        search.recall_at_k(ranked, relevant, k=3),
        search.mrr(ranked, relevant),
        search.ndcg_at_k(ranked, relevant, k=3),
    ]

    # --- 2. Recommendation ----------------------------------------------
    recs = [["i1", "i2", "i3"], ["i7", "i8", "i9"]]
    holdout = [{"i2"}, {"i9"}]
    catalog = {f"i{n}" for n in range(1, 11)}
    rec_results = [
        rec.hit_rate_at_k(recs, holdout, k=2),
        rec.ndcg_at_k(recs, holdout, k=3),
        rec.catalog_coverage_at_k(recs, catalog, k=3),
    ]

    # --- 3. LLM / agentic (deterministic, offline) ----------------------
    predictions = ["The capital of France is Paris.", "It is roughly 384,400 km."]
    references = ["Paris", "384,400 kilometers"]
    llm_results = [
        llm.exact_match(predictions, references),
        llm.token_f1(predictions, references),
    ]

    report = EvalReport(results=[*search_results, *rec_results, *llm_results])
    for name, value in report.as_dict().items():
        print(f"{name:>22}: {value:.4f}")

    print("\nRegistered metrics:")
    print("  " + ", ".join(ds_llm_eval.list_metrics()))


if __name__ == "__main__":
    main()
