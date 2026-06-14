"""Tests for the registry-driven metric reference generator."""

from ds_llm_eval import list_metrics
from ds_llm_eval.reference import MetricReferenceGenerator


def test_render_covers_all_registered_metrics():
    md = MetricReferenceGenerator().render()
    assert md.startswith("# Metric reference")
    for family in ["ranking", "beyond_accuracy", "text", "agentic"]:
        assert f"## `{family}`" in md
    # every registered metric has its own section
    for name in list_metrics():
        assert f"### `{name}`" in md


def test_render_includes_signature_and_description():
    md = MetricReferenceGenerator().render()
    assert "ranking.ndcg_at_k(ranked" in md  # signature rendered
    assert "NDCG" in md  # docstring first line pulled in


def test_write_creates_file(tmp_path):
    out = MetricReferenceGenerator().write(tmp_path / "sub" / "reference.md")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("# Metric reference")
