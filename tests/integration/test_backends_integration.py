"""Live integration tests for IR backends (parity vs. our pure-Python ranking).

Marked ``integration`` and skipped by default in CI (``pytest -m "not integration"``).
They auto-skip when the optional engine is not installed, so the default suite stays
green; install ``ds-llm-eval[backends]`` to exercise them.
"""

import pytest

from ds_llm_eval import ranking

pytestmark = pytest.mark.integration

_RANKED = [["a", "b", "c", "d"], ["x", "y", "z"]]
_RELEVANT = [{"a", "c"}, {"z"}]


def test_ranx_ndcg_parity():
    pytest.importorskip("ranx")
    from ds_llm_eval.backends import RanxBackend

    ours = ranking.ndcg_at_k(_RANKED, _RELEVANT, k=4).value
    theirs = RanxBackend().evaluate(_RANKED, _RELEVANT, ["ndcg@4"])["ndcg@4"]
    assert ours == pytest.approx(theirs, abs=1e-6)


def test_pytrec_eval_map_parity():
    pytest.importorskip("pytrec_eval")
    from ds_llm_eval.backends import PyTrecEvalBackend

    ours = ranking.average_precision(_RANKED, _RELEVANT).value
    theirs = PyTrecEvalBackend().evaluate(_RANKED, _RELEVANT, ["map"])["map"]
    assert ours == pytest.approx(theirs, abs=1e-6)
