"""Compare ranking runs with per-query significance testing.

Aggregate metrics hide whether a difference between two runs is real.
:class:`RunComparison` computes a metric *per query* for each run, then runs a
paired significance test (two-sided paired Student's t-test, or Wilcoxon
signed-rank via a normal approximation) of every run against a baseline. All runs
must share the same ``relevant`` judgments (and therefore the same query set), so
per-query scores are paired.

The p-value math is implemented in pure Python (no SciPy): the t-distribution via
the regularized incomplete beta function, the normal tail via ``math.erfc``.

References
----------
.. [1] Student, "The probable error of a mean", Biometrika 1908.
.. [2] W. Press et al., *Numerical Recipes*, §6.4 (incomplete beta) & §14.2 (t-test).
.. [3] F. Wilcoxon, "Individual comparisons by ranking methods", Biometrics 1945.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .core import get_metric

Test = Literal["ttest", "wilcoxon"]


# --- pure-Python statistics -------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    maxit, eps, fpmin = 200, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p-value of a Student's t statistic with ``df`` degrees of freedom."""
    if df <= 0:
        return 1.0
    return _betai(0.5 * df, 0.5, df / (df + t * t))


def _paired_ttest_p(diffs: Sequence[float]) -> float:
    """Two-sided p-value of a one-sample t-test of ``diffs`` against 0."""
    n = len(diffs)
    if n < 2:
        return 1.0
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0.0:
        return 1.0 if mean == 0.0 else 0.0
    t = mean / math.sqrt(var / n)
    return _t_two_sided_p(t, n - 1)


def _ranks(values: Sequence[float]) -> list[float]:
    """Average (fractional) ranks of ``values``, starting at 1."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average of ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _wilcoxon_p(diffs: Sequence[float]) -> float:
    """Two-sided p-value of the Wilcoxon signed-rank test (normal approximation).

    Applies the standard tie correction to the variance so tied absolute
    differences do not inflate it (which would make the test over-conservative).
    """
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n < 1:
        return 1.0
    abs_diffs = [abs(d) for d in nonzero]
    ranks = _ranks(abs_diffs)
    w_plus = sum(r for r, d in zip(ranks, nonzero, strict=True) if d > 0)
    mean = n * (n + 1) / 4.0
    tie_term = sum(t * (t * t - 1) for t in Counter(abs_diffs).values() if t > 1) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term
    if var <= 0.0:
        return 1.0
    z = (w_plus - mean) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2.0))


# --- public API -------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing runs against a baseline on one metric."""

    metric: str
    baseline: str
    n_queries: int
    test: Test
    alpha: float
    means: dict[str, float]
    comparisons: dict[str, dict[str, float | bool]]

    def significant_runs(self) -> list[str]:
        """Names of runs whose difference vs. the baseline is significant."""
        return [r for r, c in self.comparisons.items() if c["significant"]]


class RunComparison:
    """Compare ranking runs on a metric with paired significance testing.

    Parameters
    ----------
    metric : str
        Registry name of a ranking metric, e.g. ``"ranking.ndcg_at_k"``.
    params : dict, optional
        Keyword params for the metric (e.g. ``{"k": 10}``).
    test : {"ttest", "wilcoxon"}
        Paired test to apply (default ``"ttest"``).
    alpha : float
        Significance threshold (default 0.05).
    """

    def __init__(
        self,
        metric: str,
        *,
        params: Mapping[str, object] | None = None,
        test: Test = "ttest",
        alpha: float = 0.05,
    ) -> None:
        if test not in ("ttest", "wilcoxon"):
            raise ValueError(f"test must be 'ttest' or 'wilcoxon', got {test!r}.")
        self._metric = metric
        self._params = dict(params or {})
        self._test = test
        self._alpha = alpha

    def _per_query(
        self, ranked: Sequence[Sequence[Hashable]], relevant: Sequence[Collection[Hashable]]
    ) -> list[float]:
        fn = get_metric(self._metric)
        return [
            float(fn([r], [rel], **self._params).value)
            for r, rel in zip(ranked, relevant, strict=True)
        ]

    def compare(
        self,
        runs: Mapping[str, Sequence[Sequence[Hashable]]],
        relevant: Sequence[Collection[Hashable]],
        *,
        baseline: str | None = None,
    ) -> ComparisonResult:
        """Compare each run against ``baseline`` on the configured metric.

        Parameters
        ----------
        runs : mapping of str to ranked lists
            ``{run_name: ranked}``; every run is judged against the shared
            ``relevant`` and so covers the same queries.
        relevant : sequence of collection
            Relevant items per query, shared across runs.
        baseline : str, optional
            Run to compare others against; defaults to the first run.

        Returns
        -------
        ComparisonResult
        """
        names = list(runs)
        if len(names) < 2:
            raise ValueError("compare() needs at least two runs.")
        baseline = baseline if baseline is not None else names[0]
        if baseline not in runs:
            raise ValueError(f"baseline {baseline!r} not in runs {names}.")

        per = {name: self._per_query(runs[name], relevant) for name in names}
        means = {name: (sum(v) / len(v) if v else 0.0) for name, v in per.items()}
        base = per[baseline]

        comparisons: dict[str, dict[str, float | bool]] = {}
        for name in names:
            if name == baseline:
                continue
            diffs = [a - b for a, b in zip(per[name], base, strict=True)]
            p = _paired_ttest_p(diffs) if self._test == "ttest" else _wilcoxon_p(diffs)
            comparisons[name] = {
                "mean_diff": means[name] - means[baseline],
                "p_value": p,
                "significant": p < self._alpha,
            }
        return ComparisonResult(
            metric=self._metric,
            baseline=baseline,
            n_queries=len(base),
            test=self._test,
            alpha=self._alpha,
            means=means,
            comparisons=comparisons,
        )
