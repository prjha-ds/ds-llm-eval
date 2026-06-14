"""Reference-based text metrics: compare generated text to gold references.

Deterministic and offline — no LLM, no network — so they are safe defaults for
CI. They evaluate any text-generation task with a gold answer (QA, LLM/agent
final answers, summarization references), independent of pipeline type.

References
----------
.. [1] P. Rajpurkar et al., "SQuAD: 100,000+ Questions...", EMNLP 2016
   (normalization + token-level F1). https://aclanthology.org/D16-1264/
"""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from collections.abc import Sequence

from ..core import EvalResult, check_non_empty

_PUNCT = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _normalize(text: str) -> list[str]:
    """SQuAD-style normalization: lowercase, strip punctuation & articles, split."""
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return text.split()


def _tokens(text: str) -> list[str]:
    """Lowercase whitespace tokenization (for BLEU/ROUGE — keeps all words)."""
    return text.lower().split()


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence of ``a`` and ``b``."""
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _validate(predictions: Sequence[str], references: Sequence[str]) -> None:
    check_non_empty(predictions, name="predictions")
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) length mismatch."
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class TextMetrics:
    """Reference-based text-comparison metrics; stateless. Singleton :data:`text`."""

    def exact_match(self, predictions: Sequence[str], references: Sequence[str]) -> EvalResult:
        """Mean exact match after SQuAD-style normalization.

        Parameters
        ----------
        predictions : sequence of str
            Model outputs.
        references : sequence of str
            Gold answers, aligned with ``predictions``.

        Returns
        -------
        EvalResult
            ``text.exact_match`` = fraction whose normalized strings are equal.
        """
        _validate(predictions, references)
        scores = [
            1.0 if _normalize(p) == _normalize(r) else 0.0
            for p, r in zip(predictions, references, strict=True)
        ]
        return EvalResult(name="text.exact_match", value=_mean(scores), n=len(predictions))

    def token_f1(self, predictions: Sequence[str], references: Sequence[str]) -> EvalResult:
        """Mean SQuAD-style token-overlap F1 between prediction and reference.

        Parameters
        ----------
        predictions, references
            See :meth:`exact_match`.

        Returns
        -------
        EvalResult
            ``text.token_f1``. Both empty after normalization → 1.0; exactly one
            empty → 0.0.
        """
        _validate(predictions, references)
        scores = []
        for p, r in zip(predictions, references, strict=True):
            pred_toks, ref_toks = _normalize(p), _normalize(r)
            if not pred_toks and not ref_toks:
                scores.append(1.0)
                continue
            if not pred_toks or not ref_toks:
                scores.append(0.0)
                continue
            common = sum((Counter(pred_toks) & Counter(ref_toks)).values())
            if common == 0:
                scores.append(0.0)
                continue
            precision = common / len(pred_toks)
            recall = common / len(ref_toks)
            scores.append(2 * precision * recall / (precision + recall))
        return EvalResult(name="text.token_f1", value=_mean(scores), n=len(predictions))

    def bleu(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        *,
        max_n: int = 4,
        smoothing: bool = False,
    ) -> EvalResult:
        """Mean sentence-level BLEU (modified n-gram precision × brevity penalty).

        Parameters
        ----------
        predictions, references
            Candidate and reference strings (whitespace-tokenized, lowercased).
        max_n : int
            Maximum n-gram order (default 4); effective order is capped at the
            hypothesis length.
        smoothing : bool
            Add-one smoothing for orders ≥ 2 (Lin & Och / Chen & Cherry method 2);
            without it a missing n-gram order yields 0 for that sentence.

        Returns
        -------
        EvalResult
            ``text.bleu`` in [0, 1].

        References
        ----------
        Papineni et al., "BLEU", ACL 2002. https://aclanthology.org/P02-1040/
        Chen & Cherry, smoothing, WMT 2014. https://aclanthology.org/W14-3346/
        """
        _validate(predictions, references)
        scores = []
        for pred, ref in zip(predictions, references, strict=True):
            hyp, refs = _tokens(pred), _tokens(ref)
            c, r = len(hyp), len(refs)
            if c == 0:
                scores.append(1.0 if r == 0 else 0.0)
                continue
            order = min(max_n, c)
            log_p = 0.0
            zero = False
            for n in range(1, order + 1):
                hyp_ng = _ngram_counts(hyp, n)
                ref_ng = _ngram_counts(refs, n)
                clipped = sum(min(cnt, ref_ng[g]) for g, cnt in hyp_ng.items())
                total = sum(hyp_ng.values())
                if smoothing and n >= 2:
                    clipped, total = clipped + 1, total + 1
                if clipped == 0:
                    zero = True
                    break
                log_p += math.log(clipped / total)
            if zero:
                scores.append(0.0)
                continue
            geo_mean = math.exp(log_p / order)
            bp = 1.0 if c > r else math.exp(1.0 - r / c)
            scores.append(bp * geo_mean)
        return EvalResult(
            name="text.bleu", value=_mean(scores), n=len(predictions), params={"max_n": max_n}
        )

    def rouge_l(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        *,
        beta: float = 1.2,
    ) -> EvalResult:
        """Mean ROUGE-L: LCS-based F-measure between prediction and reference.

        ``R = LCS/len(ref)``, ``P = LCS/len(hyp)``,
        ``F = (1+β²)·R·P / (R + β²·P)``.

        Parameters
        ----------
        predictions, references
            Candidate and reference strings (whitespace-tokenized, lowercased).
        beta : float
            Recall/precision weighting (default 1.2, the DUC setting).

        Returns
        -------
        EvalResult
            ``text.rouge_l`` in [0, 1]. Both empty → 1.0; exactly one empty → 0.0.

        References
        ----------
        Lin, "ROUGE", ACL workshop 2004. https://aclanthology.org/W04-1013/
        """
        _validate(predictions, references)
        b2 = beta * beta
        scores = []
        for pred, ref in zip(predictions, references, strict=True):
            hyp, refs = _tokens(pred), _tokens(ref)
            if not hyp and not refs:
                scores.append(1.0)
                continue
            if not hyp or not refs:
                scores.append(0.0)
                continue
            lcs = _lcs_length(hyp, refs)
            if lcs == 0:
                scores.append(0.0)
                continue
            p = lcs / len(hyp)
            rec = lcs / len(refs)
            scores.append((1 + b2) * rec * p / (rec + b2 * p))
        return EvalResult(
            name="text.rouge_l", value=_mean(scores), n=len(predictions), params={"beta": beta}
        )

    def json_correctness(
        self,
        predictions: Sequence[str],
        *,
        schema: dict[str, object] | None = None,
    ) -> EvalResult:
        """Fraction of predictions that are valid JSON (and schema-compliant).

        Two-level: every output must parse as JSON; if ``schema`` is given, it must
        also validate against it (requires the ``jsonschema`` package, lazily
        imported). Reference-free.

        Parameters
        ----------
        predictions : sequence of str
            Raw model outputs expected to be JSON.
        schema : dict, optional
            A JSON Schema to validate parsed outputs against.

        Returns
        -------
        EvalResult
            ``text.json_correctness`` = valid(-and-compliant) rate in [0, 1].

        References
        ----------
        JSONSchemaBench (valid-JSON rate, schema compliance).
        https://arxiv.org/abs/2501.10868
        """
        check_non_empty(predictions, name="predictions")
        validate = None
        if schema is not None:
            try:
                from jsonschema import validate as _validate_json
            except ImportError as exc:  # pragma: no cover - exercised via mocks
                raise ImportError(
                    "json_correctness(schema=...) requires the 'jsonschema' package "
                    '(pip install "ds-llm-eval[llm]").'
                ) from exc
            validate = _validate_json

        ok = 0
        for pred in predictions:
            try:
                parsed = json.loads(pred)
            except (json.JSONDecodeError, TypeError):
                continue
            if validate is not None:
                try:
                    validate(instance=parsed, schema=schema)
                except Exception:
                    continue
            ok += 1
        return EvalResult(
            name="text.json_correctness", value=ok / len(predictions), n=len(predictions)
        )


text = TextMetrics()
"""Shared :class:`TextMetrics` singleton."""
