---
name: add-metric
description: Scaffold a new evaluation metric in ds-llm-eval end-to-end — implementation, registry entry, exports, tests, and quality gates. Use when the user asks to add/implement a metric for the search, recommendation, or llm domain.
---

# add-metric

Add a new metric to `ds-llm-eval` consistently and with full verification.

## Inputs to confirm first
- **domain**: `search` | `recommendation` | `llm`
- **name**: snake_case function name (e.g. `r_precision`); registry key is `<domain>.<name>`
- **formula + citation**: the exact definition and a primary source (DOI/URL)
- **signature**: which batched inputs it consumes and any params (e.g. `k`)

## Steps
1. Read `CLAUDE.md` and `src/ds_llm_eval/<domain>/metrics.py` to match existing idioms.
2. Implement the pure function returning `EvalResult`; validate inputs at the top
   (`check_k`, `check_non_empty`, module `_validate`); decorate `@register("<domain>.<name>")`.
   Lazily import any optional dependency inside the function.
3. Export it: add to the module `__all__` and `src/ds_llm_eval/<domain>/__init__.py`.
4. Write the test triad in `tests/<domain>/test_metrics.py` — worked example (hand-computed),
   edge cases, and a `hypothesis` property test. Keep everything offline/deterministic.
5. Update `README.md`/`docs/` if the public surface changed.
6. Run the gates and report results:
   ```bash
   ruff check . && ruff format --check . && mypy src && pytest
   ```

## Guardrails
- No network in core or unit tests. No silent failures — raise `ValueError` on bad input.
- Every formula needs a citation. Don't refactor unrelated code. Don't bump dependencies.
- Don't weaken a test to make it pass; fix the implementation instead.
