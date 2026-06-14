"""Regenerate ``docs/reference.md`` from the live metric registry.

Run: ``python scripts/gen_reference.py`` (or it runs in the docs build).
"""

from __future__ import annotations

from pathlib import Path

from ds_llm_eval.reference import MetricReferenceGenerator


def main() -> None:
    target = Path(__file__).resolve().parent.parent / "docs" / "reference.md"
    out = MetricReferenceGenerator().write(target)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
