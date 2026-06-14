"""Generate the metric-reference documentation from the live registry.

Keeps the docs honest: the metric reference is derived from what is actually
registered (names, signatures, first docstring line) rather than hand-maintained.
Used by the mkdocs build (``scripts/gen_reference.py``) and unit-tested.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from pathlib import Path

from .core import get_metric, list_metrics


class MetricReferenceGenerator:
    """Render a Markdown reference of every registered metric, grouped by family."""

    def __init__(self, *, title: str = "Metric reference") -> None:
        self.title = title

    def _first_doc_line(self, fn: object) -> str:
        doc = inspect.getdoc(fn) or ""
        return doc.split("\n", 1)[0].strip() if doc else "_(no description)_"

    def _signature(self, fn: object) -> str:
        try:
            return str(inspect.signature(fn))  # type: ignore[arg-type]
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return "(...)"

    def render(self) -> str:
        """Return the full Markdown document as a string."""
        groups: dict[str, list[str]] = defaultdict(list)
        for name in list_metrics():
            groups[name.split(".", 1)[0]].append(name)

        lines = [f"# {self.title}", ""]
        lines.append(
            f"Auto-generated from the registry — {len(list_metrics())} metrics "
            f"across {len(groups)} families. Do not edit by hand."
        )
        lines.append("")
        for family in sorted(groups):
            lines.append(f"## `{family}`")
            lines.append("")
            for name in groups[family]:
                fn = get_metric(name)
                lines.append(f"### `{name}`")
                lines.append("")
                lines.append(f"`{name}{self._signature(fn)}`")
                lines.append("")
                lines.append(self._first_doc_line(fn))
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def write(self, path: str | Path) -> Path:
        """Render and write the reference to ``path``; returns the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.render(), encoding="utf-8")
        return out
