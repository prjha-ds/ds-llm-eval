# Tools

How the standard development tools are used in `ds-llm-eval`, and the permission posture set in
[`settings.json`](settings.json).

## Toolchain
| Tool | Role | Command |
|------|------|---------|
| `pytest` (+ `pytest-cov`) | Unit tests & coverage | `pytest` |
| `hypothesis` | Property-based tests for metric invariants | (via `pytest`) |
| `ruff` | Lint + format | `ruff check . && ruff format .` |
| `mypy` | Strict static typing | `mypy src` |
| `hatchling` / `build` | Packaging the wheel/sdist | `python -m build` |
| `pre-commit` | Run the gates on every commit | `pre-commit run --all-files` |

## Permission posture (`settings.json`)
- **Auto-allowed:** read-only inspection and the local dev gates (`pytest`, `ruff`, `mypy`,
  editable install) plus `git status/diff/log/add` — they are safe and run constantly.
- **Denied:** reading `.env*` (secrets) and `git push` (outward-facing; must be explicitly requested).
- Anything not listed falls back to a prompt. Run the gates before declaring work done.

## Rules
- Never read or print secrets; configuration comes from env vars (`.env.example` documents them).
- Run gates locally before any commit; report failures honestly rather than working around them.
