# etc/

Miscellaneous project configuration, templates, and non-code assets that don't belong in `src/`,
`tests/`, or `docs/`. Keep secrets out — only templates and committed config live here.

## Where configuration lives
| Concern | File |
|---------|------|
| Packaging, deps, tool config (ruff/mypy/pytest) | [`../pyproject.toml`](../pyproject.toml) |
| Secrets template (copy to `.env`, git-ignored) | [`../.env.example`](../.env.example) |
| Pre-commit hooks | [`../.pre-commit-config.yaml`](../.pre-commit-config.yaml) |
| CI pipeline | [`../.github/workflows/ci.yml`](../.github/workflows/ci.yml) |
| AI-assisted workflow (agents/skills/tools/mcp/memory) | [`../.claude/`](../.claude/) |

A declarative experiment-config format (Elliot-style YAML) is planned for Milestone 4 — see
[`../docs/plan.md`](../docs/plan.md). Example configs will be added here when that lands.
