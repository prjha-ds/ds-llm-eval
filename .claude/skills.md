# Skills

Reusable, parameterized procedures for `ds-llm-eval`, defined under `.claude/skills/`.
A skill captures a multi-step workflow so it runs the same way every time.

| Skill | Purpose |
|-------|---------|
| [`add-metric`](skills/add-metric/SKILL.md) | Scaffold a new metric end-to-end: implementation, registry entry, exports, the test triad, and the quality gates (`ruff` + `mypy` + `pytest`). |

## Authoring conventions
- One skill per directory: `.claude/skills/<name>/SKILL.md` with `name` + `description` frontmatter.
- The `description` must state *what* it does and *when* to trigger it — that is how it gets matched.
- Keep steps imperative and verifiable; always end with the project's quality gates.
- Skills encode process; they defer the domain rules to `CLAUDE.md` rather than repeating them.

## Useful built-in skills for this repo
- `code-review` — review the current diff for correctness and cleanup before a commit.
- `deep-research` — multi-source, cited research used to build `docs/research.md`.
