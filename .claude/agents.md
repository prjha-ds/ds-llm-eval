# Agents

Specialized subagents for `ds-llm-eval`, defined in `.claude/agents/`. Each runs in its own
context with a scoped toolset; the main session delegates to them for focused, repeatable work.

| Agent | Purpose | When to use |
|-------|---------|-------------|
| [`metric-implementer`](agents/metric-implementer.md) | Implement a new metric following the pure-function + registry convention. | Adding a metric to `search`, `recommendation`, or `llm`. |
| [`eval-test-author`](agents/eval-test-author.md) | Write the standard test triad (worked example, edge cases, property test), fully offline. | After a metric is added or when coverage drops. |
| [`landscape-researcher`](agents/landscape-researcher.md) | Survey external eval tooling and update `docs/research.md` with cited findings. | Deciding whether to adopt or learn from a library. |

## Conventions
- Agents inherit the rules in `CLAUDE.md`; they must not override the guardrails.
- Keep toolsets minimal — read/edit/search/test by default; web tools only for the researcher.
- A typical add-a-metric flow chains `metric-implementer` → `eval-test-author`, then the main
  session runs the quality gates and reports results.
- Prefer extending an existing agent over creating a near-duplicate one.
