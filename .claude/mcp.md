# MCP (Model Context Protocol)

MCP servers extend the assistant with external context and actions. This file records which
servers are relevant to `ds-llm-eval` and the policy for adding more. Project-scoped servers
belong in a committed `.mcp.json`; personal/credentialed servers stay in user settings.

## Candidate servers (add as the project needs them)
| Server | Why it is useful here | Status |
|--------|----------------------|--------|
| GitHub | Read issues/PRs, manage the `prjha-ds/ds-llm-eval` repo and releases. | Optional — add per developer |
| Filesystem | Scoped read access to local eval datasets / fixtures. | Optional |
| Langfuse / observability | Pull traces & datasets to evaluate, push scores back. | Optional — pairs with `integrations.langfuse` |
| Web fetch/search | Source material for `landscape-researcher` and `docs/research.md`. | Provided by built-in tools |

## Policy
- **Secrets:** MCP credentials come from env vars / user settings — never commit tokens. A shared
  `.mcp.json` may reference `${ENV_VAR}` placeholders only.
- **Least privilege:** enable read-only scopes unless a write action is explicitly required.
- **Reproducibility:** unit tests must not depend on any MCP server; MCP is for interactive
  development and research, not for the test suite.
- No MCP server is required to build, test, or use the package.
