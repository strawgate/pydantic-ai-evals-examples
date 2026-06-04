# Pydantic AI Evals — Examples

End-to-end example eval suites built on
[Pydantic AI](https://ai.pydantic.dev/) and
[Pydantic Evals](https://pydantic.dev/docs/ai/evals/evals/), with
[Logfire](https://logfire.pydantic.dev/) for traces and managed-variable
optimization.

## Projects

### [`monty-data-science/`](./monty-data-science) — multi-step data-science evals

An agent running in the [Monty](https://github.com/pydantic/monty) CodeMode
sandbox (via [`pydantic-ai-harness`](https://github.com/pydantic/harness))
solves multi-step data-science tasks against a pre-seeded SQLite database.
The eval suite shows:

- **Two-layer system prompt.** A hardcoded `CODEMODE_PRELUDE` for sandbox
  mechanics (optimizer-blind) + a Logfire managed variable for DS-quality
  guidance (optimizer-evolved). The optimizer can improve data-science
  practice without ever rediscovering the sandbox rules.
- **5-dimension LLM judge.** `DataScienceJudge` scores correctness,
  method, completeness, reasoning, and rigor. `overall` is composed
  deterministically as `0.5·mean + 0.5·min(dims)` so a weak dimension
  can't be soft-pedalled.
- **Quiet CodeMode wrapper.** Sandbox runtime errors come back to the
  model as normal tool results, so Logfire traces don't fill with red
  ERROR spans for routine retry-and-fix behaviour.
- **Open-ended task framings.** Cases are posed as business questions
  ("the CFO wants to know about concentration risk") — the agent must
  decide methodology, catch embedded data-quality traps (orphan
  customer, 1-cent rounding anomalies, not-yet-observable cohort cells)
  on its own.

### [`twenty-questions/`](./twenty-questions) — multi-agent guessing game eval

An agent plays 20 Questions against a separate answerer agent. Useful as
a minimal example of agent-vs-agent evals, online evaluation with
background scorers, and a managed-variable system-prompt setup.

## Per-project quickstart

Each project is self-contained with its own `pyproject.toml`,
`Makefile`, and `.env.example`. Quickstart instructions are in the
project's own README.
