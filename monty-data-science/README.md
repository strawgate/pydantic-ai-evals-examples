# Monty Data Science Evals

Multi-step data-science evals for a Pydantic AI agent running in the
Monty CodeMode sandbox with SQL tools on a pre-seeded SQLite database.

Built on [Pydantic AI](https://ai.pydantic.dev/),
[Pydantic Evals](https://pydantic.dev/docs/ai/evals/evals/),
[pydantic-ai-harness](https://github.com/pydantic/harness) (CodeMode),
[pydantic-monty](https://github.com/pydantic/monty), and
[Logfire](https://logfire.pydantic.dev/) (traces + managed variables).

## Quick start

```bash
uv sync
cp .env.example .env       # fill in ANTHROPIC_API_KEY (or MINIMAX_*) + LOGFIRE_TOKEN
make check                 # lint + format + ty
make reset-prompt          # create the managed system_prompt variable
make eval                  # small dataset, judge ON
make eval-medium           # medium (5 offline + 1 online)
make eval-large            # large (8 offline + 2 online)
make eval-fast             # judge OFF (cheap smoke)
make verify-spans          # pull recent scores + variable resolutions from Logfire
make live                  # online cases with OnlineEvaluation
```

Every `make eval*` prints a direct URL to the variable's optimizer page.

## Layout

```
src/monty_data_science/
  agent.py           # CodeMode agent + task() + CODEMODE_PRELUDE + managed-var resolve
  quiet_code_mode.py # CodeMode wrapper that suppresses ERROR spans on retry
  tools.py           # Database + SQL tool factory + deterministic seed
  evaluators.py      # EfficientExecution + DataScienceJudge (5-dim, deterministic overall)
  datasets.py        # load_dataset + offline/online mode filter
  live.py            # online runner with OnlineEvaluation
  reset_prompt.py    # bump or wipe-and-reseed the managed variable
  verify_spans.py    # query Logfire for recent eval scores + variable spans
  logfire_links.py   # resolve project URL, print optimizer link
evals/
  conftest.py
  run_eval.py        # CLI runner
  test_eval.py       # pytest entrypoints
  datasets/{small,medium,large}.yaml
```

## What the agent does

Every case is a multi-step DS task that cannot be one-shot with a single
SQL query. The agent uses CodeMode to write Python that calls SQL tools
on a fresh, seeded SQLite DB, processes results, runs follow-up queries
informed by the first pass, and reports findings with interpretation
and a concrete recommendation.

SQL tools in the sandbox: `query(sql)`, `list_tables()`,
`describe_table(name)`, `insert_rows(table, rows)`, `table_count(table)`.

Seeded DB (deterministic, seed=42):
- 50 customers over ~18 months with `state` and `segment`.
- 25 products across 5 categories, a handful intentionally low-stock.
- ~288 orders with wide CLV spread, state-conditioned category mix
  (CA→Electronics, NY→Books), 3 pricing anomalies (overcharge, deep
  discount, 1-cent rounding decoy), 1 orphan customer_id, and a ~20%
  churn cohort.
- `meta.reference_date = 2024-07-01` for "today" semantics.

`metadata.mode: offline` cases go in batch evals; `online` cases go
through the live runner.

## Evaluators

- **`EfficientExecution`** — gates on success; `sqrt` curve vs a tight
  budget (10 requests, 60k tokens). Baseline 0.0–0.3, optimized 0.6–0.8.

- **`DataScienceJudge`** — one LLM call per case. Five subscores:
  `correctness`, `method`, `completeness`, `reasoning`, `rigor`. The
  `overall` score is computed deterministically as
  `0.5 × mean + 0.5 × min` of the subscores; the judge cannot soft-pedal
  a weak dimension.

  The `rigor` dimension is the senior-vs-junior gap — assumptions
  validated, intermediate results shown, embedded traps caught (orphan
  customer, 1-cent rounding decoy, not-yet-observable cohort cells),
  uncertainty flagged. Baseline agents score 0.3–0.6 here.

  Scoring is anchored strictly: 1.0 = exceptional, 0.9 = ship as-is,
  0.75 = needs one revision, 0.5 = junior work. Baseline runs land
  around 0.50–0.70 overall, leaving real headroom for optimization.

Judge is ON by default. `--no-judge` (or `make eval-fast`) skips it.

## Rubric structure

Each case's `expected_output` is structured for the judge:

```
CORRECTNESS: numeric / sanity-check requirements
METHOD:      approach requirements
COMPLETENESS: which steps must be addressed
RIGOR — explicit checks:
  - <embedded trap, e.g. orphan customer 999 must be excluded>
  - <embedded trap, e.g. 1-cent rounding anomaly must NOT be flagged>
  - <required validation steps>
```

Follow this when adding new cases so `rigor` has concrete hooks.

## Managed variable

The agent's system prompt is composed in two layers at run time:

- `CODEMODE_PRELUDE` (hardcoded in `agent.py`) — sandbox mechanics
  (forbidden imports, replacement patterns, tool surface). The
  optimizer can't see or evolve this.
- `DEFAULT_SYSTEM_PROMPT` (default) or the resolved Logfire managed
  variable value — DS-quality guidance. This is the only layer the
  optimizer touches.

Set `LOGFIRE_VAR_SYSTEM_PROMPT=system_prompt_data_science` in `.env`
to enable variable resolution. Every eval emits a `Resolve variable …`
span with the resolved version + label + value, so the Logfire
optimizer in the UI can evolve the prompt across runs.

Variable lifecycle:

- `make reset-prompt` — bump the variable to the next version with the
  current `DEFAULT_SYSTEM_PROMPT`. History preserved.
- `make wipe-prompt` — delete the variable server-side (drops all
  prior versions), then reseed at v1. Use to start a fully fresh
  optimization cycle.
- `make verify-spans` — query the Logfire read API and print a table
  of recent case scores plus the resolved variable versions.

## CodeMode flavour

Uses `QuietCodeMode` (`src/monty_data_science/quiet_code_mode.py`), a
thin wrapper around `pydantic-ai-harness`'s `CodeMode`. The only
behavioural difference: when sandbox code fails (`ModuleNotFoundError`,
syntax error, runtime exception), the wrapper returns the traceback as
a normal tool-return value instead of raising `ToolRetryError`. The
model still sees the error and retries on the next turn, but the
Logfire UI no longer fills with red `ERROR` spans for what is
essentially "the model typed something wrong, fixed it next turn".

## Env vars

| Variable | Purpose | Default |
|---|---|---|
| `MODEL` | Model identifier | `claude-sonnet-4-20250514` |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` | LLM provider | — |
| `MINIMAX_API_KEY` / `MINIMAX_BASE_URL` | Anthropic-compatible proxy | — |
| `LOGFIRE_TOKEN` | Logfire token (also used as `LOGFIRE_API_KEY`) | — |
| `LOGFIRE_SERVICE_NAME` | Service name in Logfire | `monty-data-science-evals` |
| `LOGFIRE_VAR_SYSTEM_PROMPT` | Managed-variable name | (unset) |
| `MAX_CONCURRENCY` | Max concurrent eval cases | `1` |
