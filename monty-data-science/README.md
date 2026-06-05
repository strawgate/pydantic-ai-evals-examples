# Monty Data Science

This repo has two things in it:

1. **The continuous-improvement demo** (`make demo-start`) — the main event. See
   below and **[DEMO.md](DEMO.md)** for the full stage runbook.
2. **An eval harness** (`make eval`) — multi-step data-science evals for the same
   agent. Separate from the demo; see [Eval harness](#eval-harness-separate-from-the-demo).

Built on [Pydantic AI](https://ai.pydantic.dev/),
[pydantic-ai-harness](https://github.com/pydantic/harness) (CodeMode),
[pydantic-monty](https://github.com/pydantic/monty),
[Pydantic Evals](https://pydantic.dev/docs/ai/evals/evals/), and
[Logfire](https://logfire.pydantic.dev/) (traces + managed variables + the
optimizer).

## The demo (start here)

A Pydantic AI data-science agent runs in the background answering a stream of
questions. Its system prompt — a **Logfire managed variable**
(`data_science_agent_prompt`) — documents a database schema that a migration
quietly renamed out from under it. So every run hits `no such column`,
introspects the live schema, recovers, and answers — wasting requests, tokens,
and latency. You then open the variable's **Optimize** tab in Logfire, click
**Generate**, and the optimizer rewrites the prompt with the correct schema
**from the traces alone** — no evals, no test suite. Restart the traffic and the
Agents-page charts drop.

### One-time setup

```bash
uv sync
cp .env.example .env       # then fill in the three secrets below
```

`.env` needs (see `.env.example`):

| Var | What | Notes |
|---|---|---|
| `PYDANTIC_AI_GATEWAY_API_KEY` | model access | Pydantic AI Gateway key (`pylf_v2_…`) |
| `LOGFIRE_TOKEN` | trace ingestion | project **write** token (`pylf_v1_…`) |
| `LOGFIRE_API_KEY` | managed variables | **user API key** with `read_variables` + `write_variables` (a plain write token does NOT have these) |

```bash
make seed                  # create/reset the managed variable to the wrong-schema "before" state
```

### Run it

```bash
make demo-start            # generate trace traffic in the BACKGROUND (-> demo.log)
make demo-status           # is it running? how many runs so far?
make demo-logs             # follow the live log
make demo-stop             # stop it
make seed                  # reset the variable to the "before" state (fresh optimizer window)
```

- `make demo-start DEMO_ARGS="--workers 5"` — more concurrency, charts fill faster.
- `uv run demo` — same traffic in the foreground (Ctrl-C to stop).
- `uv run demo --once` — a single request; prints the full answer + token usage.

The startup banner prints direct links to the **Agents page** and the variable's
**Optimize** tab (`…/managed-variables/variables/data_science_agent_prompt/details`).

> **Optimizer evidence window — the one gotcha.** The optimizer only considers
> runs sent *since the served value last changed*. **Seed first, then start
> traffic.** If you reseed or apply a new value in the UI, the clock resets and
> you need fresh traffic before the Optimize tab shows matches again. If it says
> "no matches": `make demo-start`, wait ~2 min, and check the tab's lookback
> range isn't tiny.

Full walkthrough, what to screenshot, and tuning knobs: **[DEMO.md](DEMO.md)**.

## Eval harness (separate from the demo)

```bash
make check                 # lint + format + ty
make reset-prompt          # create the managed system_prompt variable (eval path)
make eval                  # small dataset, judge ON
make eval-medium           # medium (5 offline + 1 online)
make eval-large            # large (8 offline + 2 online)
make eval-fast             # judge OFF (cheap smoke)
make verify-spans          # pull recent scores + variable resolutions from Logfire
make live                  # online cases with OnlineEvaluation
```

The eval path uses `ANTHROPIC_API_KEY` (or `MINIMAX_*`), not the gateway. Every
`make eval*` prints a direct URL to the variable's optimizer page.

## Layout

```
src/monty_data_science/
  # --- the demo (make demo-start / seed) ---
  demo.py             # background traffic runner (uv run demo); the connect4-style "spider"
  demo_config.py      # shared names: variable, agent, model, gateway; .env loading + logfire config
  demo_prompts.py     # WRONG_SCHEMA_PROMPT (old column names) + CORRECT_SCHEMA reference
  demo_questions.py   # pool of DS questions the traffic picks from
  seed_demo_variable.py # create/reset the managed variable to the before-state (uv run seed-demo-variable)
  # --- shared agent + tools ---
  agent.py           # CodeMode agent + task() + CODEMODE_PRELUDE + managed-var resolve
  quiet_code_mode.py # CodeMode wrapper that suppresses ERROR spans on retry
  tools.py           # Database + SQL tool factory + deterministic seed (POST-migration schema + "THE MIGRATION" note)
  logfire_links.py   # resolve project URL, print optimizer link
  # --- the eval harness ---
  evaluators.py      # EfficientExecution + DataScienceJudge (5-dim, deterministic overall)
  datasets.py        # load_dataset + offline/online mode filter
  live.py            # online runner with OnlineEvaluation
  reset_prompt.py    # bump or wipe-and-reseed the eval managed variable
  verify_spans.py    # query Logfire for recent eval scores + variable spans
evals/
  conftest.py
  run_eval.py        # CLI runner
  test_eval.py       # pytest entrypoints
  datasets/{small,medium,large}.yaml
```

The demo's entry points are the `[project.scripts]` in `pyproject.toml`
(`demo`, `seed-demo-variable`) and the `make demo-*` / `make seed` targets in
the `Makefile`.

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
