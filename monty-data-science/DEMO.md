# Continuous-improvement demo — runbook

A live demo of Logfire's **managed-variable optimizer**: an agent running in
production, quietly broken, fixed by one click — from traces alone. No evals,
no test suite, no regression harness. Just the trace stream and an **Optimize**
button.

## The story (what you say on stage)

> A week ago, Tom normalised some column names in the analytics database —
> `amount` → `total_amount`, `created_at` → `signup_date`, and a couple more.
> Routine migration. What nobody remembered is that a data-science agent had
> the old schema baked into its system prompt. It didn't crash — it's resilient,
> it introspects the database and recovers — so it kept answering correctly.
> It just got slow and expensive: every single request now wastes a few turns
> rediscovering that the schema moved. For a week. Nobody was watching.
>
> You don't fix this by being clever. You fix it by sending your traces to
> Logfire and clicking **Optimize**. The optimizer reads what the agent
> actually did — hit `no such column`, introspect, retry — and rewrites the
> prompt with the real schema. The wasted work disappears.

The point: optimization isn't the agent getting galaxy-brained. It's catching
the dumb, macroscopic stuff — a migration, a renamed tool, a shifted user base —
that you'd have caught yourself if you'd been looking. It does the looking.

## How it works

```
uv run demo  →  N async workers, each in a loop:
    resolve managed variable `data_science_agent_prompt`   (attribution)
    run the data-science agent on a random question        (full trajectory)
        DB has the NEW schema; the prompt describes the OLD schema
        → SQL hits `no such column` → agent introspects the live catalog
          (describe_table / PRAGMA / sqlite_master) → rewrites → answers
        → every run burns extra requests + tokens + latency
```

- The agent runs **inside** `with var.get()`, so every agent/model span carries
  `logfire.variables.data_science_agent_prompt=<label>` — the attribution the
  optimizer correlates on.
- `logfire.instrument_pydantic_ai()` captures the whole trajectory (tool calls,
  the `{'error': 'no such column: amount'}` returns, the recovery), which is the
  evidence the optimizer reads.
- `query()` also emits a `logfire.warning` tagged `schema_mismatch=true` on each
  failed SQL call, so the schema problem is searchable in Explore.

The migration lives in `src/monty_data_science/tools.py` (`seed_database` + the
big "THE MIGRATION" note). The wrong-schema prompt lives in
`src/monty_data_science/demo_prompts.py`.

## One-time setup (the demo author, not the presenter)

Needs three secrets in `.env` (see `.env.example`):

| Secret | What | Scope |
|---|---|---|
| `PYDANTIC_AI_GATEWAY_API_KEY` | model access | gateway key (`pylf_v2_…`) |
| `LOGFIRE_TOKEN` | trace ingestion | project **write** token (`pylf_v1_…`) |
| `LOGFIRE_API_KEY` | managed variables | **user API key** with `read_variables` + `write_variables` |

> A plain write token can send traces but **cannot** read or write variables
> (you'll get `403 Not enough permissions`). Use a user API key for
> `LOGFIRE_API_KEY` — org members already have both variable permissions.

Then seed the variable to the broken "before" state:

```bash
uv run seed-demo-variable          # create / set `data_science_agent_prompt`
# or, for a clean slate between rehearsals (drops optimizer history):
uv run seed-demo-variable --wipe
```

Smoke-test one request end to end (prints the answer + usage):

```bash
uv run demo --once
```

## Running the demo

Easiest — run the traffic **in the background** and leave it going:

```bash
make demo-start        # start generating traces in the background (-> demo.log)
make demo-status       # is it running? how many runs so far?
make demo-logs         # follow the live log (Ctrl-C just stops watching)
make demo-stop         # stop it
```

`make demo-start DEMO_ARGS="--workers 5"` for more concurrency (charts fill
faster). To watch it in the foreground instead:

```bash
uv run demo            # continuous traffic, Ctrl-C to stop
uv run demo --once     # single request, prints the full answer + usage
```

Either way the startup banner prints direct links to the **Agents page** and
the variable's **Optimize** tab.

> **Heads-up on the optimizer's evidence window:** the optimizer only considers
> runs sent *since the served value last changed*. So **seed first, then start
> traffic** — and if you reseed (`make seed`) or apply a new value in the UI,
> the clock resets and you need fresh traffic before the Optimize tab shows
> matches again. If Optimize says "no matches", just `make demo-start` and give
> it a couple of minutes (and check the lookback window in the tab isn't set to
> a tiny range).

### On stage

1. **Let it run a few minutes first** (do this before the segment, or run it
   during Samuel's earlier slides) so the charts have a "before" baseline.
2. **Agents page** → point at the `monty_data_science_agent` row: tokens, cost,
   avg duration. Open a trace → show the agent hitting `no such column`,
   introspecting, recovering. "It works. It's just wasting a third of its
   tokens every call."
3. **Variable → Optimize tab** → click **Optimize**. It reads the recent traces
   and proposes a new prompt with the corrected schema. Read the diff out loud:
   "It figured out the real column names just from watching it fail."
4. **Accept** the proposal (set it as the serving value).
5. The running traffic picks up the new value within ~a minute (variable
   refresh). If you want it instant, **restart `uv run demo`**.
6. **Back to the Agents page** (or compare the trace before/after): tokens,
   cost, and latency drop. Open a new trace → clean run, no errors, no
   introspection detour.

## What to screenshot (for the backup slide)

If the internet dies, the slide carries the demo. Capture, before and after
optimization:

- The **Agents page** row for `monty_data_science_agent` (tokens / cost / avg
  duration / sparkline).
- A **trace**: before = full of error→introspect→retry; after = clean.
- The **optimizer proposal diff** (old schema block → corrected schema block).

The headline numbers that always move: **input/output tokens, cost, average
duration** (sourced from the model-call spans). Put the before/after side by
side with the deltas circled.

## Tuning (if calibration needs it)

In `tools.py`'s "THE MIGRATION" note:

- **Louder before-state / more errors:** rename a whole table too (e.g.
  `orders` → `sales`, for `no such table`), or rename more columns.
- **Move the Agents-page _error rate_ tile:** that tile counts `is_exception`
  on the agent-run root span, which only happens when a run genuinely fails —
  so lower `DEMO_REQUEST_LIMIT` (e.g. 12–15) until the worst "before" runs
  exhaust their budget and fail. The "after" runs finish well under it.
  (Tokens/cost/latency move regardless; this is only if you want the error
  tile to move too.)
- **Safer recovery / fewer errors:** rename fewer columns, or add an
  introspect-on-error hint to `CODEMODE_PRELUDE` in `agent.py`.

Volume/pace: `DEMO_WORKERS`, `DEMO_MIN_DELAY`, `DEMO_MAX_DELAY`.

## Reset between runs

```bash
uv run seed-demo-variable --wipe   # back to the broken before-state at v1
```

## Troubleshooting

- **`403 Not enough permissions` when seeding** → `LOGFIRE_API_KEY` isn't a
  variable-scoped user API key (see setup table).
- **Optimizer proposal doesn't fix the schema** → make sure traffic ran long
  enough that recent traces show the failures; the optimizer reads a recent
  lookback window.
- **"After" charts don't improve** → the running traffic is still on the old
  value; restart `uv run demo` so it re-resolves the variable.
- **Prompt content muddies the story** → the seeded prompt is "wrong schema +
  the repo's intentionally-mediocre DS guidance". If the mixed signal is
  distracting, swap `WRONG_SCHEMA_PROMPT` in `demo_prompts.py` to use a clean
  `DS_GUIDANCE` and reseed — the schema mismatch is the part the demo turns on.
```
