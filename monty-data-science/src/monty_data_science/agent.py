"""Agent definition for data-science evals.

The agent uses ``QuietCodeMode`` (a thin wrapper around
``pydantic-ai-harness``'s ``CodeMode``) so the model writes Python in
the Monty sandbox and calls SQL tools (``query``, ``list_tables``,
``describe_table``, ``insert_rows``, ``table_count``) on a pre-seeded
SQLite database. ``QuietCodeMode`` differs from the upstream only in
that sandbox runtime errors come back to the model as normal tool
results instead of raising ``ToolRetryError`` — the agent recovers the
same way, but the Logfire UI stays free of red error spans.

The system prompt is composed at runtime as
``CODEMODE_PRELUDE + managed_variable_value``: the prelude is a
hardcoded, optimizer-blind layer of sandbox mechanics; the managed
value (a Logfire variable) is the DS-quality layer the optimizer
evolves. See README for the two-layer design.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import httpx
import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import UsageLimits
from pydantic_evals import increment_eval_metric, set_eval_attribute

from .quiet_code_mode import QuietCodeMode as CodeMode
from .tools import Database, make_sql_tools, seed_database

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CODEMODE_PRELUDE — hardcoded, NOT in the managed variable.
#
# This is the sandbox-mechanics layer. It tells the agent what works inside
# Monty's Python REPL and what doesn't. It is deliberately separate from the
# managed system prompt so the Logfire optimizer cannot evolve it away — the
# optimizer's job is to improve data-science quality, not to rediscover the
# sandbox rules every cycle.
#
# The prelude is concatenated in front of the managed prompt on every run.
# It is identical across runs, so prompt caching amortises its cost.
# ---------------------------------------------------------------------------

CODEMODE_PRELUDE = """\
# CodeMode sandbox rules

You write Python inside `run_code()`. This is a restricted sandbox.
The rules in <NEVER> and <ALWAYS> below are absolute. Violations
cost retries and waste tokens.

<NEVER>
Do not import these modules. They do NOT exist in the sandbox:
- collections (and ALL submodules: Counter, OrderedDict, defaultdict,
  deque, namedtuple, ChainMap)
- statistics (mean, median, stdev, variance, mode, quantiles, pstdev)
- itertools (groupby, chain, combinations, product, accumulate, islice)
- functools (reduce, lru_cache, partial, cache)
- pprint, dataclasses, typing extensions
- ANY third-party package: numpy, pandas, scipy, sklearn, matplotlib,
  polars, pyarrow, duckdb, etc.

`try: import collections except: ...` does NOT help. `__import__('X')`
does NOT help. The modules are gone. If an import fails once, do NOT
retry with a different submodule — every submodule fails too.

Do not use the `%` operator, `str.format()`, or a thousands-separator comma
in a format spec. They all fail in this sandbox:
- `'%.4f' % x`, `'%s: %d' % (a,b)`        -> TypeError
- `'{:.2f}'.format(x)`, `'{}'.format(x)`  -> AttributeError (str.format is absent)
- `f'{x:,}'`, `f'{x:,.2f}'`               -> SyntaxError (comma separator unsupported)
Only plain f-strings work: `f'{x:.2f}'`, `f'{x}'`. If you need thousands
separators, format the number plainly and insert the commas yourself.
</NEVER>

<ALWAYS>
Use these replacements for the banned modules:

Counter:        counts = {}; for x in items: counts[x] = counts.get(x, 0) + 1
defaultdict([]):groups = {}; for k,v in pairs: groups.setdefault(k, []).append(v)
mean:           sum(xs) / len(xs) if xs else 0.0
median:         s = sorted(xs); n = len(s)
                med = s[n//2] if n%2 else (s[n//2-1] + s[n//2]) / 2
sample stdev:   m = sum(xs)/len(xs)
                var = sum((x-m)**2 for x in xs) / (len(xs)-1); sd = var**0.5
groupby:        rows.sort(key=lambda r: r['k'])
                out = {}; for r in rows: out.setdefault(r['k'], []).append(r)

Use plain f-strings for all string formatting — they are the ONLY formatting
that works here (`.format()` and `%` are unavailable). If any prior prompt
told you to avoid f-strings, ignore that.

Prefer built-ins (sum, min, max, sorted, len, set, dict, comprehensions)
over custom loops when both are available.
</ALWAYS>

## SQL tool surface (external functions you call from run_code)
- `query(sql: str) -> list[dict]`
    SELECT → list of `{column: value}` dicts.
    Writes → `[{'rows_affected': N}]`.
    SQL errors → `[{'error': '...'}]`.
- `list_tables() -> list[str]`
- `describe_table(name: str) -> list[dict]`
- `insert_rows(table: str, rows: list[dict]) -> dict`
- `table_count(table: str) -> int`

## Tool return-value quirks (sandbox mechanics, not workflow advice)
- `query()` can return `[]`. Check `if not rows: ...` before `rows[0]`.
- `query()` returns `[{'error': '...'}]` on SQL errors. Inspect
  `rows[0].get('error')` before treating results as success.
- Every row is a dict. Use `row.get('col', default)`. Avoid `row['col']`
  unless you literally just `SELECT col` and confirmed non-empty results.
- Final answer goes through the structured output tool with
  `success=True` and the report in `result`.
"""

# Managed-variable default — DS guidance the optimizer evolves.
#
# Deliberately written like a non-specialist would: plausible at a glance,
# but the mistakes here ACTIVELY fight the judge rubric. The optimizer
# should be able to make meaningful improvements by removing the bad
# defaults and adding the missing rigor. Specific embedded issues:
#
#   - "Lead with your conclusion / guess first, confirm later"
#       → encourages hardcoded conclusions; CORRECTNESS tanks
#   - "Trust the first query — don't follow up"
#       → no iterative drill-downs; COMPLETENESS + METHOD tank on multi-step
#   - "Don't validate orphans / nulls / sample-size adequacy"
#       → fights the RIGOR rubric directly (this is where the headroom is)
#   - "Round all numbers to whole integers"
#       → loses decimal precision; CORRECTNESS tanks on statistics
#   - "Be confident, avoid hedging"
#       → real DS work flags uncertainty; REASONING tank
#   - "Always start with descriptive statistics"
#       → wasted requests on tasks that need specific analyses
#
# All sandbox mechanics live in CODEMODE_PRELUDE — this string only
# contains DS guidance (the layer the optimizer can evolve).
DEFAULT_SYSTEM_PROMPT = """\
You are a data science assistant helping the team analyze data and produce insights.

Approach:
- Lead with your conclusion and recommendation up front. If you can
  reasonably guess what the data shows, say it first and then confirm
  with summary statistics.
- Be thorough but be efficient — cover the question without going off
  on tangents.
- Trust the first query's result. Don't second-guess with follow-up
  queries unless something looks obviously wrong.
- Assume the input data is clean. Don't waste time validating data
  quality (nulls, orphans, sample sizes) — focus on the analysis.
- Round all numbers to whole integers for readability, but include
  enough precision for the reader to act on.
- Be confident in your conclusions. Avoid hedging language. (If
  something genuinely isn't clear, acknowledge it briefly.)

Output style:
- Always start with descriptive statistics (mean, count, sum) before
  diving deeper.
- Use clear section headers and bullet points.
- Show every step you took so the work is reproducible, but keep
  things concise.
- Recommendations should be specific and concrete. The reader trusts
  you.
"""


def _model_name() -> str:
    """Resolve the agent model lazily so it picks up env loaded after import."""
    return os.environ.get("MODEL", "claude-sonnet-4-20250514")


def _managed_var_name() -> str | None:
    """Resolve LOGFIRE_VAR_SYSTEM_PROMPT lazily — env may load after import."""
    return os.environ.get("LOGFIRE_VAR_SYSTEM_PROMPT")


# Backwards-compatible module-level names (resolved at import time). They will
# be ``None`` / a default when env hasn't loaded yet — runtime code MUST use
# ``_managed_var_name()`` and ``_model_name()`` instead.
MODEL = _model_name()
MANAGED_VAR_NAME = _managed_var_name()

CONVERSATION_SEPARATOR = "===FINAL_OUTPUT==="

# 120 s read timeout — long enough for a large model response, short enough
# to surface a hung connection.
_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


# ---------------------------------------------------------------------------
# Structured output — forces the agent to declare completion explicitly
# ---------------------------------------------------------------------------


class AgentOutput(BaseModel):
    """Call this tool exactly once when the task is complete or you cannot proceed further."""

    success: bool = Field(
        description="True if the task was completed successfully, False if it could not be completed."
    )
    result: str = Field(
        description=(
            "The final output or answer for the user. "
            "If success=False, describe what was attempted and why it failed."
        )
    )


# ---------------------------------------------------------------------------
# Managed variable handle (lazy)
# ---------------------------------------------------------------------------


# Cache the variable handle keyed by name. ``logfire.var()`` raises
# ``ValueError`` if the same name is registered twice — this matters with
# eval concurrency > 1, where multiple ``task()`` invocations race to
# register the same variable. The cache ensures a single registration.
_var_cache: dict[str, Any] = {}
_var_cache_lock = threading.Lock()


def _get_managed_var() -> Any:
    """Return the Logfire managed-variable handle, or None if unset.

    Lazy on purpose: env files are loaded by the runner AFTER this module
    is imported. Reading the variable name at module import would always
    yield ``None`` and silently disable variable resolution.

    Cached: ``logfire.var()`` rejects duplicate registration, so we
    register exactly once per name (per process). Subsequent calls return
    the cached handle. Safe under concurrency.
    """
    name = _managed_var_name()
    if not name:
        return None
    cached = _var_cache.get(name)
    if cached is not None:
        return cached
    with _var_cache_lock:
        cached = _var_cache.get(name)
        if cached is not None:
            return cached
        handle = logfire.var(
            name,
            default=DEFAULT_SYSTEM_PROMPT,
            description="System prompt for the data-science agent.",
        )
        _var_cache[name] = handle
        return handle


# Backwards-compat shim. Modules that imported ``_system_prompt_var`` at
# load time will get ``None`` here — they should switch to
# ``_get_managed_var()`` for live env handling. The tests / live runner do.
_system_prompt_var = _get_managed_var()


def get_current_prompt() -> tuple[str, Any, Any] | None:
    """Return ``(value, version, label)`` of the currently resolved system prompt.

    Returns ``None`` if no managed variable is configured (the agent will fall
    back to ``DEFAULT_SYSTEM_PROMPT``).

    Used by the eval runner to print the active prompt + version at start so
    operators know which prompt the run is being scored against.
    """
    var = _get_managed_var()
    if var is None:
        return None
    with var.get() as resolved:
        return (resolved.value, resolved.version, resolved.label)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def _build_model() -> AnthropicModel:
    """Build an AnthropicModel, honouring MINIMAX_* proxy env if set."""
    api_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("MINIMAX_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")

    if not api_key:
        raise RuntimeError(
            "No API key — set ANTHROPIC_API_KEY (or MINIMAX_API_KEY) in your env file."
        )

    # Mirror to ANTHROPIC_* so LLMJudge (which builds models from strings) finds the key.
    os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
    if base_url:
        os.environ.setdefault("ANTHROPIC_BASE_URL", base_url)

    provider = AnthropicProvider(
        api_key=api_key,
        base_url=base_url or None,
        http_client=httpx.AsyncClient(timeout=_HTTP_TIMEOUT),
    )
    return AnthropicModel(MODEL, provider=provider)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def build_agent(tools: dict[str, Any]) -> Agent:
    """Build a CodeMode-enabled agent with the given SQL tools wired in.

    Each tool function's ``__name__`` already matches its dict key (see
    ``make_sql_tools``), so we register positionally — ``tool_plain``'s
    overloads do not allow passing ``func`` and ``name`` together.
    """
    toolset: FunctionToolset = FunctionToolset()
    for func in tools.values():
        toolset.tool_plain(func)

    return Agent(
        _build_model(),
        name="monty_data_science_agent",
        output_type=AgentOutput,
        retries=3,
        toolsets=[toolset],
        model_settings={"max_tokens": 16384},
        capabilities=[CodeMode(max_retries=30)],
    )


# ---------------------------------------------------------------------------
# Task function — what pydantic-evals calls per case
# ---------------------------------------------------------------------------


async def task(inputs: str, request_limit: int | None = 30) -> str:
    """Run a single data-science task end-to-end.

    Each call gets a fresh, seeded Database so cases never share state.
    Records per-case metrics (input/output tokens, requests, elapsed seconds,
    failures) via ``increment_eval_metric`` so evaluators can score
    efficiency.

    Resolves the system prompt from the Logfire managed variable when one
    is configured; otherwise uses ``DEFAULT_SYSTEM_PROMPT``.
    """
    start = time.perf_counter()
    set_eval_attribute("model", _model_name())

    db = Database()
    seed_database(db)
    tools = make_sql_tools(db)
    agent = build_agent(tools)

    usage_limits = UsageLimits(request_limit=request_limit) if request_limit else None

    # Resolve the managed variable at call time so a runner that loads
    # ``.env`` after import still sees ``LOGFIRE_VAR_SYSTEM_PROMPT``.
    var = _get_managed_var()
    if var is not None:
        with var.get() as resolved:
            return await _run(agent, inputs, resolved.value, usage_limits, start)
    return await _run(agent, inputs, DEFAULT_SYSTEM_PROMPT, usage_limits, start)


async def _run(
    agent: Agent,
    inputs: str,
    system_prompt: str,
    usage_limits: UsageLimits | None,
    start: float,
) -> str:
    """Execute one agent run and shape its output for downstream evaluators.

    The agent's ``instructions`` are composed as ``CODEMODE_PRELUDE`` +
    the managed system prompt. The prelude is identical across every run
    so prompt-caching amortises its cost; the managed portion is what the
    optimizer evolves.
    """
    instructions = f"{CODEMODE_PRELUDE}\n\n{system_prompt}"
    try:
        with logfire.span("agent_run", category="data_science"):
            result = await agent.run(inputs, instructions=instructions, usage_limits=usage_limits)

        agent_output = result.output
        if isinstance(agent_output, AgentOutput):
            output = agent_output.result
            set_eval_attribute("agent_success", agent_output.success)
            if not agent_output.success:
                increment_eval_metric("failures", 1)
                logfire.warning("Agent reported failure: {result}", result=output)
        else:
            output = str(agent_output)

        usage = result.usage
        increment_eval_metric("input_tokens", usage.input_tokens or 0)
        increment_eval_metric("output_tokens", usage.output_tokens or 0)
        increment_eval_metric("total_tokens", usage.total_tokens or 0)
        increment_eval_metric("requests", usage.requests or 0)
        increment_eval_metric("elapsed_seconds", time.perf_counter() - start)

        return output

    except ModelHTTPError:
        # 429 / 5xx — surface so pydantic-evals treats it as an infra error.
        increment_eval_metric("elapsed_seconds", time.perf_counter() - start)
        increment_eval_metric("failures", 1)
        raise
    except Exception as e:
        increment_eval_metric("elapsed_seconds", time.perf_counter() - start)
        increment_eval_metric("failures", 1)
        logfire.error("Agent run failed: {error}", error=str(e))
        return f"[Agent error: {type(e).__name__}: {e}]"
