"""Background traffic generator for the continuous-improvement demo.

Think of this as the connect4 "spider": a steady stream of realistic requests
hitting a production agent, except here the agent is in-process (a Pydantic AI
CodeMode data-science agent) rather than behind an HTTP backend.

Each request:
  1. resolves the managed variable ``data_science_agent_prompt`` (so every
     agent/model span carries ``logfire.variables.data_science_agent_prompt``
     — the attribution the Logfire optimizer correlates on), then
  2. runs the agent on a random data-science question against a freshly seeded
     SQLite database.

The seeded DB uses the POST-migration schema; the seeded prompt documents the
PRE-migration column names. So every run trips over ``no such column``,
introspects the live catalog, recovers, and answers — burning extra
requests/tokens/latency. Open the Agents page to watch it; open the variable's
Optimize tab to fix it.

Run it:
    uv run demo                 # continuous traffic (Ctrl-C to stop)
    uv run demo --once          # one run, prints the full answer + usage
    uv run demo --workers 4     # more concurrency = faster chart data

Knobs (flags or env): --workers/DEMO_WORKERS, --min-delay/DEMO_MIN_DELAY,
--max-delay/DEMO_MAX_DELAY, --request-limit/DEMO_REQUEST_LIMIT, --model/MODEL.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from typing import Any

import logfire
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import UsageLimits

from .agent import CODEMODE_PRELUDE, AgentOutput
from .demo_config import (
    AGENT_NAME,
    VARIABLE_NAME,
    configure_logfire,
    load_env,
    model_name,
)
from .demo_prompts import WRONG_SCHEMA_PROMPT
from .demo_questions import QUESTIONS
from .logfire_links import _resolve_project_url
from .quiet_code_mode import QuietCodeMode as CodeMode
from .tools import Database, make_sql_tools, seed_database

# Module-level singleton variable handle: ``logfire.var()`` rejects duplicate
# registration, and multiple workers resolve the same name.
_var: Any = None


def _get_var() -> Any:
    global _var
    if _var is None:
        _var = logfire.var(
            VARIABLE_NAME,
            default=WRONG_SCHEMA_PROMPT,
            description="System prompt (incl. DB schema) for the data-science agent.",
        )
    return _var


def build_demo_agent() -> Agent:
    """Build a fresh CodeMode agent over a freshly seeded DB for one run.

    The seeded ``Database`` is kept alive by the tool closures held in the
    toolset, so it lives as long as the agent. Instructions are passed
    per-run (see ``run_once``) so each run picks up the latest resolved prompt.
    """
    db = Database()
    seed_database(db)
    tools = make_sql_tools(db)
    toolset: FunctionToolset = FunctionToolset()
    for func in tools.values():
        toolset.tool_plain(func)

    return Agent(
        model_name(),
        name=AGENT_NAME,
        output_type=AgentOutput,
        retries=3,
        toolsets=[toolset],
        model_settings={"max_tokens": 16384},
        capabilities=[CodeMode(max_retries=30)],
    )


async def run_once(question: str, request_limit: int) -> dict[str, Any]:
    """Resolve the variable, run the agent on ``question``, return a summary.

    The agent runs INSIDE ``with var.get()`` so its spans are attributed to
    the managed variable + serving label.
    """
    agent = build_demo_agent()
    var = _get_var()
    start = time.perf_counter()
    with var.get() as resolved:
        instructions = f"{CODEMODE_PRELUDE}\n\n{resolved.value}"
        result = await agent.run(
            question,
            instructions=instructions,
            usage_limits=UsageLimits(request_limit=request_limit),
        )
    elapsed = time.perf_counter() - start
    usage = result.usage
    output = result.output
    return {
        "ok": isinstance(output, AgentOutput) and output.success,
        "requests": usage.requests or 0,
        "input_tokens": usage.input_tokens or 0,
        "output_tokens": usage.output_tokens or 0,
        "total_tokens": usage.total_tokens or 0,
        "elapsed": elapsed,
        "label": resolved.label,
        "version": resolved.version,
        "answer": output.result if isinstance(output, AgentOutput) else str(output),
    }


async def _worker(
    name: str, stop: asyncio.Event, args: argparse.Namespace, counter: list[int]
) -> None:
    while not stop.is_set():
        question = random.choice(QUESTIONS)
        counter[0] += 1
        n = counter[0]
        try:
            summary = await run_once(question, args.request_limit)
            status = "ok " if summary["ok"] else "FAIL"
            print(
                f"[{name}] #{n} {status} "
                f"reqs={summary['requests']:>2} "
                f"tok={summary['total_tokens']:>6} "
                f"{summary['elapsed']:>5.1f}s "
                f"(v{summary['version']}/{summary['label']})  {question[:50]}…",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 — one bad run must not kill the worker
            logfire.exception("demo run failed")
            print(f"[{name}] #{n} ERROR {type(e).__name__}: {e}", flush=True)
        # Jittered delay between requests so traffic looks organic.
        await asyncio.sleep(random.uniform(args.min_delay, args.max_delay))


def _print_banner(args: argparse.Namespace) -> None:
    base = _resolve_project_url()
    print("\n─── Monty data-science demo ────────────────────────────────", flush=True)
    print(f"  Model:    {model_name()}", flush=True)
    print(f"  Variable: {VARIABLE_NAME}", flush=True)
    print(f"  Agent:    {AGENT_NAME}", flush=True)
    print(
        f"  Workers:  {args.workers}   delay {args.min_delay:.0f}–{args.max_delay:.0f}s   "
        f"request-limit {args.request_limit}",
        flush=True,
    )
    if base:
        print(f"  Agents page: {base}/agents", flush=True)
        print(
            f"  Optimize:    {base}/managed-variables/variables/{VARIABLE_NAME}/details"
            f"  (open the Optimize tab)",
            flush=True,
        )
    else:
        print("  (no Logfire project resolved — is LOGFIRE_TOKEN set?)", flush=True)
    print("────────────────────────────────────────────────────────────\n", flush=True)


async def _main_async(args: argparse.Namespace) -> None:
    if args.once:
        print("Running a single request (smoke test)…\n", flush=True)
        summary = await run_once(random.choice(QUESTIONS), args.request_limit)
        print(
            f"\nresult: {'OK' if summary['ok'] else 'FAILED'}  "
            f"requests={summary['requests']}  total_tokens={summary['total_tokens']}  "
            f"elapsed={summary['elapsed']:.1f}s  (prompt v{summary['version']}/{summary['label']})",
            flush=True,
        )
        print("\n--- agent answer ---\n", flush=True)
        print(summary["answer"], flush=True)
        logfire.force_flush(timeout_millis=10_000)
        return

    stop = asyncio.Event()
    counter = [0]
    workers = [
        asyncio.create_task(_worker(f"w{i}", stop, args, counter)) for i in range(args.workers)
    ]
    try:
        await asyncio.gather(*workers)
    except asyncio.CancelledError:
        pass
    finally:
        stop.set()
        for w in workers:
            w.cancel()
        logfire.force_flush(timeout_millis=10_000)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Background traffic for the Monty data-science demo")
    p.add_argument("--env", default=None, help="Path to .env (auto-detected if omitted)")
    p.add_argument("--once", action="store_true", help="Run one request, print the answer, exit")
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("DEMO_WORKERS", "3")),
        help="Concurrent request workers",
    )
    p.add_argument(
        "--min-delay",
        type=float,
        default=float(os.environ.get("DEMO_MIN_DELAY", "2")),
        help="Min seconds between a worker's requests",
    )
    p.add_argument(
        "--max-delay",
        type=float,
        default=float(os.environ.get("DEMO_MAX_DELAY", "8")),
        help="Max seconds between a worker's requests",
    )
    p.add_argument(
        "--request-limit",
        type=int,
        default=int(os.environ.get("DEMO_REQUEST_LIMIT", "30")),
        help="Per-run model-request cap (lower => worst runs fail => error tile moves)",
    )
    return p.parse_args(argv)


def run() -> None:
    """Console-script entrypoint (``uv run demo``)."""
    args = _parse_args()
    load_env(args.env)
    configure_logfire(with_variables=True)
    _print_banner(args)
    try:
        asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        print("\nStopping…", flush=True)


if __name__ == "__main__":
    run()
