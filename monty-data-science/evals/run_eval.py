"""Standalone eval runner — run outside of pytest.

Defaults:
- LLM judge ON (the primary signal). Use --no-judge for cheap smoke runs.
- Concurrency 1 (CodeMode runs are heavy).
- Dataset evals/datasets/small.yaml.

Examples:
    uv run python evals/run_eval.py
    uv run python evals/run_eval.py --dataset evals/datasets/medium.yaml
    uv run python evals/run_eval.py --dataset evals/datasets/large.yaml --concurrency 2
    uv run python evals/run_eval.py --no-judge --dataset evals/datasets/small.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import logfire
from dotenv import load_dotenv
from pydantic_evals.evaluators import Evaluator

from monty_data_science.agent import (
    CODEMODE_PRELUDE,
    DEFAULT_SYSTEM_PROMPT,
    MODEL,
    get_current_prompt,
    task,
)
from monty_data_science.datasets import Mode, load_dataset
from monty_data_science.evaluators import DataScienceJudge, EfficientExecution
from monty_data_science.logfire_links import print_logfire_links

DATASET_DIR = Path(__file__).parent / "datasets"


def setup(env_file: str) -> None:
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)

    minimax_key = os.environ.get("MINIMAX_API_KEY")
    if minimax_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", minimax_key)
    minimax_url = os.environ.get("MINIMAX_BASE_URL")
    if minimax_url:
        os.environ.setdefault("ANTHROPIC_BASE_URL", minimax_url)

    # Logfire managed-variables API requires LOGFIRE_API_KEY (separate from
    # LOGFIRE_TOKEN which is for traces). Mirror them when only one is set.
    logfire_token = os.environ.get("LOGFIRE_TOKEN")
    if logfire_token:
        os.environ.setdefault("LOGFIRE_API_KEY", logfire_token)

    variables = logfire.VariablesOptions() if os.environ.get("LOGFIRE_VAR_SYSTEM_PROMPT") else None
    logfire.configure(
        service_name=os.environ.get("LOGFIRE_SERVICE_NAME", "monty-data-science-evals"),
        send_to_logfire="if-token-present",
        variables=variables,
    )
    logfire.instrument_pydantic_ai()


def _print_active_prompt() -> None:
    """Print both layers of the active system prompt for this run.

    Layer 1 (hardcoded ``CODEMODE_PRELUDE``) is the sandbox-mechanics
    layer — not visible to the optimizer.
    Layer 2 (managed variable value) is the data-science portion the
    optimizer evolves.

    Flushes stdout so the banner survives redirected-stdout block buffering.
    """
    info = get_current_prompt()
    lines = ["─── Active system prompt ──────────────────────────────────"]
    lines.append("  Layer 1: CODEMODE_PRELUDE (hardcoded, optimizer-blind)")
    lines.append(f"           {len(CODEMODE_PRELUDE):,} chars of sandbox rules")
    lines.append("  Layer 2: data-science prompt")
    if info is None:
        lines.append("           source: DEFAULT_SYSTEM_PROMPT (managed variable not set)")
        body = DEFAULT_SYSTEM_PROMPT
    else:
        value, version, label = info
        lines.append(f"           source: Logfire managed variable (v{version}, label={label})")
        body = value
    lines.append("  ─")
    for line in body.splitlines() or [body]:
        lines.append(f"  {line}")
    lines.append("───────────────────────────────────────────────────────────\n")
    print("\n".join(lines), flush=True)


async def run(dataset_path: Path, mode: Mode | None, concurrency: int, use_judge: bool) -> None:
    dataset = load_dataset(dataset_path, mode=mode)
    if not dataset.cases:
        print(f"No cases found for mode={mode!r} in {dataset_path.name}")
        return

    evaluators: list[Evaluator[str, str]] = [
        EfficientExecution(max_requests=10, max_tokens=60_000),
    ]
    if use_judge:
        evaluators.append(DataScienceJudge(model=MODEL))

    dataset.evaluators = evaluators

    print(f"\n{'=' * 60}")
    print(f"  Data-science eval: {dataset_path.name}")
    print(f"  Mode: {mode or 'all'}")
    print(f"  Cases: {len(dataset.cases)}  |  Concurrency: {concurrency}  |  Judge: {use_judge}")
    print(f"{'=' * 60}")
    print_logfire_links()
    _print_active_prompt()

    report = await dataset.evaluate(task, max_concurrency=concurrency)
    report.print(include_input=True, include_output=True)

    if report.failures:
        print(f"\n{len(report.failures)} case(s) raised exceptions:")
        for f in report.failures:
            print(f"  - {f.name}: {f.error_message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data-science evals")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_DIR / "small.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default="offline",
        help="Which cases to run (default: offline)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_cases",
        help="Run all cases regardless of mode",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("MAX_CONCURRENCY", "1")),
        help="Max concurrent eval cases (default: 1 — DS runs are heavy)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM judge (cheap smoke run; the judge is ON by default).",
    )
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()

    setup(args.env)
    # argparse `choices=` narrows the runtime value, but the static type is
    # still `str`. Convert explicitly so ty is happy.
    mode: Mode | None = None if args.all_cases else args.mode
    asyncio.run(run(args.dataset, mode, args.concurrency, use_judge=not args.no_judge))


if __name__ == "__main__":
    main()
