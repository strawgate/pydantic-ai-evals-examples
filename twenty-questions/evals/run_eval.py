"""Standalone eval runner for 20 Questions.

Usage:
    uv run python evals/run_eval.py
    uv run python evals/run_eval.py --mode online
    uv run python evals/run_eval.py --mode offline --judge
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import logfire
from dotenv import load_dotenv
from pydantic_evals.evaluators import LLMJudge

from twenty_questions.agent import task
from twenty_questions.datasets import load_dataset
from twenty_questions.evaluators import GuessingAccuracy, QuestionEfficiency, StrategyQuality, YesNoQuestionQuality

DATASET_DIR = Path(__file__).parent / "datasets"


def setup(env_file: str = ".env") -> None:
    """Load env and configure Logfire."""
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)

    minimax_key = os.environ.get("MINIMAX_API_KEY")
    if minimax_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", minimax_key)
    minimax_url = os.environ.get("MINIMAX_BASE_URL")
    if minimax_url:
        os.environ.setdefault("ANTHROPIC_BASE_URL", minimax_url)

    # Logfire uses LOGFIRE_TOKEN for traces and LOGFIRE_API_KEY for variables/API access
    logfire_token = os.environ.get("LOGFIRE_TOKEN")
    if logfire_token:
        os.environ.setdefault("LOGFIRE_API_KEY", logfire_token)

    logfire.configure(
        service_name=os.environ.get("LOGFIRE_SERVICE_NAME", "twenty-questions-evals"),
        send_to_logfire="if-token-present",
        variables=logfire.VariablesOptions() if os.environ.get("LOGFIRE_VAR_SYSTEM_PROMPT") else None,
    )
    logfire.instrument_pydantic_ai()


async def run(dataset_path: Path, mode: str | None, use_judge: bool) -> None:
    """Load dataset, filter by mode, run, and print results."""
    dataset = load_dataset(dataset_path, mode=mode)  # type: ignore[arg-type]

    if not dataset.cases:
        print(f"No cases found for mode={mode!r} in {dataset_path.name}")
        return

    evaluators = [
        GuessingAccuracy(),
        QuestionEfficiency(),
        StrategyQuality(),
        YesNoQuestionQuality(),
    ]

    if use_judge:
        evaluators.append(
            LLMJudge(
                rubric=(
                    "Evaluate the agent's 20 Questions strategy.\n"
                    "Score 0.9-1.0: correct guess, efficient binary-search.\n"
                    "Score 0.6-0.8: correct but inefficient.\n"
                    "Score 0.3-0.5: reasonable strategy but wrong guess.\n"
                    "Score 0.0-0.2: random guessing, no strategy."
                ),
                include_input=True,
            )
        )

    dataset.evaluators = evaluators

    print(f"\n{'=' * 60}")
    print(f"  20 Questions Eval: {dataset_path.name}")
    print(f"  Mode: {mode or 'all'}")
    print(f"  Cases: {len(dataset.cases)}")
    print(f"  LLM Judge: {'yes' if use_judge else 'no'}")
    print(f"{'=' * 60}\n")

    # Sequential — each game involves many LLM calls
    report = await dataset.evaluate(task, max_concurrency=1)
    report.print(include_input=True, include_output=True)

    if report.failures:
        print(f"\n⚠ {len(report.failures)} case(s) raised exceptions")
        for f in report.failures:
            print(f"  - {f.name}: {f.error_message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 20 Questions evals")
    parser.add_argument(
        "--dataset", type=Path, default=DATASET_DIR / "twenty_questions.yaml",
    )
    parser.add_argument(
        "--mode", choices=["online", "offline"], default="offline",
    )
    parser.add_argument("--all", action="store_true", dest="all_cases")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()

    setup(args.env)
    mode = None if args.all_cases else args.mode
    asyncio.run(run(args.dataset, mode, args.judge))


if __name__ == "__main__":
    main()
