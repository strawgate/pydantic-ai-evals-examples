"""Dataset eval tests — pytest entrypoints.

Usage:
    uv run pytest evals/ -v
    uv run pytest evals/test_eval.py -k offline_small -v
"""

from __future__ import annotations

from pathlib import Path

from pydantic_evals.evaluators import Evaluator

from monty_data_science.agent import MODEL, task
from monty_data_science.datasets import load_dataset
from monty_data_science.evaluators import DataScienceJudge, EfficientExecution

DATASET_DIR = Path(__file__).parent / "datasets"


def _evaluators(judge: bool = True) -> list[Evaluator[str, str]]:
    evs: list[Evaluator[str, str]] = [EfficientExecution(max_requests=10, max_tokens=60_000)]
    if judge:
        evs.append(DataScienceJudge(model=MODEL))
    return evs


async def test_offline_small_with_judge() -> None:
    """One multi-step case with LLM judge — quick correctness smoke."""
    dataset = load_dataset(DATASET_DIR / "small.yaml", mode="offline")
    dataset.evaluators = _evaluators(judge=True)

    report = await dataset.evaluate(task, max_concurrency=1)
    report.print(include_input=True, include_output=True)

    assert not report.failures, f"{len(report.failures)} case(s) raised exceptions: " + ", ".join(
        f"{f.name}: {f.error_message}" for f in report.failures
    )


async def test_offline_medium_with_judge() -> None:
    """Full medium offline suite with LLM judge."""
    dataset = load_dataset(DATASET_DIR / "medium.yaml", mode="offline")
    dataset.evaluators = _evaluators(judge=True)

    report = await dataset.evaluate(task, max_concurrency=1)
    report.print(include_input=True, include_output=True)

    assert not report.failures, f"{len(report.failures)} case(s) raised exceptions: " + ", ".join(
        f"{f.name}: {f.error_message}" for f in report.failures
    )
