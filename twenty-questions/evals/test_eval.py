"""Dataset eval tests for 20 Questions."""

from __future__ import annotations

from pathlib import Path

from pydantic_evals.evaluators import LLMJudge

from twenty_questions.agent import task
from twenty_questions.datasets import load_dataset
from twenty_questions.evaluators import GuessingAccuracy, QuestionEfficiency, StrategyQuality, YesNoQuestionQuality

DATASET_DIR = Path(__file__).parent / "datasets"


async def test_offline_twenty_questions() -> None:
    """Run offline cases — batch eval for training/scoring."""
    dataset = load_dataset(DATASET_DIR / "twenty_questions.yaml", mode="offline")

    dataset.evaluators = [
        GuessingAccuracy(),
        QuestionEfficiency(),
        StrategyQuality(),
        YesNoQuestionQuality(),
    ]

    report = await dataset.evaluate(task, max_concurrency=1)
    report.print(include_input=True, include_output=True)

    assert not report.failures, (
        f"{len(report.failures)} case(s) failed: "
        + ", ".join(f"{f.name}: {f.error_message}" for f in report.failures)
    )


async def test_online_twenty_questions() -> None:
    """Run online cases — production-like evaluation."""
    dataset = load_dataset(DATASET_DIR / "twenty_questions.yaml", mode="online")

    dataset.evaluators = [
        GuessingAccuracy(),
        QuestionEfficiency(),
        StrategyQuality(),
        YesNoQuestionQuality(),
    ]

    report = await dataset.evaluate(task, max_concurrency=1)
    report.print(include_input=True, include_output=True)

    assert not report.failures, (
        f"{len(report.failures)} case(s) failed: "
        + ", ".join(f"{f.name}: {f.error_message}" for f in report.failures)
    )


async def test_offline_with_judge() -> None:
    """Offline eval with LLM judge for strategy quality."""
    dataset = load_dataset(DATASET_DIR / "twenty_questions.yaml", mode="offline")

    dataset.evaluators = [
        GuessingAccuracy(),
        QuestionEfficiency(),
        StrategyQuality(),
        YesNoQuestionQuality(),
        LLMJudge(
            rubric=(
                "Evaluate the agent's 20 Questions strategy.\n"
                "Score 0.9-1.0 if: guessed correctly using efficient binary-search strategy.\n"
                "Score 0.6-0.8 if: guessed correctly but used too many questions or poor strategy.\n"
                "Score 0.3-0.5 if: reasonable strategy but failed to guess correctly.\n"
                "Score 0.0-0.2 if: random guessing, no strategy, or never attempted a guess."
            ),
            include_input=True,
        ),
    ]

    report = await dataset.evaluate(task, max_concurrency=1)
    report.print(include_input=True, include_output=True)
