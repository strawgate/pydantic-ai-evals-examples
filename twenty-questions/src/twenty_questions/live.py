"""Live evaluation — run online cases through the agent with live evaluators."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_evals.online import OnlineEvaluator, wait_for_evaluations
from pydantic_evals.online_capability import OnlineEvaluation

from .agent import DEFAULT_SYSTEM_PROMPT, MANAGED_VAR_NAME, _parse_inputs, _system_prompt_var
from .datasets import load_dataset
from .evaluators import GuessingAccuracy, QuestionEfficiency, StrategyQuality
from .tools import GameState, get_tools

DATASET_DIR = Path(__file__).parent.parent.parent / "evals" / "datasets"


def build_live_agent(state: GameState, sample_rate: float = 1.0, system_prompt: str | None = None) -> Agent:
    """Build a guesser agent with live evaluators attached."""
    model = os.environ.get("MODEL", "claude-sonnet-4-20250514")
    toolset = get_tools(state)
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    online_evaluators = [
        OnlineEvaluator(evaluator=GuessingAccuracy(), sample_rate=sample_rate),
        OnlineEvaluator(evaluator=QuestionEfficiency(), sample_rate=sample_rate),
        OnlineEvaluator(evaluator=StrategyQuality(), sample_rate=sample_rate),
        OnlineEvaluator(evaluator=YesNoQuestionQuality(), sample_rate=sample_rate),
    ]

    return Agent(
        _build_model_instance(model),
        system_prompt=system_prompt,
        retries=2,
        toolsets=[toolset],
        model_settings={"max_tokens": 4096},
        capabilities=[OnlineEvaluation(evaluators=online_evaluators)],
    )


async def run_live(dataset_path: Path, sample_rate: float) -> None:
    """Load online cases and run them through the live agent."""
    dataset = load_dataset(dataset_path, mode="online")

    if not dataset.cases:
        print(f"No online cases found in {dataset_path.name}")
        return

    print(f"\n{'=' * 60}")
    print(f"  Live 20 Questions: {dataset_path.name}")
    print(f"  Online cases: {len(dataset.cases)}")
    print(f"  Sample rate: {sample_rate}")
    if MANAGED_VAR_NAME:
        print(f"  Managed variable: {MANAGED_VAR_NAME}")
    print(f"{'=' * 60}")

    # Resolve managed variable once for the session
    resolved_prompt = None
    var_ctx = None
    if _system_prompt_var is not None:
        var_ctx = _system_prompt_var.get()
        resolved = var_ctx.__enter__()
        resolved_prompt = resolved.value

    try:
        for case in dataset.cases:
            secret_item, hint = _parse_inputs(case.inputs)
            state = GameState(secret_item=secret_item, category_hint=hint)
            agent = build_live_agent(state, sample_rate=sample_rate, system_prompt=resolved_prompt)

            print(f"\n{'─' * 40}")
            print(f"  [{case.name}] Secret: {secret_item}")
            print(f"{'─' * 40}")

            result = await agent.run(hint)
            output = str(result.output)

            verdict = "✓ SOLVED" if state.solved else "✗ FAILED"
            print(f"  {verdict} in {len(state.questions_asked)} questions")
    finally:
        if var_ctx is not None:
            var_ctx.__exit__(None, None, None)

    await wait_for_evaluations()
    print(f"\n✓ All live evaluators completed. Check Logfire for results.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run 20 Questions with live evaluators")
    parser.add_argument(
        "--dataset", type=Path, default=DATASET_DIR / "twenty_questions.yaml",
    )
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()

    if os.path.exists(args.env):
        load_dotenv(args.env, override=True)

    # Logfire uses LOGFIRE_TOKEN for traces and LOGFIRE_API_KEY for variables/API access
    logfire_token = os.environ.get("LOGFIRE_TOKEN")
    if logfire_token:
        os.environ.setdefault("LOGFIRE_API_KEY", logfire_token)

    logfire.configure(
        service_name=os.environ.get("LOGFIRE_SERVICE_NAME", "twenty-questions-live"),
        send_to_logfire="if-token-present",
        variables=logfire.VariablesOptions() if os.environ.get("LOGFIRE_VAR_SYSTEM_PROMPT") else None,
    )
    logfire.instrument_pydantic_ai()

    await run_live(args.dataset, args.sample_rate)


if __name__ == "__main__":
    asyncio.run(main())
