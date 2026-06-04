"""Live (online) evaluation — run online cases with background evaluators.

The agent runs with the ``OnlineEvaluation`` capability attached; evaluators
fire in the background after each ``agent.run()``, emitting OTel events to
Logfire. This mimics production: real requests with background scoring at
a configurable sample rate.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path
from typing import Any

import httpx
import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.toolsets import FunctionToolset
from pydantic_evals import increment_eval_metric
from pydantic_evals.online import OnlineEvaluator, wait_for_evaluations
from pydantic_evals.online_capability import OnlineEvaluation

from .agent import (
    CODEMODE_PRELUDE,
    DEFAULT_SYSTEM_PROMPT,
    AgentOutput,
    _get_managed_var,
    _managed_var_name,
    _model_name,
)
from .datasets import load_dataset
from .evaluators import DataScienceJudge, EfficientExecution
from .quiet_code_mode import QuietCodeMode as CodeMode
from .tools import Database, make_sql_tools, seed_database

DATASET_DIR = Path(__file__).parent.parent.parent / "evals" / "datasets"

_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def _build_model() -> AnthropicModel:
    api_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("MINIMAX_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")
    provider = AnthropicProvider(
        api_key=api_key,
        base_url=base_url or None,
        http_client=httpx.AsyncClient(timeout=_HTTP_TIMEOUT),
    )
    return AnthropicModel(_model_name(), provider=provider)


def build_live_agent(tools: dict[str, Any], sample_rate: float, system_prompt: str) -> Agent:
    """Build a CodeMode agent with online evaluators wired in."""
    toolset: FunctionToolset = FunctionToolset()
    for func in tools.values():
        toolset.tool_plain(func)

    online_evaluators = [
        OnlineEvaluator(
            evaluator=EfficientExecution(max_requests=10, max_tokens=60_000),
            sample_rate=sample_rate,
        ),
        # LLM judge sampled lower by default — each call costs API tokens.
        OnlineEvaluator(
            evaluator=DataScienceJudge(model=_model_name()),
            sample_rate=min(sample_rate, 0.25),
        ),
    ]

    return Agent(
        _build_model(),
        name="monty_data_science_agent_live",
        # Sandbox-mechanics prelude is hardcoded and concatenated with the
        # managed (optimizer-evolved) data-science prompt.
        instructions=f"{CODEMODE_PRELUDE}\n\n{system_prompt}",
        output_type=AgentOutput,
        retries=3,
        toolsets=[toolset],
        model_settings={"max_tokens": 16384},
        capabilities=[
            CodeMode(max_retries=30),
            OnlineEvaluation(evaluators=online_evaluators),
        ],
    )


async def _run_case(case, sample_rate: float, system_prompt: str) -> None:
    db = Database()
    seed_database(db)
    tools = make_sql_tools(db)
    agent = build_live_agent(tools, sample_rate=sample_rate, system_prompt=system_prompt)

    start = time.perf_counter()
    result = await agent.run(case.inputs)
    usage = result.usage
    increment_eval_metric("input_tokens", usage.input_tokens or 0)
    increment_eval_metric("output_tokens", usage.output_tokens or 0)
    increment_eval_metric("total_tokens", usage.total_tokens or 0)
    increment_eval_metric("requests", usage.requests or 0)
    increment_eval_metric("elapsed_seconds", time.perf_counter() - start)

    output = result.output
    if isinstance(output, AgentOutput):
        preview = output.result[:200]
    else:
        preview = str(output)[:200]
    print(f"  Output: {preview}")


async def run_live(dataset_path: Path, sample_rate: float) -> None:
    dataset = load_dataset(dataset_path, mode="online")
    if not dataset.cases:
        print(f"No online cases found in {dataset_path.name}")
        return

    print(f"\n{'=' * 60}")
    print(f"  Live data-science eval: {dataset_path.name}")
    print(f"  Online cases: {len(dataset.cases)}")
    print(f"  Sample rate: {sample_rate}")
    var_name = _managed_var_name()
    if var_name:
        print(f"  Managed variable: {var_name}")
    print(f"{'=' * 60}")

    # Resolve the managed variable (if configured) for the whole session.
    var = _get_managed_var()
    if var is not None:
        with var.get() as resolved:
            prompt = resolved.value
            for case in dataset.cases:
                print(f"\n{'─' * 40}")
                print(f"  [{case.name}]")
                print(f"{'─' * 40}")
                await _run_case(case, sample_rate, prompt)
    else:
        for case in dataset.cases:
            print(f"\n{'─' * 40}")
            print(f"  [{case.name}]")
            print(f"{'─' * 40}")
            await _run_case(case, sample_rate, DEFAULT_SYSTEM_PROMPT)

    await wait_for_evaluations()
    print(f"\nDone. {len(dataset.cases)} online case(s) evaluated. See Logfire for results.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run online cases with live evaluators")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR / "small.yaml")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()

    if os.path.exists(args.env):
        load_dotenv(args.env, override=True)

    # Mirror MiniMax credentials onto ANTHROPIC_* so LLMJudge picks them up too.
    minimax_key = os.environ.get("MINIMAX_API_KEY")
    if minimax_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", minimax_key)
    minimax_url = os.environ.get("MINIMAX_BASE_URL")
    if minimax_url:
        os.environ.setdefault("ANTHROPIC_BASE_URL", minimax_url)

    logfire_token = os.environ.get("LOGFIRE_TOKEN")
    if logfire_token:
        os.environ.setdefault("LOGFIRE_API_KEY", logfire_token)

    logfire.configure(
        service_name=os.environ.get("LOGFIRE_SERVICE_NAME", "monty-data-science-live"),
        send_to_logfire="if-token-present",
        variables=logfire.VariablesOptions()
        if os.environ.get("LOGFIRE_VAR_SYSTEM_PROMPT")
        else None,
    )
    logfire.instrument_pydantic_ai()

    await run_live(args.dataset, args.sample_rate)


if __name__ == "__main__":
    asyncio.run(main())
