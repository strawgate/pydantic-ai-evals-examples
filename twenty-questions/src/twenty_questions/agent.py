"""Agent definition for the 20 Questions guesser.

The agent's job is to guess the secret item using as few questions as possible.
It has access to ask_question() and make_guess() tools.
"""

from __future__ import annotations

import os

from pydantic_ai import Agent

from .tools import GameState, _build_model_instance, get_tools

MODEL = os.environ.get("MODEL", "claude-sonnet-4-20250514")

DEFAULT_SYSTEM_PROMPT = (
    "You are playing 20 Questions. Your goal is to identify the secret item "
    "by asking strategic yes/no questions.\n\n"
    "STRATEGY:\n"
    "- Use a binary-search approach: start with broad category questions to "
    "eliminate large groups, then narrow down based on answers.\n"
    "- Never repeat information you already know.\n"
    "- Don't waste questions on things already ruled out.\n"
    "- When confident (>80% sure), make your guess.\n\n"
    "GOOD: 'Is it alive?' → 'Is it an animal?' → 'Does it live on land?' → ...\n"
    "BAD: 'Is it a dog?' → 'Is it a cat?' → 'Is it a bird?' (random guessing)"
)

# If set, use a Logfire managed variable for the system prompt.
# e.g. LOGFIRE_VAR_SYSTEM_PROMPT=system_prompt_twenty_questions
MANAGED_VAR_NAME = os.environ.get("LOGFIRE_VAR_SYSTEM_PROMPT")


def _get_managed_var():
    """Lazily create the managed variable handle (only if configured)."""
    if not MANAGED_VAR_NAME:
        return None
    import logfire

    return logfire.var(
        MANAGED_VAR_NAME,
        default=DEFAULT_SYSTEM_PROMPT,
        description="System prompt for the 20 Questions guesser agent.",
    )


_system_prompt_var = _get_managed_var()


def build_agent(state: GameState, system_prompt: str | None = None) -> Agent:
    """Build a guesser agent for a specific game session."""
    toolset = get_tools(state)
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    return Agent(
        _build_model_instance(os.environ.get("MODEL", MODEL)),
        system_prompt=system_prompt,
        retries=2,
        toolsets=[toolset],
        model_settings={"max_tokens": 4096},
    )


async def task(inputs: str) -> str:
    """Task function for dataset evals.

    Parses SECRET and HINT from inputs, runs the guesser agent,
    and returns the result with game stats appended.

    If LOGFIRE_VAR_SYSTEM_PROMPT is set, resolves the prompt from
    the managed variable for Logfire optimizer correlation.
    """
    secret_item, hint = _parse_inputs(inputs)
    state = GameState(secret_item=secret_item, category_hint=hint)

    if _system_prompt_var is not None:
        with _system_prompt_var.get() as resolved:
            agent = build_agent(state, system_prompt=resolved.value)
            result = await agent.run(hint)
    else:
        agent = build_agent(state)
        result = await agent.run(hint)

    output = str(result.output)

    # Append game stats for evaluators
    verdict = "Correct" if state.solved else "Incorrect"
    stats = (
        f"\n\n===GAME_STATS===\n"
        f"result: {verdict}\n"
        f"questions_asked: {len(state.questions_asked)}\n"
        f"guesses_made: {len(state.guesses_made)}\n"
        f"secret_item: {state.secret_item}\n"
    )
    return output + stats


def _parse_inputs(inputs: str) -> tuple[str, str]:
    """Parse SECRET: and HINT: from the inputs string."""
    secret_item = ""
    hint_lines = []
    for line in inputs.strip().splitlines():
        if line.startswith("SECRET:"):
            secret_item = line[len("SECRET:"):].strip()
        elif line.startswith("HINT:"):
            hint_lines.append(line[len("HINT:"):].strip())
        else:
            hint_lines.append(line)

    if not secret_item:
        raise ValueError(f"No SECRET found in inputs:\n{inputs}")

    hint = "\n".join(hint_lines).strip()
    return secret_item, hint
