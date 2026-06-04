"""Twenty Questions game tools.

The guesser agent uses ask_question() and make_guess() tools.
The answerer is a pydantic-ai Agent that knows the secret and responds
with a structured Answer enum — richer signal than plain yes/no strings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

# ---------------------------------------------------------------------------
# Answer enum — structured output from the answerer agent
# ---------------------------------------------------------------------------


class Answer(StrEnum):
    yes = "yes"
    kind_of = "kind of"
    not_really = "not really"
    no = "no"
    completely_wrong = "completely wrong"


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------


@dataclass
class GameState:
    """State for a single 20 questions game session."""

    secret_item: str
    category_hint: str
    questions_asked: list[tuple[str, str]] = field(default_factory=list)
    guesses_made: list[tuple[str, bool]] = field(default_factory=list)
    solved: bool = False
    max_questions: int = 20


# ---------------------------------------------------------------------------
# Shared model builder — used by both the answerer and questioner agents
# ---------------------------------------------------------------------------

# Answerer can use a cheaper/faster model; default to same as questioner
ANSWERER_MODEL = os.environ.get("ANSWERER_MODEL", os.environ.get("MODEL", "claude-sonnet-4-20250514"))


def _build_model_instance(model_name: str):
    """Build a pydantic-ai model instance.

    Uses MiniMax/Anthropic proxy if MINIMAX_API_KEY and MINIMAX_BASE_URL
    are set, otherwise falls back to the default Anthropic provider.
    """
    api_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("MINIMAX_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")

    if api_key and base_url:
        import httpx
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)),
        )
        return AnthropicModel(model_name, provider=provider)

    return f"anthropic:{model_name}"


# ---------------------------------------------------------------------------
# Answerer agent factory
# ---------------------------------------------------------------------------


def build_answerer_agent(secret_item: str) -> Agent:
    """Build an answerer agent that knows the secret and responds truthfully.

    The answerer returns a structured Answer enum so the questioner gets
    richer signal than plain "Yes"/"No" strings.
    """
    return Agent(
        _build_model_instance(ANSWERER_MODEL),
        instructions=(
            f"You are the answerer in a game of 20 Questions. "
            f"The secret item is: '{secret_item}'.\n\n"
            "RULES:\n"
            "- Answer truthfully about the secret item based on the question.\n"
            "- Choose the most accurate answer:\n"
            "  * 'yes' — clearly true\n"
            "  * 'kind of' — partially true or context-dependent\n"
            "  * 'not really' — mostly false but with some nuance\n"
            "  * 'no' — clearly false\n"
            "  * 'completely wrong' — the question reveals a fundamentally wrong assumption\n"
            "- Do NOT reveal the secret item directly.\n"
            "- Do NOT give additional hints beyond your answer."
        ),
        output_type=Answer,
        name="answerer_agent",
    )


# ---------------------------------------------------------------------------
# Toolset factory
# ---------------------------------------------------------------------------


def get_tools(state: GameState) -> FunctionToolset:
    """Build the toolset for a 20 questions game with the given state."""
    answerer = build_answerer_agent(state.secret_item)
    toolset = FunctionToolset()

    @toolset.tool
    async def ask_question(question: str) -> str:
        """Ask a yes/no question about the secret item.

        Returns one of: yes, kind of, not really, no, completely wrong.
        You have limited questions, so ask strategically — use a binary-search
        style to narrow down possibilities efficiently.
        """
        if state.solved:
            return "Game over — you already guessed correctly!"
        if len(state.questions_asked) >= state.max_questions:
            return (
                f"No questions remaining! You've used all {state.max_questions}. "
                "Use make_guess now."
            )

        result = await answerer.run(question)
        answer = str(result.output)
        state.questions_asked.append((question, answer))
        remaining = state.max_questions - len(state.questions_asked)
        return f"{answer} ({remaining} questions remaining)"

    @toolset.tool
    def make_guess(guess: str) -> str:
        """Submit your guess for the secret item.

        You can guess at any time. If correct, you win!
        If wrong, you can keep asking questions if you have any remaining.
        """
        if state.solved:
            return "Game already won! You correctly guessed earlier."

        def normalize(s: str) -> str:
            s = s.lower().strip()
            for prefix in ("a ", "an ", "the "):
                if s.startswith(prefix):
                    s = s[len(prefix):]
            return s

        is_correct = normalize(guess) == normalize(state.secret_item)
        state.guesses_made.append((guess, is_correct))

        if is_correct:
            state.solved = True
            return f"Correct! The answer is: {state.secret_item}"
        else:
            remaining = state.max_questions - len(state.questions_asked)
            if remaining > 0:
                return f"Incorrect. You have {remaining} questions remaining. Keep trying!"
            return f"Incorrect. No questions remaining. The answer was: {state.secret_item}"

    return toolset

