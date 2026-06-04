"""Evaluators for 20 Questions games.

Evaluators assess:
1. Did the agent guess correctly?
2. How efficient was the strategy (number of questions)?
3. Was the questioning strategy logical (binary search vs random guessing)?
4. Were the questions proper yes/no questions?
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


def _parse_game_stats(output: str) -> dict[str, str]:
    """Extract GAME_STATS section from output."""
    stats = {}
    if "===GAME_STATS===" in output:
        stats_section = output.split("===GAME_STATS===", 1)[1].strip()
        for line in stats_section.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                stats[key.strip()] = value.strip()
    return stats


def _extract_questions(output: str) -> list[str]:
    """Extract the questions asked from the output narrative.

    Looks for patterns like "**Question?**" or numbered "1. Question?" in the output.
    Falls back to any sentence ending in '?' that appears before GAME_STATS.
    """
    # Only look at the narrative before game stats
    narrative = output.split("===GAME_STATS===")[0] if "===GAME_STATS===" in output else output

    questions = []

    # Pattern: bold questions like **Is it alive?** → Yes
    bold_qs = re.findall(r"\*\*([^*]+\?)\*\*", narrative)
    if bold_qs:
        questions.extend(bold_qs)
    else:
        # Fallback: any sentence ending in ? (but not rhetorical phrasing)
        for line in narrative.splitlines():
            line = line.strip()
            # Skip lines that are just formatting or context
            if "?" in line and not line.startswith("#"):
                # Extract the question portion
                match = re.search(r'["\']?([^"\']*\?)', line)
                if match:
                    questions.append(match.group(1).strip())

    return questions


def _is_yes_no_question(question: str) -> bool:
    """Heuristic: does this look like a yes/no question (not open-ended)?"""
    q_lower = question.lower().strip()

    # Yes/no question indicators
    yn_starters = [
        "is ", "are ", "was ", "were ", "do ", "does ", "did ",
        "can ", "could ", "would ", "will ", "has ", "have ", "had ",
        "should ", "might ", "may ",
    ]
    if any(q_lower.startswith(s) for s in yn_starters):
        return True

    # Also accept "Is it..." style with leading context stripped
    if re.match(r"^(it\s+)?is\s+", q_lower):
        return True

    # Open-ended starters are NOT yes/no
    open_starters = ["what ", "which ", "where ", "when ", "who ", "why ", "how "]
    if any(q_lower.startswith(s) for s in open_starters):
        return False

    # Questions containing "or" with alternatives are not pure yes/no
    if re.search(r"\bor\b", q_lower) and "?" in q_lower:
        return False

    # If it ends in ? and doesn't match open-ended, give benefit of doubt
    return q_lower.endswith("?")


@dataclass
class YesNoQuestionQuality(Evaluator[str, str]):
    """Check that the agent asks proper yes/no questions (not open-ended).

    Returns:
    - yes_no_ratio: fraction of questions that are valid yes/no format
    - all_yes_no: assertion that all questions were yes/no
    - question_quality_score: score (0.0–1.0)
    """

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> dict[str, bool | float]:
        output = str(ctx.output)
        questions = _extract_questions(output)

        if not questions:
            # No questions found — can't evaluate
            return {
                "all_yes_no": True,
                "yes_no_ratio": 1.0,
                "question_quality_score": 0.5,  # Neutral — no data
            }

        yn_count = sum(1 for q in questions if _is_yes_no_question(q))
        ratio = yn_count / len(questions)

        return {
            "all_yes_no": yn_count == len(questions),
            "yes_no_ratio": round(ratio, 4),
            "question_quality_score": round(ratio, 4),
        }


@dataclass
class GuessingAccuracy(Evaluator[str, str]):
    """Did the agent guess the secret item correctly?"""

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        output = str(ctx.output)
        stats = _parse_game_stats(output)

        if stats.get("result") == "Correct":
            return 1.0

        # Partial credit if close — check if the secret item appears in guesses
        secret = stats.get("secret_item", "").lower()
        if secret and secret in output.lower():
            return 0.5

        return 0.0


@dataclass
class QuestionEfficiency(Evaluator[str, str]):
    """Score based on how few questions were needed to solve."""

    max_questions: int = 20

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        output = str(ctx.output)
        stats = _parse_game_stats(output)

        if stats.get("result") != "Correct":
            return 0.0  # No efficiency credit for incorrect guesses

        questions = int(stats.get("questions_asked", self.max_questions))

        # Scoring curve: fewer questions = higher score
        if questions <= 5:
            return 1.0
        elif questions <= 8:
            return 0.9
        elif questions <= 12:
            return 0.75
        elif questions <= 15:
            return 0.55
        elif questions <= 20:
            return 0.35
        return 0.1


@dataclass
class StrategyQuality(Evaluator[str, str]):
    """Heuristic check for binary-search strategy vs random guessing.

    Looks for signs of good strategy (broad → narrow) vs bad strategy (random item guesses).
    """

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> dict[str, bool | float]:
        output = str(ctx.output)
        stats = _parse_game_stats(output)

        solved = stats.get("result") == "Correct"
        questions_asked = int(stats.get("questions_asked", 0))
        guesses_made = int(stats.get("guesses_made", 0))

        # Bad sign: too many guesses relative to questions (random guessing style)
        excessive_guesses = guesses_made > 3 and guesses_made > questions_asked * 0.3

        # Good sign: used questions before guessing
        used_questions_first = questions_asked >= 3 or solved

        # Score composition
        strategy_score = 0.0
        if solved:
            strategy_score += 0.5
        if used_questions_first:
            strategy_score += 0.25
        if not excessive_guesses:
            strategy_score += 0.25

        return {
            "solved": solved,
            "used_questions_first": used_questions_first,
            "no_excessive_guesses": not excessive_guesses,
            "strategy_score": strategy_score,
        }
