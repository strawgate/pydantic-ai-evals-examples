"""Evaluators for the data-science suite.

Two evaluators:

- ``EfficientExecution`` — gates on success; scores requests + tokens vs a
  tight budget on a sqrt curve. Failed runs score 0.
- ``DataScienceJudge`` — one LLM call returning five subscores
  (``correctness``, ``method``, ``completeness``, ``reasoning``, ``rigor``).
  ``overall`` is composed deterministically by the harness as
  ``0.5 × mean + 0.5 × min`` of the subscores so the judge cannot
  soft-pedal weak dimensions. The rubric in each case's
  ``expected_output`` is the reference the judge compares against.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

# ---------------------------------------------------------------------------
# Failure detection (used by EfficientExecution to gate scoring)
# ---------------------------------------------------------------------------

_FAILURE_PATTERNS = (
    "max turns",
    "max retries",
    "[agent error:",
    "i cannot complete",
    "i'm unable to complete",
    "unable to complete the task",
    "i was unable to",
)


def _looks_like_failure(text_lower: str) -> bool:
    return any(p in text_lower for p in _FAILURE_PATTERNS)


# ---------------------------------------------------------------------------
# EfficientExecution — efficiency, gated on task success
# ---------------------------------------------------------------------------


@dataclass
class EfficientExecution(Evaluator[str, str]):
    """Reward solving the task in few requests/tokens; zero out on failure.

    Tight defaults (10 requests, 60k tokens) so a competent run is rewarded
    and a wasteful one is visibly penalised. A failed run scores 0
    regardless — we don't reward giving up fast.
    """

    max_requests: int = 10
    max_tokens: int = 60_000

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> dict[str, bool | float]:
        requests = ctx.metrics.get("requests", 0)
        total_tokens = ctx.metrics.get("total_tokens", 0) or (
            ctx.metrics.get("input_tokens", 0) + ctx.metrics.get("output_tokens", 0)
        )
        failures = ctx.metrics.get("failures", 0)

        output = str(ctx.output)
        task_failed = _looks_like_failure(output.lower()) or len(output.strip()) < 3 or failures > 0

        if task_failed:
            return {
                "succeeded": False,
                "within_request_budget": requests <= self.max_requests,
                "within_token_budget": total_tokens <= self.max_tokens,
                "efficiency_score": 0.0,
            }

        req_ratio = min(requests / self.max_requests, 1.0) if requests > 0 else 0.0
        tok_ratio = min(total_tokens / self.max_tokens, 1.0) if total_tokens > 0 else 0.0
        request_score = max(0.0, (1.0 - req_ratio) ** 0.5)
        token_score = max(0.0, (1.0 - tok_ratio) ** 0.5)

        return {
            "succeeded": True,
            "within_request_budget": requests <= self.max_requests,
            "within_token_budget": total_tokens <= self.max_tokens,
            "efficiency_score": round((request_score + token_score) / 2, 4),
        }


# ---------------------------------------------------------------------------
# DataScienceJudge — multi-dimensional LLM scoring of DS quality
# ---------------------------------------------------------------------------


_JUDGE_SYSTEM_PROMPT = (
    "You are a senior data scientist grading another data scientist's work "
    "STRICTLY. Most submissions will NOT be at senior level — your job is to "
    "say so clearly. Anchor your scores low: a 0.5 means 'junior DS work, "
    "needs significant rework'; a 0.9 means 'I would ship this to a stakeholder'.\n"
    "\n"
    "You will be shown the task, a rubric (in 'expected_output'), and the "
    "agent's report. Score across five dimensions on a 0.0–1.0 scale.\n"
    "\n"
    "DIMENSIONS:\n"
    "- correctness: Are the numbers right? Suspiciously round / hardcoded / "
    "implausible numbers cap this at 0.5. Internal-consistency failures (totals "
    "don't add up, percentages don't sum to 100) cap at 0.4. If the agent "
    "asserts a number but shows no derivation, treat with skepticism.\n"
    "- method: Right statistic for the question. Right granularity (per-group "
    "vs aggregate as the task demands). Correct joins, no obvious leakage or "
    "double-counting. Wrong-method-but-correct-output still caps at 0.6.\n"
    "- completeness: Every numbered step in the task is addressed with "
    "evidence. 4-of-5 steps done caps at 0.7. Missing the follow-up query "
    "where the task explicitly asks for one caps at 0.6.\n"
    "- reasoning: Interpreted findings, not just dumped numbers. Specific "
    "actionable recommendations when asked (generic 'send an email' caps "
    "this at 0.4). Identified what's surprising vs expected. Flagged "
    "uncertainty or limitations where relevant.\n"
    "- rigor: The senior-vs-junior gap. Did the agent validate assumptions "
    "before computing (check for nulls / outliers / sample size adequacy)? "
    "Show intermediate results, not just final numbers? Identify and address "
    "the EXPLICIT traps listed in the rubric (e.g. excluding the orphan "
    "customer, marking non-observable cohort cells as N/A, not flagging the "
    "1-cent rounding anomaly as a real anomaly)? Without these, score 0.3–0.5; "
    "with most of them, 0.7–0.85; with all of them plus extra sanity checks, "
    "0.9+. Most baseline submissions score 0.3–0.5 here.\n"
    "\n"
    "STRICT RULES:\n"
    "- Output ONLY a single JSON object. No markdown, no prose, no code fences.\n"
    "- Keys: correctness, method, completeness, reasoning, rigor, notes.\n"
    "- DO NOT include 'overall' — the harness computes it deterministically "
    "from your subscores so you can't soft-pedal weaknesses.\n"
    "- 'notes' is one sentence (≤ 240 chars) naming the BIGGEST missing rigor "
    "item (the thing that would most move the score if fixed).\n"
    "\n"
    "SCORING ANCHORS (apply per dimension):\n"
    "- 1.0: exceptional — exceeds what I'd produce myself.\n"
    "- 0.9: ship to stakeholder; minor polish only.\n"
    "- 0.75: solid but needs one round of revision.\n"
    "- 0.5: addresses the task but missing rigor / has gaps; junior work.\n"
    "- 0.3: significant errors or omissions; would not ship.\n"
    "- 0.1: fundamentally wrong or did not engage with the task.\n"
    "- 0.0: did not attempt this dimension at all.\n"
    "\n"
    "Be honest. Most agent submissions on first attempt score 0.4–0.6 overall "
    "after composition. That's correct calibration. Do not inflate scores."
)


def _compose_overall(subscores: dict[str, float]) -> float:
    """Compose the overall score from sub-dimension scores.

    ``0.5 × mean + 0.5 × min`` — a weak dimension drags the score down half
    as much as it would in pure min(), but still meaningfully. This stops
    a strong correctness from masking a tanked rigor / reasoning.
    """
    vals = [v for v in subscores.values() if isinstance(v, int | float)]
    if not vals:
        return 0.0
    return round(0.5 * (sum(vals) / len(vals)) + 0.5 * min(vals), 4)


# Cap on how much of the agent's output we paste into the judge prompt.
# A senior reviewer doesn't need 30 KB of report to score it.
_MAX_AGENT_OUTPUT_CHARS = 8_000

# Judge token budget. Sonnet emits thinking tokens before text, so the cap
# needs headroom over the JSON itself. 4096 handles the cohort case (long
# matrix output) reliably.
_JUDGE_MAX_TOKENS = 4096


def _truncate_for_judge(output: str) -> str:
    """Truncate huge agent reports so the judge can read input + emit JSON.

    Keeps the head and tail of the report — the head usually has the framing
    and the tail usually has the conclusion/recommendation, both important
    for scoring.
    """
    if len(output) <= _MAX_AGENT_OUTPUT_CHARS:
        return output
    head_chars = int(_MAX_AGENT_OUTPUT_CHARS * 0.6)
    tail_chars = _MAX_AGENT_OUTPUT_CHARS - head_chars
    head = output[:head_chars]
    tail = output[-tail_chars:]
    omitted = len(output) - head_chars - tail_chars
    return f"{head}\n\n...[{omitted} chars omitted]...\n\n{tail}"


def _judge_call(
    *,
    inputs: str,
    expected: str,
    output: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    """One synchronous Anthropic call returning the parsed judge JSON.

    Uses the same MINIMAX_*/ANTHROPIC_* env as the agent so the proxy is
    honoured. Falls back gracefully if the response is unparseable.
    """
    api_key = os.environ.get("MINIMAX_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = (
        os.environ.get("MINIMAX_BASE_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or "https://api.anthropic.com"
    )

    user_msg = (
        f"TASK:\n{inputs}\n\n"
        f"RUBRIC (expected_output):\n{expected or '(no rubric — judge on general DS quality)'}\n\n"
        f"AGENT REPORT:\n{_truncate_for_judge(output)}\n\n"
        "Return the JSON object now."
    )

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{base_url.rstrip('/')}/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": _JUDGE_MAX_TOKENS,
                "temperature": 0,
                "system": _JUDGE_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_msg}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text_blocks = [b for b in data.get("content", []) if b.get("type") == "text"]
        if not text_blocks:
            raise ValueError(f"Judge returned no text. stop_reason={data.get('stop_reason')!r}")
        raw = text_blocks[0]["text"].strip()

    # Strip any stray ```json fences just in case.
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Judge returned non-object JSON: {type(parsed).__name__}")
    return parsed


@dataclass
class DataScienceJudge(Evaluator[str, str]):
    """LLM-as-judge for data-science quality with strict anchoring.

    One LLM call per case. Subscores: ``correctness``, ``method``,
    ``completeness``, ``reasoning``, ``rigor`` (all 0.0–1.0). The
    ``overall`` score is computed deterministically from those subscores
    as ``0.5 × mean + 0.5 × min`` — this stops a strong dimension from
    masking a weak one, which is what happens when the judge picks its
    own "overall".

    The ``rigor`` dimension is the senior-vs-junior gap: did the agent
    validate assumptions, show intermediate results, handle the embedded
    traps from the seed, flag uncertainty? Baseline runs typically tank
    this dimension (0.3–0.5), which is where the optimizer earns its
    keep.
    """

    model: str = "claude-sonnet-4-6"
    # Sonnet may emit thinking tokens for long reports — 60 s was hitting
    # ReadTimeout on the cohort case's matrix output. 180 s is comfortable.
    timeout: float = 180.0

    _DIMENSIONS = ("correctness", "method", "completeness", "reasoning", "rigor")

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> dict[str, float | str]:
        inputs = str(ctx.inputs)
        expected = str(ctx.expected_output) if ctx.expected_output is not None else ""
        output = str(ctx.output)

        # Don't waste a judge call on an obvious failure — score 0 directly.
        if _looks_like_failure(output.lower()) or len(output.strip()) < 20:
            zeros = dict.fromkeys(self._DIMENSIONS, 0.0)
            return {
                **zeros,
                "overall": 0.0,
                "judge_notes": "Output was empty or a hard failure; skipped judge call.",
            }

        try:
            parsed = _judge_call(
                inputs=inputs,
                expected=expected,
                output=output,
                model=self.model,
                timeout=self.timeout,
            )
        except Exception as e:
            # Judge failed — return neutral 0.5 so we don't crash the eval,
            # but flag it loudly so a human notices.
            neutrals = dict.fromkeys(self._DIMENSIONS, 0.5)
            return {
                **neutrals,
                "overall": 0.5,
                "judge_notes": f"Judge call failed: {type(e).__name__}: {e}",
            }

        def _f(key: str, default: float = 0.0) -> float:
            v = parsed.get(key, default)
            try:
                return float(max(0.0, min(1.0, float(v))))
            except (TypeError, ValueError):
                return default

        subscores = {dim: _f(dim) for dim in self._DIMENSIONS}
        overall = _compose_overall(subscores)

        return {
            **subscores,
            "overall": overall,
            "judge_notes": str(parsed.get("notes", ""))[:240],
        }
