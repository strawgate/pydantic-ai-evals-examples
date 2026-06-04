"""CodeMode wrapper that suppresses spurious error spans.

The default ``pydantic-ai-harness`` ``CodeModeToolset`` raises
``ModelRetry`` when sandbox code fails (`ModuleNotFoundError`, syntax
error, runtime exception, etc.). pydantic-ai then converts that to a
``ToolRetryError``, and the OTel instrumentation records the
``running tool`` span as ``is_exception=true, level=ERROR``. The agent
recovers fine — the model sees the retry prompt, writes new code — but
the Logfire UI fills with red ERROR spans for what is functionally
"the model typed something wrong, fixed it on the next turn".

This wrapper catches the ``ModelRetry`` inside ``call_tool`` and
returns the message as a normal tool-result string instead. Outcome:

- pydantic-ai sees a successful tool return → no exception recorded
  → the ``running tool`` span stays at OK status (no red flag)
- the model receives a ``ToolReturnPart`` whose content is the
  Python traceback ("Runtime error: …"), which it can read and react
  to exactly as it would have reacted to a retry prompt
- the agent's overall request_limit / usage_limits still bound any
  pathological retry loops

This is purely a presentation fix for traces; behaviour is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.code_mode._toolset import CodeModeToolset, ToolsetTool

if TYPE_CHECKING:
    from pydantic_ai import AbstractToolset


@dataclass
class QuietCodeModeToolset(CodeModeToolset[AgentDepsT]):
    """CodeModeToolset that converts ``ModelRetry`` into a returned string.

    Drop-in replacement for ``CodeModeToolset``. The model still sees the
    full traceback and retries naturally; the trace stops being littered
    with ERROR spans.
    """

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: Any,
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        try:
            return await super().call_tool(name, tool_args, ctx, tool)
        except ModelRetry as exc:
            # Convert the retry signal into a normal tool return value.
            # The model reads this as the tool's output, sees the
            # traceback, and writes corrected code on its next turn.
            return exc.message
        except BaseException as exc:
            # pyo3 PanicException is a BaseException, not Exception —
            # standard handlers miss it. Return its repr as a tool
            # result so the model can react.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            try:
                json.dumps(repr(exc))
            except Exception:
                pass
            return f"Sandbox failure ({type(exc).__name__}): {exc}"


@dataclass
class QuietCodeMode(CodeMode[AgentDepsT]):
    """CodeMode capability that uses :class:`QuietCodeModeToolset`."""

    def get_wrapper_toolset(
        self, toolset: AbstractToolset[AgentDepsT]
    ) -> AbstractToolset[AgentDepsT] | None:
        return QuietCodeModeToolset(
            wrapped=toolset,
            tool_selector=self.tools,
            max_retries=self.max_retries,
        )
