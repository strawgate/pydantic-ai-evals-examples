"""Shared configuration for the continuous-improvement demo.

Centralises the few names the demo turns on so the traffic runner
(``demo.py``) and the variable seeder (``seed_demo_variable.py``) can't drift
apart: the managed-variable name, the agent name shown on the Agents page, the
model string, and how we find/load the ``.env`` and configure Logfire.
"""

from __future__ import annotations

import os
from pathlib import Path

import logfire
from dotenv import load_dotenv

# The managed variable the optimizer evolves. This is what you open the
# "Optimize" tab on. Both the runner and the seeder reference this name.
VARIABLE_NAME = "data_science_agent_prompt"

# The agent name that shows up on the Logfire Agents page. Stable on purpose:
# the before/after charts are grouped by this name.
AGENT_NAME = "monty_data_science_agent"

# Service name for the traces.
DEFAULT_SERVICE_NAME = "monty-data-science"

# Model the agent runs on, via the Pydantic AI Gateway (reads
# PYDANTIC_AI_GATEWAY_API_KEY). Override with MODEL in the env.
DEFAULT_MODEL = "gateway/anthropic:claude-sonnet-4-5"

_PKG_DIR = Path(__file__).resolve().parent  # .../src/monty_data_science
_SRC_DIR = _PKG_DIR.parent  # .../src
_PROJECT_ROOT = _SRC_DIR.parent  # repo root


def find_env_file(explicit: str | None = None) -> Path | None:
    """Locate the ``.env`` file, trying the most likely places in order.

    Order: an explicit path, the ``ENV_FILE`` env var, then ``.env`` in the
    current dir, the project root, and ``src/`` (where the keys currently
    live). Returns the first that exists, or ``None``.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("ENV_FILE"):
        candidates.append(Path(os.environ["ENV_FILE"]))
    candidates += [
        Path.cwd() / ".env",
        _PROJECT_ROOT / ".env",
        _SRC_DIR / ".env",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_env(explicit: str | None = None) -> Path | None:
    """Load the demo's ``.env`` and mirror tokens where the SDKs expect them.

    Returns the path that was loaded (or ``None`` if no file was found — env
    vars may still be set in the shell).
    """
    path = find_env_file(explicit)
    if path is not None:
        load_dotenv(path, override=True)
    # The variables API reuses the Logfire token as its API key.
    token = os.environ.get("LOGFIRE_TOKEN")
    if token:
        os.environ.setdefault("LOGFIRE_API_KEY", token)
    return path


def model_name() -> str:
    return os.environ.get("MODEL", DEFAULT_MODEL)


def configure_logfire(*, with_variables: bool = True) -> None:
    """Configure Logfire for the demo and instrument Pydantic AI.

    Region is inferred from the token (the demo tokens are ``…_us_…``), so we
    don't set a base URL here. ``with_variables`` enables managed-variable
    resolution; the seeder turns it on too so it can push config.
    """
    logfire.configure(
        service_name=os.environ.get("LOGFIRE_SERVICE_NAME", DEFAULT_SERVICE_NAME),
        send_to_logfire="if-token-present",
        variables=logfire.VariablesOptions() if with_variables else None,
    )
    logfire.instrument_pydantic_ai()
