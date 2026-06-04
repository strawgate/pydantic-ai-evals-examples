"""Pytest fixtures for twenty questions evals."""

from __future__ import annotations

import os

import logfire
from dotenv import load_dotenv


def pytest_configure(config) -> None:
    """Load environment and configure Logfire."""
    env_file = os.environ.get("ENV_FILE", ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file, override=True)

    # Map MiniMax keys for Anthropic SDK compatibility
    minimax_key = os.environ.get("MINIMAX_API_KEY")
    if minimax_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", minimax_key)
    minimax_url = os.environ.get("MINIMAX_BASE_URL")
    if minimax_url:
        os.environ.setdefault("ANTHROPIC_BASE_URL", minimax_url)

    # Logfire uses LOGFIRE_TOKEN for traces and LOGFIRE_API_KEY for variables/API access
    logfire_token = os.environ.get("LOGFIRE_TOKEN")
    if logfire_token:
        os.environ.setdefault("LOGFIRE_API_KEY", logfire_token)

    logfire.configure(
        service_name=os.environ.get("LOGFIRE_SERVICE_NAME", "twenty-questions-evals"),
        send_to_logfire="if-token-present",
        variables=logfire.VariablesOptions() if os.environ.get("LOGFIRE_VAR_SYSTEM_PROMPT") else None,
    )
    logfire.instrument_pydantic_ai()
