"""Reset the managed ``system_prompt_data_science`` variable.

Two modes:

- Default: bump the variable forward to a NEW version containing the
  hardcoded ``DEFAULT_SYSTEM_PROMPT``. History (previous versions) is
  preserved. This is what you want during normal iteration.

- ``--wipe``: delete the variable from Logfire entirely (dropping all
  prior versions), then re-create it at v1 with the hardcoded default.
  Use this when starting a fresh optimization cycle and you do not want
  the optimizer to see or compare against the old version history.

Usage:
    uv run python -m monty_data_science.reset_prompt --env .env
    uv run python -m monty_data_science.reset_prompt --env .env --wipe
"""

from __future__ import annotations

import argparse
import json
import os

import httpx
import logfire
from dotenv import load_dotenv

from .agent import DEFAULT_SYSTEM_PROMPT, MANAGED_VAR_NAME

# The Logfire variables-API base. Mirror the SDK's region default; allow
# override via LOGFIRE_BASE_URL for non-US deployments.
_DEFAULT_API = "https://logfire-us.pydantic.dev"


def _build_var_config(variable_name: str, *, version: int):
    """Build a fresh VariableConfig at the given version."""
    from logfire.variables.config import LabeledValue, Rollout, VariableConfig

    return VariableConfig(
        name=variable_name,
        labels={
            "default": LabeledValue(
                version=version,
                serialized_value=json.dumps(DEFAULT_SYSTEM_PROMPT),
            )
        },
        rollout=Rollout(labels={"default": 1.0}),
        overrides=[],
        json_schema={"type": "string"},
    )


def _bump_version(variable_name: str) -> int:
    """Push the default as the next version of an existing variable; create
    fresh at v1 if the variable doesn't exist. Returns the new version.
    """
    config = logfire.variables_pull_config()
    var_config = config.variables.get(variable_name)
    if var_config is None:
        config.variables[variable_name] = _build_var_config(variable_name, version=1)
        new_version = 1
    else:
        from logfire.variables.config import LabeledValue

        current = var_config.latest_version.version if var_config.latest_version else 0
        new_version = current + 1
        var_config.labels["default"] = LabeledValue(
            version=new_version,
            serialized_value=json.dumps(DEFAULT_SYSTEM_PROMPT),
        )
    logfire.variables_push_config(config, mode="merge", yes=True)
    return new_version


def _wipe_and_reseed(variable_name: str) -> None:
    """Delete the variable server-side, then recreate it at v1.

    Uses the direct ``DELETE /v1/variables/<name>`` REST endpoint (returns
    204 on success, 404 if already gone — both are fine). The variables
    SDK's ``push_config(mode='replace', variables={})`` does NOT drop
    existing rows, so this is the only path that actually wipes history.
    """
    base = os.environ.get("LOGFIRE_BASE_URL") or _DEFAULT_API
    token = os.environ["LOGFIRE_TOKEN"]
    resp = httpx.delete(
        f"{base}/v1/variables/{variable_name}",
        headers={"Authorization": f"Bearer {token}"},
        follow_redirects=True,
        timeout=15,
    )
    if resp.status_code not in (204, 404):
        raise RuntimeError(
            f"Failed to delete variable '{variable_name}': {resp.status_code} {resp.text[:200]}"
        )

    # Recreate fresh at v1 via the SDK (merge mode — variable doesn't exist).
    config = logfire.variables_pull_config()
    config.variables[variable_name] = _build_var_config(variable_name, version=1)
    logfire.variables_push_config(config, mode="merge", yes=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset the managed system_prompt variable to the hardcoded default"
    )
    parser.add_argument("--env", default=".env")
    parser.add_argument(
        "--variable",
        default=None,
        help=(
            "Variable name to reset (default: LOGFIRE_VAR_SYSTEM_PROMPT from the env "
            "file, or 'system_prompt_data_science' if unset)."
        ),
    )
    parser.add_argument(
        "--wipe",
        action="store_true",
        help=(
            "Wipe the variable server-side (drop all prior versions) and "
            "recreate it at v1. Use to start a fully fresh optimization cycle."
        ),
    )
    args = parser.parse_args()

    if os.path.exists(args.env):
        load_dotenv(args.env, override=True)

    logfire_token = os.environ.get("LOGFIRE_TOKEN")
    if logfire_token:
        os.environ.setdefault("LOGFIRE_API_KEY", logfire_token)

    variable_name = (
        args.variable
        or MANAGED_VAR_NAME
        or os.environ.get("LOGFIRE_VAR_SYSTEM_PROMPT")
        or "system_prompt_data_science"
    )

    logfire.configure(
        service_name=os.environ.get("LOGFIRE_SERVICE_NAME", "monty-data-science-evals"),
        send_to_logfire="if-token-present",
        variables=logfire.VariablesOptions(),
    )

    if args.wipe:
        _wipe_and_reseed(variable_name)
        print(
            f"Wiped managed variable '{variable_name}' and reseeded at v1 "
            f"with DEFAULT_SYSTEM_PROMPT."
        )
    else:
        new_version = _bump_version(variable_name)
        print(
            f"Bumped managed variable '{variable_name}' to v{new_version} "
            f"with DEFAULT_SYSTEM_PROMPT."
        )


if __name__ == "__main__":
    main()
