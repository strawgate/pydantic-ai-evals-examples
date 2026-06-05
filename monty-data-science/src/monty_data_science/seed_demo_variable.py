"""Seed the demo's managed variable to the "before" (wrong-schema) state.

This is a SETUP step, run by whoever prepares the demo — it needs
``LOGFIRE_API_KEY`` set to a credential with ``read_variables`` +
``write_variables`` scopes (a plain write/ingest token does NOT have them).
The person *running* the demo (``uv run demo``) doesn't seed; the variable
already lives server-side.

Running it (idempotent — always resets to the clean before-state):

    uv run seed-demo-variable

Each run DELETEs any existing ``data_science_agent_prompt`` and recreates it at
v1 with ``WRONG_SCHEMA_PROMPT`` and a single ``default`` label serving 100% of
traffic. Recreating also resets the optimizer's lookback clock, so only traffic
sent *after* you seed is considered — seed first, then run traffic.

Done over the Logfire REST API directly (not the SDK's config-merge), which is
deterministic and matches the platform's own managed-variables walkthrough.
"""

from __future__ import annotations

import argparse
import json
import os

import httpx

from .demo_config import VARIABLE_NAME, load_env
from .demo_prompts import BEFORE_PROMPT, IDEAL_AFTER_PROMPT

# Logfire variables REST base. The demo tokens are US-region; override with
# LOGFIRE_BASE_URL for other deployments.
_DEFAULT_API = "https://logfire-us.pydantic.dev"


def _base_and_token() -> tuple[str, str]:
    base = (os.environ.get("LOGFIRE_BASE_URL") or _DEFAULT_API).rstrip("/")
    token = os.environ.get("LOGFIRE_API_KEY")
    if not token:
        raise SystemExit(
            "LOGFIRE_API_KEY is not set. The variables API needs a credential with "
            "'read_variables' + 'write_variables' scopes (a user API key); a plain "
            "write/ingest token does not have them."
        )
    return base, token


def _variable_body(prompt: str) -> dict[str, object]:
    serialized = json.dumps(prompt)
    return {
        "name": VARIABLE_NAME,
        "description": "System prompt (incl. DB schema) for the data-science agent.",
        "json_schema": {"type": "string"},
        "rollout": {"labels": {"default": 1.0}},
        "overrides": [],
        "example": serialized,
        "labels": {
            "default": {"target_type": "version", "serialized_value": serialized},
        },
    }


def reset_variable(prompt: str) -> None:
    base, token = _base_and_token()
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=base, headers=headers, timeout=20, follow_redirects=True) as client:
        delete = client.delete(f"/v1/variables/{VARIABLE_NAME}/")
        if delete.status_code not in (204, 404):
            raise SystemExit(
                f"Failed to delete '{VARIABLE_NAME}': {delete.status_code} {delete.text[:300]}\n"
                f"(403 => LOGFIRE_API_KEY lacks write_variables on this project.)"
            )
        create = client.post("/v1/variables/", json=_variable_body(prompt))
        if not create.is_success:
            raise SystemExit(
                f"Failed to create '{VARIABLE_NAME}': {create.status_code} {create.text[:300]}"
            )

        # Verify it actually landed (and is readable for runtime resolution).
        check = client.get(f"/v1/variables/{VARIABLE_NAME}/")
        if not check.is_success:
            raise SystemExit(
                f"Created '{VARIABLE_NAME}' but read-back failed: "
                f"{check.status_code} {check.text[:300]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset the demo managed variable to the wrong-schema before-state"
    )
    parser.add_argument("--env", default=None, help="Path to .env (auto-detected if omitted)")
    parser.add_argument(
        "--after",
        action="store_true",
        help="Seed the IDEAL 'after' value (schema baked in) instead of the before-state. "
        "For A/B measuring the optimization's effect.",
    )
    # Accepted for back-compat; seeding is always a clean reset now.
    parser.add_argument("--wipe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    load_env(args.env)
    prompt = IDEAL_AFTER_PROMPT if args.after else BEFORE_PROMPT
    state = (
        "ideal AFTER (migration applied: renames + customers JSON)"
        if args.after
        else "BEFORE (stale schema — pre-migration column names + flat customers)"
    )
    reset_variable(prompt)
    print(
        f"Reset '{VARIABLE_NAME}' to the {state} state (v1, label 'default' serving 100%). "
        f"Run traffic now — the optimizer only sees runs sent after this."
    )


if __name__ == "__main__":
    main()
