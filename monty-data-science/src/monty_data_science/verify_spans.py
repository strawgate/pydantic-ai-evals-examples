"""Verify recent eval activity in Logfire.

Queries the Logfire read API for:
- ``Resolve variable …`` spans (proving the managed variable resolved)
- ``case: {case_name}`` spans (per-case scores from the latest run)
- ``agent_run`` spans (count, for sanity)

Useful after a `make eval` run to confirm traces actually landed and the
managed-variable wiring is correct without bouncing through the UI.

Usage:
    uv run python -m monty_data_science.verify_spans --env .env
    uv run python -m monty_data_science.verify_spans --since 5m
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any

import httpx
from dotenv import load_dotenv

from .logfire_links import _resolve_project_url  # type: ignore[reportPrivateUsage]

_DEFAULT_SINCE = "30 minutes"

# Logfire's hosting region is encoded in the token; the SDK derives the
# URL itself, but for a thin query helper we read the env var Logfire's
# own clients set, falling back to the US region.
_DEFAULT_API = "https://logfire-us.pydantic.dev"


def _api_base() -> str:
    return os.environ.get("LOGFIRE_BASE_URL") or _DEFAULT_API


def _parse_since(s: str) -> str:
    """Turn '5m'/'1h'/'30s' into a Postgres interval expression."""
    m = re.fullmatch(r"\s*(\d+)\s*([smhd]?)\s*", s)
    if m:
        n, unit = m.groups()
        unit_word = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "": "minutes"}[unit]
        return f"{n} {unit_word}"
    return s  # already an interval expression like "30 minutes"


def _query(token: str, sql: str, retries: int = 3) -> dict[str, Any]:
    """GET /v1/query with simple retry on transient 5xx errors."""
    last_err = ""
    for attempt in range(retries):
        try:
            resp = httpx.get(
                f"{_api_base()}/v1/query",
                params={"sql": sql},
                headers={"Authorization": f"Bearer {token}"},
                timeout=30.0,
            )
        except httpx.HTTPError as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code >= 500 and attempt < retries - 1:
            last_err = f"{resp.status_code}: {resp.text[:120]}"
            continue
        raise RuntimeError(f"Logfire query failed ({resp.status_code}): {resp.text[:200]}")
    raise RuntimeError(f"Logfire query failed after {retries} attempts: {last_err}")


def _columns_to_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert columnar response into row-oriented dicts."""
    cols = payload.get("columns", [])
    if not cols:
        return []
    n = len(cols[0]["values"])
    return [{c["name"]: c["values"][i] for c in cols} for i in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify eval spans landed in Logfire")
    parser.add_argument("--env", default=".env", help="Env file with LOGFIRE_TOKEN")
    parser.add_argument(
        "--since",
        default="30m",
        help="How far back to query, e.g. 5m / 1h / 30 minutes (default: 30m)",
    )
    parser.add_argument(
        "--service",
        default=None,
        help="Service name to filter by (default: LOGFIRE_SERVICE_NAME or "
        "monty-data-science-evals)",
    )
    args = parser.parse_args()

    if os.path.exists(args.env):
        load_dotenv(args.env, override=True)

    token = os.environ.get("LOGFIRE_TOKEN")
    if not token:
        print("LOGFIRE_TOKEN not set — cannot query.", file=sys.stderr)
        return 2

    service = args.service or os.environ.get("LOGFIRE_SERVICE_NAME", "monty-data-science-evals")
    interval = _parse_since(args.since)

    base = _resolve_project_url() or "(project URL not resolved)"
    var_name = os.environ.get("LOGFIRE_VAR_SYSTEM_PROMPT")

    print(f"Logfire project: {base}")
    print(f"Service:         {service}")
    print(f"Window:          last {interval}")
    print()

    # 1. Variable resolution spans
    sql_vars = (
        "SELECT attributes->>'name' AS var, attributes->>'version' AS ver, "
        "attributes->>'label' AS lbl, attributes->>'reason' AS reason, "
        "created_at "
        "FROM records "
        f"WHERE service_name = '{service}' "
        f"AND span_name LIKE 'Resolve variable%' "
        f"AND created_at > now() - interval '{interval}' "
        "ORDER BY created_at DESC LIMIT 20"
    )
    var_rows = _columns_to_rows(_query(token, sql_vars))
    print(f"[1] Managed-variable resolutions: {len(var_rows)} span(s)")
    if not var_rows:
        if var_name:
            print(f"    No resolutions for {var_name!r} found. Possible reasons:")
            print("      - No eval has run in this window")
            print("      - LOGFIRE_VAR_SYSTEM_PROMPT not set when the agent imported")
            print("      - variables=VariablesOptions() not passed to logfire.configure")
        else:
            print("    LOGFIRE_VAR_SYSTEM_PROMPT not set in env — set it to enable.")
    else:
        seen: set[tuple[str, str, str]] = set()
        for row in var_rows:
            key = (row.get("var") or "?", row.get("ver") or "?", row.get("lbl") or "?")
            if key in seen:
                continue
            seen.add(key)
            print(
                f"    - {row['var']} v{row['ver']} ({row['lbl']}, reason={row['reason']})"
                f"  last @ {row['created_at']}"
            )
    print()

    # 2. Case-level scores from the most recent run
    sql_cases = (
        "SELECT attributes->>'case_name' AS case_name, "
        "attributes->'scores'->'overall'->>'value' AS overall, "
        "attributes->'scores'->'correctness'->>'value' AS corr, "
        "attributes->'scores'->'method'->>'value' AS method, "
        "attributes->'scores'->'completeness'->>'value' AS comp, "
        "attributes->'scores'->'reasoning'->>'value' AS reason, "
        "attributes->'scores'->'rigor'->>'value' AS rigor, "
        "attributes->'scores'->'efficiency_score'->>'value' AS eff, "
        "attributes->'metrics'->>'requests' AS req, "
        "attributes->'metrics'->>'total_tokens' AS toks, "
        "attributes->'labels'->'judge_notes'->>'value' AS notes "
        "FROM records "
        f"WHERE service_name = '{service}' "
        "AND span_name = 'case: {case_name}' "
        f"AND created_at > now() - interval '{interval}' "
        "ORDER BY created_at DESC LIMIT 10"
    )
    case_rows = _columns_to_rows(_query(token, sql_cases))
    print(f"[2] Recent eval cases: {len(case_rows)}")
    if case_rows:
        header = (
            f"    {'case':<34s} {'over':>5s} {'cor':>5s} {'mth':>5s} "
            f"{'cmp':>5s} {'rsn':>5s} {'rig':>5s} {'eff':>5s} {'req':>4s} {'tok':>7s}"
        )
        print(header)
        for row in case_rows:
            name = (row.get("case_name") or "?")[:34]
            vals = [
                row.get(k) or ""
                for k in (
                    "overall",
                    "corr",
                    "method",
                    "comp",
                    "reason",
                    "rigor",
                    "eff",
                    "req",
                    "toks",
                )
            ]
            print(
                f"    {name:<34s} {vals[0]:>5s} {vals[1]:>5s} {vals[2]:>5s} "
                f"{vals[3]:>5s} {vals[4]:>5s} {vals[5]:>5s} {vals[6]:>5s} "
                f"{vals[7]:>4s} {vals[8]:>7s}"
            )
            # Print judge_notes inline so the operator sees what to fix.
            notes = row.get("notes")
            if notes:
                print(f"      ↳ {str(notes)[:150]}")
    else:
        print("    No case spans found. Has `make eval` run in this window?")
    print()

    # 3. agent_run sanity count
    sql_agent = (
        "SELECT count(*) AS n FROM records "
        f"WHERE service_name = '{service}' AND span_name = 'agent_run' "
        f"AND created_at > now() - interval '{interval}'"
    )
    agent_rows = _columns_to_rows(_query(token, sql_agent))
    n = agent_rows[0]["n"] if agent_rows else 0
    print(f"[3] agent_run spans in window: {n}")

    print()
    if base.startswith("http") and var_name:
        print(f"Open the optimizer: {base}/variables/{var_name}")
    return 0 if var_rows else 1


if __name__ == "__main__":
    sys.exit(main())
