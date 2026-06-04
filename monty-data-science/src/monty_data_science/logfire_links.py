"""Helpers for surfacing useful Logfire URLs to the operator.

After ``logfire.configure()`` resolves the project, the SDK knows the
project's web URL (e.g. ``https://logfire-us.pydantic.dev/<org>/<project>``).
From that base we derive direct deep-links to:

- The managed-variables page (where the optimizer lives) when
  ``LOGFIRE_VAR_SYSTEM_PROMPT`` is set.
- The project trace view.

The URL resolution path is robust to the fact that ``configure()`` validates
the token in a background thread and may not have populated
``_project_url`` by the time we read it: we re-validate synchronously
ourselves if needed.
"""

from __future__ import annotations

import os
import time
from typing import Any

import logfire


def _resolve_project_url() -> str | None:
    """Return the project's web URL, or None if it can't be determined.

    Tries (in order):
    1. ``logfire._project_url`` — populated by configure() after the bg
       token-validation thread completes. Fast path when it's already set.
    2. ``logfire._initialize_credentials_from_token(token)`` — same call
       configure() makes, but run synchronously here.
    3. ``LOGFIRE_PROJECT_URL`` env var — escape hatch for unusual setups.
    """
    # 1. Already populated?
    config: Any = getattr(logfire, "_config", None) or getattr(
        logfire.DEFAULT_LOGFIRE_INSTANCE, "_config", None
    )
    if config is not None:
        url = getattr(config, "_project_url", None)
        if isinstance(url, str) and url.startswith("http"):
            return url.rstrip("/")

        # 2. Validate token synchronously to fetch credentials (project_url).
        token = getattr(config, "token", None) or os.environ.get("LOGFIRE_TOKEN")
        if token and hasattr(config, "_initialize_credentials_from_token"):
            try:
                creds = config._initialize_credentials_from_token(token)
            except Exception:
                creds = None
            if creds is not None:
                url = getattr(creds, "project_url", None)
                if isinstance(url, str) and url.startswith("http"):
                    return url.rstrip("/")

        # 3. Final fallback: poll briefly in case the bg thread is about to
        # finish. Don't block more than ~1s total.
        for _ in range(10):
            time.sleep(0.1)
            url = getattr(config, "_project_url", None)
            if isinstance(url, str) and url.startswith("http"):
                return url.rstrip("/")

    explicit = os.environ.get("LOGFIRE_PROJECT_URL")
    if explicit:
        return explicit.rstrip("/")
    return None


def print_logfire_links(*, banner: bool = True) -> None:
    """Print useful Logfire URLs to stdout. Flushes so the banner survives
    redirected-stdout block buffering. Cheap and safe to call repeatedly.
    """
    base = _resolve_project_url()
    var_name = os.environ.get("LOGFIRE_VAR_SYSTEM_PROMPT")

    lines: list[str] = []
    if banner:
        lines.append("")
        lines.append("─── Logfire ───────────────────────────────────────────────")

    if base is None:
        lines.append("  (no Logfire project resolved — set LOGFIRE_TOKEN to enable)")
    else:
        lines.append(f"  Traces:   {base}")
        if var_name:
            lines.append(f"  Variable: {base}/variables/{var_name}")
            lines.append("            ↑ open here to run the Logfire optimizer")
        else:
            lines.append(
                "  Variable: (LOGFIRE_VAR_SYSTEM_PROMPT not set — optimizer "
                "won't be able to correlate runs)"
            )

    if banner:
        lines.append("───────────────────────────────────────────────────────────\n")

    print("\n".join(lines), flush=True)
