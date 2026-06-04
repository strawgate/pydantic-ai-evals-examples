"""Dataset utilities — load YAML datasets and filter by metadata.mode.

Each case has a ``metadata.mode`` field:

- ``offline`` — only included in batch dataset evals (training / scoring)
- ``online``  — only included in live (production-like) evaluation
- ``both`` (or missing) — included in either mode

This lets you keep all cases in one file but ensures online cases never
get used to train the offline scorer (and vice-versa).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_evals import Dataset

Mode = Literal["online", "offline"]


def load_dataset(path: Path, mode: Mode | None = None) -> Dataset[str, str]:
    """Load a dataset from YAML, optionally filtering cases by mode."""
    dataset: Dataset[str, str] = Dataset.from_file(path)
    if mode is None:
        return dataset

    filtered = []
    for case in dataset.cases:
        case_mode = (case.metadata or {}).get("mode", "both")
        if case_mode == mode or case_mode == "both":
            filtered.append(case)

    dataset.cases = filtered
    return dataset
