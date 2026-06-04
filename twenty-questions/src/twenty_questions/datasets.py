"""Dataset utilities — load and filter cases by mode."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_evals import Dataset

Mode = Literal["online", "offline"]


def load_dataset(path: Path, mode: Mode | None = None) -> Dataset[str, str]:
    """Load a dataset, optionally filtering cases by metadata.mode."""
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
