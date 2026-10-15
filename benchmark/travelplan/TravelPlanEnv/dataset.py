"""Dataset access for the TravelPlanner ``train.csv`` corpus.

Thin helpers to read a single query row (1-based) and count rows, plus the
default CSV path. Cell values are optionally parsed from their Python-literal
string form (lists / dicts stored as text in the CSV).
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any

# TravelPlanEnv/ lives inside benchmark/travelplan/, so data/ is one level up.
DEFAULT_TRAIN_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "train.csv"
# Backwards-compatible alias exported for evaluate.py.
DEFAULT_CSV_PATH = DEFAULT_TRAIN_CSV_PATH


def maybe_parse_value(value: str) -> Any:
    if value is None:
        return None

    stripped = value.strip()
    if not stripped:
        return value

    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return value


def load_nth_row(csv_path: Path, index: int, parse_literals: bool) -> dict[str, Any]:
    if index < 1:
        raise ValueError("Row indices are 1-based.")

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for current_index, row in enumerate(reader, start=1):
            if current_index == index:
                if not parse_literals:
                    return row
                return {key: maybe_parse_value(value) for key, value in row.items()}

    raise IndexError(f"Row {index} does not exist.")


def count_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        return sum(1 for _ in reader)
