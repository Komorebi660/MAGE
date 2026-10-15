"""Map multi-traveler samples to their originating ``train.csv`` query rows.

Normalize the base traveler's query before validating the sample-to-row mapping.
Pure data-access layer: no agent / session state.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .dataset import DEFAULT_CSV_PATH, count_rows, load_nth_row

TRAVELER_INTRO_RE = re.compile(r"^\s*I am\s+[A-Za-z][A-Za-z' -]*\.\s*", flags=re.IGNORECASE)


def strip_traveler_intro(query: str) -> str:
    return TRAVELER_INTRO_RE.sub("", str(query).strip(), count=1).strip()


def normalize_query_text(query: str) -> str:
    normalized = strip_traveler_intro(query)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().lower()


def validate_sample_query_mapping(
    sample: dict[str, Any],
    query_row: dict[str, Any],
    *,
    sample_index: int | None = None,
    query_id: int | None = None,
) -> None:
    sample_query = normalize_query_text(sample["base_person"]["query"])
    train_query = normalize_query_text(str(query_row["query"]))
    if sample_query == train_query:
        return

    sample_label = f"sample_index={sample_index}" if sample_index is not None else "sample"
    query_label = f"query_id={query_id}" if query_id is not None else "query"
    raise ValueError(
        "Multitraveler sample does not match the train.csv query: "
        f"{sample_label}, {query_label}. "
        f"sample_base_query={sample_query!r}, train_query={train_query!r}"
    )


def infer_query_id_for_sample(
    sample: dict[str, Any],
    *,
    sample_index: int | None = None,
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> int:
    sample_id = sample.get("id")
    if not isinstance(sample_id, int) or sample_id < 1:
        raise ValueError(f"sample_index={sample_index} has no valid positive integer id; cannot map to a train.csv query_id.")

    total_rows = count_rows(Path(csv_path))
    query_id = ((sample_id - 1) % total_rows) + 1
    query_row = load_nth_row(Path(csv_path), query_id, parse_literals=True)
    validate_sample_query_mapping(sample, query_row, sample_index=sample_index, query_id=query_id)
    return query_id
