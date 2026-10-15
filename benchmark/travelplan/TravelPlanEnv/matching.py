"""Low-level value-normalization and constraint-matching helpers.

These are pure functions (no state, no I/O) shared by the search filters in
``TravelPlanEnv``. They normalize cities/text/numbers and implement the
range/string/term constraint checks.
"""

from __future__ import annotations

import re
from typing import Any

FLOAT_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
# Pattern to strip state suffix like "(Oregon)" from city names for matching.
_CITY_STATE_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")


def normalize_city(value: Any) -> str:
    """Normalize city name: strip state suffix and lowercase."""
    text = str(value).strip()
    text = _CITY_STATE_SUFFIX_RE.sub("", text)
    return text.lower()


def normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or not FLOAT_RE.match(text):
        return None
    return float(text)


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def matches_string_constraint(value: str, expected: Any) -> bool:
    if expected is None:
        return True
    normalized_value = normalize_text(value)
    return any(normalize_text(option) == normalized_value for option in ensure_list(expected))


def contains_all_terms(value: str, required_terms: Any) -> bool:
    if required_terms is None:
        return True
    normalized_value = normalize_text(value)
    return all(normalize_text(term) in normalized_value for term in ensure_list(required_terms))


def within_range(value: float | None, minimum: Any = None, maximum: Any = None) -> bool:
    if value is None:
        return False
    min_value = to_float(minimum)
    max_value = to_float(maximum)
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True


def slugify(value: Any) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return normalized.strip("_")
