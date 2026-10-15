"""Decoders for the raw TravelPlanner ``reference_information`` text blobs.

Each query row in ``train.csv`` carries a ``reference_information`` list whose
``Content`` fields contain fixed-width text tables. This module decodes the
flight, restaurant and accommodation rows into typed record dicts and assembles
the per-query candidate catalog.
"""

from __future__ import annotations

import ast
import re
from typing import Any

FLIGHT_DESCRIPTION_RE = re.compile(
    r"^Flight from (?P<from_city>.+?) to (?P<to_city>.+?) on (?P<date>\d{4}-\d{2}-\d{2})$"
)
CITY_DESCRIPTION_RE = re.compile(
    r"^(?P<kind>Restaurants|Accommodations) in (?P<city>.+)$"
)


def parse_restaurant_line(line: str, city: str) -> dict[str, Any] | None:
    stripped = line.rstrip()
    if not stripped:
        return None

    city_suffix = f" {city}"
    if not stripped.endswith(city_suffix):
        return None

    body = stripped[: -len(city_suffix)].rstrip()
    match = re.match(
        r"^(?P<row_id>\d+)\s+(?P<name>.+?)\s{2,}(?P<price>\d+)\s{2,}"
        r"(?P<cuisines>.+?)\s{2,}(?P<rating>\d+\.\d+)$",
        body,
    )
    if not match:
        return None

    cuisines = [item.strip() for item in match.group("cuisines").split(",") if item.strip()]
    return {
        "row_id": int(match.group("row_id")),
        "name": match.group("name").strip(),
        "price": int(match.group("price")),
        "cuisines": cuisines,
        "rating": float(match.group("rating")),
        "city": city,
    }


def parse_accommodation_line(line: str, city: str) -> dict[str, Any] | None:
    stripped = line.rstrip()
    if not stripped:
        return None

    city_suffix = f" {city}"
    if not stripped.endswith(city_suffix):
        return None

    body = stripped[: -len(city_suffix)].rstrip()
    match = re.match(
        r"^(?P<name>.+?)\s+(?P<price>\d+\.\d+)\s+"
        r"(?P<room_type>Entire home/apt|Private room|Shared room)\s+"
        r"(?P<rules>.+?)\s+(?P<minimum_nights>\d+\.\d+)\s+"
        r"(?P<max_occupancy>\d+)\s+(?P<rating>\d+\.\d+)$",
        body,
    )
    if not match:
        return None

    rules = [item.strip() for item in match.group("rules").split("&") if item.strip()]
    return {
        "name": match.group("name").strip(),
        "price": float(match.group("price")),
        "room_type": match.group("room_type").strip(),
        "rules": rules,
        "minimum_nights": float(match.group("minimum_nights")),
        "max_occupancy": int(match.group("max_occupancy")),
        "rating": float(match.group("rating")),
        "city": city,
    }


def parse_flight_line(line: str, from_city: str, to_city: str, date: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if "There is no flight" in stripped or "No valid information." in stripped:
        return None

    tokens = stripped.split()
    if len(tokens) < 7:
        return None

    date_index = -1
    for index, token in enumerate(tokens):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
            date_index = index
            break

    if date_index == -1:
        return None

    try:
        distance = float(tokens[-1])
        dep_time = tokens[2]
        arr_time = tokens[3]
        price = int(tokens[1])
    except ValueError:
        return None

    elapsed = " ".join(tokens[4:date_index]).strip()
    return {
        "flight_number": tokens[0],
        "price": price,
        "departure_time": dep_time,
        "arrival_time": arr_time,
        "actual_elapsed_time": elapsed,
        "date": date,
        "from": from_city,
        "to": to_city,
        "distance": distance,
    }


def parse_reference_information(raw_value: Any) -> list[dict[str, Any]]:
    """Coerce the ``reference_information`` cell into a list of dicts."""
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, str):
        parsed = ast.literal_eval(raw_value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError("Unable to parse reference_information.")


def build_catalog(reference_information: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Parse a query's ``reference_information`` into a candidate catalog.

    Returns ``{"flights": [...], "restaurants": [...], "hotels": [...]}``.
    """
    catalog: dict[str, list[dict[str, Any]]] = {
        "flights": [],
        "restaurants": [],
        "hotels": [],
    }

    for item in reference_information:
        description = str(item.get("Description", "")).strip()
        content = str(item.get("Content", "")).strip("\n")
        if not description or not content:
            continue

        city_match = CITY_DESCRIPTION_RE.match(description)
        flight_match = FLIGHT_DESCRIPTION_RE.match(description)

        if city_match:
            kind = city_match.group("kind")
            city = city_match.group("city")
            lines = [line for line in content.splitlines()[1:] if line.strip()]

            if kind == "Restaurants":
                parse_line, records = parse_restaurant_line, catalog["restaurants"]
            else:
                parse_line, records = parse_accommodation_line, catalog["hotels"]
            for line in lines:
                record = parse_line(line, city)
                if record:
                    records.append(record)
            continue

        if flight_match:
            lines = [line for line in content.splitlines()[1:] if line.strip()]
            for line in lines:
                record = parse_flight_line(
                    line,
                    from_city=flight_match.group("from_city"),
                    to_city=flight_match.group("to_city"),
                    date=flight_match.group("date"),
                )
                if record:
                    catalog["flights"].append(record)

    return catalog
