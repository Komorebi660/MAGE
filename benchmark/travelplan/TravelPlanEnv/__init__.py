"""Low-level TravelPlanner dataset access, parsing and matching utilities.

Split out of ``travelplan_env`` so the agent-facing session environment stays
focused on stateful search/select/book logic, while the corpus-decoding layer
(CSV access, ``reference_information`` text parsing, constraint matching) lives
here and can be tested independently.
"""

from .dataset import (
    DEFAULT_CSV_PATH,
    DEFAULT_TRAIN_CSV_PATH,
    count_rows,
    load_nth_row,
    maybe_parse_value,
)
from .matching import (
    contains_all_terms,
    ensure_list,
    matches_string_constraint,
    normalize_city,
    normalize_text,
    slugify,
    to_float,
    within_range,
)
from .parsers import (
    build_catalog,
    parse_accommodation_line,
    parse_flight_line,
    parse_reference_information,
    parse_restaurant_line,
)
from .multitraveler import (
    infer_query_id_for_sample,
    normalize_query_text,
    strip_traveler_intro,
    validate_sample_query_mapping,
)

__all__ = [
    "DEFAULT_CSV_PATH",
    "DEFAULT_TRAIN_CSV_PATH",
    "count_rows",
    "load_nth_row",
    "maybe_parse_value",
    "contains_all_terms",
    "ensure_list",
    "matches_string_constraint",
    "normalize_city",
    "normalize_text",
    "slugify",
    "to_float",
    "within_range",
    "build_catalog",
    "parse_accommodation_line",
    "parse_flight_line",
    "parse_reference_information",
    "parse_restaurant_line",
    "infer_query_id_for_sample",
    "normalize_query_text",
    "strip_traveler_intro",
    "validate_sample_query_mapping",
]
