import copy
import json
import re
import threading
from pathlib import Path
from typing import Any

from TravelPlanEnv import (
    DEFAULT_TRAIN_CSV_PATH,
    build_catalog,
    contains_all_terms,
    load_nth_row,
    matches_string_constraint,
    normalize_city,
    parse_reference_information,
    slugify,
    within_range,
)

__all__ = [
    "DEFAULT_TRAIN_CSV_PATH",
    "load_nth_row",
    "derive_traveler_name",
    "build_session_id",
    "TravelPlanEnv",
    "TravelPlanRuntime",
]

ITINERARY_SLOT_TYPES = {
    "transportation",
    "breakfast",
    "lunch",
    "dinner",
    "accommodation",
}
RESTAURANT_SLOT_TYPES = {"breakfast", "lunch", "dinner"}
PUBLIC_OPTION_FIELDS = {
    "restaurant": ("option_id", "name", "city", "price", "rating", "cuisines"),
    "hotel": ("option_id", "name", "city", "price", "rating", "room_type", "rules", "max_occupancy"),
}


def _public_option_details(option: dict[str, Any]) -> dict[str, Any]:
    """Use the same independent public details for search and selection."""
    return copy.deepcopy({
        key: option[key]
        for key in PUBLIC_OPTION_FIELDS[option["candidate_type"]]
        if key in option
    })


def derive_traveler_name(query: str) -> str:
    match = re.search(r"\bI am\s+([A-Za-z][A-Za-z' -]*)[.\n]", query)
    return match.group(1).strip() if match else "Traveler"


def build_session_id(traveler_name: str, turn_index: int) -> str:
    return f"{slugify(traveler_name) or 'traveler'}_turn_{turn_index}"


class TravelPlanEnv:
    def __init__(self, query_id: int, csv_path: str | Path = DEFAULT_TRAIN_CSV_PATH) -> None:
        self.query_id = query_id
        self.csv_path = Path(csv_path)
        self.query_row = load_nth_row(self.csv_path, query_id, parse_literals=True)
        self.reference_information = parse_reference_information(self.query_row["reference_information"])
        self.catalog = build_catalog(self.reference_information)
        self.option_registry: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.default_session_id = "default"
        self.create_session(self.default_session_id, overwrite=True)

    def get_query_context(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "org": self.query_row["org"],
            "dest": self.query_row["dest"],
            "date": self.query_row["date"],
            "people_number": self.query_row["people_number"],
            "local_constraint": self.query_row["local_constraint"],
            "budget": self.query_row["budget"],
            "query": self.query_row["query"],
        }

    def _initialize_itinerary(self) -> list[dict[str, Any]]:
        total_days = int(self.query_row["days"])
        annotated_plan = self.query_row.get("annotated_plan", [])
        source_days: list[dict[str, Any]] = []

        if (
            isinstance(annotated_plan, list)
            and len(annotated_plan) >= 2
            and isinstance(annotated_plan[1], list)
        ):
            source_days = [
                item
                for item in annotated_plan[1]
                if isinstance(item, dict) and item.get("days") is not None
            ]

        itinerary: list[dict[str, Any]] = []
        for day in range(1, total_days + 1):
            source_day = next((item for item in source_days if item.get("days") == day), {})
            itinerary.append(
                {
                    "days": day,
                    "current_city": source_day.get("current_city", self._default_current_city(day)),
                    "transportation": "-",
                    "breakfast": "-",
                    "attraction": "-",
                    "lunch": "-",
                    "dinner": "-",
                    "accommodation": "-",
                    "selected_options": {},
                }
            )
        return itinerary

    def _default_current_city(self, day: int) -> str:
        total_days = int(self.query_row["days"])
        org = self.query_row["org"]
        dest = self.query_row["dest"]
        if day == 1:
            return f"from {org} to {dest}"
        if day == total_days:
            return f"from {dest} to {org}"
        return str(dest)

    def _compose_session_id(self, agent_id: str | None, task_id: str | None) -> str:
        parts = [part for part in [agent_id, task_id] if part]
        if not parts:
            raise ValueError("Provide session_id, agent_id, or task_id.")
        return "__".join(slugify(part) for part in parts)

    def _build_session_state(
        self,
        session_id: str,
        *,
        agent_id: str | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        budget = float(self.query_row["budget"])
        return {
            "session_id": session_id,
            "agent_id": agent_id,
            "task_id": task_id,
            "itinerary": self._initialize_itinerary(),
            "current_total_cost": 0.0,
            "remaining_budget": budget,
            "budget": budget,
            "slot_costs": {},
            "selection_history": [],
        }

    def create_session(
        self,
        session_id: str | None = None,
        *,
        agent_id: str | None = None,
        task_id: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        resolved_session_id = session_id or self._compose_session_id(agent_id, task_id)
        with self._lock:
            if resolved_session_id in self.sessions and not overwrite:
                raise ValueError(f"session_id={resolved_session_id!r} already exists.")
            self.sessions[resolved_session_id] = self._build_session_state(
                resolved_session_id,
                agent_id=agent_id,
                task_id=task_id,
            )
            return self.get_session_summary(resolved_session_id)

    def _get_session(self, session_id: str | None) -> dict[str, Any]:
        resolved_session_id = session_id or self.default_session_id
        session = self.sessions.get(resolved_session_id)
        if session is None:
            raise KeyError(f"session_id={resolved_session_id!r} does not exist; call create_session first.")
        return session

    def get_session_summary(self, session_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            session = self._get_session(session_id)
            return {
                "session_id": session["session_id"],
                "agent_id": session["agent_id"],
                "task_id": session["task_id"],
                "current_total_cost": session["current_total_cost"],
                "remaining_budget": session["remaining_budget"],
                "budget": session["budget"],
                "selection_count": len(session["selection_history"]),
            }

    def get_itinerary(self, session_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            session = self._get_session(session_id)
            return copy.deepcopy(session["itinerary"])

    def get_total_cost(self, session_id: str | None = None) -> float:
        with self._lock:
            session = self._get_session(session_id)
            return float(session["current_total_cost"])

    def _get_day_plan(self, session_id: str | None, day: int) -> dict[str, Any]:
        session = self._get_session(session_id)
        itinerary = session["itinerary"]
        if day < 1 or day > len(itinerary):
            raise IndexError(f"day={day} is outside the itinerary range.")
        return itinerary[day - 1]

    def _validate_slot_type(self, slot_type: str, allowed_slot_types: set[str] | None = None) -> str:
        normalized = slot_type.strip().lower()
        if normalized not in ITINERARY_SLOT_TYPES:
            raise ValueError(f"Unsupported slot_type: {slot_type}")
        if allowed_slot_types is not None and normalized not in allowed_slot_types:
            raise ValueError(f"{slot_type} is not valid for this selection type; allowed values: {sorted(allowed_slot_types)}")
        return normalized

    def _resolve_option(self, option_id: str, expected_prefix: str) -> dict[str, Any]:
        option = self.option_registry.get(option_id)
        if option is None:
            raise KeyError(f"option_id={option_id!r} does not exist.")
        if option.get("candidate_type") != expected_prefix:
            raise ValueError(f"option_id={option_id!r} is not a {expected_prefix} candidate.")
        return option

    def _estimate_price_delta(self, candidate_type: str, option: dict[str, Any]) -> float:
        people_number = int(self.query_row["people_number"])
        if candidate_type == "restaurant":
            return float(option["price"]) * people_number
        if candidate_type == "hotel":
            return float(option["price"])
        raise ValueError(f"Unknown candidate_type: {candidate_type}")

    def _assign_slot(
        self,
        session_id: str | None,
        day: int,
        slot_type: str,
        value: str,
        *,
        option_id: str,
        option: dict[str, Any],
    ) -> dict[str, Any]:
        session = self._get_session(session_id)
        day_plan = self._get_day_plan(session["session_id"], day)
        day = int(day)
        normalized_slot = self._validate_slot_type(slot_type)
        slot_cost = self._estimate_price_delta(option["candidate_type"], option)
        selected_option = _public_option_details(option)

        day_plan[normalized_slot] = value
        day_plan["selected_options"][normalized_slot] = selected_option
        slot_key = f"{day}:{normalized_slot}"
        previous_total_cost = float(session["current_total_cost"])
        previous_slot_cost = float(session["slot_costs"].get(slot_key, 0.0))
        session["slot_costs"][slot_key] = slot_cost
        current_total_cost = previous_total_cost - previous_slot_cost + slot_cost
        session["current_total_cost"] = current_total_cost
        session["remaining_budget"] = float(session["budget"]) - current_total_cost
        price_delta = current_total_cost - previous_total_cost
        history_entry = {
            "session_id": session["session_id"],
            "day": day,
            "slot_type": normalized_slot,
            "option_id": option_id,
            "candidate_type": option["candidate_type"],
            "selected_value": value,
            "selected_option": copy.deepcopy(selected_option),
            "price_delta": price_delta,
            "current_total_cost": current_total_cost,
        }
        session["selection_history"].append(history_entry)

        result = {
            "session_id": session["session_id"],
            "day": day,
            "slot_type": normalized_slot,
            "selected_value": value,
            "selected_option": copy.deepcopy(selected_option),
            "day_plan": copy.deepcopy(day_plan),
            "price_delta": price_delta,
            "previous_total_cost": previous_total_cost,
            "current_total_cost": current_total_cost,
            "remaining_budget": float(session["remaining_budget"]),
            "within_budget": session["remaining_budget"] >= 0,
        }
        if not result["within_budget"]:
            result["warning"] = "Current selections exceed the budget."
        return result

    def select_restaurant(
        self,
        option_id: str,
        *,
        session_id: str | None = None,
        day: int,
        slot_type: str,
    ) -> dict[str, Any]:
        with self._lock:
            option = self._resolve_option(option_id, "restaurant")
            normalized_slot = self._validate_slot_type(slot_type, RESTAURANT_SLOT_TYPES)
            selected_value = f"{option['name']}, {option['city']}"
            result = self._assign_slot(
                session_id,
                day,
                normalized_slot,
                selected_value,
                option_id=option_id,
                option=option,
            )
            return result

    def select_hotel(
        self,
        option_id: str,
        *,
        session_id: str | None = None,
        day: int,
        slot_type: str = "accommodation",
    ) -> dict[str, Any]:
        with self._lock:
            option = self._resolve_option(option_id, "hotel")
            normalized_slot = self._validate_slot_type(slot_type, {"accommodation"})
            selected_value = f"{option['name']}, {option['city']}"
            result = self._assign_slot(
                session_id,
                day,
                normalized_slot,
                selected_value,
                option_id=option_id,
                option=option,
            )
            return result

    def search_restaurants(self, city: str, constraints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        constraints = constraints or {}
        results: list[dict[str, Any]] = []

        for record in self.catalog["restaurants"]:
            if normalize_city(record["city"]) != normalize_city(city):
                continue
            if constraints.get("name_contains") and not contains_all_terms(record["name"], constraints["name_contains"]):
                continue
            if not within_range(
                record["price"],
                constraints.get("min_price"),
                constraints.get("max_price"),
            ):
                continue
            if not within_range(
                record["rating"],
                constraints.get("min_rating"),
                constraints.get("max_rating"),
            ):
                continue
            cuisines = ", ".join(record["cuisines"])
            if constraints.get("cuisine") and not contains_all_terms(cuisines, constraints["cuisine"]):
                continue
            if constraints.get("cuisines_all") and not contains_all_terms(cuisines, constraints["cuisines_all"]):
                continue
            results.append(record)

        return self._finalize_results("restaurant", results, constraints)

    def search_hotels(self, city: str, constraints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        constraints = constraints or {}
        results: list[dict[str, Any]] = []

        for record in self.catalog["hotels"]:
            if normalize_city(record["city"]) != normalize_city(city):
                continue
            if constraints.get("name_contains") and not contains_all_terms(record["name"], constraints["name_contains"]):
                continue
            if not within_range(record["price"], constraints.get("min_price"), constraints.get("max_price")):
                continue
            if not within_range(
                record["rating"],
                constraints.get("min_rating"),
                constraints.get("max_rating"),
            ):
                continue
            if constraints.get("room_type") and not matches_string_constraint(
                record["room_type"], constraints["room_type"]
            ):
                continue
            rules = " & ".join(record["rules"])
            if constraints.get("house_rule") and not contains_all_terms(rules, constraints["house_rule"]):
                continue
            if constraints.get("house_rules_all") and not contains_all_terms(
                rules, constraints["house_rules_all"]
            ):
                continue
            if not within_range(
                record["max_occupancy"],
                constraints.get("min_occupancy"),
                constraints.get("max_occupancy"),
            ):
                continue
            if not within_range(
                record["minimum_nights"],
                constraints.get("min_minimum_nights"),
                constraints.get("max_minimum_nights"),
            ):
                continue
            results.append(record)

        return self._finalize_results("hotel", results, constraints)

    def _finalize_results(
        self,
        prefix: str,
        results: list[dict[str, Any]],
        constraints: dict[str, Any],
    ) -> list[dict[str, Any]]:
        normalized = copy.deepcopy(results)
        sort_by = constraints.get("sort_by")
        if sort_by:
            reverse = bool(constraints.get("descending", False))
            normalized.sort(key=lambda row: row.get(sort_by), reverse=reverse)

        limit = constraints.get("limit")
        if limit is not None:
            normalized = normalized[: int(limit)]

        for item in normalized:
            item["candidate_type"] = prefix
            if prefix == "restaurant":
                option_id = f"restaurant:{self.query_id}:{item['row_id']}"
            elif prefix == "hotel":
                option_id = f"hotel:{self.query_id}:{slugify(item['name'])}"
            else:
                raise ValueError(f"Unknown prefix: {prefix}")
            item["option_id"] = option_id
            self.option_registry[item["option_id"]] = copy.deepcopy(item)

        return normalized


class TravelPlanRuntime:
    """Session bookkeeping around a query-scoped environment.

    Construction never resets an injected environment. ``csv_path`` controls
    new environments created by select_query; checkpoints also retain the
    active environment's source when it was injected from a different CSV.
    The session itinerary is the plan. Full snapshots cover every traveler
    without copying the immutable catalog.
    """

    def __init__(
        self,
        query_id: int | None = None,
        env: TravelPlanEnv | None = None,
        csv_path: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path) if csv_path is not None else (
            env.csv_path if env is not None else DEFAULT_TRAIN_CSV_PATH
        )
        self.query_id = env.query_id if env is not None else query_id
        self.env = env if env is not None else (
            TravelPlanEnv(query_id, csv_path=self.csv_path) if query_id is not None else None
        )
        self.current_query: dict[str, Any] | None = (
            copy.deepcopy(self.env.get_query_context()) if self.env else None
        )
        self.current_session_id = self.env.default_session_id if self.env else "default"

    def _require_env(self) -> TravelPlanEnv:
        if self.env is None:
            raise ValueError("Call select_query() first, or initialize TravelPlanRuntime with query_id or env.")
        return self.env

    def _resolve_session_id(self, session_id: str | None = None) -> str:
        return session_id or self.current_session_id

    def _make_search_payload(
        self,
        *,
        kind: str,
        session_id: str,
        result: list[dict[str, Any]],
        **extra: Any,
    ) -> dict[str, Any]:
        self.current_session_id = session_id
        return {
            "kind": kind,
            "session_id": session_id,
            "result": {"records": result},
            "result_count": len(result),
            **extra,
        }

    def _make_selection_payload(
        self,
        *,
        kind: str,
        session_id: str,
        selection: dict[str, Any],
    ) -> dict[str, Any]:
        self.current_session_id = session_id
        return {
            "kind": kind,
            "session_id": session_id,
            **selection,
        }

    def _parse_constraints(self, raw_value: Any) -> dict[str, Any]:
        if raw_value is None:
            return {}
        if isinstance(raw_value, dict):
            return raw_value
        text = str(raw_value).strip()
        if not text:
            return {}
        if text.startswith("{"):
            return json.loads(text)
        raise ValueError(f"Unable to parse constraints: {raw_value}")

    def select_query(self, query_index: int) -> dict[str, Any]:
        self.env = TravelPlanEnv(query_index, csv_path=self.csv_path)
        self.query_id = query_index
        self.current_query = copy.deepcopy(self.env.get_query_context())
        self.current_session_id = self.env.default_session_id
        return {
            "kind": "query",
            "query_index": query_index,
            "query_context": self.current_query,
            "default_session_id": self.current_session_id,
        }

    def create_session(
        self,
        session_id: str | None = None,
        *,
        agent_id: str | None = None,
        task_id: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        env = self._require_env()
        summary = env.create_session(
            session_id=session_id,
            agent_id=agent_id,
            task_id=task_id,
            overwrite=overwrite,
        )
        resolved_session_id = summary["session_id"]
        self.current_session_id = resolved_session_id
        return {
            "kind": "session_created",
            **summary,
        }

    def prepare_session(
        self,
        session_id: str,
        *,
        query_index: int | None = None,
        traveler_name: str | None = None,
    ) -> dict[str, Any]:
        """Prepare one fresh traveler without assigning or executing an agent task.

        Selecting the same query preserves the other travelers. A different
        query starts a new query-scoped environment using the configured CSV.
        """
        if query_index is not None and (self.env is None or query_index != self.query_id):
            self.select_query(query_index)
        return self.create_session(session_id, agent_id=traveler_name, overwrite=True)

    def search_hotels(
        self,
        *,
        city: str,
        constraints: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        env = self._require_env()
        resolved_session_id = self._resolve_session_id(session_id)
        result = env.search_hotels(city, constraints or {})
        return self._make_search_payload(
            kind="hotel_search",
            session_id=resolved_session_id,
            result=result,
            city=city,
            constraints=constraints or {},
        )

    def search_restaurants(
        self,
        *,
        city: str,
        constraints: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        env = self._require_env()
        resolved_session_id = self._resolve_session_id(session_id)
        result = env.search_restaurants(city, constraints or {})
        return self._make_search_payload(
            kind="restaurant_search",
            session_id=resolved_session_id,
            result=result,
            city=city,
            constraints=constraints or {},
        )

    def select_restaurant(
        self,
        *,
        option_id: str,
        day: int,
        slot_type: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        env = self._require_env()
        resolved_session_id = self._resolve_session_id(session_id)
        selection = env.select_restaurant(
            option_id,
            session_id=resolved_session_id,
            day=day,
            slot_type=slot_type,
        )
        return self._make_selection_payload(
            kind="restaurant_selection",
            session_id=resolved_session_id,
            selection=selection,
        )

    def select_hotel(
        self,
        *,
        option_id: str,
        day: int,
        slot_type: str = "accommodation",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        env = self._require_env()
        resolved_session_id = self._resolve_session_id(session_id)
        selection = env.select_hotel(
            option_id,
            session_id=resolved_session_id,
            day=day,
            slot_type=slot_type,
        )
        return self._make_selection_payload(kind="hotel_selection", session_id=resolved_session_id, selection=selection)

    def get_current_cost(self, session_id: str | None = None) -> float:
        env = self._require_env()
        return env.get_total_cost(self._resolve_session_id(session_id))

    def snapshot_full(self) -> dict[str, Any]:
        """Copy all mutable runtime state, excluding immutable query catalogs."""
        env = self._require_env()
        with env._lock:
            return copy.deepcopy({
                "query_id": self.query_id,
                "csv_path": str(self.csv_path),
                "env_csv_path": str(env.csv_path),
                "current_query": self.current_query,
                "default_session_id": env.default_session_id,
                "sessions": env.sessions,
                "option_registry": env.option_registry,
                "current_session_id": self.current_session_id,
            })

    def restore_full(self, snapshot: dict[str, Any]) -> None:
        """Restore a full checkpoint, removing sessions created after it.

        The snapshot remains independent of subsequent runtime mutations. For
        checkpoints of the same query, the environment and catalog are reused.
        """
        state = copy.deepcopy(snapshot)
        query_id = state["query_id"]
        csv_path = Path(state["csv_path"])
        env_csv_path = Path(state["env_csv_path"])
        if self.env is None or self.env.query_id != query_id or self.env.csv_path != env_csv_path:
            self.env = TravelPlanEnv(query_id, csv_path=env_csv_path)
        env = self.env
        with env._lock:
            self.query_id = query_id
            self.csv_path = csv_path
            self.current_query = state["current_query"]
            env.default_session_id = state["default_session_id"]
            env.sessions = state["sessions"]
            env.option_registry = state["option_registry"]
            self.current_session_id = state["current_session_id"]

    @staticmethod
    def summarize_observation(payload: dict[str, Any] | str) -> str:
        if isinstance(payload, str):
            return payload

        kind = payload.get("kind")
        if kind == "current_cost":
            return json.dumps({"cost": payload["cost"]}, ensure_ascii=False)

        if kind == "session_created":
            return json.dumps({
                key: value for key, value in payload.items()
                if key not in {"session_id", "agent_id", "task_id"}
            }, ensure_ascii=False)

        if kind and kind.endswith("_search"):
            records = payload.get("result", {}).get("records", [])
            return json.dumps({
                "results": [_public_option_details(record) for record in records],
            }, ensure_ascii=False)

        fields = ("day", "slot_type", "selected_value", "selected_option")
        return json.dumps({key: payload[key] for key in fields if key in payload}, ensure_ascii=False)
