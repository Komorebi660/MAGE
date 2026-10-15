"""TravelPlan BaseAction tools bound to one traveler's runtime session.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agentcore.actions import BaseAction, FinishAct

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from travelplan_env import TravelPlanRuntime


class _TravelTypedAction(BaseAction):
    """Bind a session and render the payload returned by a domain operation."""

    NAME: str = ""
    DESC: str = ""
    PARAMS: dict = {}

    def __init__(self, session_id: str, runtime: TravelPlanRuntime) -> None:
        self.session_id = session_id
        self.runtime = runtime
        super().__init__(self.NAME, self.DESC, self.PARAMS)

    def _run(self, **kwargs: Any) -> dict[str, Any] | str:
        raise NotImplementedError

    def __call__(self, **kwargs: Any) -> str:
        # The session is bound, not selected by model-supplied arguments.
        clean_kwargs = {
            key: value
            for key, value in kwargs.items()
            if not key.startswith("_") and key != "session_id"
        }
        try:
            payload = self._run(**clean_kwargs)
            return self.runtime.summarize_observation(payload)
        except (KeyError, ValueError, TypeError, IndexError, FileNotFoundError) as exc:
            return f"Error: {exc}."


class RestaurantSearchAction(_TravelTypedAction):
    NAME = "RestaurantSearch"
    DESC = "Search for restaurant options in a city using optional filters."
    PARAMS = {
        "city": "String: the name of the city to search for restaurants in.",
        "constraints": """Optional JSON object containing the filters below. Omit it or use {} for no extra filters.
Each supplied filter must be satisfied. Text matching treats uppercase and lowercase letters as the same; for example, "Chinese" matches "chinese". All minimum and maximum limits include equality.

name_contains: string or list of strings to find within a restaurant's name. Keep restaurants whose names contain every supplied string. For example, ["Willow", "Table"] matches "Willow Table Bistro".
cuisine, cuisines_all: string or list of strings to find in the cuisine descriptions. Every value in a supplied list is required. For ["Chinese", "Seafood"], return restaurants serving both; restaurants that also serve Italian are included. Text can match part of a cuisine name, so "BBQ" also matches "Korean BBQ".
min_price, max_price: numbers setting the lowest and highest allowed price per person for one meal.
min_rating, max_rating: numbers setting the lowest and highest allowed rating.
sort_by: string naming the field to sort by, such as "price" or "rating". Sort the matches before applying limit.
descending: boolean used with sort_by; false (default) sorts from low to high, and true sorts from high to low.
limit: integer setting the maximum number of search results to return.""",
    }

    def _run(self, city, constraints=None):
        return self.runtime.search_restaurants(
            city=str(city),
            constraints=self.runtime._parse_constraints(constraints),
            session_id=self.session_id,
        )


class AccommodationSearchAction(_TravelTypedAction):
    NAME = "AccommodationSearch"
    DESC = "Search for accommodation options in a city using optional filters."
    PARAMS = {
        "city": "String: the name of the city to search for accommodation in.",
        "constraints": """Optional JSON object containing the filters below. Omit it or use {} for no extra filters.
Each supplied filter must be satisfied. Text matching treats uppercase and lowercase letters as the same; for example, "private room" matches "Private room". All minimum and maximum limits include equality.

name_contains: string or list of strings to find within an accommodation's name. Keep accommodations whose names contain every supplied string.
room_type: string or list of strings specifying allowed room types. Match the complete room type against any supplied value. For example, ["Private room", "Shared room"] allows either type; "room" alone does not match "Private room".
house_rule, house_rules_all: string or list of strings specifying required rule text. Return accommodations whose rule descriptions contain every supplied string. For example, ["No pets", "No smoking"] requires both rules; other rules may also be present.
min_price, max_price: numbers setting the lowest and highest allowed price per night.
min_rating, max_rating: numbers setting the lowest and highest allowed rating.
min_occupancy, max_occupancy: numbers limiting the accommodation's guest capacity. For example, min_occupancy=2 keeps accommodations that can hold at least two guests; max_occupancy sets the upper limit on that capacity.
min_minimum_nights, max_minimum_nights: numbers limiting the property's minimum required stay. For example, max_minimum_nights=2 keeps properties that require at most two consecutive nights as their minimum booking length.
sort_by: string naming the field to sort by, such as "price" or "rating". Sort the matches before applying limit.
descending: boolean used with sort_by; false (default) sorts from low to high, and true sorts from high to low.
limit: integer setting the maximum number of search results to return.""",
    }

    def _run(self, city, constraints=None):
        return self.runtime.search_hotels(
            city=str(city),
            constraints=self.runtime._parse_constraints(constraints),
            session_id=self.session_id,
        )


class SelectRestaurantAction(_TravelTypedAction):
    NAME = "SelectRestaurant"
    DESC = "Select a restaurant for the specified day and meal in the current plan."
    PARAMS = {
        "option_id": "String: the option_id of a restaurant returned by RestaurantSearch.",
        "day": "Integer: the trip day to update, starting at 1.",
        "slot_type": 'String: the meal to update, one of "breakfast", "lunch", or "dinner".',
    }

    def _run(self, option_id, day, slot_type):
        return self.runtime.select_restaurant(
            option_id=str(option_id),
            day=int(day),
            slot_type=str(slot_type),
            session_id=self.session_id,
        )


class SelectHotelAction(_TravelTypedAction):
    NAME = "SelectHotel"
    DESC = "Select accommodation for the night of the specified trip day."
    PARAMS = {
        "option_id": "String: the option_id of an accommodation returned by AccommodationSearch.",
        "day": "Integer: the trip day to update, starting at 1.",
    }

    def _run(self, option_id, day, slot_type="accommodation"):
        return self.runtime.select_hotel(
            option_id=str(option_id),
            day=int(day),
            slot_type=str(slot_type),
            session_id=self.session_id,
        )


class CurrentCostAction(_TravelTypedAction):
    NAME = "CurrentCost"
    DESC = "Show the total cost of the selections in the current session."
    PARAMS: dict = {}

    def _run(self):
        return {
            "kind": "current_cost",
            "session_id": self.session_id,
            "cost": self.runtime.get_current_cost(self.session_id),
        }


def build_travel_actions(session_id: str, runtime: TravelPlanRuntime) -> list[BaseAction]:
    """Build the session-bound domain tools and the shared Finish action."""
    return [
        RestaurantSearchAction(session_id, runtime),
        AccommodationSearchAction(session_id, runtime),
        SelectRestaurantAction(session_id, runtime),
        SelectHotelAction(session_id, runtime),
        CurrentCostAction(session_id, runtime),
        FinishAct,
    ]
