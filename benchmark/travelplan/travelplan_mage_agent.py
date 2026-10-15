"""MAGE's traveler memory, selection checks, maintain and suffix replay.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
for path in (CURRENT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from agentcore.agents.MAGEAgent import MAGEAgent
from agentcore.commons import AgentAct, TaskPackage
from agentcore.llm.LLMBackend import BaseLLM
from agentcore.logging.terminal_logger import AgentLogger
from agentcore.memory.ExplorationTree import ExplorationTree

from travelplan_actions import build_travel_actions
from travelplan_env import TravelPlanRuntime, build_session_id, derive_traveler_name
from TravelPlanEnv.constraints import (
    _analyze_joining_query_slots,
    _parse_query_slot_constraints,
    verify_slot_constraints,
)
from TravelPlanEnv.scoring import score_travelers


MAGE_ROLE = r"""You are a travel planning agent. Complete the current traveler's plan according to their requirements.

Use the supplied base plan and prior travelers' plans in the current attempt to resolve references.
Compressed context contains this traveler's selected results saved with Store. Previous attempts provide historical alternatives; the current attempt's selections determine the current plan.

For each requested meal or accommodation:
1. Identify the target day and meal or accommodation. A different day or meal mentioned for comparison is a reference, not another target.
2. Use the city from the base plan for that exact day and meal or accommodation.
3. Find the referenced traveler's correct item. Read its existing tool observations, or search its name when you need its price, rating, room type, rules, or cuisines.
4. Calculate the search bounds and pass every known condition supported by the search tool.
5. Check the returned options against the complete request, including exact names and any conditions the search cannot express.
6. Record each new or changed arrangement with SelectRestaurant or SelectHotel, using an option_id from the actual search results.

You do not need to search or select again when the request explicitly keeps the base traveler's arrangement for the same day and the same meal or accommodation. Search and select arrangements for the meals or accommodation with explicit new requirements.
For a comparison such as a higher rating or a lower price, first find the referenced item's rating or price, then search and select an option that meets the comparison.

Comparison rules:
- "Within", "at least", and "at most" include equality. An option exactly at an allowed boundary qualifies.
- "Higher", "lower", "more expensive", and "less expensive" are strict unless the request also permits equality.
- Search min/max price and rating bounds include equality. For a strict comparison, search with the reference value as the bound, then check the strict relation on the returned options.
- Calculate from the stated numbers. Do not shift a boundary by a fixed amount such as 0.1.
- Use the returned per-person meal price or nightly accommodation price for the request's price comparisons.

Tool format:
- Emit one action per turn: Action: ToolName[{"parameter":"value"}]. Keep the entire action on one physical line.
- Use Action: Think[{"response":"brief reasoning"}] for reasoning. Do not write a separate Thought line or add text after an action.
- Zero-argument tools use {}.
- Encode any newline inside a string as \n.
- If a search returns no options, check the request and reference facts, then continue with a valid tool call.

A saved selection is shown with a [Step N] marker. To change a saved selection, call Backtrace with that step and a reason, then search and select again.
After each successful SelectRestaurant or SelectHotel action, you may call Store to save the selected results.
When the traveler's requirements are satisfied, call Finish with a brief summary of the completed arrangements.
"""
JUDGE_RETRY_GUIDANCE = (
    "You should be careful to check the options more thoroughly and select an itinerary "
    "that satisfies the traveler's requirements and is compatible with prior travelers' plans."
)
_SELECT_ACTIONS = {"SelectRestaurant", "SelectHotel"}


@dataclass
class TravelerRecord:
    """A traveler's plan and tree entry, retained even off the active path."""
    index: int = -1
    traveler_name: str = ""
    session_id: str = ""
    plan_json: list = field(default_factory=list)
    completed: bool = False
    summary_text: str = ""
    parent_node_id: int | None = None
    parent_summary_id: str | None = None
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    children_record_ids: list[str] = field(default_factory=list)
    visit_count: int = 1
    verify_feedback: str | None = None

    @property
    def merge_key(self) -> str:
        return json.dumps([self.plan_json, self.completed], sort_keys=True)


class TravelerTree:
    """Traveler completions branch under their active predecessor, not their session ID."""

    def __init__(self) -> None:
        self.root_id = "traveler_root"
        self.records = {self.root_id: TravelerRecord(record_id=self.root_id)}
        self._active_path: list[str] = []

    def active_records(self) -> list[TravelerRecord]:
        return [self.records[rid] for rid in self._active_path]

    def parent_for_index(self, index: int) -> TravelerRecord:
        return next((rec for rec in reversed(self.active_records()) if rec.index == index - 1),
                    self.records[self.root_id])

    def trim_active_path(self, before_index: int) -> None:
        self._active_path = [rid for rid in self._active_path if self.records[rid].index < before_index]

    def add_record(self, record: TravelerRecord) -> TravelerRecord:
        parent = self.parent_for_index(record.index)
        for rid in parent.children_record_ids:
            previous = self.records[rid]
            if previous.index == record.index and previous.merge_key == record.merge_key:
                record.record_id = rid
                record.children_record_ids = previous.children_record_ids
                record.visit_count = previous.visit_count + 1
                break
        else:
            parent.children_record_ids.append(record.record_id)
        self.records[record.record_id] = record
        self.trim_active_path(record.index)
        self._active_path.append(record.record_id)
        return record

    def previous_attempts(self, index: int) -> list[str]:
        best: dict[str, TravelerRecord] = {}
        for record in self.records.values():
            if record.index == index:
                previous = best.get(record.merge_key)
                if previous is None or record.visit_count > previous.visit_count:
                    best[record.merge_key] = record
        summaries = []
        for record in best.values():
            text = record.summary_text
            if record.visit_count > 1:
                text += f" This traveler plan has been generated in {record.visit_count} independent attempts."
            if record.verify_feedback:
                text += f" But may have issues: {' '.join(record.verify_feedback.split())}"
            summaries.append(text)
        return summaries


@dataclass
class TravelBundleSharedState:
    """Create once per dataset sample; reference answers stay out of model prompts."""
    sample_id: int
    traveler_queries: list[str] = field(default_factory=list)
    traveler_answers: list[list] = field(default_factory=list)
    base_traveler_query: str = ""
    base_traveler_plan: list = field(default_factory=list)
    max_judge_retries: int = 0
    exploration_tree: ExplorationTree = field(default_factory=ExplorationTree, init=False)
    traveler_tree: TravelerTree = field(default_factory=TravelerTree, init=False)
    global_step: int = field(default=1, init=False)
    judge_count: int = field(default=0, init=False)
    agent_backtrace_count: int = field(default=0, init=False)
    pending_backtrace_reasons: list[tuple[int, str]] = field(default_factory=list, init=False)
    traveler_settings: dict[int, dict] = field(default_factory=dict, init=False)

    def get_active_results(self) -> list[dict[str, Any]]:
        """Copy the ordered raw active records, including pending verify failures."""
        return [
            {
                "turn_index": record.index,
                "traveler_name": record.traveler_name,
                "session_id": record.session_id,
                "plan_json": copy.deepcopy(record.plan_json),
                "completed": record.completed,
                "status": "success" if record.completed else "incomplete",
            }
            for record in self.traveler_tree.active_records()
        ]


def summarize_traveler_plan(record: TravelerRecord) -> str:
    """Show the saved plan, including the details captured when selecting."""
    return "Selected Plan: " + json.dumps(record.plan_json, ensure_ascii=False)


class TravelPlanMAGEAgent(MAGEAgent):
    previous_results_title = "Previous attempts for the current traveler:"

    def __init__(
        self,
        llm: BaseLLM,
        runtime: TravelPlanRuntime,
        turn_index: int,
        traveler_query: str,
        shared_state: TravelBundleSharedState,
        max_steps: int = 40,
        PROMPT_DEBUG_FLAG: bool = False,
        log_file: str = os.devnull,
        max_verify_retries: int = 0,
        max_check_retries: int = 1,
        max_backtrace: int = 5,
    ) -> None:
        self.runtime = runtime
        self.turn_index = turn_index
        self.traveler_query = traveler_query
        self.shared_state = shared_state
        self._task_id = shared_state.sample_id
        self._log_dir = os.path.dirname(log_file)
        self.runtime_state: dict[str, Any] = {"session_id": None, "traveler_name": "Traveler"}
        self._query_row: dict = {}
        self._replay_target_turn_index: int | None = None
        self._store_check_failed = False
        self.verification_feedback: str | None = None

        if turn_index == len(shared_state.traveler_queries):
            shared_state.traveler_queries.append(traveler_query)
        elif not 0 <= turn_index < len(shared_state.traveler_queries):
            raise ValueError("Travelers must be registered in order.")
        elif shared_state.traveler_queries[turn_index] != traveler_query:
            raise ValueError("The traveler query does not match the registered turn.")

        super().__init__(
            name="travelplan_mage_agent",
            role=MAGE_ROLE,
            llm=llm,
            actions=build_travel_actions("__unbound__", runtime),
            logger=AgentLogger(log_file_name=log_file, FLAG_PRINT=False, PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG),
            max_exec_steps=max_steps,
            max_verify_retries=max_verify_retries,
            max_check_retries=max_check_retries,
            max_backtrace=max_backtrace,
        )
        self.exploration_tree = shared_state.exploration_tree
        self._global_step = shared_state.global_step
        self._agent_backtrace_count = shared_state.agent_backtrace_count
        self._build_examples()

    def _build_examples(self) -> None:
        """Build the static MAGE example with Store turns after each selection."""
        base_query = (
            "I am Rowan. Please arrange a two-day solo trip from Atlanta to Savannah and back on May 8–9, 2027. "
            "I will return after lunch on the second day. The overall trip budget is $1,100."
        )
        base_plan = [
            {
                "days": 1, "current_city": "from Atlanta to Savannah",
                "dinner": "Bayglass Kitchen, Savannah", "accommodation": "Maple Quay Rooms, Savannah",
            },
            {"days": 2, "current_city": "from Savannah to Atlanta", "lunch": "Willow Table, Savannah"},
        ]
        query = (
            "I am Mira. I am joining Rowan for this trip.\n\n"
            "Day 1 accommodation: I need a different room category from Rowan's. The nightly rate should be higher "
            "than Rowan's accommodation that night but within $55 of it. The rating should be within 0.3 of Rowan's, "
            "and the accommodation must have the same No pets rule.\n\n"
            "Day 1 dinner: I would like to eat at the same restaurant Rowan chose for lunch on day 2.\n\n"
            "Day 2 lunch: I would like a restaurant serving Chinese and Seafood cuisines and no other cuisine types, "
            "costing at least 25% less than Rowan's dinner on day 1, with a rating at least as high as that restaurant."
        )
        maple = {
            "option_id": "hotel:1:maple_quay_rooms", "name": "Maple Quay Rooms", "city": "Savannah",
            "price": 180.0, "rating": 4.5, "room_type": "Private room",
            "rules": ["No pets", "No parties"], "max_occupancy": 1,
        }
        cedar = {
            "option_id": "hotel:1:cedar_walk_flat", "name": "Cedar Walk Flat", "city": "Savannah",
            "price": 205.0, "rating": 4.2, "room_type": "Entire home/apt",
            "rules": ["No pets"], "max_occupancy": 2,
        }
        birch = {
            "option_id": "hotel:1:birch_harbor_loft", "name": "Birch Harbor Loft", "city": "Savannah",
            "price": 180.0, "rating": 4.5, "room_type": "Entire home/apt",
            "rules": ["No pets"], "max_occupancy": 2,
        }
        bistro = {
            "option_id": "restaurant:1:2", "name": "Willow Table Bistro", "city": "Savannah",
            "price": 56, "rating": 4.6, "cuisines": ["Italian", "Seafood"],
        }
        willow = {
            "option_id": "restaurant:1:4", "name": "Willow Table", "city": "Savannah",
            "price": 48, "rating": 4.4, "cuisines": ["Italian", "Seafood"],
        }
        bayglass = {
            "option_id": "restaurant:1:1", "name": "Bayglass Kitchen", "city": "Savannah",
            "price": 44, "rating": 4.4, "cuisines": ["Italian", "Seafood"],
        }
        jade = {
            "option_id": "restaurant:1:6", "name": "Jade Spoon", "city": "Savannah",
            "price": 33, "rating": 4.4, "cuisines": ["Chinese", "Seafood"],
        }
        orchid = {
            "option_id": "restaurant:1:7", "name": "Orchid Lantern", "city": "Savannah",
            "price": 30, "rating": 4.4, "cuisines": ["Chinese", "Seafood", "Italian"],
        }
        result = "Day 1 accommodation: Cedar Walk Flat. Day 1 dinner: Willow Table. Day 2 lunch: Jade Spoon."
        chain = [
            (AgentAct(name="Think", params={"response": "First, I check Rowan's Day 1 accommodation."}), "OK"),
            (AgentAct(name="AccommodationSearch", params={
                "city": "Savannah", "constraints": {"name_contains": "Maple Quay Rooms"},
            }), json.dumps({"results": [maple]})),
            (AgentAct(name="Think", params={
                "response": "Use a different room type, price above 180 up to 235, and rating 4.2–4.8, including both rating boundaries.",
            }), "OK"),
            (AgentAct(name="AccommodationSearch", params={
                "city": "Savannah", "constraints": {
                    "min_price": 180, "max_price": 235, "min_rating": 4.2, "max_rating": 4.8,
                    "room_type": ["Entire home/apt", "Shared room"], "house_rules_all": ["No pets"],
                    "min_occupancy": 1, "max_minimum_nights": 1,
                },
            }), json.dumps({"results": [cedar, birch]})),
            (AgentAct(name="Think", params={
                "response": "Cedar's rating 4.2 is allowed. Birch costs 180, so it is not more expensive than Rowan's. I select Cedar.",
            }), "OK"),
            (AgentAct(name="SelectHotel", params={"option_id": cedar["option_id"], "day": 1}), json.dumps({
                "day": 1, "slot_type": "accommodation", "selected_value": "Cedar Walk Flat, Savannah",
                "selected_option": cedar,
            })),
            (AgentAct(name="Store", params={
                "response": f"Day 1 accommodation selected: Cedar Walk Flat, Savannah. Option ID: {cedar['option_id']}. "
                "Nightly price: $205. Rating: 4.2. Entire home/apt; No pets; capacity 2.",
            }), "Information stored."),
            (AgentAct(name="Think", params={
                "response": "Next, I check Rowan's Day 2 lunch for Mira's Day 1 dinner.",
            }), "OK"),
            (AgentAct(name="RestaurantSearch", params={
                "city": "Savannah", "constraints": {"name_contains": "Willow Table"},
            }), json.dumps({"results": [bistro, willow]})),
            (AgentAct(name="Think", params={
                "response": "Willow Table is the exact restaurant in Rowan's plan; Willow Table Bistro is different.",
            }), "OK"),
            (AgentAct(name="SelectRestaurant", params={
                "option_id": willow["option_id"], "day": 1, "slot_type": "dinner",
            }), json.dumps({
                "day": 1, "slot_type": "dinner", "selected_value": "Willow Table, Savannah", "selected_option": willow,
            })),
            (AgentAct(name="Store", params={
                "response": f"Day 1 dinner selected: Willow Table, Savannah. Option ID: {willow['option_id']}. "
                "Price per person: $48. Rating: 4.4. Cuisines: Italian, Seafood.",
            }), "Information stored."),
            (AgentAct(name="Think", params={
                "response": "Finally, I check the price and rating of Rowan's Day 1 dinner for Mira's Day 2 lunch.",
            }), "OK"),
            (AgentAct(name="RestaurantSearch", params={
                "city": "Savannah", "constraints": {"name_contains": "Bayglass Kitchen"},
            }), json.dumps({"results": [bayglass]})),
            (AgentAct(name="Think", params={
                "response": "The price limit is 44 × 0.75 = 33, and rating 4.4 is allowed. Require Chinese and Seafood only.",
            }), "OK"),
            (AgentAct(name="RestaurantSearch", params={
                "city": "Savannah", "constraints": {"cuisines_all": ["Chinese", "Seafood"], "max_price": 33, "min_rating": 4.4},
            }), json.dumps({"results": [jade, orchid]})),
            (AgentAct(name="Think", params={
                "response": "Jade meets both allowed boundaries: price 33 and rating 4.4. Orchid adds Italian, which the request excludes.",
            }), "OK"),
            (AgentAct(name="SelectRestaurant", params={
                "option_id": jade["option_id"], "day": 2, "slot_type": "lunch",
            }), json.dumps({
                "day": 2, "slot_type": "lunch", "selected_value": "Jade Spoon, Savannah", "selected_option": jade,
            })),
            (AgentAct(name="Store", params={
                "response": f"Day 2 lunch selected: Jade Spoon, Savannah. Option ID: {jade['option_id']}. "
                "Price per person: $33. Rating: 4.4. Cuisines: Chinese, Seafood.",
            }), "Information stored."),
            (AgentAct(name="Think", params={"response": "All three requested arrangements are selected."}), "OK"),
            (AgentAct(name="Finish", params={"response": result}), result),
        ]
        task = TaskPackage(instruction=(
            f"Supplied base traveler\n{base_query}\n\n"
            f"The fixed base traveler plan:\n{json.dumps(base_plan, ensure_ascii=False, indent=2)}\n\n"
            f"Current traveler's requirements\n{query}"
        ))
        self.add_example(task, chain)

    def _rebind_actions(self, session_id: str) -> None:
        for action in self.actions:
            if hasattr(action, "session_id"):
                action.session_id = session_id

    def __action_parser__(self, raw_action: str) -> AgentAct:
        action = super().__action_parser__(raw_action)
        if action.name not in {tool.action_name for tool in self.actions}:
            action.valid = False
        if action.name == "Store" and not isinstance((action.params or {}).get("response", ""), str):
            action.valid = False
        if action.name == "Backtrace" and action.valid:
            try:
                int((action.params or {}).get("step_idx", -1))
            except (TypeError, ValueError, OverflowError):
                action.valid = False
        return action

    def forward(self, task: TaskPackage, agent_act: AgentAct) -> str:
        if agent_act.valid and agent_act.name in _SELECT_ACTIONS:
            params = agent_act.params or {}
            default_slot = "accommodation" if agent_act.name == "SelectHotel" else None
            try:
                day = int(params["day"])
                slot = str(params.get("slot_type", default_slot)).strip().lower()
            except (KeyError, TypeError, ValueError, OverflowError):
                return super().forward(task, agent_act)
            allowed_slots = {default_slot} if default_slot else {"breakfast", "lunch", "dinner"}
            if "option_id" in params and slot in allowed_slots:
                for summary_id in self._stored_node_ids:
                    summary = self.exploration_tree.summaries[summary_id]
                    if any((selection["day"], selection["slot_type"]) == (day, slot)
                           for selection in json.loads(summary.content)):
                        return (
                            "You have already selected this slot. To change the selection of it, "
                            f"Backtrace to Step {summary.revisit_step} and provide the reason and new choice "
                            "in the reason parameter. Re-execute the selection after Backtrace."
                        )
        return super().forward(task, agent_act)

    def _prior_traveler_queries(self) -> dict[str, str] | None:
        """Expose complete public queries for the active predecessors only."""
        ss = self.shared_state
        previous = [record for record in ss.traveler_tree.active_records()
                    if 0 <= record.index < self.turn_index]
        if sorted(record.index for record in previous) != list(range(self.turn_index)):
            return None
        queries = [ss.base_traveler_query, *(ss.traveler_queries[record.index] for record in previous)]
        result: dict[str, str] = {}
        names: set[str] = set()
        for query in queries:
            name = derive_traveler_name(query)
            normalized = " ".join(name.lower().replace('’', "'").split())
            if not query.strip() or normalized == 'traveler' or normalized in names:
                return None
            names.add(normalized)
            result[name] = query
        return result

    def _build_memory_context(self) -> dict:
        ss = self.shared_state
        previous = [rec for rec in ss.traveler_tree.active_records() if rec.index < self.turn_index]
        lines = []
        for record in previous:
            lines.append(f"[Step {record.parent_node_id}] Query: {ss.traveler_queries[record.index]}\n{summarize_traveler_plan(record)}")
        return {
            "base_traveler_query": ss.base_traveler_query,
            "base_traveler_plan": copy.deepcopy(ss.base_traveler_plan),
            "all_previous_travelers": "\n".join(lines),
        }

    def build_task(
        self,
        query: str | None = None,
        *,
        query_index: int | None = None,
        traveler_name: str | None = None,
        session_id: str | None = None,
        query_row: dict[str, Any] | None = None,
    ) -> TaskPackage:
        """Prepare one fresh traveler, without resetting other travelers or assigning."""
        if query is not None and query != self.traveler_query:
            raise ValueError("The task query must match this agent's registered traveler.")
        name = traveler_name or derive_traveler_name(self.traveler_query)
        session_id = session_id or build_session_id(name, self.turn_index + 2)
        self.runtime.prepare_session(session_id, query_index=query_index, traveler_name=name)
        self.runtime_state = {"session_id": session_id, "traveler_name": name}
        row = query_row if query_row is not None else self.runtime.env.query_row
        self._query_row = {key: row[key] for key in ("days", "budget") if key in row}
        self._rebind_actions(session_id)
        self._backtrace_context = [
            (None, None, reason) for owner, reason in self.shared_state.pending_backtrace_reasons
            if owner > self.turn_index
        ]
        self._previous_results = self.shared_state.traveler_tree.previous_attempts(self.turn_index)
        self.shared_state.traveler_settings[self.turn_index] = {
            "query_index": self.runtime.query_id, "traveler_name": name,
            "session_id": session_id, "query_row": copy.deepcopy(self._query_row),
        }
        self._exploration_hint = ""
        memory = self._build_memory_context()
        sections = [
            f"Supplied base traveler\n{memory['base_traveler_query'] or '-'}",
            f"The fixed base traveler plan:\n{json.dumps(memory['base_traveler_plan'], ensure_ascii=False, indent=2)}",
        ]
        if memory["all_previous_travelers"]:
            sections.append(f"Prior travelers' current plans\n{memory['all_previous_travelers']}")
        sections.append(f"Current traveler's requirements\n{self.traveler_query}")
        return TaskPackage(task_id=str(uuid.uuid4()), instruction="\n\n".join(sections))

    # Counters and traveler records stay outside environment snapshots.
    def _get_env_snapshot(self) -> dict:
        return self.runtime.snapshot_full()

    def _env_compare(self, snap_a: Any, snap_b: Any) -> bool:
        session_id = self.runtime_state["session_id"]
        plans = [
            [day for day in (snapshot or {}).get("sessions", {}).get(session_id, {}).get("itinerary", [])
             if day.get("selected_options")]
            for snapshot in (snap_a, snap_b)
        ]
        return plans[0] == plans[1]

    def _on_backtrace_env_restore(self, env_snapshot: dict) -> None:
        self.runtime.restore_full(env_snapshot)
        session_id = self.runtime_state["session_id"]
        if self.runtime.current_session_id != session_id:
            self.runtime.prepare_session(session_id, traveler_name=self.runtime_state["traveler_name"])
        self._rebind_actions(session_id)
        self._pre_verify_snapshot = None

    def _restore_context(self, target_step: int, task: TaskPackage | None = None):
        """Resolve checkpoint aliases; the current task start always wins ties."""
        ss = self.shared_state
        tree = self.exploration_tree
        node = tree.nodes.get(target_step)
        canonical_step = node.node_id if node is not None else target_step
        checkpoints: dict[int, tuple] = {}

        # Previous traveler boundaries; later records win shared-node ties.
        for record in ss.traveler_tree.active_records():
            if record.index >= self.turn_index or record.parent_node_id is None:
                continue
            parent = tree.nodes.get(record.parent_node_id)
            if parent is not None:
                checkpoints[parent.node_id] = (
                    "traveler", record.index, parent.node_id, parent.env_snapshot, record.parent_summary_id,
                )

        # Store and raw segment starts retain only the preceding Store prefix.
        for index, summary_id in enumerate(self._stored_node_ids):
            parent = tree.get_summary_parent(summary_id)
            if parent is not None:
                restore_summary_id = self._stored_node_ids[index - 1] if index > 0 else self._task_start_summary_id
                checkpoints[parent.node_id] = (
                    "store", index, parent.node_id, parent.env_snapshot, restore_summary_id,
                )
        if self._raw_node_ids:
            parent = tree.get_parent(self._raw_node_ids[0])
            if parent is not None:
                restore_summary_id = self._stored_node_ids[-1] if self._stored_node_ids else self._task_start_summary_id
                checkpoints[parent.node_id] = (
                    "store", len(self._stored_node_ids), parent.node_id, parent.env_snapshot, restore_summary_id,
                )

        # Task start is last: undo this attempt, even at a shared traveler boundary.
        if self._task_start_node_id is not None:
            start = tree.nodes.get(self._task_start_node_id)
            if start is not None:
                checkpoints[start.node_id] = (
                    "store", 0, start.node_id, start.env_snapshot, self._task_start_summary_id,
                )

        if canonical_step not in checkpoints:
            return (
                f"Backtrace failed because step {target_step} is not valid. "
                f"Valid steps are {sorted(checkpoints.keys())}."
            )

        kind, index, target_node_id, target_env, target_summary_id = checkpoints[canonical_step]
        if kind == "store":
            self._stored_context = self._stored_context[:index]
            self._stored_node_ids = self._stored_node_ids[:index]
            self._raw_context = []
            self._raw_node_ids = []
        else:
            ss.traveler_tree.trim_active_path(index)
            self._replay_target_turn_index = max(self.turn_index, self._replay_target_turn_index or 0)
            task.completion = "replay"
        return target_node_id, target_summary_id, target_env

    def _do_store(self, task: TaskPackage, action: AgentAct) -> tuple[bool, str]:
        """Compress successful tool returns; core owns merge/check/backtrace."""
        selections = {}
        for node_id in self._raw_node_ids:
            node = self.exploration_tree.nodes[node_id]
            if node.action is None or node.action.name not in _SELECT_ACTIONS:
                continue
            try:
                selection = json.loads(node.observation)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(selection, dict) or not selection.get("selected_option"):
                continue
            selections[(selection["day"], selection["slot_type"])] = selection
        if not selections:
            return False, "No new restaurant or accommodation selections to store."
        content = json.dumps([selections[key] for key in sorted(selections)], ensure_ascii=False, sort_keys=True)
        action = action.model_copy(deep=True)
        action.params = {"response": content}
        self._store_check_failed = False
        success, observation = super()._do_store(task, action)
        # A failed check with a successful backtrace also returns success=True.
        if success and not self._store_check_failed:
            self.exploration_tree.current_summary_node.metadata["session_id"] = self.runtime_state["session_id"]
        return success, observation

    def _extract_summary_key(self, content: str) -> str:
        choices = sorted(
            (selection["day"], selection["slot_type"], selection["selected_option"]["option_id"])
            for selection in json.loads(content)
        )
        return json.dumps([self.runtime_state["session_id"], choices], ensure_ascii=False)

    def _check_store(self, task: TaskPackage, content: str) -> tuple[bool, str]:
        """Check only this segment's successful selections, without an LLM call."""
        plan = [
            {"days": selection["day"], "selected_options": {selection["slot_type"]: selection["selected_option"]}}
            for selection in json.loads(content)
        ]
        violations = verify_slot_constraints(
            self.traveler_query, plan,
            base_plan=self.shared_state.base_traveler_plan,
            base_traveler_name=derive_traveler_name(self.shared_state.base_traveler_query),
            prior_traveler_queries=self._prior_traveler_queries(),
        )
        self._store_check_failed = bool(violations)
        if not violations:
            return True, ""
        reason = "; ".join(" ".join(violation.split()).rstrip(".") for violation in violations[:3])
        return False, (
            f"CONSTRAINT VIOLATIONS: {reason}. "
            "Search for alternatives that satisfy the constraints, then re-select the slots in this segment."
        )

    def _compute_exploration_hint(self) -> str:
        children = self.exploration_tree.get_summary_children(self.exploration_tree.current_summary_id)
        session_id = self.runtime_state.get("session_id")
        requested = _parse_query_slot_constraints(
            self.traveler_query, base_traveler_name=derive_traveler_name(self.shared_state.base_traveler_query),
            prior_traveler_queries=self._prior_traveler_queries(),
        )
        candidates: dict[tuple[int, str], dict[str, dict]] = {}
        for child in children:
            if child.metadata.get("session_id") != session_id:
                continue
            for selection in json.loads(child.content):
                key = (selection["day"], selection["slot_type"])
                if key not in requested:
                    continue
                option = selection["selected_option"]
                candidates.setdefault(key, {})[option["option_id"]] = option
        blocks = []
        for (day, slot), options in sorted(candidates.items()):
            lines = [f"- {json.dumps(option, ensure_ascii=False)}" for option in options.values()]
            if slot == "accommodation":
                search, select = "AccommodationSearch", "SelectHotel"
            else:
                search, select = "RestaurantSearch", "SelectRestaurant"
            blocks.append(
                f"Here are selections for Day {day} {slot} in your previous attempts:\n"
                + "\n".join(lines)
                + "\n\nThese selections are from previous attempts, not the current attempt. "
                f"If you think one of the previous selections is correct, use {search} to find it, "
                f"then {select} to select it; if not, search for a better matched selection."
            )
        return "\n\n".join(blocks)

    def _extract_curr_env(self, snapshot) -> str | None:
        session_id = self.runtime_state.get("session_id")
        if not session_id:
            return None
        plan = self.runtime.env.get_itinerary(session_id)
        _, missing, _ = _analyze_joining_query_slots(
            self.traveler_query, plan,
            base_plan=self.shared_state.base_traveler_plan,
            base_traveler_name=derive_traveler_name(self.shared_state.base_traveler_query),
            prior_traveler_queries=self._prior_traveler_queries(),
        )
        if not missing:
            return None
        return "TODO:\n" + "\n".join(f"- {requirement}" for requirement in missing)

    # Verification is called by the core only after this traveler's Finish.
    def _verify_task(self, task: TaskPackage) -> tuple[bool, str]:
        session_id = self.runtime_state["session_id"]
        plan = self.runtime.env.get_itinerary(session_id)
        query = self.traveler_query
        base_name = derive_traveler_name(self.shared_state.base_traveler_query)
        prior_queries = self._prior_traveler_queries()
        _, missing, _ = _analyze_joining_query_slots(
            query, plan, base_plan=self.shared_state.base_traveler_plan, base_traveler_name=base_name,
            prior_traveler_queries=prior_queries,
        )
        if missing:
            reason = "; ".join(item.rstrip(".") for item in missing[:5])
            passed, reason = False, (
                f"Missing requested selections: {reason}. "
                'Search and select the requested slots, then call Finish[{"response":"done"}].'
            )
        else:
            violations = verify_slot_constraints(
                query, plan, base_plan=self.shared_state.base_traveler_plan, base_traveler_name=base_name,
                prior_traveler_queries=prior_queries,
            )
            if violations:
                reason = "; ".join(" ".join(violation.split()).rstrip(".") for violation in violations[:5])
                passed, reason = False, (
                    f"CONSTRAINT VIOLATIONS: {reason}. "
                    "Search for alternatives that satisfy the constraints, then select the requested slots."
                )
            else:
                passed, reason = self.verify_llm(query, plan)
        self.verification_feedback = None if passed else reason
        return passed, reason

    def verify_llm(self, query_text: str, plan: list[dict[str, Any]]) -> tuple[bool, str]:
        if self.max_verify_retries <= 0:
            return True, ""
        memory = self._build_memory_context()
        sections = [
            f"Supplied base traveler\n{memory['base_traveler_query']}",
            f"The fixed base traveler plan:\n{json.dumps(memory['base_traveler_plan'], ensure_ascii=False)}",
        ]
        if memory["all_previous_travelers"]:
            sections.append(f"Prior travelers' current plans\n{memory['all_previous_travelers']}")
        sections.extend([
            f"Current traveler's requirements\n{query_text}",
            f"Submitted plan\n{json.dumps(plan, ensure_ascii=False)}",
            "Check whether the requested meals and accommodation satisfy the current traveler's requirements. "
            "Resolve references using the supplied base plan and the prior travelers' current plans. "
            "For a requirement that explicitly keeps the base arrangement for the same day and item, "
            "check that supplied arrangement when there is no replacement in the submitted plan.",
            "Check the stated price conditions, rating conditions, cuisines, room type, house rules, and comparisons with other travelers. "
            '"Within", "at least", and "at most" include equality. Strictly higher or lower values exclude equality. '
            "When a request permits no additional cuisine types, check the complete cuisines list.",
            "Check the conditions stated in the current traveler's requirements using the supplied facts. "
            "Identify the affected day and meal or accommodation for any failure.",
            "Respond with YES if the arrangements satisfy the requirements, or NO followed by which requested items are wrong and why.",
        ])
        prompt = "\n\n".join(sections)
        self.logger.get_prompt(f"[Verify-LLM]: {prompt}")
        response = self.llm.run(prompt=prompt).strip()
        self.logger.get_llm_output(f"[Verify-LLM]: {response}")
        match = re.match(r"(YES|NO)\b[.,:\s]*(.*)", response, re.IGNORECASE | re.DOTALL)
        if match and match.group(1).upper() == "YES":
            return True, ""
        if match:
            reason = " ".join(match.group(2).split())[:300]
        else:
            reason = "Expected a YES or NO verdict. " + " ".join(response.split())[:300]
        return False, f"LLM verification failed. {reason} Search for alternatives that satisfy the constraints, then select the requested slots."

    def _on_task_complete(self, task: TaskPackage, last_obs: str = "") -> None:
        ss = self.shared_state
        ss.global_step = self._global_step
        ss.agent_backtrace_count = self._agent_backtrace_count
        if task.completion == "replay":
            if self._backtrace_context:
                ss.pending_backtrace_reasons.append((self.turn_index, self._backtrace_context[-1][2]))
            self._run_next_replay_turn(new_log=False)
            return
        session_id = self.runtime_state["session_id"]
        plan = self.runtime.env.get_itinerary(session_id)
        name = self.runtime_state["traveler_name"]
        record = TravelerRecord(
            index=self.turn_index, traveler_name=name, session_id=session_id,
            plan_json=copy.deepcopy(plan), completed=task.completion == "completed",
            parent_node_id=self._task_start_node_id, parent_summary_id=self._task_start_summary_id,
        )
        record.summary_text = summarize_traveler_plan(record)
        ss.traveler_tree.add_record(record)
        if self._is_verify_retry:
            record.verify_feedback = self.verification_feedback or last_obs
            self.verification_feedback = None
            self._previous_results = ss.traveler_tree.previous_attempts(self.turn_index)
            return
        if self._replay_target_turn_index is not None and self.turn_index < self._replay_target_turn_index:
            self._run_next_replay_turn(new_log=ss.judge_count > 0)
        else:
            ss.pending_backtrace_reasons.clear()
            if self._should_run_judge():
                self._run_judge()

    def _run_next_replay_turn(self, new_log: bool = False) -> None:
        ss = self.shared_state
        next_index = len(ss.traveler_tree.active_records())
        if self._replay_target_turn_index is None or next_index > self._replay_target_turn_index:
            return
        log_file = (
            os.path.join(self._log_dir, f"task_{self._task_id}_subtask_{next_index}_retry{ss.judge_count}.log")
            if new_log else self.logger.log_file_name
        )
        replay_agent = TravelPlanMAGEAgent(
            llm=self.llm, runtime=self.runtime, turn_index=next_index,
            traveler_query=ss.traveler_queries[next_index], shared_state=ss,
            max_steps=self.max_exec_steps, max_verify_retries=self.max_verify_retries,
            max_check_retries=self.max_check_retries, max_backtrace=self.max_backtrace,
            log_file=log_file, PROMPT_DEBUG_FLAG=self.logger.PROMPT_DEBUG_FLAG,
        )
        replay_agent._replay_target_turn_index = self._replay_target_turn_index
        task = replay_agent.build_task(**ss.traveler_settings.get(next_index, {}))
        replay_agent(task)

    def _should_run_judge(self) -> bool:
        ss = self.shared_state
        return (
            bool(ss.traveler_answers)
            and len(ss.traveler_queries) == len(ss.traveler_answers)
            and self.turn_index == len(ss.traveler_queries) - 1
            and len(ss.traveler_tree.active_records()) == len(ss.traveler_queries)
            and ss.judge_count < ss.max_judge_retries
        )

    def _run_judge(self) -> bool:
        ss = self.shared_state
        active = ss.traveler_tree.active_records()
        result = score_travelers(ss.base_traveler_plan, ss.traveler_answers, ss.get_active_results())
        mismatches = [index for index, passed in enumerate(result["sub_successes"], 1) if not passed]
        if not mismatches:
            return False
        self.logger.get_log(
            f"Judge: mismatched travelers: {mismatches}\n"
            f"Groundtruth answers: {[pair[0] for pair in result['answer_pairs']]}\n"
            f"Agent answers: {[pair[1] for pair in result['answer_pairs']]}"
        )
        lines = [
            "You are checking a group travel planning task. Each step planned one traveler's requirements. "
            "Identify which traveler to redo to correct the plans.",
            "",
            "For each traveler:",
            "- **CURRENT ATTEMPT** is the selected plan in the current group attempt.",
            "- **PREVIOUS ATTEMPT** is a plan tried before for this traveler, not the current plan.",
            "- visit_count is the number of independent attempts that produced that plan.",
            "",
            "Check each traveler's requested meals and accommodation against that traveler's requirements, "
            "using the base plan and the prior travelers' current plans for references. "
            "Check the stated price conditions, ratings, cuisines, room types, house rules, and comparisons. "
            '"Within", "at least", and "at most" include equality; strictly higher or lower values exclude equality. '
            "Use an explicitly retained base arrangement when that item has no replacement.",
            "",
            "Supplied base traveler",
            ss.base_traveler_query,
            "",
            "The fixed base traveler plan:",
            json.dumps(ss.base_traveler_plan, ensure_ascii=False),
            "",
        ]
        for record in active:
            lines.append(f"Traveler {record.index + 1} requirements\n{ss.traveler_queries[record.index]}")
            feedback = f" But may have issues: {' '.join(record.verify_feedback.split())}" if record.verify_feedback else ""
            lines.append(f"**CURRENT ATTEMPT** {summarize_traveler_plan(record)} visit_count: {record.visit_count}{feedback}")
            parent = ss.traveler_tree.parent_for_index(record.index)
            for rid in parent.children_record_ids:
                previous = ss.traveler_tree.records[rid]
                if previous.index == record.index and rid != record.record_id:
                    feedback = f" But may have issues: {' '.join(previous.verify_feedback.split())}" if previous.verify_feedback else ""
                    lines.append(f"**PREVIOUS ATTEMPT** {previous.summary_text} visit_count: {previous.visit_count}{feedback}")
            lines.append("")
        lines.append(
            "Choose ONE traveler whose replanning can fix the issues when that traveler and all later travelers are redone. "
            "If several choices can fix the issues, choose the earliest traveler. "
            "Respond with ONE traveler index in list format, followed by a short reason, for example, [1]. Reason: ..."
        )
        prompt = "\n".join(lines)
        self.logger.get_prompt(f"[Judge]: {prompt}")
        response = self.llm.run(prompt=prompt).strip()
        self.logger.get_llm_output(f"[Judge]: {response}")
        match = re.search(r"\[\s*(\d+)\s*\]", response)
        if match is None:
            self.logger.get_log("Warning: Do not match any traveler index in judge response. Skipping judge.")
            return False
        index = int(match.group(1)) - 1
        if not 0 <= index < len(ss.traveler_queries):
            self.logger.get_log(f"Warning: Backtrack index {index} out of range. Skipping judge.")
            return False

        # The Judge request belongs to the old round; replay starts a fresh trace.
        retry_suffix = f"_retry{ss.judge_count}" if ss.judge_count > 0 else ""
        self.llm.save_trace(os.path.join(self._log_dir, f"task_{self._task_id}_token_usage{retry_suffix}.json"))
        self.llm.reset_trace()

        selected = active[index]
        ss.judge_count += 1
        ss.traveler_tree.trim_active_path(index)
        self.exploration_tree.set_current(selected.parent_node_id)
        self.exploration_tree.set_current_summary(selected.parent_summary_id)
        self.runtime.restore_full(self.exploration_tree.nodes[selected.parent_node_id].env_snapshot)
        ss.pending_backtrace_reasons.append((index + 1, JUDGE_RETRY_GUIDANCE))
        self._replay_target_turn_index = len(ss.traveler_queries) - 1
        self._run_next_replay_turn(new_log=True)
        return True
