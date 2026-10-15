"""Online constraint scoring for multi-traveler TravelPlan outputs.

Pure scoring layer (no agentcore, no IO): given an agent's ``plan_json`` and the
ground-truth ``reference_answers`` for a multi-traveler sample, compute the
constraint-satisfaction metrics. ``score_travelers`` aggregates the active
follow-up answers for both the evaluator and the programmatic Judge gate;
``evaluate_session`` defines the shared constraint-matching rules.

Metrics:
  SR  — Success Rate: 1 if ALL sessions pass, else 0
  PS  — Pass-rate per Session: #passed / #total

A "constraint" is a slot (transportation, breakfast, lunch, dinner,
accommodation) where the ground-truth answer for a traveler differs from the
base traveler's plan for that same day. Slots where GT == base are inherited and
NOT scored.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Slots we compare (order doesn't matter)
SCORED_SLOTS = ["transportation", "breakfast", "lunch", "dinner", "accommodation"]

# Pattern to strip state suffixes like "(Pennsylvania)", "(Texas)", "(Montana)" from city names
_STATE_SUFFIX_RE = re.compile(r'\((?:Pennsylvania|Texas|Montana|California|Florida|Colorado|'
                              r'Kentucky|Louisiana|Illinois|Iowa|Georgia|New York|Ohio|'
                              r'Virginia|Maryland|Indiana|Michigan|Wisconsin|Minnesota|'
                              r'Missouri|Tennessee|North Carolina|South Carolina|Alabama|'
                              r'Mississippi|Arkansas|Oklahoma|Kansas|Nebraska|Oregon|'
                              r'Washington|Arizona|Nevada|Utah|New Mexico|Idaho|Wyoming|'
                              r'Hawaii|Alaska|Connecticut|Delaware|Maine|Massachusetts|'
                              r'New Hampshire|New Jersey|Rhode Island|Vermont|West Virginia|'
                              r'South Dakota|North Dakota|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\)'
                              r'\s*$')


def normalize(value: str) -> str:
    """Normalize a slot value for comparison.

    Strips whitespace, removes state suffixes from city names so that
    'Restaurant, Philadelphia' matches 'Restaurant, Philadelphia(Pennsylvania)'.
    Also normalizes multiple spaces to one.
    """
    if value is None:
        return "-"
    v = value.strip()
    if v == "-":
        return v
    # Remove state suffix in parentheses from city names
    v = _STATE_SUFFIX_RE.sub('', v).strip()
    # Normalize extra whitespace (e.g. "MIDTOWN 45  NYC" vs "MIDTOWN 45 NYC")
    v = re.sub(r'\s+', ' ', v)
    return v


def get_day_key(plan_item: dict) -> int:
    """Extract day number from a plan item (handles both 'days' and 'day' keys)."""
    if "days" in plan_item:
        return int(plan_item["days"])
    if "day" in plan_item:
        return int(plan_item["day"])
    raise KeyError(f"No 'days' or 'day' key in plan item: {list(plan_item.keys())}")


def plans_to_dict(plans: list) -> Dict[int, dict]:
    """Convert a list of daily plans to {day_number: plan_dict}."""
    result = {}
    for p in plans:
        day = get_day_key(p)
        result[day] = p
    return result


def extract_constraints(
    base_plans: Dict[int, dict],
    gt_plans: Dict[int, dict],
) -> List[Tuple[int, str, str]]:
    """
    Identify the explicit constraints for a traveler session.

    Returns list of (day, slot, gt_value) where gt_value != base_value.
    """
    constraints = []
    for day in sorted(gt_plans.keys()):
        gt_day = gt_plans[day]
        base_day = base_plans.get(day, {})
        for slot in SCORED_SLOTS:
            gt_val = normalize(gt_day.get(slot, "-"))
            base_val = normalize(base_day.get(slot, "-"))
            if gt_val != base_val:
                constraints.append((day, slot, gt_val))
    return constraints


def evaluate_session(
    base_plans: Dict[int, dict],
    gt_plans: Dict[int, dict],
    agent_plans: Dict[int, dict],
    status: str,
) -> dict:
    """
    Evaluate a single session (traveler turn).

    Returns:
      {
        "constraints": [(day, slot, gt_val, agent_val, match)],
        "n_constraints": int,
        "n_satisfied": int,
        "n_unfilled": int,
        "n_wrong": int,
        "passed": bool,
      }
    """
    constraint_defs = extract_constraints(base_plans, gt_plans)
    n_constraints = len(constraint_defs)

    if n_constraints == 0:
        # No explicit constraints — trivially passed
        return {
            "constraints": [],
            "n_constraints": 0,
            "n_satisfied": 0,
            "n_unfilled": 0,
            "n_wrong": 0,
            "sPS": 1.0,
            "passed": True,
        }

    if status == "error" or len(agent_plans) == 0:
        # Agent crashed — zero score
        return {
            "constraints": [
                (day, slot, gt_val, "<error>", False)
                for day, slot, gt_val in constraint_defs
            ],
            "n_constraints": n_constraints,
            "n_satisfied": 0,
            "n_unfilled": n_constraints,
            "n_wrong": 0,
            "sPS": 0.0,
            "passed": False,
        }

    results = []
    n_satisfied = 0
    n_unfilled = 0
    n_wrong = 0
    for day, slot, gt_val in constraint_defs:
        agent_day = agent_plans.get(day, {})
        agent_val = normalize(agent_day.get(slot, "-"))
        match = agent_val == gt_val
        if match:
            n_satisfied += 1
        elif agent_val == "-":
            n_unfilled += 1
        else:
            n_wrong += 1
        results.append((day, slot, gt_val, agent_val, match))

    return {
        "constraints": results,
        "n_constraints": n_constraints,
        "n_satisfied": n_satisfied,
        "n_unfilled": n_unfilled,
        "n_wrong": n_wrong,
        "sPS": n_satisfied / n_constraints,
        "passed": n_satisfied == n_constraints,
    }


def score_travelers(base_plan: list, reference_answers: list[list], active_results: list[dict]) -> dict:
    """Score every expected follow-up without changing the raw active records.

    Missing/empty plans and unfinished shells without selected options score as
    empty answers. Completed submissions and genuine partial selections retain
    their full plans; zero-constraint answers still use evaluate_session's pass.
    """
    base = plans_to_dict(base_plan)
    by_index = {record["turn_index"]: record for record in active_results}
    successes, pairs = [], []
    for index, reference in enumerate(reference_answers):
        record = by_index.get(index, {})
        prediction = record.get("plan_json") or []
        if not record.get("completed", False) and not any(day.get("selected_options") for day in prediction):
            prediction = []
        score = evaluate_session(base, plans_to_dict(reference), plans_to_dict(prediction),
                                 record.get("status", "missing"))
        successes.append(int(score["passed"]))
        pairs.append([reference, prediction])
    return {
        "sub_successes": successes,
        "task_success": int(all(successes)),
        "answer_pairs": pairs,
    }
