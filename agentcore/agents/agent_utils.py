"""functions or objects shared by agents"""

import re
import json

from agentcore.actions.BaseAction import BaseAction


def name_checking(name: str):
    """ensure no white space in name"""
    white_space = [" ", "\n", "\t"]
    for w in white_space:
        if w in name:
            return False
    return True


def act_match(input_act_name: str, act: BaseAction):
    if input_act_name == act.action_name:  # exact match
        return True
    ## To-Do More fuzzy match
    return False


def _truncate_tokens(text: str, max_tokens: int = 20) -> str:
    """Truncate text to at most max_tokens tokens."""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens]) + " ..."


def parse_action(string: str) -> tuple[str, dict, bool]:
    """
    Parse an action string of the form ActionName[{"key": "value", ...}].
    Returns (action_name, params_dict, success_bool).
    Tolerates minor bracket mismatches but rejects concatenated action calls.
    """
    string = string.strip().strip(".").strip(":")

    if string.startswith("Action:"):
        string = string[len("Action:"):].strip()

    if string.startswith("thinkAction:"):
        string = string[len("thinkAction:"):].strip()

    # Take only the first line AFTER prefix stripping
    string = string.split("\n")[0].strip()

    prefix_match = re.match(r"^(\w+)\[", string)
    if not prefix_match:
        return _truncate_tokens(string), {}, False

    action_type = prefix_match.group(1)
    rest = string[prefix_match.end():]

    def _has_trailing_actions(pos: int) -> bool:
        """Check if content after *pos* contains concatenated action calls."""
        tail = rest[pos:].strip().strip(']').strip()
        return bool(tail and re.search(r'\w+\[', tail))

    # Forward scan: try '}' and ']' positions left-to-right to find the
    # earliest valid JSON dict.  This naturally handles extra/missing trailing
    # brackets and rejects concatenated action calls.
    for i, ch in enumerate(rest):
        if ch == '}':
            attempts = (rest[:i + 1],)
        elif ch == ']':
            seg = rest[:i].rstrip()
            attempts = (seg, seg + '}')
        else:
            continue
        for candidate in attempts:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                if _has_trailing_actions(i + 1):
                    return _truncate_tokens(string), {}, False
                return action_type, parsed, True

    # No '}' or ']' produced valid JSON – try appending '}' for a missing brace
    body = rest.rstrip().rstrip(']').rstrip()
    if body:
        try:
            parsed = json.loads(body + '}')
            if isinstance(parsed, dict):
                return action_type, parsed, True
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: fix unescaped double quotes inside a single-key JSON value.
    # E.g. search[{"query": "individual "first child" speculation 2023"}]
    if body:
        m = re.match(r'\{\s*"(\w+)"\s*:\s*"(.*)"?\s*\}$', body, re.DOTALL)
        if m:
            key, raw = m.group(1), m.group(2).rstrip('"').rstrip()
            if re.search(r'\w+\[\{', raw):
                return _truncate_tokens(string), {}, False
            fixed = re.sub(r'(?<!\\)"', '\\"', raw)
            try:
                parsed = json.loads('{' + f'"{key}": "{fixed}"' + '}')
                if isinstance(parsed, dict):
                    return action_type, parsed, True
            except (json.JSONDecodeError, ValueError):
                pass

    return _truncate_tokens(string), {}, False


ACION_NOT_FOUND_MESS = (
    """Wrong action name or format. Correct action format should be ActionName[{"arg1": "value1", ...}]. Check available action list in [Action_Doc]."""
)