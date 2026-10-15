"""
Memory-related actions for MAGEAgent.
"""

from agentcore.actions.BaseAction import BaseAction


STORE_SUMMARY_KEY = "response"
BACKTRACE_STEP_KEY = "step_idx"
BACKTRACE_REASON_KEY = "reason"


class StoreAction(BaseAction):
    """Store the current context segment into a short summary.

    The agent provides the summary as the action parameter (``response``).
    Calling this action triggers the MAGEAgent to:

    1. Read the summary text from ``params["response"]``.
    2. Store a SummaryNode in the ExplorationTree's upper layer.
    3. Replace the current context segment with a single entry
       ``(StoreAction, "[Summary]: <text>")``, clearing the segment.
    4. Mark the beginning of a fresh segment from this point onward.

    The agent should call Store after it has gathered enough
    information to record its findings, so the context does not
    grow indefinitely during long explorations.
    """
    action_name: str = "Store"  # used by StoreAction.action_name

    def __init__(self) -> None:
        action_name = "Store"
        action_desc = (
            "Store previous gathered information into a concise summary."
        )
        params_doc = {
            STORE_SUMMARY_KEY: (
                "your gathered information"
            )
        }
        super().__init__(action_name, action_desc, params_doc)

    def __call__(self, response: str = "") -> str:
        return f"Stored: {response}"


class BacktraceAction(BaseAction):
    """Backtrace the agent's execution to a previous step.

    Calling this action triggers the MAGEAgent to:

    1. Locate the TreeNode whose ``step_idx`` equals the requested value.
    2. Reconstruct the visible context (context_window) as the sequence
       of action-observation pairs along the path from the root to that
       node.
    3. Call ``_on_backtrace_env_restore(node)`` so that task-specific
       subclasses can restore environment state (e.g. webshop session).
    4. Set the ExplorationTree's current pointer to that node so
       that new actions will branch off from there.

    The agent should call Backtrace when it detects that a previous
    decision was suboptimal and it needs to retry from an earlier point.
    """
    action_name: str = "Backtrace"  # used by BacktraceAction.action_name

    def __init__(self) -> None:
        action_name = "Backtrace"
        action_desc = (
            "Backtrace to a specific previous step to undo recent actions and retry from that point."
        )
        params_doc = {
            BACKTRACE_STEP_KEY: "the step index (integer) to backtrace to",
            BACKTRACE_REASON_KEY: "brief explanation of why you backtrace",
        }
        super().__init__(action_name, action_desc, params_doc)

    def __call__(self, step_idx: int, reason: str = "") -> str:
        msg = f"Restored context to step {step_idx}."
        if reason:
            msg += f" Reason: {reason}"
        return msg
