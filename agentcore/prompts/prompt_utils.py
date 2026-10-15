import json

from agentcore.actions.BaseAction import BaseAction
from agentcore.commons import AgentAct, TaskPackage

PROMPT_TASK_KEY = "task"
PROMPT_ACT_OBS_KEY = "act_obs"

PROMPT_TOKENS = {
    "instruction": {"begin": "[Instruction]", "end": "[End of Instruction]"},
    "role": {"begin": "[Role]", "end": "[End of Role]"},
    "constraint": {"begin": "[Constraint]", "end": "[End of Constraint]"},
    "action": {"begin": "[Action_Doc]", "end": "[End of Action_Doc]"},
    "example": {"begin": "[Example]", "end": "[End of Example]"},
    "action_format": {
        "begin": "[ActionFormatExample]",
        "end": "[End of ActionFormatExample]",
    },
    "execution": {"begin": "[Execution]", "end": "[End of Execution]"},
}


DEFAULT_PROMPT = {
    "agent_instruction": f"""You are an intelligent agent. You should follow your {PROMPT_TOKENS["role"]['begin']}, {PROMPT_TOKENS["action"]['begin']} to take actions. Your generation should follow the example format. Finish the task as best as you can.""",
    "constraint": "You generation should be simple and clear.",
    "action_format": "Using the following action format example to generate well formatted actions.\n",
    "not_completed": "I cannot help with that. Please be more specific.",
}


def format_act_params_example(actions: list[BaseAction]):
    """
    format the api call parameters with the provided api doc
    """
    act_params_example_str = ""
    for act in actions:
        if act.params_doc is None:
            raise KeyError("No API call params doc provided")
        agent_act = AgentAct(name=act.action_name, params=act.params_doc)
        act_str = action_format(agent_act)
        act_params_example_str += act_str
        act_params_example_str += "\n"
    return act_params_example_str


def action_format(act: AgentAct, action_trigger: bool = True) -> str:
    """unified format the action as a string"""
    str_params = json.dumps(act.params)
    action = f"{act.name}[{str_params}]"
    return f"Action:{action}" if action_trigger else action


def action_chain_format(action_chain: list[tuple[AgentAct, str]]):
    """Unified format of action generation of inner actions and outer actions.

    If the action in a pair is ``None``,
    only the ``Observation:`` line is emitted, with no ``Action:`` header.
    """
    history = ""
    for act, obs in action_chain:
        if act is not None:
            history += f"""{action_format(act)}\nObservation: {obs}\n"""
        else:
            history += f"""{obs}\n"""
    return history


def task_chain_format(task: TaskPackage, action_chain: list[tuple[AgentAct, str]]):
    context = f"Task:{task.instruction}\n"
    context += action_chain_format(action_chain)
    return context
