from typing import List

from agentcore.actions.BaseAction import BaseAction
from agentcore.prompts.prompt_utils import (
    DEFAULT_PROMPT,
    PROMPT_TOKENS,
    format_act_params_example,
    task_chain_format,
)
from agentcore.commons import AgentAct, TaskPackage


class PromptGen:
    """Prompt Generator Class"""

    def __init__(self) -> None:
        self.prompt_type = "BasePrompt"
        self.examples: dict[str, list] = {}

    def add_example(
        self,
        task: TaskPackage,
        action_chain: List[tuple[AgentAct, str]],
        example_type: str = "action",
    ):
        example_context = task_chain_format(task, action_chain)
        if example_type in self.examples:
            self.examples[example_type].append(example_context)
        else:
            self.examples[example_type] = [example_context]

    def __get_examples__(self, example_type: str) -> str:
        """get multiple examples for prompt"""
        if example_type not in self.examples:
            return None
        return "\n".join(self.examples[example_type])


class BasePromptGen(PromptGen):
    """
    this is the BasePrompt for agent to use.
    """

    def __init__(
        self,
        agent_role: str = None,
        constraint: str = DEFAULT_PROMPT["constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
    ):
        """Prompt Generator for Base Agent
        :param agent_role: the role of this agent, defaults to None
        :type agent_role: str, optional
        :param constraint: the constraint of this agent, defaults to None
        :type constraint: str, optional
        """
        super().__init__()
        self.prompt_type = "BaseAgentPrompt"
        self.agent_role = agent_role
        self.constraint = constraint
        self.instruction = instruction

    def __constraint_prompt__(self):
        if self.constraint:
            return f"""{PROMPT_TOKENS["constraint"]['begin']}\n{self.constraint}\n{PROMPT_TOKENS["constraint"]['end']}"""
        else:
            return ""

    def __role_prompt__(self, agent_role):
        prompt = f"""{PROMPT_TOKENS["role"]['begin']}\n{agent_role}\n{PROMPT_TOKENS["role"]['end']}"""
        return prompt

    def __act_doc_prompt__(self, actions: List[BaseAction]):
        action_doc = [
            {
                "name": act.action_name,
                "description": act.action_desc,
                "parameters": act.params_doc,
            }
            for act in actions
        ]
        prompt = f"""{PROMPT_TOKENS["action"]['begin']}\n{action_doc}\n{PROMPT_TOKENS["action"]['end']}"""
        return prompt

    def __prompt_example__(self, prompt_example: str):
        prompt = f"""{PROMPT_TOKENS["example"]['begin']}\n{prompt_example}{PROMPT_TOKENS["example"]['end']}\n"""
        return prompt

    def __act_format_example__(self, act_call_example: str):
        prompt = f"""{DEFAULT_PROMPT["action_format"]}{PROMPT_TOKENS["action_format"]['begin']}\n{act_call_example}{PROMPT_TOKENS["action_format"]['end']}\n"""
        return prompt

    def action_prompt(
        self,
        task: TaskPackage,
        actions: List[BaseAction],
        action_chain: List[tuple[AgentAct, str]],
        example_type: str = "action",
        example: str = None,
        **kwargs,
    ) -> str:
        """return the action generation prompt for agent
        :param task: the task to finish
        :type task: TaskPackage
        :param actions: the actions to take
        :type actions: List[BaseAction]
        :param action_chain: the history action-obs chain of this task
        :type action_chain: List[tuple[AgentAct, str]]
        :param example_type: the type of example, defaults to "action"
        :type example_type: str, optional
        :param example: the example string, defaults to None
        :type example: str, optional
        :return: the prompt for agent to take action
        :rtype: str
        """
        # adding roles into prompt
        prompt = f"""{self.instruction}\n{self.__role_prompt__(self.agent_role)}\n"""
        # adding constraint into prompt
        prompt += f"""{self.__constraint_prompt__()}\n"""
        # adding action doc into prompt
        prompt += (
            f"""{self.__act_doc_prompt__(actions=actions)}\n"""
        )
        # get task example
        if example:  # get from input
            prompt_example = example
        else:  # get from self.examples
            prompt_example = self.__get_examples__(example_type)

        if prompt_example:  # if have example, put into prompt
            prompt += self.__prompt_example__(prompt_example)
        else:  # no example provided, only add the format example
            # adding format example
            act_call_example = format_act_params_example(actions)
            prompt += self.__act_format_example__(act_call_example)
        # adding action observation chain
        cur_session = task_chain_format(task, action_chain)
        prompt += f"""{PROMPT_TOKENS["execution"]['begin']}\n{cur_session}"""
        # adding inference token
        prompt += """Action:"""
        return prompt
