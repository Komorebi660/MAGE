from agentcore.actions.BaseAction import BaseAction

DEF_INNER_ACT_OBS = "OK"
INNER_ACT_KEY = "response"


class ThinkAction(BaseAction):
    def __init__(self) -> None:
        action_name = "Think"
        action_desc = "Conduct thinking and reasoning process for solving task."
        params_doc = {
            INNER_ACT_KEY: "This is your thinking response. Be specific and critical."
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(self, **kwargs):
        return DEF_INNER_ACT_OBS


class FinishAction(BaseAction):
    def __init__(self) -> None:
        action_name = "Finish"
        action_desc = "Complete the task with a response."
        params_doc = {
            INNER_ACT_KEY: "This is the finish response. Respond towards the task instruction. Be concise and direct."
        }
        super().__init__(
            action_name=action_name,
            action_desc=action_desc,
            params_doc=params_doc,
        )

    def __call__(self, response):
        return response
