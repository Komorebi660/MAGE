from typing import Optional
from pydantic import BaseModel


class AgentAct(BaseModel):
    """Using AgentAct class to design the agent self-actions and API-call actions

    :param name: action name
    :type name: str
    :param desc: the description/documents of this action
    :type desc: str, optional
    :param params: the parameters of this action
    :type params: dict, optional
    :param valid: whether this action is valid (e.g. parse successfully, match an action/agent), default True
    :type valid: bool, optional
    """

    name: str
    desc: str = None
    params: dict = None
    valid: bool = True


ActObsChainType = list[tuple[Optional[AgentAct], str]]
