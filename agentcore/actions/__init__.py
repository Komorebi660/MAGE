from .BaseAction import BaseAction
from .InnerActions import FinishAction, ThinkAction
from .MAGEActions import StoreAction, BacktraceAction

ThinkAct = ThinkAction()
FinishAct = FinishAction()
StoreAct = StoreAction()
BacktraceAct = BacktraceAction()
INNER_ACTIONS = [ThinkAct, FinishAct]
