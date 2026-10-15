"""
MAGEAgent - a general-purpose agent with hierarchical memory.

Layer 1: ExplorationTree
    Every action-observation pair is stored as a TreeNode.

Layer 2: Leaf SummaryNodes
    Created each time the agent calls the Store action.

Special actions
---------------
Store  (StoreAction)
    Store the context segment accumulated since the last ``Store``
    call.  The agent writes a summary inline as the action parameter.
    The segment is replaced in the visible context by a single
    (store_act, summary) entry;

Backtrace  (BacktraceAction)
    Restore the visible context and the tree's current-pointer to the
    state at a previous step index.  The hook ``_on_backtrace_env_restore``
    is called so task-specific sub-classes can restore environment state.
"""

from __future__ import annotations

from typing import Any, List

from agentcore.actions import BaseAction, ThinkAct, INNER_ACTIONS
from agentcore.actions.MAGEActions import BacktraceAction, StoreAction, \
    STORE_SUMMARY_KEY, BACKTRACE_STEP_KEY, BACKTRACE_REASON_KEY
from agentcore.agents.agent_utils import act_match
from agentcore.agents.BaseAgent import BaseAgent
from agentcore.commons import AgentAct, TaskPackage
from agentcore.commons.AgentAct import ActObsChainType
from agentcore.llm.LLMBackend import BaseLLM
from agentcore.logging import DefaultLogger
from agentcore.logging.terminal_logger import AgentLogger
from agentcore.memory.ExplorationTree import ExplorationTree
from agentcore.prompts.prompt_utils import DEFAULT_PROMPT


class MAGEAgent(BaseAgent):
    """
    Agent with hierarchical memory (ExplorationTree + Summaries).
    """

    previous_results_title = "Here are your previous results for the current task:"

    def __init__(
        self,
        name: str,
        role: str,
        llm: BaseLLM,
        actions: List[BaseAction] = [],
        constraint: str = DEFAULT_PROMPT["constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
        logger: AgentLogger = DefaultLogger,
        max_verify_retries: int = 0,
        max_check_retries: int = 0,
        max_backtrace: int = 0,
        **kwargs,
    ) -> None:
        # StoreAction and BacktraceAction are always available.
        actions = list(actions) + [StoreAction(), BacktraceAction()]
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            actions=actions,
            constraint=constraint,
            instruction=instruction,
            logger=logger,
            **kwargs,
        )

        # Hierarchical memory: ExplorationTree + leaf SummaryNodes
        self.exploration_tree = ExplorationTree()

        #===========================================================================================================#
        # Context visible to the LLM:
        # _backtrace_context + _previous_results + _stored_context + _exploration_hint + _raw_context + _current_env
        #===========================================================================================================#
        # Backtrace reasons: (caller_depth, target_depth, obs).
        #   (None, None, obs) = cross-subtask or verification, always at prompt start.
        #   (int,  int,  obs) = intra-subtask, inserted at target_depth position,
        #                       removed when len(_stored_context) >= caller_depth.
        self._backtrace_context: List[tuple[int | None, int | None, str]] = []
        # _previous_results holds finish responses from failed verification attempts
        self._previous_results: List[str] = []
        # _stored_context holds all historical summary entries
        self._stored_context: ActObsChainType = []
        # summary_ids of SummaryNodes created for _stored_context.
        self._stored_node_ids: List[str] = []
        # summary content of previously explored paths from the current summary node.
        self._exploration_hint: str = ""
        # _raw_context holds the current unprocessed action-observation trace
        self._raw_context: ActObsChainType = []
        # step indices of TreeNodes in the current _raw_context.
        self._raw_node_ids: List[int] = []
        # store current environment from observations.
        self._current_env: str = None

        # global step counter, exclude inner and invalid actions.
        # starts at 1 since step 0 is reserved for the root node.
        self._global_step: int = 1

        # Maximum number of verification-triggered retries (0 = no verification).
        self.max_verify_retries: int = max_verify_retries
        # Max store-check-triggered retries (0 = disable store checks).
        self.max_check_retries: int = max_check_retries
        self._check_retry_count: int = 0
        # Max successful agent-initiated backtrace calls (0 = no agent backtrace).
        self.max_backtrace: int = max_backtrace
        self._agent_backtrace_count: int = 0

        # index of the task start TreeNode
        self._task_start_node_id: int | None = None
        # index of the task start summary node
        self._task_start_summary_id: str | None = None

        # Whether _do_store() can update in-place. False--Yes, True--No
        self._store_dirty: bool = False

        # True while _on_task_complete is called from a verification retry.
        self._is_verify_retry: bool = False


    def _get_env_snapshot(self) -> Any:
        """Return a snapshot of current environment state."""
        return None

    def _env_compare(self, snap_a: Any, snap_b: Any) -> bool:
        """Compare two env_snapshots for merge decisions. Default: no comparison."""
        return True

    def _extract_curr_env(self, snapshot) -> str:
        """Extract a string description of the environment from snapshot."""
        return None


    def _check_store(self, task: TaskPackage, content: str) -> tuple[bool, str]:
        """Check whether stored content is valid. Returns (passed, reason).
        """
        return True, ""

    def _post_action_hook(self) -> None:
        """Called after a valid non-Store/non-Backtrace action is appended to _raw_context."""
        pass


    def _on_task_complete(self, task: TaskPackage, last_obs: str = "") -> None:
        """Called when task is completed or max steps reached.
        ``last_obs`` is the last observation returned during the execute loop
        """
        # If this is called from a verification retry, record the last observation
        if self._is_verify_retry:
            self._previous_results.append(last_obs)

    def _verify_task(self, task: TaskPackage) -> tuple[bool, str]:
        """Verify whether the task was completed correctly.
        """
        return True, ""


    def execute(self, task: TaskPackage) -> None:
        """Multi-step execution with memory management.
        """
        step_size = 0
        verify_attempts = 0
        self.logger.execute_task(task=task, agent_name=self.name)

        # Populate the root node's env_snapshot once the env is ready
        root_node = self.exploration_tree.root_node
        if root_node is not None and root_node.env_snapshot is None:
            root_node.env_snapshot = self._get_env_snapshot()

        # Record the task start node for backtrace
        self._task_start_node_id = self.exploration_tree.current_node_id
        self._task_start_summary_id = self.exploration_tree.current_summary_id

        last_obs: str = ""
        while True:
            # -- Inner execution loop --
            while task.completion == "active" and step_size < self.max_exec_steps:
                # Build finish results context
                finish_results_entries: ActObsChainType = []
                if self._previous_results:
                    results_text = "\n".join(
                        f"Attempt {i+1}: {r}"
                        for i, r in enumerate(self._previous_results)
                    )
                    finish_results_entries = [(None, f"{self.previous_results_title}\n{results_text}\nYou should decide whether to follow or explore new paths to fix the issues.")]

                # Split backtrace: external at top, intra interleaved into stored_context
                external_bt: ActObsChainType = [(None, f"- {o}") for d, _, o in self._backtrace_context if d is None]
                if len(external_bt) > 0:
                    external_bt.insert(0, (None, "\nBacktrace feedback from previous attempts:"))
                    if not finish_results_entries:
                        external_bt.append((None, "Use the feedback as guidance and try to resolve the issues and correctly complete the task."))
                intra_bt: dict[int, ActObsChainType] = {}
                for d, td, o in self._backtrace_context:
                    if d is not None:
                        intra_bt.setdefault(td, []).append((None, o))

                # Interleave intra backtrace notes into stored_context by target_depth
                interleaved_stored: ActObsChainType = list(intra_bt.get(0, []))
                for i, entry in enumerate(self._stored_context):
                    interleaved_stored.append(entry)
                    interleaved_stored.extend(intra_bt.get(i + 1, []))

                # Build exploration hint entry
                hint_entries: ActObsChainType = []
                if self._exploration_hint:
                    hint_entries = [(None, self._exploration_hint)]

                # Build the full action-observation chain for the LLM input
                step_hint = f", you can backtrace to Step {self._task_start_node_id} to undo your current attempt if needed" if not self._stored_context else ""
                action_obs_chain = (
                      external_bt
                    + finish_results_entries
                    + [(None, f"\nBelow are your current attempt for the current task{step_hint}:")]
                    + interleaved_stored
                    + ([(None, "Move to next step.\n")] if interleaved_stored else [])
                    + hint_entries
                    + self._raw_context
                )

                # Show _current_env when the last action is INNER, Store, Backtrace, or an invalid action,
                # or when raw_context is empty after a Store (stored_context is not empty).
                last_act = next((act for act, _ in reversed(self._raw_context) if act is not None), None)
                if self._current_env is not None:
                    if last_act is not None:
                        is_inner = any(act_match(last_act.name, a) for a in INNER_ACTIONS)
                        is_st_or_bt = act_match(last_act.name, StoreAction()) or act_match(last_act.name, BacktraceAction())
                        if is_inner or is_st_or_bt or not last_act.valid:
                            action_obs_chain += [(None, self._current_env)]
                    elif not self._raw_context and self._stored_context:
                        action_obs_chain += [(None, self._current_env)]

                # Warn agent when running low on steps
                remaining = self.max_exec_steps - step_size
                if remaining <= 3:
                    action_obs_chain += [(None, f"Warning: You have only {remaining} action budgets remaining. Based on what you have found so far, take the most effective actions to get the correct answer and call Finish action with the answer as soon as possible.")]

                # call LLM to infer next action
                action = self.__next_act__(task, action_obs_chain)
                self.logger.take_action(action, agent_name=self.name, step_idx=step_size)

                # parse actions
                if action.valid and act_match(action.name, StoreAction()):
                    if not self._raw_node_ids:
                        observation = "Store is not needed now because there is no new information to store. Take concrete actions to gather new information first."
                        self._raw_context.append((action, observation))
                    else:
                        success, observation = self._do_store(task, action)
                        if not success:
                            self._raw_context.append((action, observation))
                elif action.valid and act_match(action.name, BacktraceAction()):
                    if self.max_backtrace <= 0 or self._agent_backtrace_count >= self.max_backtrace:
                        observation = f"Backtrace is no longer available, you must complete the task without backtracking."
                        self._raw_context.append((action, observation))
                    else:
                        # Add prefix/suffix to agent-provided reason
                        raw_reason = (action.params or {}).get(BACKTRACE_REASON_KEY, "")
                        if raw_reason:
                            action.params[BACKTRACE_REASON_KEY] = f"Backtracked because: {raw_reason}"
                        success, observation = self._do_backtrace(task, action)
                        if not success:
                            self._raw_context.append((action, observation))
                        else:
                            self._agent_backtrace_count += 1
                else:
                    observation = self.forward(task, action)
                    is_inner = any(act_match(action.name, a) for a in INNER_ACTIONS)

                    # Discourage consecutive Think loops
                    if act_match(action.name, ThinkAct):
                        consec = 0
                        for prev_act, _ in reversed(self._raw_context):
                            if prev_act is not None and prev_act.valid and act_match(prev_act.name, ThinkAct):
                                consec += 1
                            else:
                                break
                        if consec >= 4:
                            observation = "You have been thinking multiple times in a row. Stop repeating the thinking and take a concrete action now."

                    if not is_inner and action.valid:
                        snapshot = self._get_env_snapshot()
                        self.exploration_tree.add_node(
                            action, observation, self._global_step, snapshot, self._env_compare,
                        )
                        self._global_step += 1
                        self._raw_node_ids.append(
                            self.exploration_tree.current_node_id
                        )
                        self._current_env = self._extract_curr_env(snapshot)

                    if not action.valid:
                        # Check if the last real action was also invalid
                        last_act = next((e[0] for e in reversed(self._raw_context) if e[0] is not None), None)
                        is_consecutive = last_act is not None and not last_act.valid
                        # Remove all previous error entries and format hints
                        self._raw_context = [
                            e for e in self._raw_context
                            if e[0] is not None and e[0].valid
                        ]
                        # Keep only the latest error
                        self._raw_context.append((action, observation))
                        # Add format hint on consecutive errors
                        if is_consecutive:
                            self._raw_context.append((None, 'Your Response should in the format like ActionName[{param dict}], for example: Finish[{"response": "..."}]'))
                    else:
                        self._raw_context.append((action, observation))
                        self._post_action_hook()

                self.__st_memorize__(task, action, observation)
                self.logger.get_obs(obs=observation)
                last_obs = observation
                step_size += 1

            # -- Verification --
            if self.max_verify_retries > 0 and verify_attempts < self.max_verify_retries:
                if task.completion == "completed":
                    passed, reason = self._verify_task(task)
                elif task.completion == "active" and step_size >= self.max_exec_steps:
                    passed, reason = False, "Your last attempt for this task took too many steps without completing the task, e.g., failed to call correct Finish action."
                    # Save a snapshot of the current env state so that
                    # the retry backtrace can preserve already-filled
                    # selections instead of reverting to the start state.
                    self._pre_verify_snapshot = self._get_env_snapshot()
                else:  # early exit due to cross-task backtrace calling (replay)
                    passed, reason = True, ""

                if not passed:
                    verify_attempts += 1
                    self._is_verify_retry = True
                    self._on_task_complete(task, last_obs + "But not pass verification: " + reason)
                    self._is_verify_retry = False
                    # Construct backtrace action to retry the task from the start node.
                    start_step = self._task_start_node_id if self._task_start_node_id is not None else 0
                    bt_action = AgentAct(
                        name=BacktraceAction.action_name,
                        params={
                            BACKTRACE_STEP_KEY: start_step,
                            BACKTRACE_REASON_KEY: reason,
                        },
                    )
                    _saved_completion_status = task.completion # temporary save of completion status
                    task.completion = "verify_failed"   # ensure _do_backtrace uses (None, None, obs)
                    success, bt_obs = self._do_backtrace(task, bt_action) # should always succeed
                    task.completion = _saved_completion_status # restore original completion status
                    self.__st_memorize__(task, bt_action, bt_obs)
                    self.logger.get_obs(obs=bt_obs)
                    task.completion = "active"  # reset task status to active for the retry
                    step_size = 0  # reset step size for the retry
                    continue

            # verification passed or not enabled
            break

        self._on_task_complete(task, last_obs)
        self.logger.end_execute(task=task, agent_name=self.name)


    def _compute_exploration_hint(self) -> str:
        """Compute hint from children of the current summary node.

        Shows the content of previously explored summary nodes that are
        children of the current summary, informing the agent what was
        previously done at this step.
        """
        current_summary_node = self.exploration_tree.current_summary_node
        children = self.exploration_tree.get_summary_children(current_summary_node.summary_id)
        if not children:
            return ""
        parts = [f"{i+1}. {child.content.strip()}" for i, child in enumerate(children)]
        return (
            "Here are your previous results for the current step:\n"
            + "\n".join(parts)
            + "\nYou can refer and decide whether to follow or explore new paths."
        )


    def _extract_summary_key(self, content: str) -> str:
        """Extract a unique summary key from the content for Summary Tree merge."""
        return content.strip()

    def _do_store(self, task: TaskPackage, action: AgentAct) -> str:
        """Record a store event and update state.

        Steps:
        1. Read summary text from action params.
        2. Add a SummaryNode to the exploration tree.
        3. Append a ``(None, "[backtrace to step N to revisit]: <text>")`` entry to
           ``_stored_context`` and clear ``_raw_context``.
        4. Reset the pending-segment tracking.

        The observation text includes the revisit step in order to redo
        the just-compressed segment from scratch.

        Returns
        -------
        tuple[bool, str]
            ``(True, observation)`` on success, ``(False, error_message)`` on failure.
        """
        covered_ids = list(self._raw_node_ids)

        # Summary is provided inline by the agent as an action parameter.
        summary_text = (
            (action.params or {}).get(STORE_SUMMARY_KEY, "").strip()
            or "NULL"
        )
        if summary_text == "NULL":
            return False, "Store failed because no summary text provided in action parameters."

        # Update in-place
        if not covered_ids and self._stored_node_ids and not self._store_dirty:
            cur_sn = self.exploration_tree.current_summary_node
            if cur_sn is not None:
                cur_sn.content = summary_text  # replace the summary text
                observation = f"[Step {cur_sn.revisit_step}] Stored content: {summary_text}"
                self._stored_context[-1] = (None, observation)
                self._raw_context = []
                self._raw_node_ids = []
                return True, observation

        # Add SummaryNode to the tree's upper layer.
        summary_id = self.exploration_tree.add_summary(
            content=summary_text,
            covers_node_ids=covered_ids,
            key=self._extract_summary_key(summary_text)
        )
        revisit_step = self.exploration_tree.summaries[summary_id].revisit_step
        observation = f"[Step {revisit_step}] Stored content: {summary_text}"

        # Detect parent merge:
        is_parent_merge = (self._stored_node_ids and summary_id == self._stored_node_ids[-1])
        if is_parent_merge:
            # Update existing stored entry in-place instead of appending.
            self._stored_context[-1] = (None, observation)
        else:
            # append new summary entry
            self._stored_node_ids.append(summary_id)
            self._stored_context.append((None, observation))

        self._raw_context = []
        self._raw_node_ids = []

        self._store_dirty = False

        # Update exploration hint for the new summary node.
        self._exploration_hint = self._compute_exploration_hint()

        # Check stored content only when backtrace is possible.
        if self.max_check_retries > 0 and self._check_retry_count < self.max_check_retries:
            passed, reason = self._check_store(task, summary_text)
            if not passed:
                self._check_retry_count += 1
                bt_action = AgentAct(
                    name=BacktraceAction.action_name,
                    params={
                        BACKTRACE_STEP_KEY: revisit_step,
                        BACKTRACE_REASON_KEY: reason,
                    },
                )
                _, bt_obs = self._do_backtrace(task, bt_action)
                return True, f"Try to store {bt_obs}. But check failed: {reason}"

        self._check_retry_count = 0  # reset check retry count after accepted store
        # clean up backtrace context
        depth = len(self._stored_context)
        self._backtrace_context = [e for e in self._backtrace_context if e[0] is None or e[0] > depth]
        return True, observation


    def _do_backtrace(self, task: TaskPackage, action: AgentAct) -> str:
        """Restore context and env to the step specified in action params.

        Steps:
        1. Extract ``target_step`` and ``reason`` from *action* params.
        2. Validate that target_step is a valid checkpoint (step 0 for root,
           or the last step of a Store segment); return error if not.
        3. Clean context and reset to the state at target_step.
        4. Construct a backtrace observation with reason; update ``_exploration_hint``.
        5. Restore environment state from that node's env_snapshot.

        Returns
        -------
        tuple[bool, str]
            ``(True, observation)`` on success, ``(False, error_message)`` on failure.
        """
        # Extract parameters from action.
        target_step = int(action.params.get(BACKTRACE_STEP_KEY, -1) if action.params else -1)
        reason_text = (action.params or {}).get(BACKTRACE_REASON_KEY, "")

        # Mark that context has diverged; next Store must not update in-place.
        self._store_dirty = True

        # Capture depth before _restore_context truncates _stored_context.
        depth = len(self._stored_context)

        # Restore context lists to the target checkpoint (returns error str on failure).
        result = self._restore_context(target_step, task)
        if isinstance(result, str):
            return False, result   # failed to backtrace: return error message as observation
        target_node_id, target_summary_id, target_env = result

        # Append backtrace annotation
        backtrace_obs = reason_text
        target_depth = len(self._stored_context)
        if task.completion == "active":
            # Intra-subtask: inserted at target_depth, cleaned when reaches depth.
            self._backtrace_context.append((depth, target_depth, backtrace_obs))
        else:
            # Cross-subtask or verify backtrace: always at prompt start.
            self._backtrace_context.append((None, None, backtrace_obs))

        # Move tree and summary pointers
        self.exploration_tree.set_current(target_node_id)
        self.exploration_tree.set_current_summary(target_summary_id)

        # Update exploration hint for the new position.
        self._exploration_hint = self._compute_exploration_hint()

        # Restore environment
        self._on_backtrace_env_restore(target_env)
        # Reset text description of the environment
        if target_node_id == self.exploration_tree.root_id:
            self._current_env = None
        else:
            # _on_backtrace_env_restore have restored the environment state
            self._current_env = self._extract_curr_env(self._get_env_snapshot())

        return True, backtrace_obs

    def _restore_context(self, target_step: int, task: TaskPackage = None) -> tuple[int, str, Any] | str:
        """Reset context to target step and return (node_id, summary_id, env_snapshot).
        If target_step is not a valid checkpoint, return an error message string.

        Valid checkpoints:
        - The parent node of each Store segment's first covered tree node.
        - The parent of the first raw-context node (start of the current uncompressed segment)
        - The task start node (fallback when no Store or raw nodes exist)
        """
        ckpt_map: dict[int, tuple] = {}
        for i, sid in enumerate(self._stored_node_ids):
            parent_node = self.exploration_tree.get_summary_parent(sid)
            if parent_node is not None:
                restore_sid = self._stored_node_ids[i - 1] if i > 0 else self._task_start_summary_id
                ckpt_map[parent_node.node_id] = (i, parent_node.node_id, parent_node.env_snapshot, restore_sid)

        if self._raw_node_ids:
            raw_parent = self.exploration_tree.get_parent(self._raw_node_ids[0])
            if raw_parent is not None:
                restore_sid = self._stored_node_ids[-1] if self._stored_node_ids else self._task_start_summary_id
                ckpt_map[raw_parent.node_id] = (len(self._stored_node_ids), raw_parent.node_id, raw_parent.env_snapshot, restore_sid)

        # Fallback: task start node (ensures backtrace works even with no store/raw nodes)
        if self._task_start_node_id is not None and self._task_start_node_id not in ckpt_map:
            start_node = self.exploration_tree.nodes.get(self._task_start_node_id)
            if start_node is not None:
                ckpt_map[self._task_start_node_id] = (0, start_node.node_id, start_node.env_snapshot, self._task_start_summary_id)

        if target_step not in ckpt_map:
            return (
                f"Backtrace failed because step {target_step} is not valid. "
                f"Valid steps are {sorted(ckpt_map.keys())}"
            )

        target_idx, target_node_id, target_env, target_summary_id = ckpt_map[target_step]
        self._stored_context  = self._stored_context[:target_idx]
        self._stored_node_ids = self._stored_node_ids[:target_idx]
        self._raw_context  = []
        self._raw_node_ids = []
        return target_node_id, target_summary_id, target_env

    def _on_backtrace_env_restore(self, env_snapshot: Any) -> None:
        """Restore environment state from the given env_snapshot."""
        pass