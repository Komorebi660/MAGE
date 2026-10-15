"""
SearchMAGEAgent - Memory-enhanced agent for BrowseComp-Plus progressive search.

Each SearchMAGEAgent handles one subtask (question).  Cross-subtask state is
held in a BundleSharedState created once per bundle and shared across all
subtask agents.  A SubtaskTree tracks completed answers so that subsequent
subtasks see prior results without full execution traces.
"""

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agentcore.actions import FinishAct, ThinkAct, INNER_ACTIONS
from agentcore.actions.InnerActions import INNER_ACT_KEY
from agentcore.actions.MAGEActions import StoreAction, STORE_SUMMARY_KEY
from agentcore.agents.agent_utils import act_match
from agentcore.agents.MAGEAgent import MAGEAgent
from agentcore.commons import AgentAct, TaskPackage
from agentcore.llm.LLMBackend import BaseLLM
from agentcore.logging.terminal_logger import AgentLogger
from agentcore.memory.ExplorationTree import ExplorationTree

from search_actions import GetDocumentAction, SearchAction
from search_env import BrowseCompPlusEnv, extract_exact_answer, answer_matches, _STOP_WORDS


EMPTY_TAG = "EMPTY_ANSWER"


def _parse_finish_response(raw: str) -> tuple[str, List[str]]:
    """Parse 'answer: xxx; docids: [a, b]' format from Finish response.
    Returns (answer_text, docids_list). Gracefully falls back if format is wrong."""
    if not raw:
        return "", []
    text = raw.strip()
    # Try structured format
    m_answer = re.search(r'answer:\s*(.+?)(?:[;,]\s*docids:|\s*docids:|$)', text, re.IGNORECASE)
    m_docids = re.search(r'docids:\s*\[([^\]]*)\]', text, re.IGNORECASE)
    if m_answer:
        answer = m_answer.group(1).strip().rstrip(';').strip()
        docids = []
        if m_docids:
            docids = [d.strip().strip('"\' ') for d in m_docids.group(1).split(',') if d.strip()]
            # Filter out fabricated non-numeric docids (real corpus docids are always numeric)
            docids = [d for d in docids if d.isdigit()]
        return answer, docids
    # Fallback: treat entire text as answer
    return text, []


@dataclass
class SubtaskRecord:
    """One completed answer for a subtask within a bundle."""
    subtask_idx: int
    answer_text: str
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_node_id: Optional[int] = None
    parent_summary_id: Optional[str] = None
    children_record_ids: List[str] = field(default_factory=list)
    visit_count: int = 1
    verify_feedback: Optional[str] = None
    is_soft_reject: bool = False  # True = evidence insufficient; False = hard contradiction
    soft_reject_count: int = 0  # escalate to hard after threshold
    docids: List[str] = field(default_factory=list)  # support docids for this answer
    checked_docids: List[str] = field(default_factory=list)  # checked docids for this task
    summary_ids: List[str] = field(default_factory=list)  # exploration tree summary ids from this attempt


class SubtaskTree:
    """Tree of subtask answer records.

    Root → subtask 0 records → subtask 1 records → …
    Merging by normalized answer prevents duplicate records.
    """

    def __init__(self) -> None:
        self.records: Dict[str, SubtaskRecord] = {}
        self.root_id = "subtask_root"
        self.records[self.root_id] = SubtaskRecord(
            record_id=self.root_id, subtask_idx=-1, answer_text=""
        )
        self._active_path: List[str] = []

    @property
    def root(self) -> SubtaskRecord:
        return self.records[self.root_id]

    def parent_for_subtask(self, subtask_idx: int) -> SubtaskRecord:
        if subtask_idx == 0:
            return self.root
        for rid in reversed(self._active_path):
            if self.records[rid].subtask_idx == subtask_idx - 1:
                return self.records[rid]
        return self.root

    def previous_attempts(self, subtask_idx: int) -> List[str]:
        """Summary texts of all previous answer attempts for this subtask."""
        # Aggregate visit_count, feedback, and docids for same normalized answer.
        agg: dict[str, tuple[str, int, Optional[str], List[str]]] = {}
        for rec in self.records.values():
            if rec.subtask_idx != subtask_idx:
                continue

            norm = rec.answer_text.strip().lower()
            if norm == EMPTY_TAG.lower():  # ignore empty answers in previous attempts
                continue

            if norm in agg:
                raw, total, fb, docs = agg[norm]
                agg[norm] = (raw, total+rec.visit_count, rec.verify_feedback or fb, docs | set(rec.docids))
            else:
                agg[norm] = (rec.answer_text, rec.visit_count, rec.verify_feedback, set(rec.docids))

        # Format results.
        results: List[str] = []
        for norm, (raw, total, fb, docs) in agg.items():
            text = raw
            if docs:
                text += f" (supported by docs {sorted(docs)})"
            if total > 1:
                text += f" You have given this answer {total} times."
            if fb:
                text += f" But may be wrong: {fb}"
            results.append(text)
        return results

    def add_record(
        self,
        subtask_idx: int,
        answer_text: str,
        parent_node_id: Optional[int] = None,
        parent_summary_id: Optional[str] = None,
        docids: Optional[List[str]] = None,
        checked_docids: Optional[List[str]] = None,
        summary_ids: Optional[List[str]] = None,
    ) -> SubtaskRecord:
        parent = self.parent_for_subtask(subtask_idx)
        norm = answer_text.strip().lower()
        docids = docids or []
        checked_docids = checked_docids or []
        summary_ids = summary_ids or []
        # Merge if same normalized answer already exists under this parent.
        for cid in parent.children_record_ids:
            child = self.records.get(cid)
            if child and child.answer_text.strip().lower() == norm:
                child.answer_text = answer_text
                child.parent_node_id = parent_node_id
                child.parent_summary_id = parent_summary_id
                child.visit_count += 1
                child.verify_feedback = None
                child.is_soft_reject = False
                for previous, incoming in (
                    (child.docids, docids),
                    (child.checked_docids, checked_docids),
                    (child.summary_ids, summary_ids),
                ):
                    seen = set(previous)
                    for value in incoming:
                        if value not in seen:
                            previous.append(value)
                            seen.add(value)
                self._set_active(subtask_idx, cid)
                return child
        rec = SubtaskRecord(
            subtask_idx=subtask_idx,
            answer_text=answer_text,
            parent_node_id=parent_node_id,
            parent_summary_id=parent_summary_id,
            docids=list(docids),
            checked_docids=list(checked_docids),
            summary_ids=list(summary_ids),
        )
        self.records[rec.record_id] = rec
        parent.children_record_ids.append(rec.record_id)
        self._set_active(subtask_idx, rec.record_id)
        return rec

    def _set_verify_feedback(self, subtask_idx: int, feedback: str, is_soft: bool = False) -> None:
        for rec in self.active_records():
            if rec.subtask_idx == subtask_idx:
                rec.verify_feedback = feedback
                rec.is_soft_reject = is_soft
                if is_soft:
                    rec.soft_reject_count += 1
                return

    def _set_active(self, subtask_idx: int, record_id: str) -> None:
        self.trim_active_path(subtask_idx)
        self._active_path.append(record_id)

    def trim_active_path(self, before_subtask_idx: int) -> None:
        self._active_path = [
            r for r in self._active_path
            if self.records[r].subtask_idx < before_subtask_idx
        ]

    def active_records(self) -> List[SubtaskRecord]:
        return [self.records[r] for r in self._active_path]

    def active_steps(self) -> List[int]:
        return [
            self.records[r].parent_node_id if self.records[r].parent_node_id is not None else -1
            for r in self._active_path
        ]


@dataclass
class BundleSharedState:
    """Mutable state shared across all subtask agents in a bundle."""
    exploration_tree: ExplorationTree = field(default_factory=ExplorationTree)
    global_step: int = 1
    subtask_goals: List[str] = field(default_factory=list)
    subtask_answers: List[str] = field(default_factory=list)
    subtask_tree: SubtaskTree = field(default_factory=SubtaskTree)
    pending_backtrace_reasons: List[tuple] = field(default_factory=list)
    judge_count: int = 0
    max_judge_retries: int = 0
    full_question: str = ""  # The complete main question (last one in the list)

    # Shared counter for agent-initiated backtrace calls across subtasks.
    agent_backtrace_count: int = 0



class SearchMAGEAgent(MAGEAgent):
    """BrowseComp-Plus agent with hierarchical memory for bundled progressive search.
    """

    def __init__(
        self,
        question: str,
        subtask_idx: int,
        shared_state: BundleSharedState,
        env: BrowseCompPlusEnv,
        llm: BaseLLM,
        max_steps: int = 30,
        PROMPT_DEBUG_FLAG: bool = False,
        log_file: str = "agent.log",
        max_verify_retries: int = 1,
        max_check_retries: int = 1,
        max_backtrace: int = 5,
    ) -> None:
        self.env = env
        self.question = question
        self.subtask_idx = subtask_idx
        self.shared_state = shared_state
        self._replay_target_subtask_idx: Optional[int] = None
        self._cross_update_triggered_replay: bool = False
        self.is_final_question = (subtask_idx == len(shared_state.subtask_answers) - 1)
        self._log_dir = os.path.dirname(log_file)
        # Parse task_id from log filename (e.g. "task_42_subtask_0.log")
        m = re.search(r'task_(\d+)_', os.path.basename(log_file))
        self._task_id = m.group(1) if m else "0"

        search_action = SearchAction(env=env)
        get_document_action = GetDocumentAction(env=env)

        super().__init__(
            name="search_mage_agent",
            role=(
                "You are a deep research agent solving a complex question decomposed into progressive sub-questions. All sub-questions constrain the same entity. Answer each question by interacting with a search engine, i.e., searching the corpus and reading documents; never Finish from internal knowledge or lock onto a candidate without corpus evidence.\n\n"
                # Backtrace only when a document clearly proves a prior sub-question's answer should be a different person/entity. Missing confirmation, absent attributes, or your own knowledge do not count as contradiction. The backtrace reason must cite the new candidate, doc ID, and the explicit incompatible statement.
                "**Search strategy:**\n"
                "- Use focused 3-6 word queries. Try different search angles with synonyms, related concepts, broader/narrower terms, or different clue combinations.\n"
                "- Do not repeat search by only reordering keywords. If a search keyword pollutes results, drop it. If no relevant results are found, rethink entity type, wording, region/culture, or assumptions.\n"
                "- Each search may return multiple relevant documents; read several promising ones, not just the first.\n"
                "- Store brief notes for failed searches and each document you read, whether relevant or not.\n\n"
                "Keep your reasoning process compact and focused. Your generated output should NOT include any Observation lines or simulated tool results.\n"
                "**Finish format:** Finish[{\"response\": \"answer: <complete_full_name>; docids: [<id1>, <id2>]\"}]"
            ),
            llm=llm,
            actions=[search_action, get_document_action],
            constraint=(
                "1. You must search the corpus and read a document before answering. "
                "Never use internal knowledge as your answer. Find it in documents first.\n"
                "2. Keep your reasoning process brief and output exactly one next action only; do not re-solve prior sub-questions, enumerate many candidates/search queries, or simulate observations.\n"
                "3. Return the answer exactly as it appears in the document, using the complete, unabbreviated form (e.g., full legal name, full title, full official designation). Do not shorten or paraphrase. "
                "Finish format: Finish[{\"response\": \"answer: <answer>; docids: [<id1>, ...]\"}]"
            ),
            logger=AgentLogger(log_file_name=log_file, PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG),
            max_exec_steps=max_steps,
            max_verify_retries=max_verify_retries,
            max_check_retries=max_check_retries,
            max_backtrace=max_backtrace,
            enable_action_thinking=False,
        )

        # Share the exploration tree across all subtask agents.
        self.exploration_tree = shared_state.exploration_tree
        self._global_step = shared_state.global_step
        self._agent_backtrace_count = shared_state.agent_backtrace_count

        # Propagate cross-subtask backtrace reasons.
        visible = [obs for owner, obs in shared_state.pending_backtrace_reasons if owner > subtask_idx]
        if visible:
            self._backtrace_context = [(None, None, obs) for obs in visible]

        # Record this subtask's goal.
        if len(shared_state.subtask_goals) <= subtask_idx:
            shared_state.subtask_goals.append(question)

        # Populate previous attempts from SubtaskTree.
        prev = shared_state.subtask_tree.previous_attempts(subtask_idx)
        if prev:
            self._previous_results = list(prev)

        # temporary store verification feedback for the current subtask
        self.verification_feedback: Optional[str] = None
        self._is_soft_verification: bool = False
        self._verified_pass: bool = False  # True when _verify_task genuinely passes

        # Track docids for the current session
        self._session_docids: List[str] = []

        # populate exploration hint based on previous attempts for this subtask
        self._exploration_hint = self._compute_exploration_hint()

        self._build_examples()


    def _build_task_instruction(self) -> str:
        ss = self.shared_state
        active = ss.subtask_tree.active_records()
        steps = ss.subtask_tree.active_steps()
        is_final = self.is_final_question
        total_subqs = len(ss.subtask_answers) - 1  # sub-questions excluding the final full question

        parts = []

        # Main question context
        parts.append(f"**Main Question:** {ss.full_question}\n")

        if not is_final:
            parts.append(f"This question is decomposed into {total_subqs} sub-questions to narrow down the answer progressively. All sub-questions and the main question describe the same entity.\n")

        # Show answered sub-questions
        if active:
            parts.append("**Answered sub-questions** (you can Backtrace to a step to undo if needed):")
            for i, rec in enumerate(active):
                q = ss.subtask_goals[rec.subtask_idx]
                support = f" (docs: {rec.docids})" if rec.docids else ""
                parts.append(f"  [Step {steps[i]}] Sub-Q {rec.subtask_idx}: {q}")
                parts.append(f"    -> Your Answer: {rec.answer_text}{support}")
            parts.append("")

        if is_final:
            parts.append("All sub-questions have been answered. Now answer the **main question** above directly. Your answer should satisfy ALL constraints from the sub-questions and the main question.")
            if active and not self._backtrace_context:
                last_answer = active[-1].answer_text
                if last_answer and last_answer != EMPTY_TAG:
                    parts.append(f"The current candidate is \"{last_answer}\" but it is not yet confirmed. Search to verify it against the full question with document evidence. Also search for alternative candidates.")
        else:
            parts.append(f"Now your task is to find the answer to the sub-question {self.subtask_idx}:\n{self.question}\n\nThe answer should satisfy all constraints imposed by the current and prior sub-questions, but there is no need to find the final answer that satisfies the main question's constraints right now. Only incorporate clues from the main question if clues from current and prior sub-questions are not enough to narrow down the answer and find relevant results.")

            current_best = None
            if active and not self._backtrace_context:
                current_best = active[-1].answer_text

            if current_best and current_best != EMPTY_TAG:
                parts.append(
                    "Search using the current sub-question's keywords combined with clues from prior sub-questions. Do not get stuck searching for the prior candidate's name; treat it as one hypothesis in case a different candidate fits better. "
                    f"If your independent search also points to \"{current_best}\", Finish with it. "
                    "If results point to a different entity that fits the constraints better, read that document and Finish with the better-supported entity. Do not assume the prior answer is correct."
                )
            elif active:
                parts.append("Search using the current sub-question's keywords combined with clues from prior sub-questions. Only bring in clues from the main question when the sub-question clues are not enough. Think about what entity type, region, and domain could satisfy all constraints together.")

        # Show rejected candidates across ALL subtasks
        active_answers = {rec.answer_text.strip().lower() for rec in ss.subtask_tree.active_records()}
        hard_rejected = set()
        for rec in ss.subtask_tree.records.values():
            if rec.subtask_idx < 0:
                continue
            ans = rec.answer_text.strip()
            if not ans or ans == EMPTY_TAG:
                continue
            if ans.lower() in active_answers:
                continue
            if rec.verify_feedback and not rec.is_soft_reject:
                hard_rejected.add(ans)  # hard rejection: fabricated docs, answer not in doc
            elif rec.is_soft_reject and rec.soft_reject_count >= 3:
                hard_rejected.add(ans)  # soft escalated to hard after repeated failures
        if hard_rejected:
            parts.append(f"Rejected answers from prior attempts: {', '.join(sorted(hard_rejected))}. These were rejected because their cited documents were fabricated or did not contain the answer; avoid repeating them unless you find strong new evidence.")

        return "\n".join(parts)

    def build_task(self) -> TaskPackage:
        return TaskPackage(instruction=self._build_task_instruction())


    def _extract_summary_key(self, content: str) -> str:
        """Use docid as the merge key so summaries about the same document merge."""
        m = re.search(r"\b(?:docid|doc)\s*[:\s]+(\d+)", content, re.IGNORECASE)
        if m:
            return m.group(1)
        return None


    def _compute_exploration_hint(self):
        """Build hint for retry attempts from exploration tree summaries.

        Only shows summaries from previous attempts of the SAME subtask,
        using summary_ids stored in SubtaskRecord.
        """
        ss = self.shared_state
        cur_stored = set(self._stored_node_ids)
        tree = self.exploration_tree

        # Collect summary_ids from previous attempts of this subtask
        prev_sids: set[str] = set()
        for rec in ss.subtask_tree.records.values():
            if rec.subtask_idx == self.subtask_idx:
                prev_sids.update(rec.summary_ids)
        prev_sids -= cur_stored

        notes: list[str] = []
        for sid in prev_sids:
            sn = tree.summaries.get(sid)
            if sn and sn.content:
                notes.append(sn.content)

        if not notes:
            return ""

        parts: list[str] = ["Here are your searched queries and read documents for the current sub-question from previous attempts:"]
        for n in notes:
            for line in n.strip().splitlines():
                parts.append(f"{line.strip()}")
        parts.append("Review the notes from your previous attempts and avoid repeating the same search or reading the same documents. Try different search angles or a different interpretation of the clues.")
        return "\n".join(parts)


    # ------------------------------------------------------------------
    # Auto-compression: search (→ get_document) → search pattern
    # ------------------------------------------------------------------

    def _post_action_hook(self):
        """When a new search arrives after old search(+get_document) blocks,
        auto-compress everything before it into a stored summary."""
        action, _ = self._raw_context[-1]

        # Track docids for the current session
        if action.valid and action.name == "get_document":
            docid = (action.params or {}).get("docid", "")
            if docid and docid.isdigit():  # filter out fabricated non-numeric docids
                self._session_docids.append(docid)

        if not (action.valid and action.name == "search"):
            return

        # Collect real (non-Think) action indices
        real = [(i, self._raw_context[i][0]) for i, (act, _) in enumerate(self._raw_context)
                if act is not None and act.valid and act.name in ("search", "get_document")]
        if len(real) < 3 or real[-1][1].name != "search":  # allow three raw actions
            return

        # Split at the new search; compress everything before it
        cut = real[-1][0]
        node_count = len(real) - 1
        self._raw_context, to_compress = self._raw_context[cut:], self._raw_context[:cut]
        self._raw_node_ids, compressed_ids = self._raw_node_ids[node_count:], self._raw_node_ids[:node_count]

        # Build summary: group by search → checked docs, with Think analysis
        segments, cur_q, cur_docs, cur_notes = [], None, [], []
        for act, _ in to_compress:
            if act is None or not act.valid:
                continue
            if act.name == "search":
                if cur_q is not None:
                    segments.append((cur_q, cur_docs, cur_notes))
                cur_q, cur_docs, cur_notes = (act.params or {}).get("query", ""), [], []
            elif act.name == "get_document":
                docid = (act.params or {}).get("docid", "")
                if docid:
                    cur_docs.append(docid)
            elif act.name == "Think" and cur_q is not None:
                think_text = (act.params or {}).get(INNER_ACT_KEY, "").strip()
                if think_text:
                    cur_notes.append(think_text)
        if cur_q is not None:
            segments.append((cur_q, cur_docs, cur_notes))

        parts = []
        for q, docs, notes in segments:
            base = f"I have searched '{q}' and checked docs {docs}." if docs \
                   else f"I have searched '{q}'."
            if notes:
                base += " Notes: " + "; ".join(notes)
            else:
                base += " But they were not relevant."
            parts.append(base)
        summary = "\n".join(parts)
        summary_id = self.exploration_tree.add_summary(content=summary, covers_node_ids=compressed_ids)
        self._stored_node_ids.append(summary_id)
        self._stored_context.append((None, summary))


    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def _do_store(self, task: TaskPackage, action: AgentAct) -> str:
        """Record a store event and update state.

        When the raw context contains a ``search`` followed by at least one
        ``get_document``, only the portion **after** the last search is
        compressed into a summary.  The search action and its results stay
        in ``_raw_context`` so the agent can continue reading other
        documents from the same result list.

        If there is no such pattern (e.g. storing after a fruitless search
        with no get_document, or when only get_document calls remain after
        a previous partial store), the full raw context is compressed
        using the default logic.
        """
        # Locate the last search action in the current raw context.
        last_search_idx = -1
        for i in range(len(self._raw_context) - 1, -1, -1):
            act, _obs = self._raw_context[i]
            if act is not None and act.valid and act.name == "search":
                last_search_idx = i
                break

        # Check whether a get_document follows that search.
        has_get_doc_after = last_search_idx >= 0 and any(
            act is not None and act.valid and act.name == "get_document"
            for act, _ in self._raw_context[last_search_idx + 1:]
        )

        keep_context, keep_node_ids = [], []
        if has_get_doc_after:
            # _raw_node_ids only has entries for non-inner valid actions
            node_split = sum(
                1 for act, _ in self._raw_context[:last_search_idx + 1]
                if act is not None and act.valid and not any(act_match(act.name, a) for a in INNER_ACTIONS)
            )
            keep_context = self._raw_context[:last_search_idx + 1].copy()
            keep_node_ids = self._raw_node_ids[:node_split].copy()
            self._raw_context = self._raw_context[last_search_idx + 1:]
            self._raw_node_ids = self._raw_node_ids[node_split:]

        # Below is original Store behavior
        covered_ids = list(self._raw_node_ids)

        # Summary is provided inline by the agent as an action parameter.
        summary_text = (action.params or {}).get(STORE_SUMMARY_KEY, "").strip() or "NULL"
        if summary_text == "NULL":
            return False, "Store failed because no summary text provided in action parameters."

        # Update in-place
        if not covered_ids and self._stored_node_ids and not self._store_dirty:
            cur_sn = self.exploration_tree.current_summary_node
            if cur_sn is not None:
                cur_sn.content = summary_text  # replace the summary text
                observation = summary_text
                self._stored_context[-1] = (None, observation)
                self._raw_context = keep_context
                self._raw_node_ids = keep_node_ids
                return True, observation

        # Add SummaryNode to the tree's upper layer.
        summary_id = self.exploration_tree.add_summary(
            content=summary_text,
            covers_node_ids=covered_ids,
            key=self._extract_summary_key(summary_text)
        )
        observation = summary_text

        # Detect parent merge:
        is_parent_merge = (self._stored_node_ids and summary_id == self._stored_node_ids[-1])
        if is_parent_merge:
            # Update existing stored entry in-place instead of appending.
            self._stored_context[-1] = (None, observation)
        else:
            # append new summary entry
            self._stored_node_ids.append(summary_id)
            self._stored_context.append((None, observation))

        self._raw_context = keep_context
        self._raw_node_ids = keep_node_ids

        self._store_dirty = False

        # Update exploration hint for the new summary node.
        self._exploration_hint = self._compute_exploration_hint()

        # # clean up backtrace context
        # depth = len(self._stored_context)
        # self._backtrace_context = [e for e in self._backtrace_context if e[0] is None or e[0] > depth]
        return True, observation


    # ------------------------------------------------------------------
    # Backtrace
    # ------------------------------------------------------------------

    def _restore_context(self, target_step: int, task: TaskPackage = None):
        """Restore context to a checkpoint at target_step.

        Valid checkpoints:
        1. Subtask-boundary checkpoints (previous subtasks only).
        2. Task start node.

        Do NOT allow intra-subtask backtrace.
        """
        ss = self.shared_state

        ckpt_map: Dict[int, tuple] = {}
        # Source 1: subtask-boundary checkpoints (previous subtasks only).
        for i, rec in enumerate(ss.subtask_tree.active_records()):
            if rec.subtask_idx >= self.subtask_idx:
                continue
            if rec.parent_node_id is None:
                continue
            fpn = self.exploration_tree.nodes.get(rec.parent_node_id)
            if fpn is not None:
                restore_sid = rec.parent_summary_id or self._task_start_summary_id
                ckpt_map[fpn.node_id] = ("subtask", i, fpn.node_id, fpn.env_snapshot, restore_sid)

        # Source 2: task start node (OVERRIDES subtask boundary at same node).
        if self._task_start_node_id is not None:
            start_node = self.exploration_tree.nodes.get(self._task_start_node_id)
            if start_node is not None:
                ckpt_map[self._task_start_node_id] = (
                    "store", 0, start_node.node_id,
                    start_node.env_snapshot, self._task_start_summary_id,
                )

        if target_step not in ckpt_map:
            return f"Backtrace failed since step {target_step} is not valid. Valid steps are {sorted(ckpt_map.keys())}."

        kind, idx, target_node_id, target_env, target_summary_id = ckpt_map[target_step]

        if kind == "store":
            self._stored_context = self._stored_context[:idx]
            self._stored_node_ids = self._stored_node_ids[:idx]
            self._raw_context = []
            self._raw_node_ids = []
        else:  # "subtask"
            matched_rec = ss.subtask_tree.active_records()[idx]
            ss.subtask_tree.trim_active_path(matched_rec.subtask_idx)
            self._replay_target_subtask_idx = max(
                self.subtask_idx,
                self._replay_target_subtask_idx if self._replay_target_subtask_idx is not None else 0,
            )
            task.completion = "replay"

        return target_node_id, target_summary_id, target_env


    # ------------------------------------------------------------------
    # Task completion
    # ------------------------------------------------------------------

    def _on_task_complete(self, task: TaskPackage, last_obs: str = "") -> None:
        ss = self.shared_state
        ss.agent_backtrace_count = self._agent_backtrace_count

        # Cross-subtask replay: agent exited early without completing.
        if task.completion == "replay":
            ss.global_step = self._global_step
            if self._replay_target_subtask_idx is not None:
                if self._backtrace_context:
                    ss.pending_backtrace_reasons.append((self.subtask_idx, self._backtrace_context[-1][2]))
                self._run_next_replay_subtask(new_log=False)
            return

        # Parse answer and docids from Finish response, then record.
        raw_answer = (task.answer or EMPTY_TAG).strip()
        answer, docids = _parse_finish_response(raw_answer)
        if not answer: answer = EMPTY_TAG

        task.answer = answer  # store clean answer back
        # Filter out docids that don't exist in the corpus
        docids = [d for d in docids if "error" not in json.loads(self.env.get_document(d))]
        ss.subtask_tree.add_record(
            subtask_idx=self.subtask_idx,
            answer_text=answer,
            parent_node_id=self._task_start_node_id,
            parent_summary_id=self._task_start_summary_id,
            docids=docids,
            checked_docids=self._session_docids,
            summary_ids=list(self._stored_node_ids),
        )
        ss.global_step = self._global_step

        # Hard → Soft: downgrade hard-rejected answers if a newly-read doc contains them.
        if self._session_docids:
            # Build lookup: docid → lowercased full text
            new_doc_texts = {}
            for did in set(self._session_docids):
                resp = json.loads(self.env.get_document(did))
                if "error" not in resp:
                    new_doc_texts[did] = resp.get("text", "").lower()

            # Check each hard-rejected, non-active record
            active_ids = {r.record_id for r in ss.subtask_tree.active_records()}
            for rec in ss.subtask_tree.records.values():
                if rec.record_id in active_ids:
                    continue  # skip currently active answers
                if not rec.verify_feedback or rec.is_soft_reject:
                    continue  # only target hard rejections
                # Never downgrade fabricated-docid rejections
                if "NOT exist in the corpus" in rec.verify_feedback:
                    continue
                ans = rec.answer_text.strip()
                if not ans or ans == EMPTY_TAG:
                    continue
                # Only for multi-word answers to avoid false matches on short/common names
                if len(ans.split()) < 2:
                    continue
                # Search new docs (excluding ones this record already checked)
                previously_checked = set(rec.checked_docids)
                for did, text in new_doc_texts.items():
                    if did not in previously_checked and ans.lower() in text:
                        rec.is_soft_reject = True
                        if did not in set(rec.docids):
                            rec.docids.append(did)
                        break

        # Verification retry: update previous_results and return.
        if self._is_verify_retry:
            ss.subtask_tree._set_verify_feedback(self.subtask_idx, self.verification_feedback,
                                                 is_soft=self._is_soft_verification)
            self.verification_feedback = None  # reset
            self._is_soft_verification = False
            self._previous_results = list(ss.subtask_tree.previous_attempts(self.subtask_idx))
            # Reset session tracking so the next attempt gets a clean slate
            self._session_docids = []
            return

        # Fail → Correct: when this answer genuinely passed verification,
        # clear verify_feedback on matching answers across all subtasks.
        if self._verified_pass and answer and answer != EMPTY_TAG:
            for rec in ss.subtask_tree.records.values():
                if rec.subtask_idx < 0 or not rec.verify_feedback:
                    continue
                if (answer_matches(answer, rec.answer_text)
                    or answer_matches(rec.answer_text, answer)):
                    rec.verify_feedback = None  # reset feedback
                    rec.is_soft_reject = False
            self._verified_pass = False

        # Cross-verify: if this answer differs from prior sub-Qs, check documents
        # for consistency — update matching answers, detect contradictions.
        if self.subtask_idx > 0 and answer and answer != EMPTY_TAG and docids:
            self._cross_subtask_update(answer, docids)
            if self._cross_update_triggered_replay:
                self._replay_target_subtask_idx = max(
                    self.subtask_idx,
                    self._replay_target_subtask_idx if self._replay_target_subtask_idx is not None else 0,
                )
                self._run_next_replay_subtask(new_log=False)
                return

        if self._replay_target_subtask_idx is not None and self.subtask_idx < self._replay_target_subtask_idx:
            self._run_next_replay_subtask(new_log=self.shared_state.judge_count > 0)
        else:
            ss.pending_backtrace_reasons.clear()
            # Fill empty answers in prior subtasks with current answer
            if self.is_final_question and answer and answer != EMPTY_TAG:
                for rec in ss.subtask_tree.active_records():
                    if rec.subtask_idx < self.subtask_idx and rec.answer_text.strip() == EMPTY_TAG:
                        rec.answer_text = answer
                        if docids:
                            seen = set(rec.docids)
                            for d in docids:
                                if d not in seen:
                                    rec.docids.append(d)
                                    seen.add(d)
            # Run judge if all subtasks finished
            if self._should_run_judge():
                self._run_judge()


    def _cross_subtask_update(self, answer: str, docids: List[str]) -> None:
        """Check current answer's documents against prior sub-Q answers.

        Two purposes:
        1. If the new answer directly answers a prior sub-Q → update that prior answer.
        2. If the documents reveal that a prior sub-Q's answer should be DIFFERENT from
           what was recorded → detect contradiction and trigger replay with specific reason.

        Principle: never reject the current (correct) answer. Only correct prior errors.
        """
        ss = self.shared_state
        prior_active = [rec for rec in ss.subtask_tree.active_records() if rec.subtask_idx < self.subtask_idx]
        if not prior_active:
            return

        prior_answer = prior_active[-1].answer_text
        if (not prior_answer or prior_answer == EMPTY_TAG
            or answer_matches(answer, prior_answer)
            or answer_matches(prior_answer, answer)):
            return  # same candidate, nothing to do

        # Collect document texts from cited docids of current answer
        all_doc_texts: Dict[str, str] = {}
        for did in docids:
            doc_json = self.env.get_document(did)
            text = json.loads(doc_json).get("text", "")
            if text:
                all_doc_texts[did] = text
        if not all_doc_texts:
            return

        # Build prior sub-Q info for the check, skipping those already matching.
        prior_info = []
        for rec in prior_active:
            prior_ans = rec.answer_text
            if (prior_ans and prior_ans != EMPTY_TAG
                and (answer_matches(answer, prior_ans) or answer_matches(prior_ans, answer))):
                # Already the same answer — directly update docids, no LLM needed.
                seen = set(rec.docids)
                for d in docids:
                    if d not in seen:
                        rec.docids.append(d)
                        seen.add(d)
                continue
            # Not match -- send to LLM for verification
            prior_info.append((rec.subtask_idx, ss.subtask_goals[rec.subtask_idx], rec.answer_text))

        if not prior_info:
            return  # all prior sub-Qs already match

        # for each prior sub-Q, classify the relationship.
        doc_excerpts = "\n\n".join(f"[Doc {did}]:\n{text}" for did, text in all_doc_texts.items())
        check_prompt = (
            "A complex question was decomposed into progressive sub-questions. All sub-questions are related and they constrain the same underlying entity. "
            f"You found '{answer}' as the answer to the current sub-question, supported by the documents below.\n\n"
            f"Documents:\n{doc_excerpts}\n\n"
            "Now check each prior sub-question and its recorded answer:\n\n"
        )
        for idx, q, prior_ans in prior_info:
            check_prompt += f"Sub-Question {idx}: {q}\n    Recorded answer: {prior_ans}\n\n"
        check_prompt += (
            "For each prior sub-question, based ONLY on the documents above, determine one of:\n"
            f"- SAME_ENTITY: '{answer}' is also the correct answer to this sub-question\n"
            "- CONFIRMED: The documents confirm that the recorded answer is correct for this sub-question\n"
            "- DIFFERENT_ANSWER: The documents suggest that the recorded answer is WRONG for this sub-question, and provide evidence for a different answer. State the correct answer and which doc supports it.\n"
            "- NO_INFO: The documents don't contain enough information to verify this sub-question.\n\n"
            "Response format — one line per sub-question:\n"
            "[N] SAME_ENTITY / [N] CONFIRMED / [N] DIFFERENT_ANSWER: correct answer should be xxx and it is supported by doc X. / [N] NO_INFO, where N is the sub-question index.\n"
            "For example:\n"
            "[1] SAME_ENTITY\n"
            "[2] CONFIRMED\n"
            "[4] DIFFERENT_ANSWER: correct answer should be Apple and it is supported by doc 5.\n"
            "[6] NO_INFO"
        )

        # Call LLM to check prior sub-questions
        response = self.llm.run(prompt=check_prompt).strip()
        self.logger.get_llm_output(f"[Cross-check]: {response}")

        # Parse results
        same_entity_indices = set()
        different_answers: List[tuple[int, str]] = []  # (idx, reason_text)
        for idx, q, prior_ans in prior_info:
            if re.search(rf"\[{idx}\]\s*SAME_ENTITY", response, re.IGNORECASE):
                same_entity_indices.add(idx)
            m = re.search(rf"\[{idx}\]\s*DIFFERENT_ANSWER\s*:\s*(.+)", response, re.IGNORECASE)
            if m:
                different_answers.append((idx, m.group(1).strip()))
            # CONFIRMED and NO_INFO require no action.

        # SAME_ENTITY: both sub-Qs ask the same attribute → update answer text directly
        if same_entity_indices:
            for rec in prior_active:
                if rec.subtask_idx in same_entity_indices:
                    old = rec.answer_text
                    rec.answer_text = answer
                    rec.docids = list(docids)
                    self.logger.get_log(f"[Cross-update] Sub-Q {rec.subtask_idx}: '{old}' -> '{answer}'")

        # DIFFERENT_ANSWER: document evidence suggests a different answer for a prior sub-Q
        if different_answers:
            # Check backtrace budget before triggering cross-subtask replay
            if ss.agent_backtrace_count >= self.max_backtrace:
                self.logger.get_log(f"[Cross-update] Skipping replay: backtrace limit reached ({ss.agent_backtrace_count}/{self.max_backtrace}).")
                return

            # Record feedback on all flagged records before trimming
            for da_idx, da_reason in different_answers:
                ss.subtask_tree._set_verify_feedback(da_idx, f"After searching later sub-questions, found a better candidate: {da_reason}", is_soft=True)

            earliest_idx, reason_text = min(different_answers, key=lambda x: x[0])
            reason = f"After searching later sub-questions, may found a better answer for Sub-Q {earliest_idx}: {reason_text} Read docs {docids} to confirm."
            self.logger.get_log(f"[Different answer detected] {reason}")

            ss.subtask_tree.trim_active_path(earliest_idx)
            ss.pending_backtrace_reasons.append((earliest_idx + 1, reason))
            ss.agent_backtrace_count += 1
            self._cross_update_triggered_replay = True


    def _run_next_replay_subtask(self, new_log: bool = False) -> None:
        """Execute the next intermediate subtask during cross-subtask replay."""
        ss = self.shared_state
        next_sub_idx = len(ss.subtask_tree.active_records())

        log_file = os.path.join(self._log_dir, f"task_{self._task_id}_subtask_{next_sub_idx}_retry{ss.judge_count}.log") if new_log else self.logger.log_file_name

        replay_agent = SearchMAGEAgent(
            question=ss.subtask_goals[next_sub_idx],
            subtask_idx=next_sub_idx,
            shared_state=ss,
            env=self.env,
            llm=self.llm,
            log_file=log_file,
            PROMPT_DEBUG_FLAG=self.logger.PROMPT_DEBUG_FLAG,
            max_verify_retries=self.max_verify_retries,
            max_steps=self.max_exec_steps,
            max_check_retries=self.max_check_retries,
            max_backtrace=self.max_backtrace,
        )
        replay_agent._replay_target_subtask_idx = self._replay_target_subtask_idx

        task = replay_agent.build_task()
        replay_agent(task)


    # ------------------------------------------------------------------
    # Judge & Verify: verify subtask answers
    # ------------------------------------------------------------------

    def _extract_evidence(self, answer: str, docids: List[str], query: str = "", max_chars: int = 500) -> str:
        """
        Extract evidence snippets from cited docs around answer and query keyword matches.

        For each doc: find positions of answer + query keywords → expand by PAD →
        merge overlapping windows → join with ' ... ' → cap at max_chars.
        """
        answer_lower = answer.lower()
        keywords = {w.lower() for w in query.split() if len(w) > 3} - _STOP_WORDS
        PAD = 80
        snippets = []

        for docid in docids:
            text = json.loads(self.env.get_document(docid)).get("text", "")
            if not text:
                continue
            # Strip YAML frontmatter, metadata lines, normalize whitespace
            text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, count=1, flags=re.DOTALL)
            text = re.sub(r'^(?:[a-z_]+:\s*[^\n]*\n)+', '', text, count=1)
            text = re.sub(r'\s+', ' ', text).strip()
            tl = text.lower()
            if answer_lower not in tl:
                continue

            # Collect match start positions
            centers = sorted({p for term in {answer_lower} | keywords
                              if (p := tl.find(term)) >= 0})

            # Expand to windows and merge overlapping
            windows = []
            for c in centers:
                s, e = max(0, c - PAD), min(len(text), c + PAD)
                if windows and s <= windows[-1][1] + PAD:
                    windows[-1] = (windows[-1][0], max(windows[-1][1], e))
                else:
                    windows.append((s, e))

            # Extract chunks within budget
            parts, budget = [], max_chars
            for s, e in windows:
                parts.append(text[s:min(s + budget, e)])
                budget -= len(parts[-1])
                if budget <= 0:
                    break

            prefix = "..." if windows[0][0] > 0 else ""
            suffix = "..." if windows[len(parts) - 1][1] < len(text) else ""
            snippets.append(f"(doc {docid}) {prefix}{' ... '.join(parts)}{suffix}")

        return " ".join(snippets)


    def _should_run_judge(self) -> bool:
        """Determine if all subtasks finished and judge should run."""
        ss = self.shared_state
        total = len(ss.subtask_answers)
        return (
            total > 0
            and self.subtask_idx == total - 1
            and ss.judge_count < ss.max_judge_retries
            and len(ss.subtask_tree.active_records()) == total
        )

    def _run_judge(self) -> bool:
        """Review all answers after bundle completion.
        Ask LLM to decide which sub-question to redo if final answer not correct."""
        ss = self.shared_state
        active = ss.subtask_tree.active_records()

        # Only check the final answer (main question) for correctness
        final_idx = len(ss.subtask_answers) - 1
        final_rec = None
        for rec in active:
            if rec.subtask_idx == final_idx:
                final_rec = rec
                break
        if final_rec and answer_matches(extract_exact_answer(final_rec.answer_text), ss.subtask_answers[final_idx]):
            return False  # Final answer is correct

        mismatches = [
            rec.subtask_idx + 1 for rec in active
            if not answer_matches(extract_exact_answer(rec.answer_text), ss.subtask_answers[rec.subtask_idx])
        ]
        if not mismatches:
            return False

        agent_answers = ["" for _ in ss.subtask_answers]
        for rec in active:
            if 0 <= rec.subtask_idx < len(agent_answers):
                agent_answers[rec.subtask_idx] = extract_exact_answer(rec.answer_text)
        self.logger.get_log(
            f"Judge: wrong sub-questions: {mismatches}\n"
            f"Groundtruth answers: {ss.subtask_answers}\n"
            f"Agent answers: {agent_answers}"
        )

        # Build judge prompt — show answers + evidence, let LLM find problems
        lines = [
            "You are reviewing a progressive search task. A main question was decomposed into sub-questions "
            "about the same entity. The agent answered each one. Your job is to check if all answers are correct "
            "and well-supported. Find the most likely error.",
            "",
            f"**Main Question:** {ss.full_question}",
            "",
            "Review each answer below. For each, you see:",
            "- The question and current answer",
            "- Document evidence: snippets from the corpus that the agent used",
            "- visit_count: number of independent attempts to give this answer",
            "- Feedback from verification (if any)",
            "- Alternative answers from prior attempts (if any), they are not the current attempt's answer",
            "",
        ]
        for rec in active:
            idx = rec.subtask_idx
            label = f"[{idx}] Final Full Question" if idx == final_idx else f"[{idx}] Sub-Q {idx}"
            lines.append(f"**{label}**: {ss.subtask_goals[idx]}")
            evidence = self._extract_evidence(rec.answer_text, rec.docids, ss.subtask_goals[idx])
            evidence_str = f" Evidence: {evidence}" if evidence else " Evidence: NONE"
            feedback_str = f" Feedback: {rec.verify_feedback}" if rec.verify_feedback else ""
            lines.append(f"  Answer: {rec.answer_text} visit_count: {rec.visit_count}{evidence_str}{feedback_str}")
            # Show alternative attempts
            parent = ss.subtask_tree.parent_for_subtask(idx)
            for cid in parent.children_record_ids:
                child = ss.subtask_tree.records.get(cid)
                if child and child.answer_text != EMPTY_TAG and child.record_id != rec.record_id:
                    alt_evidence = self._extract_evidence(child.answer_text, child.docids, ss.subtask_goals[idx])
                    alt_str = f" Evidence: {alt_evidence}" if alt_evidence else " Evidence: NONE"
                    alt_fb = f" Feedback: {child.verify_feedback}" if child.verify_feedback else ""
                    lines.append(f"  Alternative Answer: {child.answer_text} visit_count: {child.visit_count}{alt_str}{alt_fb}")
            lines.append("")

        lines.append(
            "Sub-questions where the current answer lacks document evidence, contradicts other sub-questions, or doesn't match the main question's constraints are likely incorrect. "
            "All sub-questions describe the same entity, so answers should be consistent. "
            "If all questions are correct and consistent, then consider use visit_count as a hint, higher visit_count may indicate more reliable answers."
        )
        lines.append("")
        lines.append("IMPORTANT: If the same candidate appears across most sub-questions but the final answer is wrong, the root cause is likely in the EARLIEST sub-question (e.g., [0]) that locked in the wrong entity. In that case, choose that earliest sub-question so the entire search restarts with a different candidate.")
        lines.append("")
        lines.append(
            "Choose ONE sub-question that is most likely wrong and give a short but concise reason. "
            "Respond format: [N]. Reason: ..., where N is the index of the sub-question."
        )
        prompt = "\n".join(lines)

        self.logger.get_prompt(f"[Judge]: {prompt}")
        response = self.llm.run(prompt=prompt).strip()
        self.logger.get_llm_output(f"[Judge]: {response}")

        # Parse chosen index
        m = re.search(r"\[\s*(\d+)\s*\]", response)
        if not m:
            self.logger.get_log("Warning: no subtask index found in judge response. Skipping judge.")
            return False
        backtrack_idx = int(m.group(1))
        if backtrack_idx < 0 or backtrack_idx >= len(ss.subtask_goals):
            self.logger.get_log(f"Warning: backtrack index {backtrack_idx} out of range. Skipping judge.")
            return False

        # Save token trace for this attempt and reset for next
        retry_suffix = f"_retry{ss.judge_count}" if ss.judge_count > 0 else ""
        self.llm.save_trace(os.path.join(self._log_dir, f"task_{self._task_id}_token_usage{retry_suffix}.json"))
        self.llm.reset_trace()

        # Execute backtrace and replay
        ss.judge_count += 1
        ss.subtask_tree.trim_active_path(backtrack_idx)
        self._replay_target_subtask_idx = self.subtask_idx
        self._run_next_replay_subtask(new_log=True)
        return True


    def _verify_task(self, task):
        """Basic verification: reject obviously wrong answers."""
        answer = (task.answer or "").strip()

        self._is_soft_verification = False  # default hard reject

        # 1. Reject refusals
        if answer:
            refusal_patterns = [
                r"can['\u2019]?t\s+(find|answer|determine|identify|locate)",
                r"unable\s+to\s+(find|answer|determine|identify|locate)",
                r"no\s+answer\s+found",
                r"could\s+not\s+(find|determine|identify|locate)",
                r"not\s+able\s+to\s+(find|answer)",
                r"i\s+don['\u2019]?t\s+know",
                r"not\s+determinable",
                r"not\s+enough\s+verifiable\s+information",
                r"unable\s+to\s+determine\s+from\s+corpus",
            ]
            for pat in refusal_patterns:
                if re.search(pat, answer, re.IGNORECASE):
                    task.answer = EMPTY_TAG
                    self.verification_feedback = "Not a valid answer."
                    return False, f"'{answer}' is not acceptable. Search with different queries."

        # 2. Reject empty
        if not answer:
            task.answer = EMPTY_TAG
            self.verification_feedback = "Empty answer."
            return False, "Empty answer. Search the corpus to find the answer."

        # Parse answer
        answer_text, docids = _parse_finish_response(answer)

        # 3. Check search/get_document usage (skip if confirming prior candidate)
        has_search = False
        has_get_document = False
        for act, _ in self.short_term_memory.get_action_chain(task):
            if act is not None and act.valid:
                if act.name == "search":
                    has_search = True
                elif act.name == "get_document":
                    has_get_document = True
                if has_search and has_get_document:
                    break

        confirming_prior = False
        if self.subtask_idx > 0:
            prior_active = self.shared_state.subtask_tree.active_records()
            if prior_active:
                prior_ans = prior_active[-1].answer_text
                if (prior_ans and prior_ans != EMPTY_TAG
                    and (answer_matches(answer_text, prior_ans)
                        or answer_matches(prior_ans, answer_text))):
                    confirming_prior = True

        if not has_search and not confirming_prior:
            self.verification_feedback = "No search performed."
            return False, (
                "You MUST search the corpus before answering. Do NOT use internal knowledge. "
                "Start with: Think (analyze constraints) → search (3-6 word query) → get_document → ... → Finish."
            )
        if not has_search and not has_get_document:  # must confirm from prior
            self.verification_feedback = "No document read."
            self._is_soft_verification = True  # procedural failure, not answer rejection
            return False, (
                "You must call get_document to read a full document before Finishing. "
                f"Even when confirming the prior candidate '{answer_text}', you must "
                "re-read a prior doc or search for a new one, call get_document, and confirm it is the correct answer for the current sub-question, then Finish."
            )

        # 4. Check document support: answer must appear in cited doc
        supporting_doc_text = ""
        missing_docids = []
        if docids:
            # Split comma/and-separated answers so each part is checked individually
            answer_parts = [p.strip() for p in re.split(r',\s*(?:and\s+)?|\s+and\s+', answer_text) if p.strip()]
            for docid in docids:
                doc_resp = json.loads(self.env.get_document(docid))
                if "error" in doc_resp:
                    missing_docids.append(docid)
                    continue
                doc_text = doc_resp.get("text", "")
                if doc_text:
                    doc_lower = doc_text.lower()
                    if any(part.lower() in doc_lower for part in answer_parts):
                        supporting_doc_text = doc_text
                        break
            if missing_docids and not supporting_doc_text:
                self.verification_feedback = f"Cited docs {missing_docids} do NOT exist in the corpus."
                return False, (
                    f"The document(s) you cited ({missing_docids}) do NOT exist in the corpus. "
                    "You fabricated non-existent docids. This is NOT allowed. "
                    "You MUST read real documents, and ONLY Finish with real supporting documents. "
                    "Search and find correct answer with real supporting documents."
                )
            if not supporting_doc_text:
                self.verification_feedback = f"'{answer_text}' not appear in cited docs {docids}."
                return False, (
                    f"'{answer_text}' does not appear in any document you cited ({docids}). "
                    "This means your answer is NOT supported by evidence. "
                    "You may have cited the wrong docid or your answer is wrong. Search and read documents that actually contain the correct answer, then Finish with the correct answer and docid."
                )
        else:
            self.verification_feedback = "No supporting document cited."
            self._is_soft_verification = True
            return False, "You must cite at least one real document that supports your answer. Search the corpus, read a document containing the answer, then Finish with the correct docid."

        # 5. Semantic check: does the document support the answer for this sub-question?
        if supporting_doc_text:
            sem_resp = self.llm.run(prompt=(
                f"Does the document below support that '{answer_text}' is the answer to this question?\n\n"
                f"Question: {self.question}\n\nDocument:\n{supporting_doc_text}\n\n"
                "IMPORTANT: The document does NOT need to state the answer verbatim. If the document provides facts from which the answer can be reasonably inferred, that counts as SUPPORTED.\n"
                "Answer SUPPORTED or NOT_SUPPORTED with one short and concise sentence explanation. "
                "E.g., SUPPORTED: the answer is correct and supported by the document."
            )).strip()
            self.logger.get_llm_output(f"[Verify]: {sem_resp}")
            if "NOT_SUPPORTED" in sem_resp.upper():
                reason = sem_resp.split("NOT_SUPPORTED")[-1].strip().lstrip(":.").strip()
                self.verification_feedback = f"Doc does not support this answer because {reason}"
                self._is_soft_verification = True
                return False, (
                    f"Document does not support the answer '{answer_text}' "
                    f"Reason: {reason}. Search for better evidence."
                )

        self.verification_feedback = None
        self._verified_pass = True
        return True, ""


    # ------------------------------------------------------------------
    # Examples for the agent to learn from
    # ------------------------------------------------------------------

    def _build_examples(self) -> None:

        search_act = SearchAction(env=self.env)
        get_doc_act = GetDocumentAction(env=self.env)
        store_act = StoreAction()

        _task = ("Question: What is the name of the individual born in the 1950s who grew up near the coast and dedicated their career to protecting marine wildlife?")

        chain: list[tuple] = []

        chain.append((
            AgentAct(
                name=ThinkAct.action_name,
                params={INNER_ACT_KEY: "I should start with factual attributes from the question."},
            ),
            "OK",
        ))

        # Search 1: direct wording — no useful results
        chain.append((
            AgentAct(
                name=search_act.action_name,
                params={"query": "born 1950s coast protecting marine wildlife"},
            ),
            '[\n  {"docid": "901", "snippet": "Marine conservation efforts in the 1950s focused primarily on commercial fisheries regulation"},\n  {"docid": "402", "snippet": "Coastal erosion patterns during the post-war era affected wildlife habitats across"}\n]',
        ))

        chain.append((
            AgentAct(
                name=ThinkAct.action_name,
                params={INNER_ACT_KEY: "No useful results. Only generic articles about marine policy. Store this failed query and try a different subset."},
            ),
            "OK",
        ))

        # Store: record the failed search
        chain.append((
            AgentAct(
                name=store_act.action_name,
                params={STORE_SUMMARY_KEY: "Searched 'born 1950s coast protecting marine wildlife', no useful results. Only generic articles about marine policy and coastal erosion."},
            ),
            "Information Stored.",
        ))

        # Search 2: different subset + biography
        chain.append((
            AgentAct(
                name=ThinkAct.action_name,
                params={INNER_ACT_KEY: "Try a different clue subset. Focus on the coastal upbringing and add 'biography' to target biographical articles."},
            ),
            "OK",
        ))

        chain.append((
            AgentAct(
                name=search_act.action_name,
                params={"query": "born 1950s coastal town biography"},
            ),
            '[\n  {"docid": "612", "snippet": "Tourism in coastal towns boomed during the 1950s as families sought affordable seaside holidays"},\n  {"docid": "308", "snippet": "Born in a small fishing village on the Atlantic coast, she spent her childhood exploring tide pools"}\n]',
        ))

        chain.append((
            AgentAct(
                name=ThinkAct.action_name,
                params={INNER_ACT_KEY: "Doc 308 mentions someone born in a coastal fishing village. Read it. Doc 612 is about tourism, not relevant."},
            ),
            "OK",
        ))

        # Read doc 308 — not the right person
        chain.append((
            AgentAct(
                name=get_doc_act.action_name,
                params={"docid": "308"},
            ),
            '{"docid": "308", "text": "Clara Whitfield (1953-2019) was born in a small fishing village on the Atlantic coast. She spent her childhood exploring tide pools. She became a schoolteacher and raised her family in the same village."}',
        ))

        chain.append((
            AgentAct(
                name=ThinkAct.action_name,
                params={INNER_ACT_KEY: "Doc 308 mentions Clara Whitfield who was born 1953, but became a schoolteacher not a wildlife protector. Store this irrelevant document."},
            ),
            "OK",
        ))

        # Store: doc 308 not relevant
        chain.append((
            AgentAct(
                name=store_act.action_name,
                params={STORE_SUMMARY_KEY: "Doc 308: Clara Whitfield, born 1953, coastal village, became schoolteacher. Not a match (no wildlife career)."},
            ),
            "Information Stored.",
        ))

        # Search 3: changed angle — rethink then new entity type + document language
        chain.append((
            AgentAct(
                name=ThinkAct.action_name,
                params={INNER_ACT_KEY: "Two searches failed. 'Protecting marine wildlife' may be phrased as 'ocean conservation' or 'marine biology' in documents. Try adding 'marine biologist' as entity type."},
            ),
            "OK",
        ))

        chain.append((
            AgentAct(
                name=search_act.action_name,
                params={"query": "marine biologist born 1950s ocean conservation"},
            ),
            '[\n  {"docid": "567", "snippet": "Born in 1956 in a seaside town, she developed an early passion for the ocean and went on to pioneer coral reef restoration"},\n  {"docid": "213", "snippet": "The field of marine biology expanded rapidly during the 1960s with new funding from"}\n]',
        ))

        chain.append((
            AgentAct(
                name=ThinkAct.action_name,
                params={INNER_ACT_KEY: "Doc 567 looks promising: born 1956 in a seaside town, coral reef restoration. Read it."},
            ),
            "OK",
        ))

        # Read doc 567 — relevant
        chain.append((
            AgentAct(
                name=get_doc_act.action_name,
                params={"docid": "567"},
            ),
            '{"docid": "567", "text": "Elena Marchetti (born 12 March 1956) is an Italian marine biologist. Born in a seaside town on the Adriatic coast, she grew up fascinated by ocean life. She dedicated her career to protecting marine wildlife and pioneered coral reef restoration techniques in the Mediterranean."}',
        ))

        chain.append((
            AgentAct(
                name=ThinkAct.action_name,
                params={INNER_ACT_KEY: "Doc 567 matches all criteria: born 1956, grew up near coast, dedicated career to protecting marine wildlife. Store this finding."},
            ),
            "OK",
        ))

        # Store: doc 567 is relevant
        chain.append((
            AgentAct(
                name=store_act.action_name,
                params={STORE_SUMMARY_KEY: "Doc 567: Elena Marchetti, Italian marine biologist born 1956, grew up on Adriatic coast, dedicated to marine wildlife protection. All criteria match."},
            ),
            "Information Stored.",
        ))

        chain.append((
            AgentAct(
                name=ThinkAct.action_name,
                params={INNER_ACT_KEY: "Based on doc 567, the answer is Elena Marchetti."},
            ),
            "OK",
        ))

        # Finish
        chain.append((
            AgentAct(
                name=FinishAct.action_name,
                params={INNER_ACT_KEY: "answer: Elena Marchetti; docids: [567]"},
            ),
            "Task Completed.",
        ))

        self.add_example(
            task=TaskPackage(instruction=_task),
            action_chain=chain,
        )
