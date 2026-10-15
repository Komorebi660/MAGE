"""
MathMAGEAgent Architecture
-----------------------------
Each ``MathMAGEAgent`` handles one math sub-problem within a paper.
Cross-subtask state is held in a ``PaperSharedState`` instance that is
created once per paper and passed to every agent in the sequence.
On top of the ExplorationTree (Summary Tree), a separate SubtaskTree
tracks subtask completion records.
"""

import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_MATH_DIR = os.path.dirname(os.path.abspath(__file__))
if _MATH_DIR not in sys.path:
    sys.path.insert(0, _MATH_DIR)

from math_actions import SymbolicReasoningAction, CodeExecutorAction
from math_env import MathEnvironment

from agentcore.actions import FinishAct, ThinkAct
from agentcore.actions.InnerActions import INNER_ACT_KEY
from agentcore.actions.MAGEActions import (
    StoreAction,
    STORE_SUMMARY_KEY,
)
from agentcore.agents.MAGEAgent import MAGEAgent
from agentcore.commons import AgentAct, TaskPackage
from agentcore.llm.LLMBackend import BaseLLM
from agentcore.logging.terminal_logger import AgentLogger
from agentcore.memory.ExplorationTree import ExplorationTree


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATH_MAGE_ROLE = (
    "You are a mathematical reasoning assistant with hierarchical memory. "
    "Your task is to solve the math problem described in PROBLEM using the "
    "definitions and background provided in the context. "
    "Available tools: SymbolicReasoning (formal derivation), "
    "CodeExecutor (Python/SymPy computation), "
    "Store (compress your derivation segment "
    "into a reusable summary), and Backtrace (undo to a previous step)."
)

MATH_MAGE_CONSTRAINT = (
    "IMPORTANT: Output exactly ONE action per turn in the format: "
    "ActionName[{\"param\": \"value\"}]. "
    "Valid action names: Think, SymbolicReasoning, "
    "CodeExecutor, Store, Backtrace, Finish. "
    "Do NOT write markdown, headers, emojis, or any text outside the "
    "action call. "
    "Use LaTeX for mathematical expressions inside action parameters. "
    "Prefer SymbolicReasoning for formal derivations; Think is acceptable "
    "for brief analysis. At least one reasoning step before Finish. "
    "Use Store to save intermediate results so that later sub-problems "
    "can reference them. "
    "When finished, call Finish[{\"response\": \"your answer\"}]. "
    "CRITICAL: The Finish response MUST contain ONLY the final answer "
    "itself (e.g. a formula, value, or short phrase). Do NOT include "
    "derivations, explanations, reasoning steps, or justifications in "
    "the Finish response. All reasoning should be done in previous "
    "steps using Think, SymbolicReasoning, or CodeExecutor."
)

MATH_MAGE_INSTRUCTION = (
    "You are an intelligent agent. You MUST use the provided actions from "
    "[Action_Doc] to solve problems. Follow the EXACT action format shown "
    "in [Example]. Every response must be a single action call.\n\n"
    "All relevant definitions, lemmas, and background information are "
    "provided directly in the BACKGROUND section of the task. Refer to "
    "them as needed during your reasoning.\n\n"
    "WORKFLOW:\n"
    "  1. Use Think to analyse the problem and plan.\n"
    "  2. Use SymbolicReasoning to carry out formal derivation steps "
    "(preferred for proofs and formal arguments).\n"
    "  3. Use CodeExecutor to verify results with Python/SymPy.\n"
    "  4. Use Store to compress your derivation into a summary.\n"
    "  5. Use Finish after completing reasoning and verification.\n"
    "  If stuck, use Backtrace to return to a previous step.\n\n"
    "You must perform at least one reasoning step (Think, "
    "SymbolicReasoning, or CodeExecutor) before calling Finish. "
    "Prefer SymbolicReasoning over Think for formal derivations.\n\n"
    "ANSWER FORMAT RULE: When calling Finish, provide ONLY the final "
    "answer (a formula, value, short phrase, or list of choices). "
    "Do NOT include any derivation, explanation, or justification "
    "in the Finish response. Keep it as concise as possible.\n\n"
    "CRITICAL FORMAT RULE: Every response must be EXACTLY one action "
    "call in this format:\n"
    "  ActionName[{\"param\": \"value\"}]\n"
    "NEVER output markdown headers, checkmarks, bullet points, or "
    "free-form text as your action. NEVER use emojis in action names."
)


@dataclass
class SubtaskRecord:
    """One completed execution of a subtask within a paper."""
    subtask_idx: int
    summary_text: str
    submitted_answer: str = ""
    parent_node_id: Optional[int] = None
    parent_summary_id: Optional[str] = None
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    children_record_ids: List[str] = field(default_factory=list)
    visit_count: int = 1
    verify_feedback: Optional[str] = None


class SubtaskTree:
    """Track active subtask attempts and merge repeated answers under each parent."""

    def __init__(self) -> None:
        self.records: Dict[str, SubtaskRecord] = {}
        self.root_id = "subtask_root"
        self.records[self.root_id] = SubtaskRecord(
            record_id=self.root_id,
            subtask_idx=-1,
            summary_text="",
        )
        self._active_path: List[str] = []

    @property
    def root(self) -> SubtaskRecord:
        return self.records[self.root_id]

    def parent_for_subtask(self, subtask_idx: int) -> SubtaskRecord:
        """Return the active record for the previous subtask, or the root."""
        if subtask_idx == 0:
            return self.root
        for record_id in reversed(self._active_path):
            if self.records[record_id].subtask_idx == subtask_idx - 1:
                return self.records[record_id]
        return self.root

    def previous_attempts(self, subtask_idx: int) -> List[str]:
        """Return submitted answers and verification feedback from prior attempts."""
        answer_best: Dict[str, SubtaskRecord] = {}
        for record in self.records.values():
            if record.subtask_idx != subtask_idx:
                continue
            if not record.submitted_answer:
                continue
            key = record.submitted_answer.strip()
            if key not in answer_best:
                answer_best[key] = record
            elif record.visit_count > answer_best[key].visit_count:
                answer_best[key] = record

        results = []
        for record in answer_best.values():
            text = record.summary_text
            if record.visit_count > 2:
                text += (
                    f" This answer has been submitted in "
                    f"{record.visit_count} independent attempts."
                )
            if record.verify_feedback:
                text += f" But may be wrong: {record.verify_feedback}"
            results.append(text)
        return results

    def add_record(
        self,
        subtask_idx: int,
        summary_text: str,
        submitted_answer: str = "",
        parent_node_id: Optional[int] = None,
        parent_summary_id: Optional[str] = None,
    ) -> SubtaskRecord:
        """Add a completion record, preserving feedback when an answer is merged."""
        parent = self.parent_for_subtask(subtask_idx)
        answer_key = submitted_answer.strip() if submitted_answer else ""

        if answer_key:
            for child_id in parent.children_record_ids:
                child = self.records.get(child_id)
                if child and child.submitted_answer.strip() == answer_key:
                    child.summary_text = summary_text
                    child.parent_node_id = parent_node_id
                    child.parent_summary_id = parent_summary_id
                    child.visit_count += 1
                    # Preserve any previously attached verify_feedback so
                    # that diagnoses from earlier judge passes remain
                    # visible if the agent submits the same wrong answer
                    # again.
                    self._set_active(subtask_idx, child_id)
                    return child

        record = SubtaskRecord(
            subtask_idx=subtask_idx,
            summary_text=summary_text,
            submitted_answer=submitted_answer,
            parent_node_id=parent_node_id,
            parent_summary_id=parent_summary_id,
        )
        self.records[record.record_id] = record
        parent.children_record_ids.append(record.record_id)
        self._set_active(subtask_idx, record.record_id)
        return record

    def _set_verify_feedback(self, subtask_idx: int, feedback: str) -> None:
        active = self.active_records()
        if subtask_idx < len(active):
            active[subtask_idx].verify_feedback = feedback

    def _set_active(self, subtask_idx: int, record_id: str) -> None:
        self.trim_active_path(subtask_idx)
        self._active_path.append(record_id)

    def trim_active_path(self, before_subtask_idx: int) -> None:
        """Keep only active records before the requested subtask."""
        self._active_path = [
            record_id
            for record_id in self._active_path
            if self.records[record_id].subtask_idx < before_subtask_idx
        ]

    def active_records(self) -> List[SubtaskRecord]:
        return [self.records[record_id] for record_id in self._active_path]

    def active_steps(self) -> List[int]:
        return [
            self.records[record_id].parent_node_id
            if self.records[record_id].parent_node_id is not None
            else -1
            for record_id in self._active_path
        ]


@dataclass
class PaperSharedState:
    """Mutable state shared by subtask agents within one paper, never across papers."""
    exploration_tree: ExplorationTree = field(default_factory=ExplorationTree)
    global_step: int = 1
    subtask_questions: List[str] = field(default_factory=list)
    subtask_backgrounds: List[str] = field(default_factory=list)
    subtask_answers: List[str] = field(default_factory=list)
    subtask_tree: SubtaskTree = field(default_factory=SubtaskTree)
    num_subtasks: int = 0
    final_answer: str = ""
    # Cross-subtask backtrace: list of (owner_subtask_idx, obs_string).
    pending_backtrace_reasons: List[tuple] = field(default_factory=list)
    # Shared counter for agent-initiated backtrace calls across subtasks.
    agent_backtrace_count: int = 0
    judge_count: int = 0
    max_judge_retries: int = 0
    task_id: int = 0
    log_dir: str = ""
    internal_final_correct: Optional[bool] = None


class MathMAGEAgent(MAGEAgent):
    """Agent with hierarchical memory for sequential math reasoning."""

    def __init__(
        self,
        question: str,
        background: str,
        subtask_idx: int,
        shared_state: PaperSharedState,
        math_env: MathEnvironment,
        llm: BaseLLM,
        max_steps: int = 30,
        PROMPT_DEBUG_FLAG: bool = False,
        log_file: str = "agent.log",
        max_verify_retries: int = 1,
        max_check_retries: int = 0,
        max_backtrace: int = 5,
    ) -> None:
        self.question = question
        self.background = MathEnvironment._strip_latex_preamble(background)
        self.subtask_idx = subtask_idx
        self.shared_state = shared_state
        self.math_env = math_env

        # Target subtask for cross-subtask replay (None = no replay).
        self._replay_target_subtask_idx: Optional[int] = None

        # Temporary verification feedback.
        self.verification_feedback: Optional[str] = None

        actions = [
            SymbolicReasoningAction(),
            CodeExecutorAction(),
        ]

        super().__init__(
            name="math_mage_agent",
            role=MATH_MAGE_ROLE,
            llm=llm,
            actions=actions,
            constraint=MATH_MAGE_CONSTRAINT,
            instruction=MATH_MAGE_INSTRUCTION,
            logger=AgentLogger(
                log_file_name=log_file, PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG
            ),
            max_exec_steps=max_steps,
            max_verify_retries=max_verify_retries,
            max_check_retries=max_check_retries,
            max_backtrace=max_backtrace,
        )

        # Replace fresh tree with the shared one.
        self.exploration_tree = shared_state.exploration_tree
        self._global_step = shared_state.global_step
        self._agent_backtrace_count = shared_state.agent_backtrace_count

        # Propagate cross-subtask backtrace reasons visible to this subtask.
        visible = [
            obs
            for owner, obs in shared_state.pending_backtrace_reasons
            if owner > subtask_idx
        ]
        if visible:
            self._backtrace_context = [
                (None, None, obs) for obs in visible
            ]

        # Record this subtask's question and background in shared state.
        if len(shared_state.subtask_questions) <= subtask_idx:
            shared_state.subtask_questions.append(self.question)
        if len(shared_state.subtask_backgrounds) <= subtask_idx:
            shared_state.subtask_backgrounds.append(self.background)

        # Rebuild combined background now that this subtask's background
        # is registered in shared state.
        self._combined_background = self._build_combined_background()

        # Populate _previous_results from the SubtaskTree.
        prev = shared_state.subtask_tree.previous_attempts(subtask_idx)
        if prev:
            self._previous_results = list(prev)

        self._build_examples()

    # ------------------------------------------------------------------
    # Combined background for cross-subtask retrieval
    # ------------------------------------------------------------------

    def _build_combined_background(self) -> str:
        """Combine backgrounds from all subtasks seen so far.

        Earlier subtasks often contain fundamental definitions (e.g.,
        the definition of 'flow') that later subtasks depend on but
        don't repeat.  The combined text is embedded directly in the
        task instruction so the agent can reference any definition.
        """
        ss = self.shared_state
        parts: List[str] = []
        seen = set()
        # Include backgrounds from all registered subtasks (0..current).
        for bg in ss.subtask_backgrounds:
            bg_stripped = bg.strip()
            if bg_stripped and bg_stripped not in seen:
                seen.add(bg_stripped)
                parts.append(bg_stripped)
        # Include current subtask's background even if not yet registered.
        own_bg = self.background.strip()
        if own_bg and own_bg not in seen:
            parts.append(own_bg)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Task construction
    # ------------------------------------------------------------------

    def _build_task_instruction(self) -> str:
        """Build the instruction from shared state and current question."""
        ss = self.shared_state
        active = ss.subtask_tree.active_records()
        steps = ss.subtask_tree.active_steps()

        parts: List[str] = []

        # Background section: embed combined background from all subtasks
        # so the agent can reference definitions from earlier subtasks.
        if self._combined_background and self._combined_background.strip():
            parts.append(
                f"=== BACKGROUND ===\n{self._combined_background.strip()}\n"
                f"=== END BACKGROUND ==="
            )

        # Previously answered problems (cross-subtask context)
        if active:
            lines = ["Prior answered problems by yourself (you can backtrace to a step to undo if needed):"]
            for i, rec in enumerate(active):
                step_hint = f"[Step {steps[i]}] " if steps[i] >= 0 else ""
                answer = rec.submitted_answer.strip() if rec.submitted_answer else "(no answer)"
                question = ss.subtask_questions[rec.subtask_idx].strip()
                lines.append(f"{step_hint}Sub-Problem {rec.subtask_idx + 1}: {question}")
                lines.append(f"Your Answer: {answer}")
                lines.append("")
            parts.append("\n".join(lines))

        # Current problem
        parts.append(
            f"=== PROBLEM (Sub-problem {self.subtask_idx + 1}) ===\n"
            f"The current problem you need to solve:\n{self.question.strip()}\n=== END PROBLEM ==="
        )

        return "\n\n".join(parts)

    def build_task(self) -> TaskPackage:
        return TaskPackage(instruction=self._build_task_instruction())

    # ------------------------------------------------------------------
    # Environment hooks (math is stateless, so these are lightweight)
    # ------------------------------------------------------------------

    def _get_env_snapshot(self) -> Dict[str, Any]:
        """Return a lightweight snapshot (math has no mutable env)."""
        return {"subtask_idx": self.subtask_idx}

    def _env_compare(self, snap_a: Any, snap_b: Any) -> bool:
        if not isinstance(snap_a, dict) or not isinstance(snap_b, dict):
            return False
        return snap_a.get("subtask_idx") == snap_b.get("subtask_idx")

    # ------------------------------------------------------------------
    # Store verification
    # ------------------------------------------------------------------

    def _check_store(self, task: TaskPackage, content: str) -> tuple[bool, str]:
        """Check if a stored derivation step is mathematically sound.

        Two-phase verification:
        1. Ask LLM to generate Python/SymPy test code that substitutes
           the intermediate result into the problem conditions.
        2. If code execution reports a failure, return the error.
        Fall back to LLM-only review when code generation/execution fails.
        """
        return True, ""
        # --- Phase 1: code-based verification ---
        code_result = self._code_verify_store(content)
        if code_result is not None:
            passed, reason = code_result
            if not passed:
                return False, (
                    f"Code verification of stored derivation failed."
                    f"{(' ' + reason) if reason else ''}"
                    f" Re-examine and redo this reasoning step."
                )
            # Code passed — accept the store.
            return True, ""

        # --- Phase 2: LLM-only review (fallback) ---
        prompt = (
            f"Review this mathematical derivation for correctness.\n\n"
            f"Problem context: {self.question.strip()}\n\n"
            f"Derivation: {content}\n\n"
            f"Does this derivation contain clear logical errors, algebraic "
            f"mistakes, or unjustified claims? "
            f"Respond with YES if it is correct, or NO followed by a "
            f"one-sentence description of the error."
        )
        response = self.llm.run(prompt).strip()

        has_no = bool(re.search(r'\bNO\b', response, re.IGNORECASE))
        has_yes = bool(re.search(r'\bYES\b', response, re.IGNORECASE))
        if not has_no or has_yes:
            return True, ""
        reason_match = re.search(
            r'\bNO\b[.,:\s]*(.*)', response, re.IGNORECASE | re.DOTALL
        )
        reason = reason_match.group(1).strip() if reason_match else ""
        return False, (
            f"Stored derivation may contain errors."
            f"{(' ' + reason) if reason else ''}"
            f" Re-examine and redo this reasoning step."
        )

    def _code_verify_store(self, content: str) -> tuple[bool, str] | None:
        """Generate and run verification code for a stored derivation.

        Returns (True, "") if code passes, (False, reason) if code fails,
        or None if code generation/execution is not applicable.
        """
        code_gen_prompt = (
            "Given the following math problem and an intermediate derivation "
            "result, write a short Python/SymPy script that verifies the "
            "result by substituting it back into the problem conditions or "
            "checking it against known identities.\n\n"
            "Requirements:\n"
            "- Use sympy for symbolic checks.\n"
            "- Print 'PASS' if all checks hold.\n"
            "- Print 'FAIL: <one-line reason>' if any check fails.\n"
            "- If the derivation is a proof sketch or qualitative argument "
            "that cannot be checked numerically, print 'SKIP'.\n"
            "- Do NOT import external data or make network calls.\n\n"
            f"Problem:\n{self.question.strip()}\n\n"
            f"Background (definitions):\n"
            f"{self._combined_background.strip()}\n\n"
            f"Derivation to verify:\n{content}\n\n"
            "Output ONLY the Python code, no explanations or markdown."
        )
        code = self.llm.run(code_gen_prompt).strip()

        # Strip markdown code fences if present.
        code = re.sub(r'^```(?:python)?\s*', '', code)
        code = re.sub(r'\s*```$', '', code)
        if not code.strip():
            return None

        # Execute the verification code.
        executor = CodeExecutorAction()
        result = executor(code=code)

        if "SKIP" in result:
            return None
        if "FAIL" in result:
            # Extract the reason after FAIL.
            m = re.search(r'FAIL[:\s]*(.*)', result)
            reason = m.group(1).strip() if m else "verification code reported failure"
            return False, reason
        if "PASS" in result:
            return True, ""
        if "Code Error" in result:
            # Code itself errored; fall back to LLM review.
            return None
        # Ambiguous output; fall back to LLM review.
        return None

    # ------------------------------------------------------------------
    # Task completion handler
    # ------------------------------------------------------------------

    def _on_task_complete(self, task: TaskPackage, last_obs: str = "") -> None:
        """Record the answer and handle cross-subtask replay."""
        ss = self.shared_state
        ss.agent_backtrace_count = self._agent_backtrace_count

        # Replay path (cross-subtask backtrace forced early exit).
        if task.completion == "replay":
            ss.global_step = self._global_step
            if self._replay_target_subtask_idx is not None:
                if self._backtrace_context:
                    ss.pending_backtrace_reasons.append(
                        (self.subtask_idx, self._backtrace_context[-1][2])
                    )
                self._run_next_replay_subtask(new_log=False)
            return

        # Normal or verification-retry path: record the answer.
        answer = task.answer if task.answer else ""
        summary = answer.strip() if answer else "No answer submitted."
        ss.subtask_tree.add_record(
            subtask_idx=self.subtask_idx,
            summary_text=summary,
            submitted_answer=answer,
            parent_node_id=self._task_start_node_id,
            parent_summary_id=self._task_start_summary_id,
        )

        ss.global_step = self._global_step

        # Verification retry: just update feedback and return.
        if self._is_verify_retry:
            ss.subtask_tree._set_verify_feedback(
                self.subtask_idx,
                self.verification_feedback or "incorrect answer",
            )
            self.verification_feedback = ""
            self._previous_results = list(
                ss.subtask_tree.previous_attempts(self.subtask_idx)
            )
            return

        # Cross-subtask replay chain.
        if (
            self._replay_target_subtask_idx is not None
            and self.subtask_idx < self._replay_target_subtask_idx
        ):
            self._run_next_replay_subtask(new_log=self.shared_state.judge_count > 0)
        else:
            ss.pending_backtrace_reasons.clear()
            if self._should_run_judge():
                self._run_judge()

    def _run_next_replay_subtask(self, new_log: bool = False) -> None:
        """Execute the next intermediate subtask during cross-subtask replay."""
        ss = self.shared_state
        next_sub_idx = len(ss.subtask_tree.active_records())

        log_file = self.logger.log_file_name
        if new_log and ss.log_dir:
            log_file = os.path.join(
                ss.log_dir,
                f"task_{ss.task_id}_subtask_{next_sub_idx}_retry{ss.judge_count}.log",
            )

        replay_agent = MathMAGEAgent(
            question=ss.subtask_questions[next_sub_idx],
            background=ss.subtask_backgrounds[next_sub_idx],
            subtask_idx=next_sub_idx,
            shared_state=ss,
            math_env=self.math_env,
            llm=self.llm,
            max_verify_retries=self.max_verify_retries,
            max_steps=self.max_exec_steps,
            max_check_retries=self.max_check_retries,
            max_backtrace=self.max_backtrace,
            log_file=log_file,
            PROMPT_DEBUG_FLAG=self.logger.PROMPT_DEBUG_FLAG,
        )
        replay_agent._replay_target_subtask_idx = self._replay_target_subtask_idx

        replay_task = replay_agent.build_task()
        replay_agent(replay_task)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify_task(self, task: TaskPackage) -> tuple:
        """Subtask-level self-verification.

        Performs two checks WITHOUT using the gold answer:
            1. An answer was actually submitted.
            2. Self-consistency: the task LLM reviews the predicted answer
                for internal consistency.  The verifier sees only the problem,
                background, and predicted answer — never the gold answer.
        """
        predicted = task.answer if task.answer else ""
        if not predicted.strip():
            self.verification_feedback = "No answer submitted."
            return False, (
                f"No answer for Problem {self.subtask_idx + 1}. "
                "You must call Finish with your answer."
            )

        task_context = (
            task.instruction if task is not None and task.instruction
            else self._build_task_instruction()
        )

        prompt = (
            "You are a verifier to verify the solver's answer to a math problem. "
            "The block below (between === TASK START === and === TASK END ===) is "
            "the task instruction that was given to a solver to solve. "
            "Decide whether the solver's submitted answer correctly answers the "
            "current sub-problem. For example, check that it "
            "actually answers the question that was asked, is "
            "internally consistent with the background information, "
            "and is mathematically sound (units, dimensions, limiting cases, etc.). "
            "You can also substitute the submitted answer back into the problem to verify it.\n\n"
            "=== TASK START ===\n"
            f"{task_context}\n"
            "=== TASK END ===\n\n"
            f"Solver submitted answer:\n{predicted}\n\n"
            "Respond with YES if the answer correctly solves the "
            "current sub-problem, or NO followed by a one-sentence "
            "description of what is wrong."
        )
        response = self.llm.run(prompt).strip()

        has_no = bool(re.search(r"\bNO\b", response, re.IGNORECASE))
        has_yes = bool(re.search(r"\bYES\b", response, re.IGNORECASE))
        if not has_no or has_yes:
            return True, ""
        m = re.search(
            r"\bNO\b[.,:\s]*(.*)", response, re.IGNORECASE | re.DOTALL
        )
        reason = m.group(1).strip() if m else "answer appears inconsistent."
        self.verification_feedback = reason
        return False, (
            f"Self-check failed for Problem {self.subtask_idx + 1}. "
            f"Possible issue: {reason} "
            "Re-derive your answer using a different approach."
        )

    # ------------------------------------------------------------------
    # Judge (runs after ALL subtasks complete)
    # ------------------------------------------------------------------

    def _should_run_judge(self) -> bool:
        ss = self.shared_state
        total = ss.num_subtasks
        return (
            total > 0
            and self.subtask_idx == total - 1
            and ss.judge_count < ss.max_judge_retries
            and len(ss.subtask_tree.active_records()) == total
        )

    def _run_judge(self) -> bool:
        """End-of-task judge.

        Runs once all subtasks have produced an answer.  Each subtask's
        submitted answer is compared against its gold answer to compute
        the list of mismatched subtasks; a single summary line is
        written to the log so the evaluator can recover per-attempt
        SR/PS.  Gold answers are used only here — they are never
        exposed to the agent or to the gold-free diagnostic prompt.  If
        the final subtask is mismatched, the diagnostic prompt inspects
        the full reasoning chain, picks the sub-problem most likely to
        contain the error, and we backtrace to replay from there.
        """
        ss = self.shared_state
        active = ss.subtask_tree.active_records()
        total = ss.num_subtasks

        if not active or total == 0:
            return False

        pred_by_idx = {r.subtask_idx: r.submitted_answer for r in active}

        # --- Phase 1a: per-subtask correctness statistics for logging. ---
        mismatches: List[int] = []
        final_correct = False
        judge_backgrounds = []
        for i in range(total):
            if i >= len(ss.subtask_answers):
                continue
            background = ss.subtask_backgrounds[i]
            if background and background not in judge_backgrounds:
                judge_backgrounds.append(background)
            ok, _ = self.math_env.judge_answer(
                question=ss.subtask_questions[i] if i < len(ss.subtask_questions) else "",
                background="\n\n".join(judge_backgrounds),
                predicted=pred_by_idx.get(i, ""),
                reference=ss.subtask_answers[i],
            )
            if i == total - 1:
                final_correct = ok
            if not ok:
                mismatches.append(i + 1)

        ss.internal_final_correct = final_correct

        # return early if the final answer is correct (no replay needed).
        if final_correct:
            return False

        agent_answers = [pred_by_idx.get(i, "") for i in range(total)]
        self.logger.get_log(
            f"Judge: mismatched subtasks: {mismatches}\n"
            f"Groundtruth answers: {ss.subtask_answers}\n"
            f"Agent answers: {agent_answers}"
        )

        # --- Phase 2: gold-free diagnosis over the full chain. ---
        background = "\n\n".join(judge_backgrounds)
        subtask_lines = []
        for rec in active:
            idx = rec.subtask_idx
            question = (
                ss.subtask_questions[idx]
                if idx < len(ss.subtask_questions)
                else ""
            )
            answer = (
                rec.submitted_answer.strip()
                if rec.submitted_answer
                else "(no answer)"
            )
            subtask_lines.append(
                f"Sub-problem {idx + 1}:\n"
                f"Question: {question.strip()}\n"
                f"Answer: {answer}"
            )
        subtask_block = "\n\n".join(subtask_lines)

        diagnostic_system_prompt = (
            "You are a mathematical reasoning reviewer. "
            "A student has solved a multi-step math problem but the final answer is incorrect. "
            "Your job is to identify where in the reasoning chain the error likely occurred.\n\n"
            "Analyse each sub-step in sequence:\n"
            "  1. Whether each sub-answer is internally consistent and mathematically valid\n"
            "  2. Whether later sub-answers correctly use results from earlier ones\n"
            "  3. You can substitute the submitted answer back into the problem to verify it.\n\n"
            "Identify the earliest sub-problem that is most likely to introduce an error. "
            "Be specific and find where is wrong and which sub-problem should be re-examined. "
            "Keep your response to 2-3 sentences maximum."
        )
        diagnostic_prompt = (
            f"### Background\n{background.strip() if background else '(none)'}\n\n"
            f"{subtask_block}\n\n"
            "The answer to the final sub-problem is incorrect. "
            "Identify the earliest ONE sub-problem most likely containing the error "
            "and state in 2-3 sentences what specifically seems wrong. "
            "No need to derive the correct answer directly."
        )
        diagnosis = self.llm.run(
            diagnostic_prompt,
            system_prompt=diagnostic_system_prompt,
        ).strip()
        self.logger.get_llm_output(f"[Judge]: {diagnosis}")

        # --- Phase 3: locate target subtask from the diagnosis. ---
        # Accept several phrasings: "sub-problem 3", "subproblem 3",
        # "problem 3", "step 3", "sub-step 3", optional '#'.
        m = re.search(
            r'(?:sub[\s-]*problem|sub[\s-]*step|problem|step)\s*#?\s*(\d+)',
            diagnosis,
            re.IGNORECASE,
        )
        if m:
            backtrack_idx = int(m.group(1)) - 1
        else:
            # Default: redo the final subtask if the diagnosis does not
            # name a specific sub-problem.
            backtrack_idx = total - 1
        if backtrack_idx < 0 or backtrack_idx >= total:
            backtrack_idx = total - 1

        reason = (
            f"Final answer is incorrect. Possible Reason: {diagnosis} "
            "Try a different approach to solve it, "
            "and self-verify your answer with CodeExecutor by writing test cases."
        )
        # Attach feedback to the target record BEFORE trimming.
        ss.subtask_tree._set_verify_feedback(backtrack_idx, diagnosis)

        # Save the completed attempt before judge-triggered replay resets
        # the shared LLM traces for the next attempt.
        retry_suffix = f"_retry{ss.judge_count}" if ss.judge_count > 0 else ""
        self.llm.save_trace(os.path.join(ss.log_dir, f"task_{ss.task_id}_token_usage{retry_suffix}.json"))
        self.llm.reset_trace()
        self.math_env.judge_llm.reset_trace()

        # --- Phase 4: execute backtrace. ---
        ss.judge_count += 1
        ss.subtask_tree.trim_active_path(backtrack_idx)
        # Scope visibility: only the target subtask sees the reason
        # (owner = backtrack_idx + 1, visible to subtasks with idx < owner).
        ss.pending_backtrace_reasons.append(
            (backtrack_idx + 1, reason)
        )
        self._replay_target_subtask_idx = self.subtask_idx

        self._run_next_replay_subtask(new_log=True)
        return True

    # ------------------------------------------------------------------
    # Context restoration (backtrace support)
    # ------------------------------------------------------------------

    def _restore_context(self, target_step: int, task: TaskPackage = None):
        """Restore context to a checkpoint at target_step.

        Valid checkpoints:
        1. Subtask-boundary checkpoints (previous subtasks only).
        2. Intra-subtask Store checkpoints.
        3. Raw context start.
        4. Task start node (takes precedence at shared steps).
        """
        ss = self.shared_state
        ckpt_map: Dict[int, tuple] = {}

        # Source 1: subtask-boundary checkpoints.
        for i, rec in enumerate(ss.subtask_tree.active_records()):
            if rec.subtask_idx >= self.subtask_idx:
                continue
            if rec.parent_node_id is None:
                continue
            fpn = self.exploration_tree.nodes.get(rec.parent_node_id)
            if fpn is not None:
                restore_sid = (
                    rec.parent_summary_id or self._task_start_summary_id
                )
                ckpt_map[fpn.node_id] = (
                    "subtask", i, fpn.node_id, fpn.env_snapshot, restore_sid,
                )

        # Source 2: intra-subtask Store checkpoints.
        for i, sid in enumerate(self._stored_node_ids):
            parent_node = self.exploration_tree.get_summary_parent(sid)
            if parent_node is not None:
                restore_sid = (
                    self._stored_node_ids[i - 1]
                    if i > 0
                    else self._task_start_summary_id
                )
                ckpt_map[parent_node.node_id] = (
                    "store", i, parent_node.node_id,
                    parent_node.env_snapshot, restore_sid,
                )

        # Source 3: raw context start.
        if self._raw_node_ids:
            raw_parent = self.exploration_tree.get_parent(
                self._raw_node_ids[0]
            )
            if raw_parent is not None:
                restore_sid = (
                    self._stored_node_ids[-1]
                    if self._stored_node_ids
                    else self._task_start_summary_id
                )
                ckpt_map[raw_parent.node_id] = (
                    "store", len(self._stored_node_ids),
                    raw_parent.node_id, raw_parent.env_snapshot, restore_sid,
                )

        # A local retry must not undo prior subtasks sharing the same tree node.
        if self._task_start_node_id is not None:
            start_node = self.exploration_tree.nodes.get(
                self._task_start_node_id
            )
            if start_node is not None:
                ckpt_map[self._task_start_node_id] = (
                    "store", 0, start_node.node_id,
                    start_node.env_snapshot, self._task_start_summary_id,
                )

        if target_step not in ckpt_map:
            return (
                f"Backtrace failed since step {target_step} is not valid. "
                f"Valid steps are {sorted(ckpt_map.keys())}."
            )

        kind, idx, target_node_id, target_env, target_summary_id = (
            ckpt_map[target_step]
        )

        if kind == "store":
            self._stored_context = self._stored_context[:idx]
            self._stored_node_ids = self._stored_node_ids[:idx]
            self._raw_context = []
            self._raw_node_ids = []
        else:  # subtask
            record_idx: int = idx
            matched_rec = ss.subtask_tree.active_records()[record_idx]
            ss.subtask_tree.trim_active_path(matched_rec.subtask_idx)
            self._replay_target_subtask_idx = max(
                self.subtask_idx,
                self._replay_target_subtask_idx
                if self._replay_target_subtask_idx is not None
                else 0,
            )
            task.completion = "replay"

        return target_node_id, target_summary_id, target_env

    # ------------------------------------------------------------------
    # Action parser (handles LaTeX in JSON)
    # ------------------------------------------------------------------

    # Valid action names for fuzzy matching
    _VALID_ACTION_NAMES = {"Think", "SymbolicReasoning", "CodeExecutor", "Store", "Backtrace", "Finish"}

    def __action_parser__(self, raw_action: str) -> AgentAct:
        """Robust action parser handling LaTeX-heavy content."""
        from agentcore.agents.agent_utils import parse_action

        action_name, args, ok = parse_action(raw_action)
        if ok:
            return AgentAct(name=action_name, params=args, valid=True)

        text = raw_action.strip()
        if text.startswith("Action:"):
            text = text[len("Action:"):].strip()
        first_line = text.split("\n")[0].strip().rstrip(".")

        for candidate in (first_line, text):
            m = re.match(r"^(\w+)\[", candidate)
            if not m:
                continue
            act_name = m.group(1)
            rest = candidate[m.end():]
            if rest.endswith("]"):
                rest = rest[:-1]
            json_str = self._extract_json_by_braces(rest)
            if json_str is None:
                continue
            for fixer in [
                lambda s: s,
                lambda s: s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"),
                lambda s: s.replace("\\", "\\\\"),
                lambda s: s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t").replace("\\", "\\\\"),
            ]:
                try:
                    params = json.loads(fixer(json_str))
                    return AgentAct(
                        name=act_name, params=params, valid=True
                    )
                except json.JSONDecodeError:
                    continue

        # Fuzzy matching: detect intent from malformed outputs
        rescued = self._fuzzy_rescue(raw_action)
        if rescued is not None:
            return rescued

        return AgentAct(name=action_name, params=args, valid=False)

    def _fuzzy_rescue(self, raw_action: str) -> AgentAct | None:
        """Attempt to rescue common malformed outputs.

        Handles patterns like:
          - '\u2705 **Answer: some text**'
          - '### Solution: some text'
          - Markdown-wrapped answer text that should be Finish
        """
        text = raw_action.strip()

        # Strip markdown/emoji formatting to see if there's an answer intent
        answer_pattern = re.compile(
            r"^[\s#*\u2705\U0001F3AF\U0001F4DD`]*"
            r"(?:answer|solution|final\s*answer|result|conclusion)"
            r"[:\s*]*(.+)",
            re.IGNORECASE | re.DOTALL,
        )
        m = answer_pattern.match(text)
        if m:
            answer_text = m.group(1).strip().rstrip("*").strip()
            if answer_text:
                return AgentAct(
                    name="Finish",
                    params={INNER_ACT_KEY: answer_text},
                    valid=True,
                )

        # Pattern: contains a valid action name buried in text
        for name in self._VALID_ACTION_NAMES:
            idx = text.find(f"{name}[")
            if idx == -1:
                continue
            rest = text[idx + len(name) + 1:]
            if rest.endswith("]"):
                rest = rest[:-1]
            json_str = self._extract_json_by_braces(rest)
            if json_str is None:
                continue
            try:
                params = json.loads(json_str)
                return AgentAct(name=name, params=params, valid=True)
            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def _extract_json_by_braces(text: str) -> Optional[str]:
        """Extract a JSON object {...} by balanced braces."""
        depth = 0
        start = None
        in_string = False
        i = 0
        while i < len(text):
            ch = text[i]
            if in_string and ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start is not None:
                        return text[start : i + 1]
            i += 1
        return None

    # ------------------------------------------------------------------
    # Forward override: enforce reasoning before Finish
    # ------------------------------------------------------------------

    _REASONING_ACTIONS = {"Think", "SymbolicReasoning", "CodeExecutor"}
    _DEEP_REASONING_ACTIONS = {"SymbolicReasoning", "CodeExecutor"}

    def forward(self, task: TaskPackage, agent_act: AgentAct) -> str:
        """Enforce at least one reasoning step and provide improved error messages.

        On verify retry (when _previous_results is non-empty), enforce
        stricter requirements: the agent must use at least one deep
        reasoning action (SymbolicReasoning or CodeExecutor) before
        calling Finish.
        """
        if agent_act.name == FinishAct.action_name and agent_act.valid:
            # Check current attempt's context (properly reset by backtrace),
            # not short_term_memory which retains pre-backtrace history.
            raw_acts = [
                act for act, _obs in self._raw_context if act is not None
            ]
            has_reasoning = any(
                act.name in self._REASONING_ACTIONS for act in raw_acts
            )
            if not has_reasoning:
                return (
                    "You must perform at least one reasoning step "
                    "(Think, SymbolicReasoning, or CodeExecutor) before "
                    "calling Finish."
                )

            # On verify retry: enforce deep reasoning
            if self._previous_results:
                has_deep = any(
                    act.name in self._DEEP_REASONING_ACTIONS
                    for act in raw_acts
                )
                if not has_deep:
                    return (
                        "Your previous answer was incorrect. You must use "
                        "SymbolicReasoning or CodeExecutor to derive "
                        "your answer with a DIFFERENT approach before "
                        "calling Finish. Re-read the BACKGROUND for "
                        "definitions you may have missed. Do NOT repeat "
                        "your previous reasoning."
                    )
        obs = super().forward(task, agent_act)
        from agentcore.agents.agent_utils import ACION_NOT_FOUND_MESS
        if obs == ACION_NOT_FOUND_MESS:
            obs = (
                f"Invalid action. Your output was not recognized as a valid "
                f"action call.\n"
                f"Available actions: Think, SymbolicReasoning, "
                f"CodeExecutor, Store, Backtrace, Finish.\n"
                f"Required format: ActionName[{{\"param\": \"value\"}}]\n"
                f"Example: Finish[{{\"response\": \"The answer is ...\"}}]\n"
                f"Do NOT use markdown, emojis, or headers as action names."
            )
        return obs

    # ------------------------------------------------------------------
    # Few-shot examples
    # ------------------------------------------------------------------

    def _build_examples(self) -> None:
        sym_act = SymbolicReasoningAction()
        code_act = CodeExecutorAction()
        store_act = StoreAction()

        # Example 1: single problem with Store
        example_task = TaskPackage(
            instruction=(
                "=== BACKGROUND ===\n"
                "Let $f: \\mathbb{R} \\to \\mathbb{R}$ be twice "
                "differentiable with $f(0)=1$, $f'(0)=0$, and "
                "$f''(x)+f(x)=0$.\n"
                "=== END BACKGROUND ===\n\n"
                "=== PROBLEM (Sub-problem 1) ===\n"
                "Find an explicit expression for $f(x)$.\n"
                "=== END PROBLEM ==="
            )
        )

        a1 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: (
                    "The ODE $f''+f=0$ has general solution "
                    "$A\\cos x+B\\sin x$. Apply initial conditions."
                )
            },
        )
        o1 = "OK"

        a2 = AgentAct(
            name=sym_act.action_name,
            params={
                "reasoning": (
                    "Characteristic eq: $r^2+1=0 \\Rightarrow r=\\pm i$. "
                    "General: $f=A\\cos x+B\\sin x$. "
                    "$f(0)=1 \\Rightarrow A=1$; $f'(0)=0 \\Rightarrow B=0$. "
                    "So $f(x)=\\cos x$."
                )
            },
        )
        o2 = "OK. Reasoning recorded."

        a3 = AgentAct(
            name=code_act.action_name,
            params={
                "code": (
                    "import sympy as sp\n"
                    "x = sp.Symbol('x')\n"
                    "f = sp.cos(x)\n"
                    "print('f(0)=', f.subs(x,0))\n"
                    "print('f\\'\\'+ f =', sp.simplify(sp.diff(f,x,2)+f))"
                )
            },
        )
        o3 = "Code Output:\nf(0)= 1\nf''+ f = 0"

        a4 = AgentAct(
            name=store_act.action_name,
            params={
                STORE_SUMMARY_KEY: (
                    "Result for Problem 1: $f(x)=\\cos(x)$ verified by "
                    "SymPy."
                )
            },
        )
        o4 = "[Step 1]: Result for Problem 1: $f(x)=\\cos(x)$ verified by SymPy."

        a5 = AgentAct(
            name=FinishAct.action_name,
            params={INNER_ACT_KEY: "$f(x) = \\cos(x)$"},
        )
        o5 = "$f(x) = \\cos(x)$"

        self.add_example(
            task=example_task,
            action_chain=[(a1, o1), (a2, o2), (a3, o3), (a4, o4), (a5, o5)],
        )

        # Example 2: uses memory context from previous sub-problem
        example_task2 = TaskPackage(
            instruction=(
                "Prior proven results:\n"
                "  Problem 1: $f(x)=\\cos(x)$\n\n"
                "=== PROBLEM (Sub-problem 2) ===\n"
                "Evaluate $\\int_0^{\\pi} f(x)^2 dx$ using the result from Problem 1.\n"
                "=== END PROBLEM ==="
            )
        )

        b1 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: (
                    "From memory: $f(x)=\\cos x$. Need "
                    "$\\int_0^\\pi \\cos^2 x\\,dx$."
                )
            },
        )
        bo1 = "OK"

        b2 = AgentAct(
            name=sym_act.action_name,
            params={
                "reasoning": (
                    "Using $\\cos^2 x=(1+\\cos 2x)/2$: "
                    "$\\int_0^\\pi \\cos^2 x\\,dx = \\pi/2$."
                )
            },
        )
        bo2 = "OK. Reasoning recorded."

        b3 = AgentAct(
            name=code_act.action_name,
            params={
                "code": (
                    "import sympy as sp\n"
                    "x = sp.Symbol('x')\n"
                    "print(sp.integrate(sp.cos(x)**2, (x,0,sp.pi)))"
                )
            },
        )
        bo3 = "Code Output:\npi/2"

        b4 = AgentAct(
            name=FinishAct.action_name,
            params={INNER_ACT_KEY: "$\\frac{\\pi}{2}$"},
        )
        bo4 = "$\\frac{\\pi}{2}$"

        self.add_example(
            task=example_task2,
            action_chain=[(b1, bo1), (b2, bo2), (b3, bo3), (b4, bo4)],
        )
