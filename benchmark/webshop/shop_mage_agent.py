"""
ShoppingMAGEAgent Architecture
---------------------------------
Each ``ShoppingMAGEAgent`` handles one subtask.
Cross-subtask state is held in a ``BundleSharedState`` instance that is
created once per bundle and passed to every agent in the sequence.
On top of the Summary Tree, a separate SubtaskTree tracks subtask completion records.
Each record corresponds to one completed execution of a subtask.
"""

import copy
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus
import requests
from bs4 import BeautifulSoup

from webshop_actions import ClickAction, SearchAction
from webshop_env import Webshop

from agentcore.actions import FinishAct, ThinkAct
from agentcore.actions.InnerActions import INNER_ACT_KEY
from agentcore.actions.MAGEActions import StoreAction, STORE_SUMMARY_KEY
from agentcore.agents.MAGEAgent import MAGEAgent
from agentcore.commons import AgentAct, TaskPackage
from agentcore.llm.LLMBackend import BaseLLM
from agentcore.logging.terminal_logger import AgentLogger
from agentcore.memory.ExplorationTree import ExplorationTree
from agentcore.prompts import DEFAULT_PROMPT



def _parse_product_goal(goal: str) -> dict:
    """Parse goal to extract compatibility notes, avoid notes, preference, and available options.

    Returns a dict with keys:
    - "compatibility_notes": str or None
    - "avoid_notes": str or None
    - "preference": str  (one of: "highest-price", "lowest-price", "highest-rating", "lowest-rating")
    - "available_options": list[str]  (each option text without the leading "- ")
    """
    # Extract Goal field (selection criterion for Product 1, or compatibility notes for Product 2+)
    goal_match = re.search(
        r'\*{0,2}Goal:\*{0,2}\s*(.+?)(?=\n\s*\*{0,2}Preference:\*{0,2}|\n\s*\*{0,2}Avoid:\*{0,2}|\n\s*\*{0,2}Available\s+Options:\*{0,2}|\Z)',
        goal, re.DOTALL | re.IGNORECASE,
    )
    goal_text = goal_match.group(1).strip() if goal_match else None

    # Extract compatibility notes (after "Compatibility notes:" up to next section)
    compat_match = re.search(
        r'[Cc]ompatibility\s+notes?:\s*(.+?)(?=\n\s*\*{0,2}Preference:\*{0,2}|\n\s*\*{0,2}Avoid:\*{0,2}|\n\s*\*{0,2}Available\s+Options:\*{0,2}|\Z)',
        goal, re.DOTALL | re.IGNORECASE,
    )
    compatibility_notes = compat_match.group(1).strip() if compat_match else None

    # Extract avoid notes
    avoid_match = re.search(
        r'\*{0,2}Avoid:\*{0,2}\s*(.+?)(?=\n\s*\*{0,2}Preference:\*{0,2}|\n\s*\*{0,2}Available\s+Options:\*{0,2}|\n\s*\*{0,2}Constraint:\*{0,2}|\Z)',
        goal, re.DOTALL | re.IGNORECASE,
    )
    avoid_notes = avoid_match.group(1).strip() if avoid_match else None

    # Extract preference
    pref_match = re.search(
        r'\*{0,2}Preference:\*{0,2}\s*(.+?)(?=\n\s*\*{0,2}Avoid:\*{0,2}|\n\s*\*{0,2}Available\s+Options:\*{0,2}|\n\s*\*{0,2}Constraint:\*{0,2}|\Z)',
        goal, re.DOTALL | re.IGNORECASE,
    )
    preference = pref_match.group(1).strip() if pref_match else None

    # Extract available options (lines starting with "- ")
    options_section = re.search(
        r'\*{0,2}Available\s+Options:\*{0,2}\s*\n(.*)',
        goal, re.DOTALL | re.IGNORECASE,
    )
    available_options = []
    if options_section:
        for line in options_section.group(1).splitlines():
            line = line.strip()
            if line.startswith("- "):
                available_options.append(line[2:].strip())

    # Normalise preference to a standard key.
    # For Product 1: Goal section contains the criterion
    # For Product 2+: Preference section has the criterion.
    # Check Goal first, then Preference as fallback.
    preference_key = "lowest-price"  # default
    for text in (goal_text, preference):
        if not text:
            continue
        p = text.lower()
        if "lowest" in p and "price" in p:
            preference_key = "lowest-price"
            break
        if "highest" in p and "price" in p:
            preference_key = "highest-price"
            break
        if "highest" in p and ("rate" in p or "rating" in p):
            preference_key = "highest-rating"
            break
        if "lowest" in p and ("rate" in p or "rating" in p):
            preference_key = "lowest-rating"
            break

    return {
        "compatibility_notes": compatibility_notes,
        "avoid_notes": avoid_notes,
        "preference": preference_key,
        "available_options": available_options,
    }



@dataclass
class SubtaskRecord:
    """One completed execution of a subtask within a bundle."""
    subtask_idx: int
    summary_text: str
    purchased_asin: str = ""
    parent_node_id: Optional[int] = None  # id of the parent TreeNode of this record
    parent_summary_id: Optional[str] = None  # id of the parent SummaryNode of this record
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    children_record_ids: List[str] = field(default_factory=list)
    visit_count: int = 1
    verify_feedback: Optional[str] = None  # feedback from judge verification, if applicable


class SubtaskTree:
    """Tree of subtask execution records.

    The root has children for subtask 0 attempts.  Each subtask 0 record
    has children for subtask 1 attempts, and so on.  Merging by purchased
    ASIN prevents duplicate records for the same product.
    """

    def __init__(self) -> None:
        self.records: Dict[str, SubtaskRecord] = {}
        self.root_id = "subtask_root"
        self.records[self.root_id] = SubtaskRecord(
            record_id=self.root_id, subtask_idx=-1, summary_text="",
        )
        self._active_path: List[str] = []

    @property
    def root(self) -> SubtaskRecord:
        return self.records[self.root_id]

    def get_children(self, record_id: str) -> List[SubtaskRecord]:
        rec = self.records.get(record_id)
        if rec is None:
            return []
        return [self.records[c] for c in rec.children_record_ids if c in self.records]

    def parent_for_subtask(self, subtask_idx: int) -> SubtaskRecord:
        """Return the active record for the previous subtask (or root for subtask 0)."""
        if subtask_idx == 0:
            return self.root
        for rid in reversed(self._active_path):
            if self.records[rid].subtask_idx == subtask_idx - 1:
                return self.records[rid]
        return self.root

    def previous_attempts(self, subtask_idx: int) -> List[str]:
        """Return summary texts of all previous attempts for this subtask."""
        # For same-ASIN records under different parents, select the one with higher visit_count
        asin_best: Dict[str, SubtaskRecord] = {}
        for rec in self.records.values():
            # only select records for this subtask
            if rec.subtask_idx != subtask_idx:
                continue
            # ignore no ASIN records
            if not rec.purchased_asin:
                continue
            # keep the one with higher visit_count as representative
            if rec.purchased_asin not in asin_best:
                asin_best[rec.purchased_asin] = rec
            else:
                existing = asin_best[rec.purchased_asin]
                if rec.visit_count > existing.visit_count:
                    asin_best[rec.purchased_asin] = rec

        results = []
        for rec in list(asin_best.values()):
            text = rec.summary_text
            if rec.visit_count > 1:
                text += f" This product has been chosen in {rec.visit_count} independent attempts."
            if rec.verify_feedback:
                text += f" But may be wrong: {rec.verify_feedback}"
            results.append(text)
        return results

    def add_record(
        self, subtask_idx: int, summary_text: str,
        purchased_asin: str = "",
        parent_node_id: Optional[int] = None,
        parent_summary_id: Optional[str] = None
    ) -> SubtaskRecord:
        """Add a subtask completion record, merging by ASIN if possible."""
        parent = self.parent_for_subtask(subtask_idx)

        if purchased_asin:
            for cid in parent.children_record_ids:
                child = self.records.get(cid)
                # skip create new record if the same product has been purchased in previous attempts of this subtask
                if child and child.purchased_asin == purchased_asin:
                    child.summary_text = summary_text
                    child.parent_node_id = parent_node_id
                    child.parent_summary_id = parent_summary_id
                    child.visit_count += 1
                    child.verify_feedback = None
                    self._set_active(subtask_idx, cid)
                    return child

        rec = SubtaskRecord(
            subtask_idx=subtask_idx,
            summary_text=summary_text,
            purchased_asin=purchased_asin,
            parent_node_id=parent_node_id,
            parent_summary_id=parent_summary_id
        )
        self.records[rec.record_id] = rec
        parent.children_record_ids.append(rec.record_id)
        self._set_active(subtask_idx, rec.record_id)
        return rec

    def _set_verify_feedback(self, subtask_idx: int, feedback: str) -> None:
        self.active_records()[subtask_idx].verify_feedback = feedback

    def _set_active(self, subtask_idx: int, record_id: str) -> None:
        self._active_path = [
            r for r in self._active_path
            if self.records[r].subtask_idx < subtask_idx
        ]
        self._active_path.append(record_id)

    def trim_active_path(self, before_subtask_idx: int) -> None:
        """Keep only active records with subtask_idx < before_subtask_idx."""
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
    subtask_answers: List[str] = field(default_factory=list)  # used for final judge
    subtask_tree: SubtaskTree = field(default_factory=SubtaskTree)
    worker_id: str = ""

    # Cross-subtask backtrace reasons: list of (owner_subtask_idx, obs_string).
    # A reason is visible to subtasks with index < owner_subtask_idx.
    # Removed when the owner subtask completes normally.
    pending_backtrace_reasons: List[tuple] = field(default_factory=list)

    # Verify all answers after all subtasks complete
    judge_count: int = 0
    max_judge_retries: int = 0

    # Shared counter for agent-initiated backtrace calls across subtasks.
    agent_backtrace_count: int = 0



class ShoppingMAGEAgent(MAGEAgent):
    def __init__(
        self,
        role: Optional[str],
        constraint: Optional[str],
        product_goal: str,
        session_id: str,
        subtask_idx: int,
        shared_state: BundleSharedState,
        webshop_env: Webshop,
        llm: BaseLLM,
        max_steps: int = 60,
        PROMPT_DEBUG_FLAG: bool = False,
        log_file: str = "agent.log",
        max_verify_retries: int = 1,
        max_check_retries: int = 1,
        max_backtrace: int = 5,
    ) -> None:
        self.webshop_env = webshop_env
        self.session_id = session_id
        self.subtask_idx = subtask_idx
        self.shared_state = shared_state
        self.product_goal = product_goal
        self._role = role
        self._constraint = constraint
        parts = session_id[len("bundle_"):].split("_") if session_id.startswith("bundle_") \
                else session_id.split("_")[1:]
        self._task_id = parts[0] if parts else "0"
        self._log_dir = os.path.dirname(log_file)

        # When a cross-subtask backtrace lands more than one subtask back,
        # this is set to the subtask_idx we ultimately need to reach. The
        # intermediate subtasks are executed automatically inside
        # _on_task_complete before returning control to that target subtask.
        self._replay_target_subtask_idx: Optional[int] = None

        task_role = "You can interact with the webshop. Your goal is to purchase a bundle of items that are technically compatible and fit the budget. Each time you will receive one instruction with multiple options and you need to buy the best product among the options. For each option, search by the option text (do not miss any keywords) and browse multiple pages to find the most matched product, e.g., containing as many keywords mentioned in that option as possible, with the same weight, size, quantity, pack, brand, etc. Each option corresponds to only one best-matching product. The valid product may appear on later pages. Do NOT miss checking any option. If given compatibility/avoid notes in the task, filter options/products by the compatibility notes (i.e., keywords of prior purchased products pair with keywords of the chosen product) and avoid notes (i.e., keywords of prior purchased products not pair with keywords of the chosen product). When applying compatibility/avoid notes, only consider exact keywords that appear in the text literally, do NOT infer the attributes of the product based on your internal knowledge. Compatibility and avoid notes should not be applied to the keywords of the chosen product itself, only apply them between prior purchased products and currently chosen product. If a product does not contain any keyword from the compatibility/avoid notes, it is compatible by default (no rule applies). Once you finish checking an option and find the corresponding product, store the information (product title, ASIN, price, rating if needed, etc) even the option is NOT compatible. Finally, compare all compatible products and navigate to the page of the product that satisfies the preference rule and buy it by clicking [Buy Now] button. You must buy exactly one product per task."
        task_constraint = constraint or DEFAULT_PROMPT["constraint"]
        task_constraint = task_constraint.replace("\n\n", "\n")

        actions = [
            ClickAction(session_idx=session_id, env=webshop_env),
            SearchAction(session_idx=session_id, env=webshop_env),
        ]

        super().__init__(
            name="shopping_mage_agent",
            role=task_role,
            llm=llm,
            actions=actions,
            constraint=task_constraint,
            logger=AgentLogger(log_file_name=log_file, PROMPT_DEBUG_FLAG=PROMPT_DEBUG_FLAG),
            max_exec_steps=max_steps,
            max_verify_retries=max_verify_retries,
            max_check_retries=max_check_retries,
            max_backtrace=max_backtrace,
        )

        # MAGEAgent.__init__ created a fresh tree; replace it with the
        # shared one so every subtask agent writes to the same tree.
        self.exploration_tree = shared_state.exploration_tree
        self._global_step = shared_state.global_step
        self._agent_backtrace_count = shared_state.agent_backtrace_count

        # Propagate cross-subtask backtrace reasons visible to this subtask.
        visible = [obs for owner, obs in shared_state.pending_backtrace_reasons
                   if owner > subtask_idx]
        if visible:
            self._backtrace_context = [(None, None, obs) for obs in visible]

        # For the very first subtask, populate the root node's env_snapshot
        # so that backtrace to step 0 can restore the initial env state.
        if subtask_idx == 0:
            root_node = self.exploration_tree.root_node
            if root_node is not None and root_node.env_snapshot is None:
                root_node.env_snapshot = {
                    "sessions": copy.deepcopy(webshop_env.sessions),
                    "session_id": session_id,
                }

        # Record this subtask's goal in shared_state for context reconstruction.
        if len(shared_state.subtask_goals) > subtask_idx:
            assert shared_state.subtask_goals[subtask_idx] == product_goal, \
                f"Subtask {subtask_idx} goal mismatch: existing '{shared_state.subtask_goals[subtask_idx]}', new '{product_goal}'"
        else:
            # len(shared_state.subtask_goals) == subtask_idx
            shared_state.subtask_goals.append(product_goal)

        # Populate _previous_results from SubtaskTree (previous attempts for this subtask).
        prev = shared_state.subtask_tree.previous_attempts(subtask_idx)
        if prev:
            self._previous_results = list(prev)

        # parsed dict of goal components for easy access
        self.goal_dict = _parse_product_goal(self.product_goal)

        # temporary store verification feedback for the current subtask
        self.verification_feedback: Optional[str] = None

        self._build_examples()


    def _build_task_instruction(self) -> str:
        """Build the task instruction string from the current active records."""
        ss     = self.shared_state
        active = ss.subtask_tree.active_records()
        goals  = ss.subtask_goals
        steps  = ss.subtask_tree.active_steps()

        prev_block = ""
        if active:
            lines = ["Here are your current purchases so far, you can backtrace to one of the steps to undo if needed:"]
            for i, rec in enumerate(active):
                redo_hint = f"[Step {steps[i]}] "
                # goal = goals[rec.subtask_idx].split("**Goal:**")[0].replace("\n", " ").strip()
                goal = f"Product {rec.subtask_idx + 1}"
                lines.append(f"{redo_hint}{goal} Purchased: {rec.summary_text}")
            lines.append("You now need to buy the next product compatible with the previous purchases. Here is the instruction of the next product:\n")
            prev_block = "\n".join(lines)

        # remove potentially misleading information
        goal = re.sub(r"^[ \t]*### Select .+\n?", "", self.product_goal, flags=re.MULTILINE)
        if self.subtask_idx == 0:
            goal = re.sub(r"^\*{0,2}Preference:\*{0,2}[^\n]*\n?", "", goal, flags=re.MULTILINE)
        return prev_block + goal.replace("\n\n", "\n")

    def build_task(self) -> TaskPackage:
        return TaskPackage(instruction=self._build_task_instruction())


    def _get_env_snapshot(self) -> Dict[str, Any]:
        return {
            "sessions": copy.deepcopy(self.webshop_env.sessions),
            "session_id": self.session_id,
        }

    def _env_compare(self, snap_a: Any, snap_b: Any) -> bool:
        """Compare webshop snapshots by session state."""
        if snap_a is None or snap_b is None:
            return snap_a is snap_b
        sa = snap_a.get("sessions", {}).get(snap_a.get("session_id", ""), {})
        sb = snap_b.get("sessions", {}).get(snap_b.get("session_id", ""), {})
        return (
            sa.get("page_type") == sb.get("page_type")
            and sa.get("asin") == sb.get("asin")
            and sa.get("query_string") == sb.get("query_string")
            and sa.get("page_num") == sb.get("page_num")
        )

    def _extract_curr_env(self, snapshot):
        """Extract a simple string description of the current environment from snapshot."""
        if snapshot is None:
            return None
        sessions = snapshot.get("sessions", {})
        sid = snapshot.get("session_id", self.session_id)
        session = sessions.get(sid)
        if not session:
            return None

        page_type = session.get("page_type", "")

        if page_type == "init":
            return "Currently on the initial page. You can use Search to search for products."

        if page_type == "search":
            query = session.get("query_string", "")
            page_num = session.get("page_num", 1)
            asins = session.get("asins", [])
            buttons = ["Back to Search", "Next >", "< Prev"]
            parts = [f"Currently on the search results page of '{query}' (page {page_num})."]
            btn_desc = ", ".join(f"[{b}]" for b in buttons)
            if asins:
                parts.append(f"You can click {btn_desc}, and product ASIN buttons.")
            else:
                parts.append(f"You can click {btn_desc}.")
            return " ".join(parts)

        if page_type == "item":
            asin = session.get("asin", "")
            buttons = ["Buy Now", "< Prev", "Back to Search",]
            parts = [f"Currently on the item page of product {asin}."]
            btn_desc = ", ".join(f"[{b}]" for b in buttons)
            parts.append(f"You can click {btn_desc}.")
            return " ".join(parts)

        if page_type == "item_sub":
            asin = session.get("asin", "")
            subpage = session.get("subpage", "")
            return (
                f"Currently on the {subpage} sub-page of product {asin}. "
                f"You can click [< Prev] to go back to the item page, or [Back to Search] to start a new search."
            )

        if page_type == "end":
            asin = session.get("asin", "")
            return f"Purchase completed for product {asin}."

        return None


    def _fetch_product_info(self, asin: str, query: str) -> dict:
        """Fetch title, price, and rating for a product from its item page.

        Returns a dict with keys: "asin", "title", "price", "price_val", "rating", "rating_val".
        price_val and rating_val are floats for comparison (0.0 if unparseable).
        """
        info = {"asin": asin, "title": "", "price": "", "price_val": 0.0, "rating": "", "rating_val": 0.0}
        if not asin or not query:
            return info
        try:
            qs = quote_plus(query)
            url = (
                f"{self.webshop_env.web_url}/item_page/{self.session_id}/"
                f"{asin}/{qs}/1/{{}}"
            )
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                return info
            soup = BeautifulSoup(resp.text, "html.parser")
            h2 = soup.find("h2")
            if h2:
                info["title"] = h2.get_text(strip=True)
            for h4 in soup.find_all("h4"):
                text = h4.get_text(strip=True)
                if text.startswith("Price:"):
                    info["price"] = text[len("Price:"):].strip()
                    m = re.search(r'[\d,]+\.?\d*', info["price"])
                    if m:
                        info["price_val"] = float(m.group().replace(",", ""))
                elif text.startswith("Rating:"):
                    info["rating"] = text[len("Rating:"):].strip()
                    m = re.search(r'[\d.]+', info["rating"])
                    if m:
                        info["rating_val"] = float(m.group())
        except Exception:
            pass
        return info


    def _extract_summary_key(self, content: str) -> Optional[str]:
        """Extract an explicitly labeled ASIN from the summary."""
        asin_match = re.search(r'ASIN(?:\s+is|[:\s])\s*([A-Z0-9]{10})', content)
        return asin_match.group(1) if asin_match else None


    def _compute_exploration_hint(self) -> str:
        """Compute hint from children of the current summary node.

        Determines the majority option via vote, deduplicates by ASIN within
        that option, fetches product info, and builds a concise hint.
        """
        current_summary_node = self.exploration_tree.current_summary_node
        children = self.exploration_tree.get_summary_children(current_summary_node.summary_id)
        if not children:
            return ""

        # collect per-option unique ASINs
        option_asins: Dict[str, set] = {}  # normalized option -> {asin, ...}
        for child in children:
            m = re.search(r'[Oo]ption\s+(\d+|[A-Za-z])\b', child.content)
            if not m:
                continue
            raw = m.group(1)
            token = str(ord(raw.upper()) - ord('A') + 1) if raw.isalpha() else raw

            asin = child.key
            if not asin:
                asin = self._extract_summary_key(child.content)
            if asin:
                option_asins.setdefault(token, set()).add(asin)

        if not option_asins:
            return ""

        # Majority option (by unique ASIN count)
        current_option = max(option_asins, key=lambda k: len(option_asins[k]))
        seen_asins = option_asins[current_option]

        # Resolve query from option text
        idx = int(current_option) - 1
        options = self.goal_dict.get("available_options", [])
        query = options[idx] if 0 <= idx < len(options) else ""
        if not query:
            return ""

        # Fetch product info and build hint
        parts = []
        for asin in seen_asins:
            info = self._fetch_product_info(asin, query)
            title = info["title"] or asin
            price = info["price"] or "N.A."
            rating = info["rating"] or "N.A."
            parts.append(f"- {title} ASIN: {asin}. Price: {price}. Rating: {rating}.")

        return (
            f"Here are candidate products for Option {current_option} in your previous attempts:\n"
            + "\n".join(parts)
            + f"\nNote that the above candidate products are not what you checked for Option {current_option} in your current attempt. If you believe one of the previous selections was correct, navigate to the corresponding product page and store the information before you move to the next option; if not, carefully review more pages to find a better matched product."
        )


    # ------------------------------------------------------------------
    # Functions after task completion
    # ------------------------------------------------------------------

    def _on_task_complete(self, task: TaskPackage, last_obs: str = "") -> None:
        """Record the purchase summary and handle intermediate subtask replay."""
        ss = self.shared_state
        ss.agent_backtrace_count = self._agent_backtrace_count

        # Replay: cross-subtask backtrace forced an early exit (no purchase to record).
        if task.completion == "replay":
            ss.global_step = self._global_step
            if self._replay_target_subtask_idx is not None:
                # Record the NEW cross-subtask reason with this subtask as owner.
                if self._backtrace_context:
                    ss.pending_backtrace_reasons.append((self.subtask_idx, self._backtrace_context[-1][2]))
                self._run_next_replay_subtask(new_log=False)
            return

        # Record the purchase into the SubtaskTree only if a purchase occurred.
        session = self.webshop_env.sessions.get(self.session_id, {})
        if session.get("page_type") == "end":
            summary_text = self._extract_purchase_result(session)
            purchased_asin = session.get("asin", "")
            ss.subtask_tree.add_record(
                subtask_idx=self.subtask_idx,
                summary_text=summary_text,
                purchased_asin=purchased_asin,
                parent_node_id=self._task_start_node_id,
                parent_summary_id=self._task_start_summary_id
            )
        else:
            # add a placeholder record for attempt with no purchase
            ss.subtask_tree.add_record(
                subtask_idx=self.subtask_idx,
                summary_text="No purchase made.",
                purchased_asin="",
                parent_node_id=self._task_start_node_id,
                parent_summary_id=self._task_start_summary_id
            )

        # Set global_step in shared_state
        ss.global_step = self._global_step

        # # Remove reasons owned by this subtask.
        # ss.pending_backtrace_reasons = [
        #     r for r in ss.pending_backtrace_reasons if r[0] != self.subtask_idx
        # ]

        # Rebuild _previous_results from the subtask tree,
        # which now includes the just-recorded failed attempt,
        # then return early — do NOT trigger the replay chain.
        if self._is_verify_retry:
            ss.subtask_tree._set_verify_feedback(self.subtask_idx,
                                                 self.verification_feedback or "violated compatibility/avoid notes.")
            self.verification_feedback = ""  # reset
            self._previous_results = list(ss.subtask_tree.previous_attempts(self.subtask_idx))
            return

        # If a cross-subtask backtrace set a replay target, replay intermediate subtasks.
        if (self._replay_target_subtask_idx is not None
            and self.subtask_idx < self._replay_target_subtask_idx):
            self._run_next_replay_subtask(new_log=self.shared_state.judge_count > 0)
        else:
            # Replay target reached or no replay active, clear all pending reasons.
            ss.pending_backtrace_reasons.clear()
            # Run judge if all subtasks finished
            if self._should_run_judge():
                self._run_judge()

    def _run_next_replay_subtask(self, new_log: bool = False) -> None:
        """Execute the next intermediate subtask during cross-subtask replay."""
        ss = self.shared_state

        # get session_id for the next subtask
        next_sub_idx = len(ss.subtask_tree.active_records())  # next subtask to run
        worker_suffix = f"_w{ss.worker_id}" if ss.worker_id else ""
        next_session_id = f"bundle_{self._task_id}_{next_sub_idx}{worker_suffix}"

        # set environment
        self.webshop_env.step(next_session_id, "reset[]")

        log_file = os.path.join(self._log_dir, f"task_{self._task_id}_subtask_{next_sub_idx}_retry{ss.judge_count}.log") if new_log else self.logger.log_file_name

        replay_agent = ShoppingMAGEAgent(
            role=self._role,
            constraint=self._constraint,
            product_goal=ss.subtask_goals[next_sub_idx],
            session_id=next_session_id,
            subtask_idx=next_sub_idx,
            shared_state=ss,
            webshop_env=self.webshop_env,
            llm=self.llm,
            PROMPT_DEBUG_FLAG=self.logger.PROMPT_DEBUG_FLAG,
            log_file=log_file,
            max_verify_retries=self.max_verify_retries,
            max_steps=self.max_exec_steps,
            max_check_retries=self.max_check_retries,
            max_backtrace=self.max_backtrace,
        )
        replay_agent._replay_target_subtask_idx = self._replay_target_subtask_idx

        replay_task = replay_agent.build_task()
        replay_agent(replay_task)

    def _extract_purchase_result(self, session: dict) -> str:
        """Extract purchased product info from the given end-page session."""
        asin = session.get("asin", "")
        query_string = session.get("query_string", "")
        info = self._fetch_product_info(asin, query_string)

        parts: List[str] = []
        if info["title"]:  parts.append(info["title"])
        if info["price"]:  parts.append(f"Price: {info['price']}")
        if asin:           parts.append(f"ASIN: {asin}")
        if info["rating"]: parts.append(f"Rating: {info['rating']}")
        return (". ".join(parts) + ".") if parts else "Product purchased (details unavailable)."


    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _check_store(self, task, content: str) -> tuple[bool, str]:
        """Check if stored content accurately reflects the referenced option.

        Extracts Option index and ASIN from content, fetches product title
        from webshop, then uses LLM to verify attribute matching.
        """
        options = self.goal_dict.get("available_options", [])
        if not options:
            return True, ""

        # Match patterns like "Option 1", "option A", "Option 2" etc.
        m = re.search(r'[Oo]ption\s+(\d+|[A-Za-z])\b', content)
        if not m:
            self.logger.get_log("Warning: Do not match any option pattern in content. Skipping store check.")
            return True, ""

        token = m.group(1)
        idx = int(token) - 1 if token.isdigit() else ord(token.upper()) - ord('A')
        if idx < 0 or idx >= len(options):
            self.logger.get_log(f"Warning: Option index {idx} out of range. Skipping store check.")
            return True, ""

        option_text = options[idx]

        # Extract ASIN and fetch actual product title
        asin = self._extract_summary_key(content)
        if asin:
            info = self._fetch_product_info(asin, option_text)
            if info["title"]:
                product_desc = info["title"]
            else:
                self.logger.get_log(f"Warning: Failed to fetch title for ASIN {asin}. Skipping store check.")
                return True, ""
        else:
            self.logger.get_log("Warning: No ASIN found in content. Skipping store check.")
            return True, ""

        prompt = (
            "Given the option text and the product description, does the product description matches all key attributes (brand, model, size, quantity, shape, color, weight, flavor, intended use, etc.) mentioned in the option?\n"
            "- Key attribute keyword in the option must appear explicitly (or as a very close synonym) in the product description.\n"
            "- Minor wording differences are allowed if they clearly refer to the same concept.\n"
            "- Use common product knowledge to recognize equivalent or closely related terms.\n"
            "- Do not reject a match solely due to missing exact wording if the meaning is clearly the same.\n\n"
            f"Option: {option_text}\n\n"
            f"Product: {product_desc}\n\n"
            # "Do not try to guess or infer any attribute that is not explicitly mentioned in the given text. "
            "If the product description satisfies all key attributes in the option, ONLY respond with YES, otherwise respond with NO followed by a short and concise reason."
        )

        self.logger.get_prompt(f"[Check]: {prompt}")
        response = self.llm.run(prompt=prompt).strip()
        self.logger.get_llm_output(f"[Check]: {response}")

        has_no = bool(re.search(r'\bNO\b', response, re.IGNORECASE))
        has_yes = bool(re.search(r'\bYES\b', response, re.IGNORECASE))
        if not has_no or has_yes:
            return True, ""
        # Extract reason after "NO"
        reason_match = re.search(r'\bNO\b[.,:\s]*(.*)', response, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).replace("\n", " ").strip() if reason_match else ""
        reason_suffix = f" Reason: {reason}" if reason else ""
        return False, f"Check failed: the product of the option {token} you found is not fully matched the option keywords.{reason_suffix} Search query should not miss any keywords. Browse more pages (5 pages in total) to find a better match. When selecting the best-matching product for an option, focus ONLY on attribute matching (brand, model, quantity, etc.) and ignore price or rating — price and rating should only be compared in the final step after all options have been evaluated. If it is indeed the best match you can find, goto its page again and store that product information as the best match for this option."


    def _verify_task(self, task: TaskPackage) -> tuple:
        """Verify the purchased product satisfies compatibility/avoid notes."""
        session = self.webshop_env.sessions.get(self.session_id, {})
        if session.get("page_type") != "end":
            self.verification_feedback = "No purchase made."
            return False, f"No purchase for Product {self.subtask_idx+1}. You must click Buy Now button to buy at least one product for this task."
        purchased_asin = session.get("asin", "")
        if not purchased_asin:
            self.verification_feedback = "Purchase made but ASIN not found."
            return False, f"Purchase ended but no product ASIN found for Product {self.subtask_idx+1}. Please try again."

        compatibility_notes = self.goal_dict["compatibility_notes"]
        avoid_notes = self.goal_dict["avoid_notes"]
        if not compatibility_notes and not avoid_notes:
            self.logger.get_log("Warning: No compatibility or avoid notes found. Skipping verification.")
            return True, ""

        # Get purchased product title
        query_string = session.get("query_string", "")
        agent_info = self._fetch_product_info(purchased_asin, query_string)
        agent_title = agent_info["title"] if agent_info["title"] else purchased_asin

        # Build previous purchases context
        active = [rec for rec in self.shared_state.subtask_tree.active_records()
                  if rec.subtask_idx < self.subtask_idx]
        prev_block = ""
        if active:
            lines = ["Here are the previous purchases:"]
            for i, rec in enumerate(active):
                lines.append(f"Product {i + 1}: {rec.summary_text}")
            prev_block = "\n".join(lines)

        notes_block = ""
        # if compatibility_notes:
        #     notes_block += f"Compatibility notes: {compatibility_notes}\n"
        if avoid_notes:
            # Split rules by period and put each on its own line for clarity
            rules = [r.strip() for r in avoid_notes.split('.') if r.strip()]
            formatted_rules = "\n".join(f"{r}." for r in rules)
            notes_block += f"Avoid notes (each line is one independent rule):\n{formatted_rules}\n"

        prompt = (
            "Given the previous purchases and avoid rules, determine whether or not the purchased product is compatible.\n\n"
            "Avoid notes format: each line is one independent rule. Each rule reads \"A avoids B1, B2, ...\" where A is one keyword and \"B1, B2, ...\" are COMMA-separated keywords. Keyword can be multi-word phrases. Always match the complete phrase as a whole, not partial words or substrings. Each rule is independent and do NOT mix avoided keywords across different rules.\n\n"
            "Checking Steps:\n"
            "1. Check if each rule's left-side keyword A appears in any previous purchases. If not, skip that rule entirely.\n"
            "2. For each applicable rule, check if any of its avoided keywords (B1, B2, ...) appears in the purchased product. Only check words that are literally present. Do not guess or infer keywords. Visually similar words are NOT matches unless they are the exact same meaning.\n\n"
            "Here are examples showing how to apply the rules:\n"
            "Example 1 (rule skipped because A not in previous purchases):\n"
            "  Previous Purchase: \"Hardwood Oak Dining Table 6ft\"\n"
            "  Avoid notes: Marble avoids Iron, Steel. Bamboo avoids Plastic, Vinyl.\n"
            "  Purchased product: \"Natural Bamboo Cutting Board with Plastic Handle\"\n"
            "  Analysis: Rule 1 \"Marble avoids Iron, Steel\" → \"Marble\" not in previous purchase → SKIP. Rule 2 \"Bamboo avoids Plastic, Vinyl\" → \"Bamboo\" not in previous purchase (it only appears in the purchased product, NOT in previous purchase) → SKIP. No rule triggered.\n"
            "  Answer: YES\n"
            "Example 2 (rule triggered, violation found):\n"
            "  Previous Purchase: \"Italian Marble Countertop Slab White\"\n"
            "  Avoid notes: Marble avoids Iron, Steel. Pine avoids Laminate, Particle Board.\n"
            "  Purchased product: \"Stainless Steel Kitchen Faucet Modern\"\n"
            "  Analysis: Rule 1 \"Marble avoids Iron, Steel\" → \"Marble\" in previous purchase (\"Italian Marble Countertop\") → rule applies → check purchased product: \"Steel\" found in purchased product → VIOLATED. \n"
            "  Answer: NO - Marble avoids Steel.\n"
            "Example 3 (partial keyword must NOT match):\n"
            "  Previous Purchase: \"Smooth Dark Chocolate Truffle Box 12pc\"\n"
            "  Avoid notes: Dark Roast avoids Decaf, Instant. Mint avoids Pepper, Wasabi.\n"
            "  Purchased product: \"Premium Decaf Coffee Beans 1lb\"\n"
            "  Analysis: Rule 1 \"Dark Roast avoids Decaf, Instant\" → \"Dark Roast\" not in previous purchase (previous has \"Dark\" alone, not \"Dark Roast\" — partial match does NOT count) → SKIP. Rule 2 \"Mint avoids Pepper, Wasabi\" → \"Mint\" not in previous purchase → SKIP. No rule triggered.\n"
            "  Answer: YES\n"
            "Example 4 (rule applies but no violation in purchased product):\n"
            "  Previous Purchase: \"Italian Marble Countertop Slab White\"\n"
            "  Avoid notes: Marble avoids Iron, Steel. Pine avoids Laminate, Particle Board.\n"
            "  Purchased product: \"Solid Walnut Bookshelf 5-Tier\"\n"
            "  Analysis: Rule 1 \"Marble avoids Iron, Steel\" → \"Marble\" in previous purchase → rule applies → check purchased product: no \"Iron\" or \"Steel\" in \"Solid Walnut Bookshelf\" → OK. Rule 2 \"Pine avoids Laminate, Particle Board\" → \"Pine\" not in previous purchase → SKIP. No violation.\n"
            "  Answer: YES\n"
            "Example 5 (keywords appear but in DIFFERENT rules — no cross-rule mixing):\n"
            "  Previous Purchase: \"Vintage Brass Floor Lamp 60in\"\n"
            "  Avoid notes: Brass avoids Chrome, Nickel. Copper avoids Zinc, Tin. Silver avoids Gold, Bronze.\n"
            "  Purchased product: \"Modern Gold Pendant Light Fixture\"\n"
            "  Analysis: Rule 1 \"Brass avoids Chrome, Nickel\" → \"Brass\" in previous purchase → rule applies → check purchased product for \"Chrome\" or \"Nickel\": neither found → OK. Rule 2 \"Copper avoids Zinc, Tin\" → \"Copper\" not in previous purchase → SKIP. Rule 3 \"Silver avoids Gold, Bronze\" → \"Silver\" not in previous purchase → SKIP. Note: \"Gold\" appears in the purchased product, but the only triggered rule is \"Brass avoids Chrome, Nickel\" which does NOT list \"Gold\". Do NOT mix keywords across different rules. No violation.\n"
            "  Answer: YES\n\n"
            "Now complete the following task. Determine if the purchased product is compatible that does not conflict with the avoid notes against previous purchases:\n"
            f"{prev_block}\n\n"
            f"{notes_block}\n"
            f"Purchased product: {agent_title}\n\n"
            "Remember: Only compare keywords that are literally present. Do not guess or infer keywords. Do not reject the purchased product by a rule that does not exist or should not apply.\n"
            "If the purchased product is compatible with previous purchases, ONLY respond with YES, otherwise respond with NO followed by a short reason citing the exact rule violated from the avoid notes."
        )

        self.logger.get_prompt(f"[Verify]: {prompt}")
        response = self.llm.run(prompt=prompt).strip()
        self.logger.get_llm_output(f"[Verify]: {response}")

        has_no = bool(re.search(r'\bNO\b', response, re.IGNORECASE))
        has_yes = bool(re.search(r'\bYES\b', response, re.IGNORECASE))
        if not has_no or has_yes:
            return True, ""
        # Extract reason after "NO"
        reason_match = re.search(r'\bNO\b[.,:\s]*(.*)', response, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).replace("\n", " ").strip() if reason_match else ""
        self.verification_feedback = reason
        reason_suffix = f" Reason: {reason}" if reason else ""
        return False, (
            f"Product {self.subtask_idx+1} you last purchased '{agent_title}' (ASIN: {purchased_asin}) "
            f"violates compatibility/avoid rules.{reason_suffix} You need to buy a compatible one."
        )


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
        """Compare active ASINs with answers.
        If mismatched, ask LLM to pick a product to redo. Returns True if replay triggered."""
        ss = self.shared_state
        active = ss.subtask_tree.active_records()

        mismatches = [
            rec.subtask_idx+1 for rec in active
            if rec.purchased_asin != ss.subtask_answers[rec.subtask_idx]
        ]
        if not mismatches:
            return False

        purchased_asins = ["" for _ in ss.subtask_answers]
        for rec in active:
            if 0 <= rec.subtask_idx < len(purchased_asins):
                purchased_asins[rec.subtask_idx] = rec.purchased_asin
        self.logger.get_log(
            f"Judge: mismatched products: {mismatches}\n"
            f"Groundtruth answers: {ss.subtask_answers}\n"
            f"Agent answers: {purchased_asins}"
        )

        # Build prompt
        lines = [
            "You are evaluating a multi-step shopping task. Each step purchased one product. Some purchases are incorrect. Identify which product to redo.",
            "",
            "For each step's product, you will be provided with the following information:",
            "- available options and compatibility/avoid notes (except for the first product) for the this product.",
            "- **BOUGHT** is the current purchase for this product.",
            "- **OTHER ATTEMPTS** is alternative tried purchases for this product if have, not the current attempt's purchase.",
            "- visit_count: number of independent attempts that chose this product.",
            "",
            "Compatibility/Avoid notes format: rules are separated by PERIODS (.) Each rule has the format \"A avoids or pairs with B1, B2, ...\" where \"A\" is a keyword and \"B1, B2, ...\" are COMMA-separated keywords. Keyword can be multi-word phrases. Always match the complete phrase as a whole, not partial words or substrings.",
            "Each rule is independent: for each rule, first check if \"A\" appears in any prior step's bought product; if not, skip that rule. If yes, check if any of \"B1, B2, ...\" appears in the current step's bought product. Do NOT mix keywords across different rules, and do NOT guess or infer keywords that are not literally present in the product descriptions.",
            ""
            "The purchased product is correct only if it matches one option (i.e., the bought product's description contains all keywords of the chosen option) and satisfies the compatibility and avoid notes between prior steps' bought products and current step's bought product. Compatibility and avoid notes should not be applied to the keywords within one step's bought product itself. When applying compatibility/avoid notes, ONLY consider exact keywords that appear in the description text, without inferring or guessing based on your internal knowledge. If a product does not contain any keyword from the compatibility/avoid notes, it is compatible by default (no rule applies).",
            "",
            "If you think all bought products passed the option match and compatibility check, then you can consider using visit_count as a hint to determine the most likely incorrectly bought product. Products with higher visit_count are more likely to be correct.",
            "",
        ]
        for rec in active:
            idx = rec.subtask_idx
            goal = re.sub(r"^[ \t]*### Select .+\n?", "", ss.subtask_goals[idx], flags=re.MULTILINE)
            goal = re.sub(r"^\*{0,2}Preference:\*{0,2}[^\n]*\n?", "", goal, flags=re.MULTILINE)
            if idx == 0:
                goal = re.sub(r"^\*{0,2}Goal:\*{0,2}[^\n]*\n?", "", goal, flags=re.MULTILINE)
            goal = goal.replace("\n\n", "\n").strip()

            lines.append(f"{goal}")
            feedback = f" But may be wrong because verification failed: {rec.verify_feedback}" if rec.verify_feedback else ""
            lines.append(f"**BOUGHT** Product description: {rec.summary_text} visit_count: {rec.visit_count}{feedback}")

            # TODO: show all attempts for this subtask instead of only sibling attempts.
            parent = ss.subtask_tree.parent_for_subtask(idx)
            for cid in parent.children_record_ids:
                child = ss.subtask_tree.records.get(cid)
                if child and child.purchased_asin and child.record_id != rec.record_id:
                    feedback = f" But may be wrong because verification failed: {child.verify_feedback}" if child.verify_feedback else ""
                    lines.append(f"**OTHER ATTEMPTS** Product description: {child.summary_text} visit_count: {child.visit_count}{feedback}")
            lines.append("")
        lines.append("Choose ONE product which you think is most likely to fix the issues if we backtrack to it and redo from there. Common issues include but not limited to: the bought product does not satisfy the compatibility/avoid rules, or the bought product is not the best match for one of the options. If you think multiple products can be redone to fix the issues, choose the earliest one.")
        # If you think all bought products appear to be correct and with the same visit_count, retry starting from the last one.
        lines.append("Respond with the ONE product index in list format, followed by a short and concise reason, for example, [1]. Reason: ...")
        prompt = "\n".join(lines)

        self.logger.get_prompt(f"[Judge]: {prompt}")
        response = self.llm.run(prompt=prompt).strip()
        self.logger.get_llm_output(f"[Judge]: {response}")

        # Extract chosen index
        m = re.search(r'\[\s*(\d+)\s*\]', response)
        if not m:
            self.logger.get_log("Warning: Do not match any product index in judge response. Skipping judge.")
            return False
        backtrack_idx = int(m.group(1)) - 1  # convert to 0-based index
        if backtrack_idx < 0 or backtrack_idx >= len(ss.subtask_goals):
            self.logger.get_log(f"Warning: Backtrack index {backtrack_idx} out of range. Skipping judge.")
            return False

        # # Extract reason
        # reason_parts = response.split("Reason:")
        # reason_suffix = ""
        # if len(reason_parts) > 1:
        #     flat_reason = reason_parts[1].replace("\n", " ").strip()
        #     reason_suffix = " Possible reason: " + flat_reason
        #     ss.subtask_tree._set_verify_feedback(backtrack_idx, flat_reason)
        # reason = f"Likely bought incorrect product.{reason_suffix} You should be careful to check the options more thoroughly and buy the product that is compatible and matches one of the options. You can buy the same product again if you think it is indeed correct."
        reason = "You should be careful to check the options more thoroughly and buy the product that is compatible with prior purchases and matches one of the given options."

        # Save token trace for this attempt and reset for next
        retry_suffix = f"_retry{ss.judge_count}" if ss.judge_count > 0 else ""
        self.llm.save_trace(os.path.join(self._log_dir, f"task_{self._task_id}_token_usage{retry_suffix}.json"))
        self.llm.reset_trace()

        # Execute backtrace
        ss.judge_count += 1
        ss.subtask_tree.trim_active_path(backtrack_idx)
        ss.pending_backtrace_reasons.append((backtrack_idx+1, reason))  # visible only to the target subtask
        self._replay_target_subtask_idx = self.subtask_idx
        self._run_next_replay_subtask(new_log=True)
        return True


    # ------------------------------------------------------------------
    # Functions for backtrace
    # ------------------------------------------------------------------

    def _restore_context(self, target_step: int, task: TaskPackage = None):
        """Restore context to a checkpoint at target_step.

        Valid checkpoints:
        1. Subtask-boundary checkpoints (previous subtasks only).
        2. Intra-subtask Store checkpoints.
        3. Raw context start.
        4. Task start node (fallback).
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
                ckpt_map[fpn.node_id] = ('subtask', i, fpn.node_id, fpn.env_snapshot, restore_sid)
        # Source 2: intra-subtask Store checkpoints.
        for i, sid in enumerate(self._stored_node_ids):
            parent_node = self.exploration_tree.get_summary_parent(sid)
            if parent_node is not None:
                restore_sid = self._stored_node_ids[i - 1] if i > 0 else self._task_start_summary_id
                ckpt_map[parent_node.node_id] = ('store', i, parent_node.node_id, parent_node.env_snapshot, restore_sid)
        # Source 3: raw context start.
        if self._raw_node_ids:
            raw_parent = self.exploration_tree.get_parent(self._raw_node_ids[0])
            if raw_parent is not None:
                restore_sid = self._stored_node_ids[-1] if self._stored_node_ids else self._task_start_summary_id
                ckpt_map[raw_parent.node_id] = ('store', len(self._stored_node_ids), raw_parent.node_id, raw_parent.env_snapshot, restore_sid)
        # Source 4: task start node (OVERRIDES subtask boundary at same node).
        if self._task_start_node_id is not None:
            start_node = self.exploration_tree.nodes.get(self._task_start_node_id)
            if start_node is not None:
                ckpt_map[self._task_start_node_id] = ('store', 0, start_node.node_id, start_node.env_snapshot, self._task_start_summary_id)

        if target_step not in ckpt_map:
            return (
                f"Backtrace failed since step {target_step} is not valid. "
                f"Valid steps are {sorted(ckpt_map.keys())}."
            )

        kind, idx, target_node_id, target_env, target_summary_id = ckpt_map[target_step]

        if kind == 'store':
            store_idx: int = idx
            self._stored_context  = self._stored_context[:store_idx]
            self._stored_node_ids = self._stored_node_ids[:store_idx]
            self._raw_context  = []
            self._raw_node_ids = []
        else:  # 'subtask'
            record_idx: int = idx
            matched_rec = ss.subtask_tree.active_records()[record_idx]
            ss.subtask_tree.trim_active_path(matched_rec.subtask_idx)
            self._replay_target_subtask_idx = max(
                self.subtask_idx,
                self._replay_target_subtask_idx if self._replay_target_subtask_idx is not None else 0,
            )
            task.completion = "replay"

        return target_node_id, target_summary_id, target_env

    def _on_backtrace_env_restore(self, env_snapshot) -> None:
        """Restore WebShop sessions from env_snapshot and sync session_id on all actions."""
        snap = env_snapshot
        if snap is None:
            return

        if "sessions" in snap:
            self.webshop_env.sessions.clear()
            self.webshop_env.sessions.update(copy.deepcopy(snap["sessions"]))

        # If the snapshot predates our session (cross-subtask backtrace), our
        # session no longer exists in the restored env.  Reset it so actions
        # can proceed immediately without a stale or missing session.
        restored_session_id = snap.get("session_id", self.session_id)
        if restored_session_id != self.session_id:
            self.webshop_env.step(self.session_id, "reset[]")

        # Always sync action objects to the current session_id.
        for act in self.actions:
            if isinstance(act, (ClickAction, SearchAction)):
                act.session_idx = self.session_id


    # ------------------------------------------------------------------
    # Few-shot example
    # ------------------------------------------------------------------

    def _build_examples(self) -> None:
        search_act  = SearchAction("fixed_999")
        click_act   = ClickAction("fixed_999")
        store_act = StoreAction()
        example_task = (
            "I want to buy a wireless computer mouse. Buy the highest-rating option from the available options:\n"
            "- A Logitech M330 Silent Plus Wireless Mouse.\n"
            "- A Microsoft Bluetooth ergonomic mouse."
        )
        exp_task_package = TaskPackage(instruction=example_task)

        a1  = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "I must evaluate all options by searching each product individually, browse pages, find the valid product, check its rating on the detail page, and then compare to find the highest-rating one."})
        o1  = "OK"
        a2  = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "I check the first option."})
        o2  = "OK"
        a3  = AgentAct(name=search_act.action_name, params={"product": "Logitech M330 Silent Plus wireless mouse"})
        o3  = "CURRENT WEB PAGE: {\n[Back to Search] \nPage 1 (Total pages: 12) \n[Next >] \n[B08KZV9Q2J] \nLogitech Wireless Mouse Bundle - M330 Silent Plus 2 Count Pack \n$27.99 \n[B07FY8C6GY] \nLogitech M220 Silent Wireless Mouse, 2.4 GHz Compact Design \n$100.0 }"
        a4  = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The first product meets most of the requirements but is a multi-pack bundle, while the user does not ask for multiple counts. I should browse to the next page to see if there is a more relevant product."})
        o4  = "OK"
        a5  = AgentAct(name=click_act.action_name, params={"button": "Next >"})
        o5  = "CURRENT WEB PAGE: {\n[Back to Search] \nPage 2 (Total pages: 12) \n[< Prev] \n[Next >] \n[B07XH3C8LQ] \nLogitech M330 Silent Plus Wireless Mouse \n$14.99 \n[B08F7PTF54] \nLogitech M590 Multi-Device Silent Wireless Mouse \n$21.50 }"
        a6  = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The first product on this page is the most relevant one. Although there are 10 pages remaining, further browsing is unnecessary because a valid match has already been identified. I need to click on it to check its rating on the detail page."})
        o6  = "OK"
        a7  = AgentAct(name=click_act.action_name, params={"button": "B07XH3C8LQ"})
        o7  = "CURRENT WEB PAGE: {\n[Back to Search] \n[< Prev] \nLogitech M330 Silent Plus Wireless Mouse \nPrice: $14.99 \nRating: 4.8 \n[Description] \n[Features] \n[Reviews] \n[Attributes] \n[Buy Now] }"
        a8  = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The product rating is 4.8. I should store this information before I move to the second option."})
        o8  = "OK"
        a9  = AgentAct(name=store_act.action_name, params={STORE_SUMMARY_KEY: "Option 1 verified. Logitech M330 Silent Plus Wireless Mouse. ASIN: B07XH3C8LQ. Price: $14.99. Rating: 4.8."})
        o9  = "Information stored"
        a10 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "Now I check the second option. Firstly I should back to the search page."})
        o10 = "OK"
        a11 = AgentAct(name=click_act.action_name, params={"button": "Back to Search"})
        o11 = "CURRENT WEB PAGE: {\nLet us begin shopping\n\n[[Search]] }"
        a12 = AgentAct(name=search_act.action_name, params={"product": "Microsoft Bluetooth ergonomic mouse"})
        o12 = "CURRENT WEB PAGE: {\n[Back to Search] \nPage 1 (Total pages: 6) \n[Next >] \n[B09KQG1L5D] \nMicrosoft Bluetooth Mobile Mouse 3600 \n$28.50 \n[B07Y41Y7SP] \nMicrosoft Bluetooth Ergonomic Mouse \n$24.99 }"
        a13 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The second product on this page meets all the requirements. I need to click on it to check its rating."})
        o13 = "OK"
        a14 = AgentAct(name=click_act.action_name, params={"button": "B07Y41Y7SP"})
        o14 = "CURRENT WEB PAGE: {\n[Back to Search] \n[< Prev] \nMicrosoft Bluetooth Ergonomic Mouse \nPrice: $24.99 \nRating: 4.6 \n[Description] \n[Features] \n[Reviews] \n[Attributes] \n[Buy Now] }"
        a15 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "The product rating is 4.6. I should store this information."})
        o15 = "OK"
        a16 = AgentAct(name=store_act.action_name, params={STORE_SUMMARY_KEY: "Option 2 verified. Microsoft Bluetooth Ergonomic Mouse. ASIN: B07Y41Y7SP. Price: $24.99. Rating: 4.6."})
        o16 = "Information stored."
        a17 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: "I have the rating of both options. The first option has rating 4.8 and the second has rating 4.6. Since the goal is to buy the highest-rating option, I should purchase the first one, which is B07XH3C8LQ. I need to navigate back to its product page."})
        o17 = "OK"
        a18 = AgentAct(name=click_act.action_name, params={"button": "Back to Search"})
        o18 = "CURRENT WEB PAGE: {\nLet us begin shopping\n\n[[Search]] }"
        a19 = AgentAct(name=search_act.action_name, params={"product": "Logitech M330 Silent Plus wireless mouse"})
        o19 = "CURRENT WEB PAGE: {\n[Back to Search] \nPage 1 (Total pages: 12) \n[Next >] \n[B08KZV9Q2J] \nLogitech Wireless Mouse Bundle - M330 Silent Plus 2 Count Pack \n$27.99 \n[B07FY8C6GY] \nLogitech M220 Silent Wireless Mouse, 2.4 GHz Compact Design \n$100.0 }"
        a20 = AgentAct(name=click_act.action_name, params={"button": "Next >"})
        o20 = "CURRENT WEB PAGE: {\n[Back to Search] \nPage 2 (Total pages: 12) \n[< Prev] \n[Next >] \n[B07XH3C8LQ] \nLogitech M330 Silent Plus Wireless Mouse \n$14.99 \n[B08F7PTF54] \nLogitech M590 Multi-Device Silent Wireless Mouse \n$21.50 }"
        a21 = AgentAct(name=click_act.action_name, params={"button": "B07XH3C8LQ"})
        o21 = "CURRENT WEB PAGE: {\n[Back to Search] \n[< Prev] \nLogitech M330 Silent Plus Wireless Mouse \nPrice: $14.99 \nRating: 4.8 \n[Description] \n[Features] \n[Reviews] \n[Attributes] \n[Buy Now] }"
        a22 = AgentAct(name=click_act.action_name, params={"button": "Buy Now"})
        o22 = "shopping is finished."
        a23 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: "Bought Logitech M330 Silent Plus Wireless Mouse B07XH3C8LQ at $14.99 with rating 4.8."})
        o23 = "Bought Logitech M330 Silent Plus Wireless Mouse B07XH3C8LQ at $14.99 with rating 4.8."

        self.add_example(task=exp_task_package, action_chain=[
            (a1,o1),(a2,o2),(a3,o3),(a4,o4),(a5,o5),
            (a6,o6),(a7,o7),(a8,o8),(a9,o9),(a10,o10),
            (a11,o11),(a12,o12),(a13,o13),(a14,o14),(a15,o15),
            (a16,o16),(a17,o17),(a18,o18),(a19,o19),(a20,o20),
            (a21,o21),(a22,o22),(a23,o23)
        ])
