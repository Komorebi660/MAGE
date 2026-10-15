"""
Exploration Tree for MAGEAgent.

The memory structure has three layers:

Bottom layer (TreeNodes)
    Every real action-observation pair is stored as a TreeNode.  Nodes are
    linked parent→children.  Same Node will merge.

Top layer (SummaryNodes)
    Created automatically whenever the agent calls the Store action.
    Each SummaryNode covers a contiguous list of TreeNode node_ids.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agentcore.commons import AgentAct


@dataclass
class TreeNode:
    """
    One action-observation pair in the exploration tree.
    Uniquely identified by ``node_id``.
    """
    # Unique identifier (node_id) assigned at creation time.
    node_id: int
    # Action-observation content
    action: AgentAct
    observation: str
    # Parent node's node_id; None for the root.
    parent_id: Optional[int]
    # Optional environment snapshot for backtrace restoration (task-specific format).
    env_snapshot: Any = None
    # List of child node_ids (supports tree branching after backtrace).
    children_ids: List[int] = field(default_factory=list)


@dataclass
class SummaryNode:
    """
    One summary entry, organized as a tree above the exploration tree.
    Uniquely identified by ``summary_id``.
    """
    # Unique identifier.
    summary_id: str
    # The summary text.
    content: str
    # All TreeNode node_ids covered by this summary.
    covers_node_ids: List[int]
    # The TreeNode node_id to backtrace to for revisiting this segment.
    revisit_step: int = 0
    # Optional key for deduplication/merging.
    key: Optional[str] = None
    # Parent SummaryNode's summary_id; None for the root.
    parent_id: Optional[str] = None
    # List of child summary_ids (supports branching after backtrace).
    children_ids: List[str] = field(default_factory=list)
    # Optional metadata dict for constraint fingerprints, slot details, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExplorationTree:
    """
    Manages the hierarchical memory structure (Tree + summaries).
    """

    def __init__(self) -> None:
        # Bottom layer: node_id → TreeNode (includes merge aliases)
        # multiple node_ids may point to the same TreeNode if they were merged
        # node.node_id is the first assigned node_id and is the canonical identifier for the node.
        self.nodes: Dict[int, TreeNode] = {}

        # Upper layer: summary_id → SummaryNode (includes merge aliases)
        self.summaries: Dict[str, SummaryNode] = {}

        # Auto-create a sentinel root node (node_id=0) so agents can always
        # backtrace to the very beginning of the task.
        self.root_id = 0
        self.current_node_id = self.root_id
        self.nodes[self.root_id] = TreeNode(
            node_id=self.root_id,
            action=None,
            observation="[Task start]",
            parent_id=None,
            env_snapshot=None,
            children_ids=[],
        )

        # Auto-create a sentinel root summary node so the summary tree
        # always has a valid root (mirrors the TreeNode sentinel).
        self.root_summary_id = "root"
        self.current_summary_id = self.root_summary_id
        self.summaries[self.root_summary_id] = SummaryNode(
            summary_id=self.root_summary_id,
            content="",
            covers_node_ids=[],
            parent_id=None,
            children_ids=[],
        )


    # ------------------------------------------------------------------
    # Bottom-layer operations
    # ------------------------------------------------------------------

    @property
    def root_node(self) -> TreeNode:
        return self.nodes[self.root_id]

    @property
    def current_node(self) -> TreeNode:
        return self.nodes[self.current_node_id]
    
    def set_current(self, node_id: int) -> None:
        """Move the current pointer to an existing node."""
        if node_id not in self.nodes:
            raise KeyError(f"Node {node_id} not found in ExplorationTree")
        self.current_node_id = node_id
    
    def get_parent(self, node_id: int) -> Optional[TreeNode]:
        """Return the parent TreeNode of the node at node_id, or None."""
        node = self.nodes.get(node_id)
        if node is None or node.parent_id is None:
            return None
        return self.nodes.get(node.parent_id)

    def get_children(self, node_id: int) -> List[TreeNode]:
        """Return child TreeNodes of the node at node_id."""
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [self.nodes[c] for c in node.children_ids if c in self.nodes]

    def add_node(
        self,
        action: AgentAct,
        observation: str,
        node_id: int,
        env_snapshot: Any = None,
        env_compare: Callable[[Any, Any], bool] = None,
    ) -> int:
        """Add a new TreeNode as a child of the current node, merging if possible.

        Before creating a new node, checks whether any existing child of the
        current node already records the same ``(action, observation)`` pair.
        If a match is found the existing node is **reused**: the new
        ``node_id`` is stored as an alias in ``nodes``.

        Parameters
        ----------
        action : AgentAct
        observation : str
        node_id : int, use as the node identifier.
        env_snapshot : Any, optional, Environment state snapshot.
        env_compare : Callable, optional, (snap_a, snap_b) -> bool.
            If provided, env_snapshots must also match for merging.

        Returns
        -------
        int, The canonical ``node_id`` of the node.
        """
        parent_id = self.current_node_id

        # Merge with an existing child
        for child_id in self.nodes[parent_id].children_ids:
            child = self.nodes[child_id]
            if (child.action is not None and action is not None
                and child.action.name == action.name
                and child.observation == observation
                and (env_compare is None or env_compare(child.env_snapshot, env_snapshot))):
                # Same action type, observation, and env: reuse the existing node.
                self.nodes[node_id] = child
                if env_snapshot is not None:
                    child.env_snapshot = env_snapshot
                self.current_node_id = child_id
                return child_id

        # No matching child — create a new node
        node = TreeNode(
            node_id=node_id,
            action=action,
            observation=observation,
            parent_id=parent_id,
            env_snapshot=env_snapshot,
            children_ids=[],
        )
        self.nodes[node_id] = node
        self.nodes[parent_id].children_ids.append(node_id)

        self.current_node_id = node_id
        return node_id

    
    
    # ------------------------------------------------------------------
    # Top-layer operations
    # ------------------------------------------------------------------

    @property
    def current_summary_node(self) -> SummaryNode:
        return self.summaries[self.current_summary_id]

    def set_current_summary(self, summary_id: str) -> None:
        """Move the current summary pointer to an existing SummaryNode."""
        if summary_id not in self.summaries:
            raise KeyError(f"SummaryNode {summary_id!r} not found")
        self.current_summary_id = summary_id
    
    def get_summary_parent(self, summary_id: str) -> Optional[TreeNode]:
        """Return the parent of the first covered node of a SummaryNode."""
        sn = self.summaries.get(summary_id)
        if sn is None or not sn.covers_node_ids:
            return None
        return self.get_parent(sn.covers_node_ids[0])

    def get_summary_children(self, summary_id: str) -> List[SummaryNode]:
        """Return child SummaryNodes of the given summary."""
        sn = self.summaries.get(summary_id)
        if sn is None:
            return []
        return [self.summaries[c] for c in sn.children_ids if c in self.summaries]

    def add_summary(
        self,
        content: str,
        covers_node_ids: List[int],
        key: Optional[str] = None,
    ) -> str:
        """Add a new SummaryNode as a child of the current SummaryNode, merging if possible.

        If an existing child of the current summary has the same ``key`` (or same content if key is not provided),
        it is reused (merge).  Otherwise a new node is created.

        Parameters
        ----------
        content : str
            The compressed summary text.
        covers_node_ids : List[int]
            Ordered list of TreeNode node_ids captured by this summary.
        key : str, optional
            A unique key for identifying summaries for merging.

        Returns
        -------
        str, The canonical ``summary_id`` of the (new or reused) SummaryNode.
        """
        parent_sid = self.current_summary_id
        parent = self.summaries[parent_sid]

        # Merge into parent itself if key matches
        if key is not None and parent.key == key:
            parent.content = content
            parent.covers_node_ids.extend(covers_node_ids) # append new covered nodes to the existing list
            # current_summary_id stays unchanged
            return parent_sid

        # Merge with an existing child
        for child_sid in parent.children_ids:
            child = self.summaries[child_sid]
            is_same = (key is not None and child.key == key) or \
                      (key is None and child.key is None and child.content == content)
            if is_same:
                child.content = content
                child.covers_node_ids.extend(covers_node_ids)
                self.current_summary_id = child_sid  # set current to the merged summary node
                return child_sid
        
        # Compute revisit_step: parent of first covered TreeNode, or current_node_id as fallback.
        revisit_step = self.current_node_id
        if covers_node_ids:
            parent_tree_node = self.get_parent(covers_node_ids[0])
            if parent_tree_node is not None:
                revisit_step = parent_tree_node.node_id

        # Create a new SummaryNode
        new_id = str(uuid.uuid4())
        summary = SummaryNode(
            summary_id=new_id,
            content=content,
            covers_node_ids=list(covers_node_ids),
            revisit_step=revisit_step,
            key=key,
            parent_id=parent_sid,
            children_ids=[],
        )
        self.summaries[new_id] = summary
        self.summaries[parent_sid].children_ids.append(new_id)

        self.current_summary_id = new_id
        return new_id
