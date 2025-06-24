"""
Tree of Thought - ThoughtTree Implementation

This module implements the core tree structure for organizing and managing thoughts
during Tree of Thought reasoning. Includes async operations, memory management,
and statistics tracking.

Based on research findings from Princeton ToT, MCTS optimization, and production best practices.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict, OrderedDict
from datetime import datetime
import weakref

from .core_models import (
    ThoughtNode, ThoughtState, ToTConfig, TreeStatistics,
    ReasoningPath, TreeTraversalStrategy
)

logger = logging.getLogger(__name__)


class ThoughtTree:
    """
    Hierarchical tree structure for organizing thoughts during reasoning.
    
    Features:
    - Async operations with thread safety
    - LRU cache for memory efficiency
    - Parent-child relationship management
    - Real-time statistics tracking
    - Memory optimization with weak references
    """
    
    def __init__(self, config: ToTConfig):
        self.config = config
        self.nodes: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None
        self.statistics = TreeStatistics()
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Memory management
        self._node_cache: OrderedDict[str, ThoughtNode] = OrderedDict()
        self._parent_to_children: Dict[str, Set[str]] = defaultdict(set)
        self._depth_index: Dict[int, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self._access_count: Dict[str, int] = defaultdict(int)
        self._last_access: Dict[str, datetime] = {}
        
        logger.info(f"Initialized ThoughtTree with config: {config}")
    
    async def create_root(self, content: str, context: Dict[str, Any] = None) -> str:
        """
        Create the root node of the thought tree.
        
        Args:
            content: The initial problem or question
            context: Additional context for the root node
            
        Returns:
            str: Root node ID
        """
        async with self._lock:
            if self.root_id is not None:
                raise ValueError("Root node already exists")
            
            root_node = ThoughtNode(
                id="",  # Will be auto-generated
                content=content,
                depth=0,
                context=context or {},
                state=ThoughtState.ACTIVE
            )
            
            self.root_id = root_node.id
            await self._add_node(root_node)
            
            logger.info(f"Created root node: {self.root_id}")
            return self.root_id
    
    async def add_child(self, parent_id: str, content: str, 
                       context: Dict[str, Any] = None) -> str:
        """
        Add a child node to the specified parent.
        
        Args:
            parent_id: ID of the parent node
            content: Content of the new child node
            context: Additional context for the child node
            
        Returns:
            str: Child node ID
        """
        async with self._lock:
            parent = await self.get_node(parent_id)
            if parent is None:
                raise ValueError(f"Parent node {parent_id} not found")
            
            if len(parent.children) >= self.config.max_children:
                raise ValueError(f"Parent node {parent_id} has reached max children limit")
            
            child_node = ThoughtNode(
                id="",  # Will be auto-generated
                content=content,
                depth=parent.depth + 1,
                parent_id=parent_id,
                context=context or {},
                state=ThoughtState.PENDING
            )
            
            # Update parent-child relationships
            parent.children.append(child_node.id)
            self._parent_to_children[parent_id].add(child_node.id)
            
            await self._add_node(child_node)
            
            logger.debug(f"Added child {child_node.id} to parent {parent_id}")
            return child_node.id
    
    async def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        """
        Retrieve a node by ID with caching.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            ThoughtNode or None if not found
        """
        # Update access statistics
        self._access_count[node_id] += 1
        self._last_access[node_id] = datetime.utcnow()
        
        # Check cache first
        if node_id in self._node_cache:
            # Move to end (most recently used)
            self._node_cache.move_to_end(node_id)
            self.statistics.cache_hits += 1
            return self._node_cache[node_id]
        
        # Check main storage
        if node_id in self.nodes:
            node = self.nodes[node_id]
            await self._cache_node(node)
            self.statistics.cache_misses += 1
            return node
        
        return None
    
    async def update_node(self, node_id: str, **updates) -> bool:
        """
        Update a node with new values.
        
        Args:
            node_id: ID of the node to update
            **updates: Fields to update
            
        Returns:
            bool: True if update successful
        """
        async with self._lock:
            node = await self.get_node(node_id)
            if node is None:
                return False
            
            # Update fields
            for field, value in updates.items():
                if hasattr(node, field):
                    setattr(node, field, value)
            
            # Update cache
            await self._cache_node(node)
            
            logger.debug(f"Updated node {node_id} with {updates}")
            return True
    
    async def prune_node(self, node_id: str) -> bool:
        """
        Prune a node and all its descendants.
        
        Args:
            node_id: ID of the node to prune
            
        Returns:
            bool: True if pruning successful
        """
        async with self._lock:
            node = await self.get_node(node_id)
            if node is None:
                return False
            
            # Get all descendants
            descendants = await self._get_all_descendants(node_id)
            
            # Mark all nodes as pruned
            for desc_id in [node_id] + descendants:
                await self.update_node(desc_id, state=ThoughtState.PRUNED)
            
            self.statistics.pruned_nodes += len(descendants) + 1
            
            logger.info(f"Pruned node {node_id} and {len(descendants)} descendants")
            return True
    
    async def get_children(self, node_id: str) -> List[ThoughtNode]:
        """
        Get all children of a node.
        
        Args:
            node_id: ID of the parent node
            
        Returns:
            List of child nodes
        """
        node = await self.get_node(node_id)
        if node is None:
            return []
        
        children = []
        for child_id in node.children:
            child = await self.get_node(child_id)
            if child and child.state != ThoughtState.PRUNED:
                children.append(child)
        
        return children
    
    async def get_path_to_root(self, node_id: str) -> List[ThoughtNode]:
        """
        Get the path from a node to the root.
        
        Args:
            node_id: ID of the starting node
            
        Returns:
            List of nodes from node to root
        """
        path = []
        current_id = node_id
        
        while current_id is not None:
            node = await self.get_node(current_id)
            if node is None:
                break
            path.append(node)
            current_id = node.parent_id
        
        return path
    
    async def get_leaves(self) -> List[ThoughtNode]:
        """
        Get all leaf nodes in the tree.
        
        Returns:
            List of leaf nodes
        """
        leaves = []
        for node in self.nodes.values():
            if node.is_leaf and node.state != ThoughtState.PRUNED:
                leaves.append(node)
        
        return leaves
    
    async def get_nodes_at_depth(self, depth: int) -> List[ThoughtNode]:
        """
        Get all nodes at a specific depth.
        
        Args:
            depth: Depth level to retrieve
            
        Returns:
            List of nodes at the specified depth
        """
        if depth not in self._depth_index:
            return []
        
        nodes = []
        for node_id in self._depth_index[depth]:
            node = await self.get_node(node_id)
            if node and node.state != ThoughtState.PRUNED:
                nodes.append(node)
        
        return nodes
    
    async def create_reasoning_path(self, leaf_id: str) -> ReasoningPath:
        """
        Create a reasoning path from root to leaf.
        
        Args:
            leaf_id: ID of the leaf node
            
        Returns:
            ReasoningPath object
        """
        path_nodes = await self.get_path_to_root(leaf_id)
        path_nodes.reverse()  # Root to leaf order
        
        # Calculate path statistics
        total_score = sum(node.quality_score for node in path_nodes if node.is_evaluated)
        avg_confidence = sum(node.confidence or 0 for node in path_nodes) / len(path_nodes)
        
        reasoning_chain = [node.reasoning or node.content for node in path_nodes]
        evidence_chain = [node.evidence for node in path_nodes]
        
        return ReasoningPath(
            path_id="",  # Will be auto-generated
            nodes=[node.id for node in path_nodes],
            total_score=total_score,
            confidence=avg_confidence,
            reasoning_chain=reasoning_chain,
            evidence_chain=evidence_chain,
            is_complete=True
        )
    
    async def get_statistics(self) -> TreeStatistics:
        """
        Get current tree statistics.
        
        Returns:
            TreeStatistics object
        """
        # Update statistics
        self.statistics.total_nodes = len(self.nodes)
        self.statistics.max_depth_reached = max(
            (node.depth for node in self.nodes.values()), default=0
        )
        
        if self.statistics.total_nodes > 0:
            self.statistics.avg_depth = sum(
                node.depth for node in self.nodes.values()
            ) / self.statistics.total_nodes
        
        self.statistics.memory_usage = len(self._node_cache)
        
        return self.statistics
    
    async def _add_node(self, node: ThoughtNode) -> None:
        """Add a node to the tree with indexing."""
        self.nodes[node.id] = node
        self._depth_index[node.depth].add(node.id)
        await self._cache_node(node)
        
        # Update statistics
        self.statistics.total_nodes += 1
        if node.depth > self.statistics.max_depth_reached:
            self.statistics.max_depth_reached = node.depth
    
    async def _cache_node(self, node: ThoughtNode) -> None:
        """Add node to cache with LRU eviction."""
        # Remove if already in cache
        if node.id in self._node_cache:
            del self._node_cache[node.id]
        
        # Add to cache
        self._node_cache[node.id] = node
        
        # Evict if over limit
        while len(self._node_cache) > self.config.cache_size:
            oldest_id, _ = self._node_cache.popitem(last=False)
            logger.debug(f"Evicted node {oldest_id} from cache")
    
    async def _get_all_descendants(self, node_id: str) -> List[str]:
        """Get all descendant node IDs."""
        descendants = []
        to_visit = [node_id]
        
        while to_visit:
            current_id = to_visit.pop()
            if current_id in self._parent_to_children:
                children = list(self._parent_to_children[current_id])
                descendants.extend(children)
                to_visit.extend(children)
        
        return descendants
    
    def __len__(self) -> int:
        """Return the number of nodes in the tree."""
        return len(self.nodes)
    
    def __contains__(self, node_id: str) -> bool:
        """Check if a node exists in the tree."""
        return node_id in self.nodes