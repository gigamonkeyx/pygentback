"""
Tree of Thought - Exploration Engine

This module implements the core exploration algorithms for Tree of Thought reasoning.
Includes BFS, DFS, Best-First, and Adaptive strategies with parallel exploration
and backtracking support.

Based on research findings from Princeton ToT, MCTS optimization, and adaptive search literature.
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from collections import deque
import heapq
from datetime import datetime
import time

from .core_models import (
    ThoughtNode, ThoughtState, ToTConfig, TreeTraversalStrategy,
    EvaluationMethod, GenerationMethod, TreeStatistics
)
from .thought_tree import ThoughtTree

logger = logging.getLogger(__name__)


class ExplorationEngine:
    """
    Multi-strategy search engine for thought space exploration.
    
    Features:
    - Multiple traversal strategies (BFS, DFS, Best-First, Adaptive)
    - Parallel exploration with controlled concurrency
    - Dynamic strategy switching based on performance
    - Backtracking when paths fail
    - Memory-efficient exploration with pruning
    """
    
    def __init__(self, config: ToTConfig, thought_tree: ThoughtTree):
        self.config = config
        self.tree = thought_tree
        self.current_strategy = config.search_strategy
        
        # Exploration state
        self.active_nodes: Set[str] = set()
        self.explored_nodes: Set[str] = set()
        self.frontier: deque = deque()
        self.priority_queue: List[Tuple[float, str]] = []
        
        # Parallel exploration
        self.semaphore = asyncio.Semaphore(config.max_parallel_thoughts)
        self.exploration_tasks: Set[asyncio.Task] = set()
        
        # Strategy performance tracking
        self.strategy_performance: Dict[TreeTraversalStrategy, Dict[str, float]] = {
            strategy: {"success_rate": 0.0, "avg_time": 0.0, "nodes_explored": 0}
            for strategy in TreeTraversalStrategy
        }
        
        # Adaptive strategy state
        self.strategy_switch_threshold = config.adaptive_threshold
        self.last_strategy_switch = datetime.utcnow()
        self.nodes_since_switch = 0
        
        logger.info(f"Initialized ExplorationEngine with strategy: {self.current_strategy}")
    
    async def explore(self, 
                     root_id: str,
                     thought_generator: Callable,
                     thought_evaluator: Callable,
                     max_iterations: int = 100,
                     target_solutions: int = 1) -> List[str]:
        """
        Main exploration method that orchestrates the search process.
        
        Args:
            root_id: ID of the root node to start exploration
            thought_generator: Function to generate new thoughts
            thought_evaluator: Function to evaluate thoughts
            max_iterations: Maximum number of exploration iterations
            target_solutions: Number of solutions to find before stopping
            
        Returns:
            List of solution node IDs
        """
        start_time = time.time()
        solutions = []
        iteration = 0
        
        # Initialize exploration
        await self._initialize_exploration(root_id)
        
        logger.info(f"Starting exploration from root {root_id}")
        
        try:
            while (iteration < max_iterations and 
                   len(solutions) < target_solutions and 
                   await self._has_nodes_to_explore()):
                
                iteration += 1
                
                # Select next nodes to explore based on current strategy
                nodes_to_explore = await self._select_nodes_for_exploration()
                
                if not nodes_to_explore:
                    logger.debug("No more nodes to explore")
                    break
                
                # Explore nodes in parallel
                exploration_results = await self._explore_nodes_parallel(
                    nodes_to_explore, thought_generator, thought_evaluator
                )
                
                # Process results
                new_solutions = await self._process_exploration_results(exploration_results)
                solutions.extend(new_solutions)
                
                # Update strategy performance and consider switching
                await self._update_strategy_performance(exploration_results)
                await self._consider_strategy_switch()
                
                # Prune low-value branches
                await self._prune_branches()
                
                if iteration % 10 == 0:
                    logger.debug(f"Iteration {iteration}: {len(solutions)} solutions found")
        
        except Exception as e:
            logger.error(f"Exploration failed: {str(e)}")
            raise
        
        finally:
            # Clean up exploration tasks
            await self._cleanup_exploration_tasks()
        
        exploration_time = time.time() - start_time
        logger.info(f"Exploration completed: {len(solutions)} solutions in {exploration_time:.2f}s")
        
        return solutions
    
    async def _initialize_exploration(self, root_id: str) -> None:
        """Initialize exploration state."""
        self.active_nodes.clear()
        self.explored_nodes.clear()
        self.frontier.clear()
        self.priority_queue.clear()
        
        # Add root to frontier
        self.frontier.append(root_id)
        self.active_nodes.add(root_id)
    
    async def _has_nodes_to_explore(self) -> bool:
        """Check if there are nodes available for exploration."""
        return bool(self.frontier or self.priority_queue or self.active_nodes)
    
    async def _select_nodes_for_exploration(self) -> List[str]:
        """Select nodes for exploration based on current strategy."""
        if self.current_strategy == TreeTraversalStrategy.BREADTH_FIRST:
            return await self._select_bfs_nodes()
        elif self.current_strategy == TreeTraversalStrategy.DEPTH_FIRST:
            return await self._select_dfs_nodes()
        elif self.current_strategy == TreeTraversalStrategy.BEST_FIRST:
            return await self._select_best_first_nodes()
        elif self.current_strategy == TreeTraversalStrategy.ADAPTIVE:
            return await self._select_adaptive_nodes()
        else:
            return await self._select_bfs_nodes()  # Default fallback
    
    async def _select_bfs_nodes(self) -> List[str]:
        """Select nodes using breadth-first strategy."""
        nodes = []
        max_nodes = min(self.config.max_parallel_thoughts, len(self.frontier))
        
        for _ in range(max_nodes):
            if self.frontier:
                node_id = self.frontier.popleft()
                if node_id not in self.explored_nodes:
                    nodes.append(node_id)
        
        return nodes
    
    async def _select_dfs_nodes(self) -> List[str]:
        """Select nodes using depth-first strategy."""
        nodes = []
        max_nodes = min(self.config.max_parallel_thoughts, len(self.frontier))
        
        for _ in range(max_nodes):
            if self.frontier:
                node_id = self.frontier.pop()  # LIFO for DFS
                if node_id not in self.explored_nodes:
                    nodes.append(node_id)
        
        return nodes
    
    async def _select_best_first_nodes(self) -> List[str]:
        """Select nodes using best-first strategy based on evaluation scores."""
        nodes = []
        max_nodes = min(self.config.max_parallel_thoughts, len(self.priority_queue))
        
        for _ in range(max_nodes):
            if self.priority_queue:
                # Pop highest priority (lowest score due to min-heap)
                neg_score, node_id = heapq.heappop(self.priority_queue)
                if node_id not in self.explored_nodes:
                    nodes.append(node_id)
        
        return nodes
    
    async def _select_adaptive_nodes(self) -> List[str]:
        """Select nodes using adaptive strategy that switches based on performance."""
        # Determine best performing strategy
        best_strategy = await self._get_best_performing_strategy()
        
        # Switch to best strategy if different from current
        if best_strategy != self.current_strategy:
            await self._switch_strategy(best_strategy)
        
        # Use current strategy for selection
        return await self._select_nodes_for_exploration()
    
    async def _explore_nodes_parallel(self, 
                                    node_ids: List[str],
                                    thought_generator: Callable,
                                    thought_evaluator: Callable) -> List[Dict[str, Any]]:
        """Explore multiple nodes in parallel."""
        if not node_ids:
            return []
        
        # Create exploration tasks
        tasks = []
        for node_id in node_ids:
            task = asyncio.create_task(
                self._explore_single_node(node_id, thought_generator, thought_evaluator)
            )
            tasks.append(task)
            self.exploration_tasks.add(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        exploration_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exploration task failed for node {node_ids[i]}: {result}")
            else:
                exploration_results.append(result)
        
        return exploration_results
    
    async def _explore_single_node(self, 
                                 node_id: str,
                                 thought_generator: Callable,
                                 thought_evaluator: Callable) -> Dict[str, Any]:
        """Explore a single node by generating and evaluating children."""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                # Mark node as being explored
                self.explored_nodes.add(node_id)
                self.active_nodes.discard(node_id)
                
                # Get the node
                node = await self.tree.get_node(node_id)
                if not node:
                    return {"node_id": node_id, "success": False, "error": "Node not found"}
                
                # Skip if already has children or is pruned
                if node.children or node.state == ThoughtState.PRUNED:
                    return {"node_id": node_id, "success": False, "error": "Node already explored or pruned"}
                
                # Generate child thoughts
                child_thoughts = await thought_generator(node)
                
                # Create child nodes
                child_ids = []
                for thought_content in child_thoughts:
                    child_id = await self.tree.add_child(node_id, thought_content)
                    child_ids.append(child_id)
                
                # Evaluate child nodes
                evaluations = []
                for child_id in child_ids:
                    evaluation = await thought_evaluator(child_id)
                    evaluations.append(evaluation)
                    
                    # Update node with evaluation
                    await self.tree.update_node(
                        child_id,
                        value_score=evaluation.value_score,
                        confidence=evaluation.confidence,
                        uncertainty=evaluation.uncertainty,
                        reasoning=evaluation.reasoning,
                        evidence=evaluation.evidence,
                        state=ThoughtState.EVALUATED,
                        evaluated_at=datetime.utcnow()
                    )
                
                # Add promising children to frontier
                await self._add_children_to_frontier(child_ids, evaluations)
                
                exploration_time = time.time() - start_time
                
                return {
                    "node_id": node_id,
                    "success": True,
                    "child_ids": child_ids,
                    "evaluations": evaluations,
                    "exploration_time": exploration_time
                }
                
            except Exception as e:
                logger.error(f"Failed to explore node {node_id}: {str(e)}")
                return {"node_id": node_id, "success": False, "error": str(e)}
    
    async def _add_children_to_frontier(self, child_ids: List[str], evaluations: List[Any]) -> None:
        """Add promising children to the exploration frontier."""
        for child_id, evaluation in zip(child_ids, evaluations):
            # Skip low-quality thoughts
            if evaluation.value_score < self.config.pruning_threshold:
                continue
            
            # Add to appropriate frontier based on strategy
            if self.current_strategy in [TreeTraversalStrategy.BREADTH_FIRST, TreeTraversalStrategy.DEPTH_FIRST]:
                self.frontier.append(child_id)
            else:
                # Use negative score for min-heap (want highest scores first)
                heapq.heappush(self.priority_queue, (-evaluation.value_score, child_id))
            
            self.active_nodes.add(child_id)
    
    async def _process_exploration_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """Process exploration results and identify solutions."""
        solutions = []
        
        for result in results:
            if not result.get("success", False):
                continue
            
            # Check if any children are solutions
            for child_id in result.get("child_ids", []):
                child_node = await self.tree.get_node(child_id)
                if child_node and await self._is_solution(child_node):
                    solutions.append(child_id)
                    await self.tree.update_node(child_id, state=ThoughtState.COMPLETED)
        
        return solutions
    
    async def _is_solution(self, node: ThoughtNode) -> bool:
        """Check if a node represents a valid solution."""
        # Basic solution criteria
        return (node.is_evaluated and 
                node.confidence and node.confidence >= self.config.confidence_threshold and
                node.value_score and node.value_score >= self.config.confidence_threshold)
    
    async def _update_strategy_performance(self, results: List[Dict[str, Any]]) -> None:
        """Update performance metrics for the current strategy."""
        if not results:
            return
        
        successful_results = [r for r in results if r.get("success", False)]
        success_rate = len(successful_results) / len(results)
        avg_time = sum(r.get("exploration_time", 0) for r in successful_results) / max(len(successful_results), 1)
        
        # Update strategy performance
        perf = self.strategy_performance[self.current_strategy]
        perf["success_rate"] = (perf["success_rate"] + success_rate) / 2
        perf["avg_time"] = (perf["avg_time"] + avg_time) / 2
        perf["nodes_explored"] += len(results)
        
        self.nodes_since_switch += len(results)
    
    async def _get_best_performing_strategy(self) -> TreeTraversalStrategy:
        """Determine the best performing strategy based on metrics."""
        best_strategy = self.current_strategy
        best_score = 0.0
        
        for strategy, perf in self.strategy_performance.items():
            # Combined score: success rate weighted by exploration efficiency
            if perf["avg_time"] > 0:
                efficiency = perf["success_rate"] / perf["avg_time"]
                if efficiency > best_score:
                    best_score = efficiency
                    best_strategy = strategy
        
        return best_strategy
    
    async def _consider_strategy_switch(self) -> None:
        """Consider switching strategies based on performance."""
        if self.current_strategy != TreeTraversalStrategy.ADAPTIVE:
            return
        
        # Only consider switching after exploring enough nodes
        if self.nodes_since_switch < 20:
            return
        
        best_strategy = await self._get_best_performing_strategy()
        current_perf = self.strategy_performance[self.current_strategy]
        best_perf = self.strategy_performance[best_strategy]
        
        # Switch if best strategy is significantly better
        if (best_perf["success_rate"] > current_perf["success_rate"] + self.strategy_switch_threshold):
            await self._switch_strategy(best_strategy)
    
    async def _switch_strategy(self, new_strategy: TreeTraversalStrategy) -> None:
        """Switch to a new exploration strategy."""
        if new_strategy == self.current_strategy:
            return
        
        logger.info(f"Switching strategy from {self.current_strategy} to {new_strategy}")
        
        old_strategy = self.current_strategy
        self.current_strategy = new_strategy
        self.last_strategy_switch = datetime.utcnow()
        self.nodes_since_switch = 0
        
        # Update tree statistics
        stats = await self.tree.get_statistics()
        stats.strategy_switches += 1
        
        # Reorganize frontier for new strategy
        await self._reorganize_frontier()
    
    async def _reorganize_frontier(self) -> None:
        """Reorganize frontier based on new strategy."""
        # Collect all frontier nodes
        all_nodes = list(self.frontier) + [node_id for _, node_id in self.priority_queue]
        
        # Clear frontiers
        self.frontier.clear()
        self.priority_queue.clear()
        
        # Reorganize based on new strategy
        for node_id in all_nodes:
            node = await self.tree.get_node(node_id)
            if node and node.is_evaluated:
                if self.current_strategy in [TreeTraversalStrategy.BREADTH_FIRST, TreeTraversalStrategy.DEPTH_FIRST]:
                    self.frontier.append(node_id)
                else:
                    heapq.heappush(self.priority_queue, (-node.quality_score, node_id))
    
    async def _prune_branches(self) -> None:
        """Prune low-value branches to conserve memory."""
        # Get all leaf nodes
        leaves = await self.tree.get_leaves()
        
        # Identify nodes to prune
        nodes_to_prune = []
        for leaf in leaves:
            if (leaf.is_evaluated and 
                leaf.value_score and 
                leaf.value_score < self.config.pruning_threshold):
                nodes_to_prune.append(leaf.id)
        
        # Prune identified nodes
        for node_id in nodes_to_prune:
            await self.tree.prune_node(node_id)
            # Remove from frontiers
            if node_id in self.active_nodes:
                self.active_nodes.remove(node_id)
    
    async def _cleanup_exploration_tasks(self) -> None:
        """Clean up any remaining exploration tasks."""
        if self.exploration_tasks:
            # Cancel remaining tasks
            for task in self.exploration_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*self.exploration_tasks, return_exceptions=True)
            self.exploration_tasks.clear()
    
    async def get_exploration_statistics(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        return {
            "current_strategy": self.current_strategy.value,
            "nodes_explored": len(self.explored_nodes),
            "active_nodes": len(self.active_nodes),
            "frontier_size": len(self.frontier) + len(self.priority_queue),
            "strategy_performance": {
                strategy.value: perf for strategy, perf in self.strategy_performance.items()
            }
        }