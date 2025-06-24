"""
Search Strategies for Tree of Thought Framework

Implements different search algorithms (BFS, DFS) for exploring
the thought tree and selecting promising reasoning paths.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Set
from abc import ABC, abstractmethod
from collections import deque
import heapq

from .models import ThoughtState, ThoughtTree, ToTConfig, SearchMethod

logger = logging.getLogger(__name__)


class SearchStrategy(ABC):
    """Abstract base class for search strategies"""
    
    def __init__(self, config: ToTConfig):
        self.config = config
        self.nodes_explored = 0
        self.nodes_pruned = 0
    
    @abstractmethod
    async def search(self, tree: ThoughtTree, 
                    generate_thoughts_fn, evaluate_states_fn,
                    task_context: Dict[str, Any]) -> List[ThoughtState]:
        """
        Execute the search strategy
        
        Args:
            tree: The thought tree to search
            generate_thoughts_fn: Function to generate new thoughts
            evaluate_states_fn: Function to evaluate states
            task_context: Task-specific context
            
        Returns:
            List of solution states found
        """
        pass
    
    def select_best_states(self, states: List[ThoughtState]) -> List[ThoughtState]:
        """Select the best states based on evaluation scores"""
        if not states:
            return []
        
        # Sort by value score (descending)
        sorted_states = sorted(states, key=lambda s: s.value_score, reverse=True)
        
        # Select top n_select_sample states
        selected = sorted_states[:self.config.n_select_sample]
        
        # Mark non-selected states as pruned
        for state in sorted_states[self.config.n_select_sample:]:
            state.is_pruned = True
            self.nodes_pruned += 1
        
        return selected
    
    def is_terminal_state(self, state: ThoughtState) -> bool:
        """Check if a state should be considered terminal"""
        return (state.is_solution or 
                state.depth >= self.config.max_depth or
                state.is_pruned)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            "nodes_explored": self.nodes_explored,
            "nodes_pruned": self.nodes_pruned,
            "strategy": self.__class__.__name__
        }


class BFSStrategy(SearchStrategy):
    """Breadth-First Search strategy for thought exploration"""
    
    async def search(self, tree: ThoughtTree, 
                    generate_thoughts_fn, evaluate_states_fn,
                    task_context: Dict[str, Any]) -> List[ThoughtState]:
        """Execute BFS search"""
        solutions = []
        queue = deque([tree.root_id])
        visited = set()
        
        while queue and len(solutions) < 10:  # Limit solutions found
            current_level = []
            level_size = len(queue)
            
            # Process all nodes at current level
            for _ in range(level_size):
                if not queue:
                    break
                    
                state_id = queue.popleft()
                if state_id in visited:
                    continue
                
                visited.add(state_id)
                current_state = tree.get_state(state_id)
                
                if not current_state:
                    continue
                
                self.nodes_explored += 1
                
                # Check if this is a solution
                if current_state.is_solution:
                    solutions.append(current_state)
                    continue
                
                # Check if terminal (but not solution)
                if self.is_terminal_state(current_state):
                    continue
                
                current_level.append(current_state)
            
            if not current_level:
                break
            
            # Generate and evaluate thoughts for current level
            await self._expand_level(current_level, tree, generate_thoughts_fn, 
                                   evaluate_states_fn, task_context, queue)
        
        return solutions
    
    async def _expand_level(self, states: List[ThoughtState], tree: ThoughtTree,
                          generate_thoughts_fn, evaluate_states_fn,
                          task_context: Dict[str, Any], queue: deque) -> None:
        """Expand all states at the current level"""
        all_new_states = []
        
        # Generate thoughts for all states in parallel
        generation_tasks = []
        for state in states:
            task = generate_thoughts_fn(state, task_context)
            generation_tasks.append((state, task))
        
        # Wait for all generations to complete
        for state, task in generation_tasks:
            try:
                thoughts = await task
                
                # Create new states for each thought
                for thought in thoughts:
                    new_state = ThoughtState(
                        content=thought,
                        depth=state.depth + 1,
                        parent_id=state.id
                    )
                    
                    # Add to tree
                    tree.add_state(new_state)
                    state.add_child(new_state.id)
                    all_new_states.append(new_state)
                    
            except Exception as e:
                logger.error(f"Error generating thoughts for state {state.id}: {e}")
        
        if not all_new_states:
            return
        
        # Evaluate all new states
        try:
            evaluated_states = await evaluate_states_fn(all_new_states, task_context)
            
            # Select best states for next level
            selected_states = self.select_best_states(evaluated_states)
            
            # Add selected states to queue
            for state in selected_states:
                queue.append(state.id)
                
        except Exception as e:
            logger.error(f"Error evaluating states: {e}")


class DFSStrategy(SearchStrategy):
    """Depth-First Search strategy for thought exploration"""
    
    async def search(self, tree: ThoughtTree, 
                    generate_thoughts_fn, evaluate_states_fn,
                    task_context: Dict[str, Any]) -> List[ThoughtState]:
        """Execute DFS search"""
        solutions = []
        stack = [tree.root_id]
        visited = set()
        
        while stack and len(solutions) < 10:  # Limit solutions found
            state_id = stack.pop()
            
            if state_id in visited:
                continue
            
            visited.add(state_id)
            current_state = tree.get_state(state_id)
            
            if not current_state:
                continue
            
            self.nodes_explored += 1
            
            # Check if this is a solution
            if current_state.is_solution:
                solutions.append(current_state)
                continue
            
            # Check if terminal (but not solution)
            if self.is_terminal_state(current_state):
                continue
            
            # Expand this state
            await self._expand_state(current_state, tree, generate_thoughts_fn,
                                   evaluate_states_fn, task_context, stack)
        
        return solutions
    
    async def _expand_state(self, state: ThoughtState, tree: ThoughtTree,
                          generate_thoughts_fn, evaluate_states_fn,
                          task_context: Dict[str, Any], stack: List[str]) -> None:
        """Expand a single state"""
        try:
            # Generate thoughts for this state
            thoughts = await generate_thoughts_fn(state, task_context)
            
            if not thoughts:
                return
            
            # Create new states
            new_states = []
            for thought in thoughts:
                new_state = ThoughtState(
                    content=thought,
                    depth=state.depth + 1,
                    parent_id=state.id
                )
                
                # Add to tree
                tree.add_state(new_state)
                state.add_child(new_state.id)
                new_states.append(new_state)
            
            # Evaluate new states
            evaluated_states = await evaluate_states_fn(new_states, task_context)
            
            # Select best states
            selected_states = self.select_best_states(evaluated_states)
            
            # Add to stack in reverse order (so highest scored is processed first)
            for state in reversed(selected_states):
                stack.append(state.id)
                
        except Exception as e:
            logger.error(f"Error expanding state {state.id}: {e}")


class PrioritySearchStrategy(SearchStrategy):
    """Priority-based search using a heap for best-first exploration"""
    
    async def search(self, tree: ThoughtTree, 
                    generate_thoughts_fn, evaluate_states_fn,
                    task_context: Dict[str, Any]) -> List[ThoughtState]:
        """Execute priority-based search"""
        solutions = []
        # Use negative scores for max-heap behavior
        heap = [(-tree.get_state(tree.root_id).value_score, tree.root_id)]
        visited = set()
        
        while heap and len(solutions) < 10:
            neg_score, state_id = heapq.heappop(heap)
            
            if state_id in visited:
                continue
            
            visited.add(state_id)
            current_state = tree.get_state(state_id)
            
            if not current_state:
                continue
            
            self.nodes_explored += 1
            
            # Check if this is a solution
            if current_state.is_solution:
                solutions.append(current_state)
                continue
            
            # Check if terminal (but not solution)
            if self.is_terminal_state(current_state):
                continue
            
            # Expand this state
            await self._expand_state_priority(current_state, tree, generate_thoughts_fn,
                                            evaluate_states_fn, task_context, heap)
        
        return solutions
    
    async def _expand_state_priority(self, state: ThoughtState, tree: ThoughtTree,
                                   generate_thoughts_fn, evaluate_states_fn,
                                   task_context: Dict[str, Any], heap: List) -> None:
        """Expand state and add children to priority queue"""
        try:
            # Generate and evaluate thoughts
            thoughts = await generate_thoughts_fn(state, task_context)
            
            if not thoughts:
                return
            
            # Create new states
            new_states = []
            for thought in thoughts:
                new_state = ThoughtState(
                    content=thought,
                    depth=state.depth + 1,
                    parent_id=state.id
                )
                
                tree.add_state(new_state)
                state.add_child(new_state.id)
                new_states.append(new_state)
            
            # Evaluate new states
            evaluated_states = await evaluate_states_fn(new_states, task_context)
            
            # Add to heap with priority
            for new_state in evaluated_states:
                if not new_state.is_pruned:
                    heapq.heappush(heap, (-new_state.value_score, new_state.id))
                    
        except Exception as e:
            logger.error(f"Error expanding state {state.id}: {e}")


def create_search_strategy(config: ToTConfig) -> SearchStrategy:
    """Factory function to create search strategy based on config"""
    if config.search_method == SearchMethod.BFS:
        return BFSStrategy(config)
    elif config.search_method == SearchMethod.DFS:
        return DFSStrategy(config)
    else:
        # Default to BFS
        return BFSStrategy(config)
