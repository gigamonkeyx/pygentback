"""
Depth-First Search implementation for Tree of Thoughts.

This module implements a depth-first search algorithm for exploring
the Tree of Thoughts, allowing for deep exploration of thought paths.
"""

from typing import Optional, Dict, Any, List, Callable
import logging
from ..core.tree import ThoughtTree
from ..core.thought import Thought
from ..core.state import ReasoningState
from ..models import ToTConfig

logger = logging.getLogger(__name__)


class DFSSearch:
    """
    Depth-First Search algorithm for Tree of Thoughts exploration.
    
    This class implements a depth-first search strategy that explores
    thought branches deeply before backtracking.
    """
    
    def __init__(self, config: ToTConfig):
        """
        Initialize the DFS search with configuration.
        
        Args:
            config: ToT configuration object
        """
        self.config = config
        self.max_depth = config.max_depth
        self.exploration_factor = getattr(config, 'exploration_factor', 0.3)
        self.visited_states = set()
        
    async def search(
        self,
        tree: ThoughtTree,
        thought_generator: Callable,
        thought_evaluator: Callable,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Thought]:
        """
        Perform depth-first search on the thought tree.
        
        Args:
            tree: The Tree of Thoughts to search
            thought_generator: Function to generate new thoughts
            thought_evaluator: Function to evaluate thoughts
            context: Additional search context
            
        Returns:
            List of best solution thoughts found
        """
        if context is None:
            context = {}
            
        logger.info(f"Starting DFS search with max_depth={self.max_depth}")
        
        if not tree.root:
            logger.warning("Tree has no root, cannot perform search")
            return []
            
        self.visited_states.clear()
        iterations = 0
        max_iter = self.config.max_iterations
        
        # Stack for DFS (thought, depth)
        stack = [(tree.root, 0)]
        solutions = []
        
        while stack and iterations < max_iter:
            current_thought, depth = stack.pop()
            iterations += 1
            
            # Skip if already visited
            state_key = self._get_state_key(current_thought)
            if state_key in self.visited_states:
                continue
                
            self.visited_states.add(state_key)
            
            logger.debug(f"Exploring thought at depth {depth}: {current_thought.content[:50]}...")
            
            # If this is a potential solution, add to solutions
            if current_thought.confidence > 0.5:  # Threshold for solution
                solutions.append(current_thought)
                logger.info(f"Found solution with confidence {current_thought.confidence}")
              # If we haven't reached max depth, generate and add children
            if depth < self.max_depth:
                try:
                    # Create reasoning state for this search
                    reasoning_state = ReasoningState(
                        problem=context.get('problem', ''),
                        session_id=context.get('session_id', 'dfs_search')
                    )
                    
                    # Generate new thoughts
                    new_thoughts = await thought_generator(current_thought, context)
                    if new_thoughts:
                        # Evaluate the new thoughts
                        evaluated_thoughts = await thought_evaluator(new_thoughts, reasoning_state, context)
                        
                        # Add children to tree and stack
                        for eval_thought, score in evaluated_thoughts:
                            tree.add_thought(eval_thought)
                            if self._should_explore_child(eval_thought, depth + 1):
                                stack.append((eval_thought, depth + 1))
                                
                except Exception as e:
                    logger.error(f"Error generating/evaluating thoughts: {e}")
                    continue
        
        logger.info(f"DFS search completed after {iterations} iterations, found {len(solutions)} solutions")
        return solutions
    
    def _get_state_key(self, thought: Thought) -> str:
        """
        Generate a unique key for a thought state.
        
        Args:
            thought: The thought to generate a key for
            
        Returns:
            A string key representing the thought state
        """
        return f"{thought.id}_{hash(thought.content)}"
    
    def _should_explore_child(self, child: Thought, depth: int) -> bool:
        """
        Determine if a child thought should be explored.
        
        Args:
            child: The child thought to consider
            depth: Current depth in the tree
            
        Returns:
            True if the child should be explored
        """
        # Skip low-confidence thoughts based on exploration factor
        if hasattr(child, 'confidence') and child.confidence < self.exploration_factor:
            return False
            
        # Skip if already visited
        state_key = self._get_state_key(child)
        if state_key in self.visited_states:
            return False
            
        return True
    
    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the last search operation.
        
        Returns:
            Dictionary containing search statistics
        """
        return {
            'algorithm': 'DFS',
            'visited_states': len(self.visited_states),
            'max_depth': self.max_depth,
            'exploration_factor': self.exploration_factor
        }
