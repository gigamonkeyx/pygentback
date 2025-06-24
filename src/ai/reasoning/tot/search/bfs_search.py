"""
Breadth-First Search for Tree of Thoughts

Implements BFS traversal strategy for the ToT framework, exploring
thoughts level by level to find optimal solutions.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from collections import deque

from ..core.thought import Thought
from ..core.state import ReasoningState
from ..core.tree import ThoughtTree
from ..models import ToTConfig

logger = logging.getLogger(__name__)


class BFSSearch:
    """
    Breadth-First Search implementation for ToT
    
    Explores thoughts level by level, ensuring all possibilities
    at each depth are considered before moving deeper.
    Good for finding optimal solutions when they exist at
    moderate depths.
    """
    
    def __init__(self, config: ToTConfig):
        self.config = config
        self.search_count = 0
        self.nodes_expanded = 0
        self.max_depth_reached = 0
        
    async def search(
        self,
        tree: ThoughtTree,
        thought_generator: Callable,
        thought_evaluator: Callable,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Thought]:
        """
        Perform BFS search on the thought tree
        
        Args:
            tree: ThoughtTree to search
            thought_generator: Function to generate new thoughts
            thought_evaluator: Function to evaluate thoughts
            context: Additional search context
            
        Returns:
            List of best solution thoughts found
        """
        if context is None:
            context = {}
            
        self.search_count += 1
        logger.info(f"Starting BFS search (iteration {self.search_count})")
          # Initialize search queue with root nodes
        search_queue = deque()
        
        # Add root node to queue
        if tree.root:
            search_queue.append(tree.root)
        else:
            logger.warning("No root node found in tree")
            return []
        
        solutions = []
        iterations = 0
        
        while search_queue and iterations < self.config.max_iterations:
            iterations += 1
            current_level_size = len(search_queue)
            
            logger.info(f"BFS Level {iterations}: Processing {current_level_size} thoughts")
            
            # Process all thoughts at current level
            current_level_thoughts = []
            for _ in range(current_level_size):
                if not search_queue:
                    break
                thought = search_queue.popleft()
                current_level_thoughts.append(thought)
            
            # Generate and evaluate new thoughts for current level
            await self._process_level(
                current_level_thoughts, tree, thought_generator, 
                thought_evaluator, search_queue, solutions, context
            )
            
            # Check for early termination conditions
            if self._should_terminate(solutions, iterations):
                logger.info(f"Early termination at iteration {iterations}")
                break
        
        logger.info(f"BFS search completed: {len(solutions)} solutions found, "
                   f"{self.nodes_expanded} nodes expanded, "
                   f"max depth: {self.max_depth_reached}")
        
        return self._select_best_solutions(solutions)
    
    async def _process_level(
        self,
        level_thoughts: List[Thought],
        tree: ThoughtTree,
        thought_generator: Callable,
        thought_evaluator: Callable,
        search_queue: deque,
        solutions: List[Thought],
        context: Dict[str, Any]
    ):
        """Process all thoughts at the current level"""
        
        for thought in level_thoughts:
            self.nodes_expanded += 1
            self.max_depth_reached = max(self.max_depth_reached, thought.depth)
            
            # Check if this thought is a solution
            if self._is_solution(thought, context):
                solutions.append(thought)
                logger.info(f"Solution found at depth {thought.depth}")
                continue
            
            # Check depth limit
            if thought.depth >= self.config.max_depth:
                logger.debug(f"Depth limit reached for thought {thought.id}")
                continue
            
            # Generate child thoughts
            try:
                # Create reasoning state for generation
                reasoning_state = ReasoningState(
                    problem=context.get('problem', ''),
                    session_id=context.get('session_id', 'bfs_search')
                )
                  # Generate new thoughts
                new_thoughts = await thought_generator(
                    thought, context
                )
                
                if not new_thoughts:
                    continue
                
                # Evaluate new thoughts
                evaluated_thoughts = await thought_evaluator(
                    new_thoughts, reasoning_state, context
                )
                  # Add evaluated thoughts to tree and queue
                for eval_thought, score in evaluated_thoughts:
                    # Update parent-child relationships
                    tree.add_thought(eval_thought)
                    
                    # Apply selection criteria
                    if self._should_explore(eval_thought, score, context):
                        search_queue.append(eval_thought)
                        logger.debug(f"Added thought to queue: score={score:.3f}, depth={eval_thought.depth}")
                
            except Exception as e:
                logger.error(f"Error processing thought {thought.id}: {e}")
    
    def _is_solution(self, thought: Thought, context: Dict[str, Any]) -> bool:
        """Check if a thought represents a complete solution"""
        
        # Check solution indicators in content
        content_lower = thought.content.lower()
        solution_indicators = [
            'solution:', 'answer:', 'final result:', 'conclusion:',
            'final implementation', 'complete solution', 'final answer'
        ]
        
        has_solution_indicator = any(
            indicator in content_lower for indicator in solution_indicators
        )
          # Check for high confidence or value score
        high_score = (
            thought.confidence >= 0.7  # Use confidence directly
        ) or (
            thought.get_evaluation_score('value_score') is not None and
            thought.get_evaluation_score('value_score') >= 0.7
        )
        
        # Check minimum depth (avoid trivial solutions)
        sufficient_depth = thought.depth >= 2
        
        # For coding tasks, check for implementation completeness
        if context.get('task_type') == 'coding':
            has_implementation = any(
                keyword in content_lower for keyword in 
                ['def ', 'class ', 'function', 'implementation', 'code:']
            )
            return has_solution_indicator and high_score and sufficient_depth and has_implementation
        
        return has_solution_indicator and high_score and sufficient_depth
    
    def _should_explore(self, thought: Thought, score: float, context: Dict[str, Any]) -> bool:
        """Determine if a thought should be added to the search queue"""
        
        # Score threshold
        if score < self.config.value_threshold * 0.7:  # Slightly lower threshold for exploration
            return False
        
        # Depth limit
        if thought.depth >= self.config.max_depth:
            return False
        
        # For BFS, we generally explore all promising thoughts
        # but may apply additional filtering for efficiency
        
        return True
    
    def _should_terminate(self, solutions: List[Thought], iterations: int) -> bool:
        """Check if search should terminate early"""
        
        # Found enough high-quality solutions
        if len(solutions) >= self.config.n_select_sample:
            high_quality_solutions = [
                s for s in solutions 
                if (hasattr(s.metrics, 'value_score') and s.metrics.value_score >= 0.8) or
                   (hasattr(s.metrics, 'vote_score') and s.metrics.vote_score >= 0.8)
            ]
            if len(high_quality_solutions) >= 2:
                return True
        
        # Iteration limit
        if iterations >= self.config.max_iterations:
            return True
        
        return False
    
    def _select_best_solutions(self, solutions: List[Thought]) -> List[Thought]:
        """Select the best solutions from all found solutions"""
        
        if not solutions:
            return []
        
        # Sort by score (value_score or vote_score)
        def get_score(thought):
            if hasattr(thought.metrics, 'value_score') and thought.metrics.value_score is not None:
                return thought.metrics.value_score
            elif hasattr(thought.metrics, 'vote_score') and thought.metrics.vote_score is not None:
                return thought.metrics.vote_score
            else:
                return 0.0
        
        solutions.sort(key=get_score, reverse=True)
        
        # Return top N solutions
        return solutions[:self.config.n_select_sample]
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search statistics"""
        return {
            'search_method': 'bfs',
            'total_searches': self.search_count,
            'nodes_expanded': self.nodes_expanded,
            'max_depth_reached': self.max_depth_reached,
            'config': {
                'max_depth': self.config.max_depth,
                'max_iterations': self.config.max_iterations,
                'n_select_sample': self.config.n_select_sample,
                'value_threshold': self.config.value_threshold
            }
        }
    
    def reset_stats(self):
        """Reset search statistics"""
        self.search_count = 0
        self.nodes_expanded = 0
        self.max_depth_reached = 0
