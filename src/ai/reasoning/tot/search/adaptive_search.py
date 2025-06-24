"""
Adaptive Search for Tree of Thoughts

Intelligent search strategy that dynamically selects between BFS and DFS
based on problem characteristics and search progress.
"""

import logging
from typing import List, Dict, Any, Optional, Callable

from ..core.thought import Thought
from ..core.tree import ThoughtTree
from ..models import ToTConfig, SearchMethod
from .bfs_search import BFSSearch
from .dfs_search import DFSSearch

logger = logging.getLogger(__name__)


class AdaptiveSearch:
    """
    Adaptive search strategy for ToT
    
    Dynamically chooses between BFS and DFS based on:
    - Problem complexity and type
    - Current search progress
    - Solution quality requirements
    - Resource constraints
    """
    
    def __init__(self, config: ToTConfig):
        self.config = config
        self.bfs_search = BFSSearch(config)
        self.dfs_search = DFSSearch(config)
        
        self.search_count = 0
        self.strategy_switches = 0
        self.current_strategy = None
        self.search_history = []
        
    async def search(
        self,
        tree: ThoughtTree,
        thought_generator: Callable,
        thought_evaluator: Callable,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Thought]:
        """
        Perform adaptive search on the thought tree
        
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
        logger.info(f"Starting adaptive search (iteration {self.search_count})")
        
        # Analyze problem characteristics
        problem_analysis = self._analyze_problem(context)
        
        # Select initial strategy
        initial_strategy = self._select_initial_strategy(problem_analysis, context)
        self.current_strategy = initial_strategy
        
        logger.info(f"Initial strategy selected: {initial_strategy}")
        
        # Perform search with potential strategy switching
        solutions = await self._adaptive_search_with_switching(
            tree, thought_generator, thought_evaluator, context, problem_analysis
        )
        
        logger.info(f"Adaptive search completed: {len(solutions)} solutions found, "
                   f"{self.strategy_switches} strategy switches")
        
        return solutions
    
    async def _adaptive_search_with_switching(
        self,
        tree: ThoughtTree,
        thought_generator: Callable,
        thought_evaluator: Callable,
        context: Dict[str, Any],
        problem_analysis: Dict[str, Any]
    ) -> List[Thought]:
        """Perform search with potential strategy switching"""
        
        solutions = []
        max_rounds = 3  # Maximum number of search rounds
        
        for round_num in range(max_rounds):
            logger.info(f"Search round {round_num + 1} using {self.current_strategy}")
            
            # Perform search with current strategy
            if self.current_strategy == SearchMethod.BFS:
                round_solutions = await self.bfs_search.search(
                    tree, thought_generator, thought_evaluator, context
                )
            else:  # DFS
                round_solutions = await self.dfs_search.search(
                    tree, thought_generator, thought_evaluator, context
                )
            
            solutions.extend(round_solutions)
            
            # Record search round results
            self.search_history.append({
                'round': round_num + 1,
                'strategy': self.current_strategy,
                'solutions_found': len(round_solutions),
                'total_solutions': len(solutions)
            })
            
            # Check if we should continue searching
            if self._should_stop_search(solutions, round_num, problem_analysis):
                logger.info(f"Stopping search after round {round_num + 1}")
                break
            
            # Decide if we should switch strategies
            new_strategy = self._should_switch_strategy(
                solutions, round_num, problem_analysis, context
            )
            
            if new_strategy and new_strategy != self.current_strategy:
                logger.info(f"Switching strategy from {self.current_strategy} to {new_strategy}")
                self.current_strategy = new_strategy
                self.strategy_switches += 1
        
        return self._select_final_solutions(solutions)
    
    def _analyze_problem(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze problem characteristics to guide strategy selection"""
        
        problem = context.get('problem', '')
        task_type = context.get('task_type', 'general')
        
        analysis = {
            'complexity_score': 0.5,  # Default medium complexity
            'depth_requirement': 0.5,  # Default medium depth
            'breadth_requirement': 0.5,  # Default medium breadth
            'creativity_requirement': 0.5,  # Default medium creativity
            'problem_length': len(problem.split()) if problem else 0,
            'task_type': task_type
        }
        
        # Analyze problem text for complexity indicators
        if problem:
            problem_lower = problem.lower()
            
            # Complexity indicators
            complex_keywords = [
                'complex', 'complicated', 'multi-step', 'algorithm',
                'optimize', 'efficient', 'advanced', 'sophisticated'
            ]
            complexity_score = sum(1 for kw in complex_keywords if kw in problem_lower)
            analysis['complexity_score'] = min(complexity_score / 4, 1.0)
            
            # Depth requirement indicators
            depth_keywords = [
                'detailed', 'thorough', 'deep', 'comprehensive',
                'step-by-step', 'implementation', 'complete'
            ]
            depth_score = sum(1 for kw in depth_keywords if kw in problem_lower)
            analysis['depth_requirement'] = min(depth_score / 3, 1.0)
            
            # Breadth requirement indicators
            breadth_keywords = [
                'alternatives', 'options', 'approaches', 'methods',
                'different', 'various', 'multiple', 'explore'
            ]
            breadth_score = sum(1 for kw in breadth_keywords if kw in problem_lower)
            analysis['breadth_requirement'] = min(breadth_score / 3, 1.0)
            
            # Creativity requirement indicators
            creativity_keywords = [
                'creative', 'innovative', 'novel', 'original',
                'brainstorm', 'generate ideas', 'think outside'
            ]
            creativity_score = sum(1 for kw in creativity_keywords if kw in problem_lower)
            analysis['creativity_requirement'] = min(creativity_score / 3, 1.0)
        
        # Task-specific adjustments
        if task_type == 'coding':
            analysis['depth_requirement'] += 0.2  # Coding often requires deep thinking
            analysis['complexity_score'] += 0.1
        elif task_type == 'creative':
            analysis['breadth_requirement'] += 0.3  # Creative tasks benefit from breadth
            analysis['creativity_requirement'] += 0.2
        elif task_type == 'mathematical':
            analysis['depth_requirement'] += 0.3  # Math often requires deep reasoning
            analysis['complexity_score'] += 0.2
        
        # Normalize scores
        for key in ['complexity_score', 'depth_requirement', 'breadth_requirement', 'creativity_requirement']:
            analysis[key] = min(analysis[key], 1.0)
        
        return analysis
    
    def _select_initial_strategy(
        self,
        problem_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> SearchMethod:
        """Select initial search strategy based on problem analysis"""
        
        # Calculate strategy scores
        bfs_score = 0.0
        dfs_score = 0.0
        
        # BFS is better for:
        bfs_score += problem_analysis['breadth_requirement'] * 0.4
        bfs_score += problem_analysis['creativity_requirement'] * 0.3
        bfs_score += (1.0 - problem_analysis['complexity_score']) * 0.2
        bfs_score += (1.0 - problem_analysis['depth_requirement']) * 0.1
        
        # DFS is better for:
        dfs_score += problem_analysis['depth_requirement'] * 0.4
        dfs_score += problem_analysis['complexity_score'] * 0.3
        dfs_score += (1.0 - problem_analysis['breadth_requirement']) * 0.2
        dfs_score += (1.0 - problem_analysis['creativity_requirement']) * 0.1
        
        # Task-specific preferences
        task_type = context.get('task_type', 'general')
        if task_type == 'coding':
            dfs_score += 0.1  # Coding often benefits from deep exploration
        elif task_type == 'creative':
            bfs_score += 0.2  # Creative tasks benefit from breadth
        
        # Select strategy
        if bfs_score > dfs_score:
            return SearchMethod.BFS
        else:
            return SearchMethod.DFS
    
    def _should_stop_search(
        self,
        solutions: List[Thought],
        round_num: int,
        problem_analysis: Dict[str, Any]
    ) -> bool:
        """Determine if search should stop"""
        
        # Stop if we have enough high-quality solutions
        if len(solutions) >= self.config.n_select_sample:
            high_quality_count = sum(
                1 for s in solutions
                if self._get_thought_score(s) >= 0.8
            )
            if high_quality_count >= 2:
                return True
        
        # Stop if we have many solutions (even if not perfect)
        if len(solutions) >= self.config.n_select_sample * 2:
            return True
        
        # Continue if no solutions found yet (unless max rounds reached)
        if len(solutions) == 0 and round_num < 2:
            return False
        
        return False
    
    def _should_switch_strategy(
        self,
        solutions: List[Thought],
        round_num: int,
        problem_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[SearchMethod]:
        """Determine if and how to switch search strategy"""
        
        # Don't switch on first round
        if round_num == 0:
            return None
        
        # Don't switch if we found good solutions
        if solutions and self._get_thought_score(solutions[0]) >= 0.8:
            return None
        
        # Switch if current strategy isn't producing results
        current_round_solutions = len(solutions) - sum(
            h['solutions_found'] for h in self.search_history[:-1]
        )
        
        if current_round_solutions == 0:
            # Switch strategy
            if self.current_strategy == SearchMethod.BFS:
                return SearchMethod.DFS
            else:
                return SearchMethod.BFS
        
        return None
    
    def _get_thought_score(self, thought: Thought) -> float:
        """Get the score of a thought"""
        if hasattr(thought.metrics, 'value_score') and thought.metrics.value_score is not None:
            return thought.metrics.value_score
        elif hasattr(thought.metrics, 'vote_score') and thought.metrics.vote_score is not None:
            return thought.metrics.vote_score
        else:
            return 0.0
    
    def _select_final_solutions(self, solutions: List[Thought]) -> List[Thought]:
        """Select final solutions from all rounds"""
        
        if not solutions:
            return []
        
        # Remove duplicates based on content similarity
        unique_solutions = []
        for solution in solutions:
            is_duplicate = False
            for existing in unique_solutions:
                if self._are_solutions_similar(solution, existing):
                    # Keep the better one
                    if self._get_thought_score(solution) > self._get_thought_score(existing):
                        unique_solutions.remove(existing)
                        unique_solutions.append(solution)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_solutions.append(solution)
        
        # Sort by score
        unique_solutions.sort(key=self._get_thought_score, reverse=True)
        
        # Return top N
        return unique_solutions[:self.config.n_select_sample]
    
    def _are_solutions_similar(self, solution1: Thought, solution2: Thought) -> bool:
        """Check if two solutions are too similar"""
        
        # Simple similarity check based on word overlap
        words1 = set(solution1.content.lower().split())
        words2 = set(solution2.content.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        
        similarity = overlap / total if total > 0 else 0
        
        # Consider similar if >70% word overlap
        return similarity > 0.7
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get comprehensive search statistics"""
        bfs_stats = self.bfs_search.get_search_stats()
        dfs_stats = self.dfs_search.get_search_stats()
        
        return {
            'search_method': 'adaptive',
            'total_searches': self.search_count,
            'strategy_switches': self.strategy_switches,
            'final_strategy': self.current_strategy,
            'search_history': self.search_history,
            'bfs_stats': bfs_stats,
            'dfs_stats': dfs_stats
        }
    
    def reset_stats(self):
        """Reset all search statistics"""
        self.search_count = 0
        self.strategy_switches = 0
        self.current_strategy = None
        self.search_history = []
        self.bfs_search.reset_stats()
        self.dfs_search.reset_stats()
