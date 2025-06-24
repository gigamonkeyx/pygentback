"""
Tree of Thought Engine

Main orchestration engine that combines thought generation, state evaluation,
and search strategies to perform deliberate multi-path reasoning using the
new ToT core components.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from .models import ToTConfig, GenerationStrategy, EvaluationMethod, SearchMethod
from .core.thought import Thought, ThoughtType
from .core.state import ReasoningState
from .core.tree import ThoughtTree
from .thought_generator import ThoughtGenerator, LLMBackend
from .evaluators import ValueEvaluator, VoteEvaluator, CodingEvaluator
from .search import BFSSearch, DFSSearch, AdaptiveSearch

logger = logging.getLogger(__name__)


class ToTSearchResult:
    """Result of ToT search operation"""
    
    def __init__(self, solutions: List[Thought], tree: ThoughtTree, 
                 search_time: float, stats: Dict[str, Any]):
        self.solutions = solutions
        self.tree = tree
        self.search_time = search_time
        self.stats = stats
        self.success = len(solutions) > 0


class ToTEngine:
    """
    Main Tree of Thought reasoning engine
    
    Orchestrates the complete ToT process using new core components:
    1. Initialize reasoning state and thought tree
    2. Generate multiple reasoning paths using generators
    3. Evaluate thoughts using specialized evaluators
    4. Search through the thought tree using search algorithms
    5. Return best solutions with detailed results
    """
    
    def __init__(self, config: ToTConfig, llm_backend: Optional[LLMBackend] = None):
        self.config = config
        self.llm_backend = llm_backend

        # Initialize core components
        self.thought_generator = ThoughtGenerator(config, llm_backend)
        
        # Initialize evaluators
        self.value_evaluator = ValueEvaluator(config, llm_backend)
        self.vote_evaluator = VoteEvaluator(config, llm_backend)
        self.coding_evaluator = CodingEvaluator(config, llm_backend)
        
        # Initialize search strategies
        self.bfs_search = BFSSearch(config)
        self.dfs_search = DFSSearch(config)
        self.adaptive_search = AdaptiveSearch(config)

        # Statistics
        self.total_runs = 0
        self.successful_runs = 0
        self.total_search_time = 0.0
    
    async def solve(self, problem: str, task_context: Optional[Dict[str, Any]] = None) -> ToTSearchResult:
        """
        Solve a problem using Tree of Thought reasoning
        
        Args:
            problem: The problem statement or initial prompt
            task_context: Task-specific context including prompts and parameters
            
        Returns:
            ToTSearchResult containing solutions, tree, and statistics
        """
        start_time = time.time()
        self.total_runs += 1
        
        # Initialize task context
        if task_context is None:
            task_context = {}
        
        # Update config with problem
        self.config.task_description = problem
        task_context['problem'] = problem
        
        try:
            logger.info(f"Starting ToT reasoning for problem: {problem[:100]}...")
            
            # Create initial reasoning state
            reasoning_state = ReasoningState(
                problem=problem,
                session_id=task_context.get('session_id', f'tot_session_{self.total_runs}')
            )
            
            # Initialize thought tree with root thought
            tree = ThoughtTree(max_depth=self.config.max_depth)
            root_thought = await self._create_root_thought(problem, task_context)
            tree.add_thought(root_thought)
            reasoning_state.add_thought(root_thought)
            
            # Select and execute search strategy
            search_strategy = self._get_search_strategy()
            
            logger.info(f"Using search strategy: {search_strategy.__class__.__name__}")
            
            solutions = await search_strategy.search(
                tree,
                self._generate_thoughts_wrapper,
                self._evaluate_thoughts_wrapper,
                task_context
            )
            
            # Calculate search time
            search_time = time.time() - start_time
            self.total_search_time += search_time
            
            # Gather statistics
            stats = self._gather_statistics(reasoning_state, tree, search_strategy, search_time)
            
            # Create result
            result = ToTSearchResult(solutions, tree, search_time, stats)
            
            if solutions:
                self.successful_runs += 1
                logger.info(f"ToT reasoning successful: {len(solutions)} solutions found in {search_time:.2f}s")
            else:
                logger.warning(f"ToT reasoning found no solutions in {search_time:.2f}s")
            
            return result
            
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"ToT reasoning failed: {e}")
            
            # Return empty result with error info
            stats = {'error': str(e), 'search_time': search_time}
            return ToTSearchResult([], ThoughtTree(), search_time, stats)
    
    async def _create_root_thought(self, problem: str, task_context: Dict[str, Any]) -> Thought:
        """Create the initial root thought for the problem"""
        
        root_thought = Thought(
            content=f"Problem: {problem}",
            thought_type=ThoughtType.PROBLEM,
            depth=0,
            context=task_context.copy(),
            metadata={
                'is_root': True,
                'problem_statement': problem,
                'task_type': task_context.get('task_type', 'general')
            }
        )
        
        return root_thought
    
    def _get_search_strategy(self):
        """Get the appropriate search strategy based on configuration"""
        
        if self.config.search_method == SearchMethod.BFS:
            return self.bfs_search
        elif self.config.search_method == SearchMethod.DFS:
            return self.dfs_search
        else:
            # Default to adaptive search
            return self.adaptive_search
    
    async def _generate_thoughts_wrapper(
        self,
        reasoning_state: ReasoningState,
        parent_thought: Optional[Thought] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Thought]:
        """Wrapper for thought generation that integrates with search algorithms"""
        
        try:
            thoughts = await self.thought_generator.generate_thoughts(
                reasoning_state, parent_thought, context
            )
            
            # Add generated thoughts to reasoning state
            for thought in thoughts:
                reasoning_state.add_thought(thought)
            
            return thoughts
            
        except Exception as e:
            logger.error(f"Error in thought generation: {e}")
            return []
    
    async def _evaluate_thoughts_wrapper(
        self,
        thoughts: List[Thought],
        reasoning_state: ReasoningState,
        context: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """Wrapper for thought evaluation that integrates with search algorithms"""
        
        if not thoughts:
            return []
        
        try:
            # Select appropriate evaluator
            evaluator = self._get_evaluator(context)
            
            # Evaluate thoughts
            evaluated_thoughts = await evaluator.evaluate_thoughts(
                thoughts, reasoning_state, context
            )
            
            return evaluated_thoughts
            
        except Exception as e:
            logger.error(f"Error in thought evaluation: {e}")
            # Return thoughts with default scores
            return [(thought, 0.5) for thought in thoughts]
    
    def _get_evaluator(self, context: Optional[Dict[str, Any]] = None):
        """Get the appropriate evaluator based on context and configuration"""
        
        if context is None:
            context = {}
        
        # Check for task-specific evaluators
        task_type = context.get('task_type', 'general')
        if task_type == 'coding':
            return self.coding_evaluator
        
        # Use configured evaluation method
        if self.config.evaluation_method == EvaluationMethod.VOTE:
            return self.vote_evaluator
        else:
            return self.value_evaluator
    
    def _gather_statistics(
        self,
        reasoning_state: ReasoningState,
        tree: ThoughtTree,
        search_strategy,
        search_time: float
    ) -> Dict[str, Any]:
        """Gather comprehensive statistics from the reasoning session"""
        
        stats = {
            'search_time': search_time,
            'total_thoughts': len(reasoning_state.thoughts),
            'tree_depth': tree.get_max_depth(),
            'tree_breadth': tree.get_max_breadth(),
            'reasoning_state': reasoning_state.get_statistics(),
            'tree_stats': tree.get_statistics(),
            'search_stats': search_strategy.get_search_stats(),
            'generator_stats': self.thought_generator.get_generation_stats(),
            'evaluator_stats': {
                'value': self.value_evaluator.get_evaluation_stats(),
                'vote': self.vote_evaluator.get_evaluation_stats(),
                'coding': self.coding_evaluator.get_evaluation_stats()
            },
            'engine_stats': {
                'total_runs': self.total_runs,
                'successful_runs': self.successful_runs,
                'success_rate': self.successful_runs / self.total_runs if self.total_runs > 0 else 0,
                'total_search_time': self.total_search_time,
                'average_search_time': self.total_search_time / self.total_runs if self.total_runs > 0 else 0
            }
        }
        
        return stats
            
        except Exception as e:
            logger.error(f"Error in ToT search: {e}")
            
            # Create error result
            result = SearchResult(
                tree=ThoughtTree(root_id="", config=self.config),
                total_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            return result
    
    async def _initialize_tree(self, problem: str, task_context: Dict[str, Any]) -> ThoughtTree:
        """Initialize the thought tree with root state"""
        # Create root state
        root_state = ThoughtState(
            content=problem,
            depth=0,
            parent_id=None
        )
        
        # Create tree
        tree = ThoughtTree(
            root_id=root_state.id,
            config=self.config
        )
        
        # Add root state to tree
        tree.add_state(root_state)
        
        # Evaluate root state
        try:
            evaluated_states = await self.state_evaluator.evaluate_states([root_state], task_context)
            if evaluated_states:
                tree.states[root_state.id] = evaluated_states[0]
        except Exception as e:
            logger.warning(f"Could not evaluate root state: {e}")
        
        return tree
    
    async def _generate_thoughts_wrapper(self, state: ThoughtState, 
                                       task_context: Dict[str, Any]) -> List[str]:
        """Wrapper for thought generation with error handling"""
        try:
            return await self.thought_generator.generate_thoughts(state, task_context)
        except Exception as e:
            logger.error(f"Error generating thoughts for state {state.id}: {e}")
            return []
    
    async def _evaluate_states_wrapper(self, states: List[ThoughtState], 
                                     task_context: Dict[str, Any]) -> List[ThoughtState]:
        """Wrapper for state evaluation with error handling"""
        try:
            return await self.state_evaluator.evaluate_states(states, task_context)
        except Exception as e:
            logger.error(f"Error evaluating states: {e}")
            return states  # Return original states if evaluation fails
    
    def _create_search_result(self, tree: ThoughtTree, solutions: List[ThoughtState], 
                            start_time: float) -> SearchResult:
        """Create comprehensive search result"""
        # Find best solution
        best_solution = None
        if solutions:
            best_solution = max(solutions, key=lambda s: s.value_score)
        
        # Calculate statistics
        total_time = time.time() - start_time
        generator_stats = self.thought_generator.get_stats()
        evaluator_stats = self.state_evaluator.get_stats()
        search_stats = self.search_strategy.get_stats()
        
        # Create result
        result = SearchResult(
            tree=tree,
            best_solution=best_solution,
            all_solutions=solutions,
            total_time=total_time,
            total_llm_calls=generator_stats["total_generations"] + evaluator_stats["total_evaluations"],
            nodes_explored=search_stats["nodes_explored"],
            nodes_pruned=search_stats["nodes_pruned"],
            max_depth_reached=tree.max_depth_reached,
            success=len(solutions) > 0
        )
        
        return result
    
    async def solve_with_multiple_strategies(self, problem: str, 
                                           strategies: List[SearchMethod],
                                           task_context: Optional[Dict[str, Any]] = None) -> Dict[str, SearchResult]:
        """
        Solve the same problem with multiple search strategies for comparison
        
        Args:
            problem: The problem to solve
            strategies: List of search strategies to try
            task_context: Task-specific context
            
        Returns:
            Dictionary mapping strategy names to search results
        """
        results = {}
        original_strategy = self.config.search_method
        
        for strategy in strategies:
            try:
                # Update strategy
                self.config.search_method = strategy
                self.search_strategy = create_search_strategy(self.config)
                
                # Solve with this strategy
                result = await self.solve(problem, task_context)
                results[strategy.value] = result
                
                logger.info(f"Strategy {strategy.value}: "
                          f"{'Success' if result.success else 'Failed'}, "
                          f"Solutions: {len(result.all_solutions)}, "
                          f"Time: {result.total_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error with strategy {strategy.value}: {e}")
                results[strategy.value] = SearchResult(
                    tree=ThoughtTree(root_id="", config=self.config),
                    success=False,
                    error_message=str(e)
                )
        
        # Restore original strategy
        self.config.search_method = original_strategy
        self.search_strategy = create_search_strategy(self.config)
        
        return results
    
    def get_solution_path(self, solution: ThoughtState, tree: ThoughtTree) -> List[ThoughtState]:
        """Get the complete reasoning path to a solution"""
        return tree.get_path_to_root(solution.id)
    
    def format_solution_path(self, solution: ThoughtState, tree: ThoughtTree) -> str:
        """Format the solution path as a readable string"""
        path = self.get_solution_path(solution, tree)
        
        formatted_path = []
        for i, state in enumerate(path):
            prefix = f"Step {i}: " if i > 0 else "Problem: "
            formatted_path.append(f"{prefix}{state.content}")
        
        return "\n".join(formatted_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "success_rate": self.successful_runs / max(1, self.total_runs),
            "config": {
                "generation_strategy": self.config.generation_strategy.value,
                "evaluation_method": self.config.evaluation_method.value,
                "search_method": self.config.search_method.value,
                "model_name": self.config.model_name
            },
            "generator_stats": self.thought_generator.get_stats(),
            "evaluator_stats": self.state_evaluator.get_stats(),
            "search_stats": self.search_strategy.get_stats()
        }
