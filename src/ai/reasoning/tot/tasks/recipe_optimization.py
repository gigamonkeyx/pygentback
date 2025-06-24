"""
Recipe Optimization Task for Tree of Thought

Specialized ToT implementation for optimizing and evolving recipes
using multi-path reasoning to explore different improvement strategies.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..models import ToTConfig, GenerationStrategy, EvaluationMethod, SearchMethod
from ..tot_engine import ToTEngine

logger = logging.getLogger(__name__)


@dataclass
class RecipeOptimizationConfig:
    """Configuration specific to recipe optimization"""
    optimization_goals: List[str]  # e.g., ["performance", "reliability", "efficiency"]
    constraints: List[str]         # e.g., ["maintain compatibility", "use existing tools"]
    evaluation_criteria: List[str] # e.g., ["feasibility", "impact", "complexity"]
    domain_knowledge: Dict[str, Any] = None


class RecipeOptimizationTask:
    """
    Tree of Thought implementation for recipe optimization
    
    Uses multi-path reasoning to explore different ways to improve recipes,
    considering various optimization strategies and constraints.
    """
    
    def __init__(self, config: Optional[ToTConfig] = None, 
                 optimization_config: Optional[RecipeOptimizationConfig] = None):
        # Default ToT configuration for recipe optimization
        if config is None:
            config = ToTConfig(
                generation_strategy=GenerationStrategy.PROPOSE,
                evaluation_method=EvaluationMethod.VALUE,
                search_method=SearchMethod.BFS,
                n_generate_sample=3,
                n_evaluate_sample=2,
                n_select_sample=3,
                max_depth=6,
                temperature=0.7
            )
        
        self.config = config
        self.optimization_config = optimization_config or RecipeOptimizationConfig(
            optimization_goals=["performance", "reliability"],
            constraints=["maintain compatibility"],
            evaluation_criteria=["feasibility", "impact"]
        )
        
        self.tot_engine = ToTEngine(config)
    
    async def optimize_recipe(self, recipe_description: str, 
                            current_issues: List[str] = None,
                            target_improvements: List[str] = None) -> Dict[str, Any]:
        """
        Optimize a recipe using Tree of Thought reasoning
        
        Args:
            recipe_description: Current recipe description
            current_issues: Known issues with the current recipe
            target_improvements: Specific improvements to target
            
        Returns:
            Dictionary containing optimization results and reasoning paths
        """
        # Prepare task context
        task_context = self._create_task_context(
            recipe_description, current_issues, target_improvements
        )
        
        # Formulate the optimization problem
        problem = self._formulate_optimization_problem(
            recipe_description, current_issues, target_improvements
        )
        
        # Execute ToT reasoning
        result = await self.tot_engine.solve(problem, task_context)
        
        # Process and format results
        return self._process_optimization_results(result, recipe_description)
    
    def _create_task_context(self, recipe_description: str, 
                           current_issues: List[str],
                           target_improvements: List[str]) -> Dict[str, Any]:
        """Create task-specific context for recipe optimization"""
        return {
            "recipe_description": recipe_description,
            "current_issues": current_issues or [],
            "target_improvements": target_improvements or [],
            "optimization_goals": self.optimization_config.optimization_goals,
            "constraints": self.optimization_config.constraints,
            "evaluation_criteria": self.optimization_config.evaluation_criteria,
            
            # Custom prompts for recipe optimization
            "propose_prompt": self._get_propose_prompt(),
            "value_prompt": self._get_value_prompt(),
            "solution_prompt": self._get_solution_prompt()
        }
    
    def _formulate_optimization_problem(self, recipe_description: str,
                                      current_issues: List[str],
                                      target_improvements: List[str]) -> str:
        """Formulate the optimization problem statement"""
        problem_parts = [
            f"Recipe to optimize: {recipe_description}",
        ]
        
        if current_issues:
            issues_text = ", ".join(current_issues)
            problem_parts.append(f"Current issues: {issues_text}")
        
        if target_improvements:
            improvements_text = ", ".join(target_improvements)
            problem_parts.append(f"Target improvements: {improvements_text}")
        
        goals_text = ", ".join(self.optimization_config.optimization_goals)
        problem_parts.append(f"Optimization goals: {goals_text}")
        
        constraints_text = ", ".join(self.optimization_config.constraints)
        problem_parts.append(f"Constraints: {constraints_text}")
        
        return "\n".join(problem_parts)
    
    def _get_propose_prompt(self) -> str:
        """Get the proposal prompt for recipe optimization"""
        return """Recipe Optimization Task: {task_description}

Current optimization step: {current_thought}
Depth: {current_depth}

Based on the recipe description, current issues, and optimization goals, propose the next specific optimization step.

Consider:
- Optimization goals: {optimization_goals}
- Constraints: {constraints}
- Current issues: {current_issues}
- Target improvements: {target_improvements}

Propose a concrete, actionable optimization step that builds on the current progress:"""
    
    def _get_value_prompt(self) -> str:
        """Get the value evaluation prompt for recipe optimization"""
        return """Recipe Optimization Evaluation

Task: {task_description}
Optimization step: {thought_content}

Evaluate this optimization step based on:
1. Feasibility: How realistic is this step to implement?
2. Impact: How much will this improve the recipe?
3. Alignment: How well does this align with the optimization goals?
4. Constraints: Does this respect the given constraints?

Optimization goals: {optimization_goals}
Constraints: {constraints}
Evaluation criteria: {evaluation_criteria}

Rate this optimization step on a scale of 0.0 to 1.0:

Score:"""
    
    def _get_solution_prompt(self) -> str:
        """Get the solution check prompt for recipe optimization"""
        return """Recipe Optimization Solution Check

Task: {task_description}
Proposed optimization: {thought_content}

Optimization goals: {optimization_goals}
Constraints: {constraints}

Is this a complete and viable optimization solution that addresses the main issues and meets the optimization goals?

Answer yes or no:"""
    
    def _process_optimization_results(self, result, original_recipe: str) -> Dict[str, Any]:
        """Process and format the optimization results"""
        processed_result = {
            "success": result.success,
            "original_recipe": original_recipe,
            "optimization_found": result.best_solution is not None,
            "total_time": result.total_time,
            "reasoning_stats": {
                "nodes_explored": result.nodes_explored,
                "max_depth_reached": result.max_depth_reached,
                "total_llm_calls": result.total_llm_calls
            }
        }
        
        if result.best_solution:
            # Format the best optimization
            processed_result["best_optimization"] = {
                "description": result.best_solution.content,
                "confidence_score": result.best_solution.value_score,
                "reasoning_path": self.tot_engine.format_solution_path(
                    result.best_solution, result.tree
                )
            }
            
            # Include all solutions if multiple found
            if len(result.all_solutions) > 1:
                processed_result["alternative_optimizations"] = [
                    {
                        "description": sol.content,
                        "confidence_score": sol.value_score
                    }
                    for sol in sorted(result.all_solutions, 
                                    key=lambda s: s.value_score, reverse=True)[1:]
                ]
        
        if not result.success:
            processed_result["error"] = result.error_message
            
            # Provide partial results if available
            if result.tree.total_nodes > 1:
                # Find the best partial solution
                all_states = [s for s in result.tree.states.values() 
                            if s.depth > 0 and not s.is_pruned]
                if all_states:
                    best_partial = max(all_states, key=lambda s: s.value_score)
                    processed_result["partial_optimization"] = {
                        "description": best_partial.content,
                        "confidence_score": best_partial.value_score,
                        "note": "Partial optimization - reasoning was incomplete"
                    }
        
        return processed_result
    
    async def compare_optimization_strategies(self, recipe_description: str,
                                            strategies: List[str]) -> Dict[str, Any]:
        """
        Compare different optimization strategies for the same recipe
        
        Args:
            recipe_description: Recipe to optimize
            strategies: List of optimization strategies to compare
            
        Returns:
            Comparison results for different strategies
        """
        results = {}
        
        for strategy in strategies:
            # Create strategy-specific context
            strategy_config = RecipeOptimizationConfig(
                optimization_goals=[strategy],
                constraints=self.optimization_config.constraints,
                evaluation_criteria=self.optimization_config.evaluation_criteria
            )
            
            # Create temporary task instance
            strategy_task = RecipeOptimizationTask(
                self.config, strategy_config
            )
            
            # Run optimization with this strategy
            result = await strategy_task.optimize_recipe(
                recipe_description,
                target_improvements=[f"Optimize for {strategy}"]
            )
            
            results[strategy] = result
        
        # Add comparison summary
        results["comparison_summary"] = self._create_strategy_comparison(results)
        
        return results
    
    def _create_strategy_comparison(self, strategy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary comparing different optimization strategies"""
        successful_strategies = [
            strategy for strategy, result in strategy_results.items()
            if isinstance(result, dict) and result.get("success", False)
        ]
        
        if not successful_strategies:
            return {"message": "No strategies produced successful optimizations"}
        
        # Find best strategy by confidence score
        best_strategy = max(
            successful_strategies,
            key=lambda s: strategy_results[s]["best_optimization"]["confidence_score"]
        )
        
        return {
            "total_strategies_tested": len([k for k in strategy_results.keys() 
                                          if k != "comparison_summary"]),
            "successful_strategies": len(successful_strategies),
            "recommended_strategy": best_strategy,
            "recommendation_reason": f"Highest confidence score: "
                                   f"{strategy_results[best_strategy]['best_optimization']['confidence_score']:.3f}"
        }
