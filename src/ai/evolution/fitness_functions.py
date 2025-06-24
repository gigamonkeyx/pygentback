"""
Fitness Functions for Recipe Evolution

Multi-objective fitness evaluation for genetic algorithm optimization.
Evaluates recipes based on success rate, performance, maintainability, and other criteria.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time

try:
    from ...testing.recipes.schema import RecipeDefinition
    from ...testing.core.framework import RecipeTestResult
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    
    @dataclass
    class RecipeDefinition:
        name: str = ""
        description: str = ""
    
    @dataclass
    class RecipeTestResult:
        success: bool = True
        score: float = 0.8
        execution_time_ms: int = 1000
        memory_usage_mb: float = 256.0


logger = logging.getLogger(__name__)


class FitnessObjective(Enum):
    """Fitness objectives for multi-objective optimization"""
    SUCCESS_RATE = "success_rate"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    RELIABILITY = "reliability"
    COMPLEXITY = "complexity"
    INNOVATION = "innovation"
    COMPATIBILITY = "compatibility"


@dataclass
class FitnessScore:
    """Comprehensive fitness score for a recipe"""
    total_score: float
    objective_scores: Dict[FitnessObjective, float] = field(default_factory=dict)
    weights: Dict[FitnessObjective, float] = field(default_factory=dict)
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    evaluation_time_ms: float = 0.0
    confidence: float = 1.0
    
    def get_weighted_score(self) -> float:
        """Calculate weighted total score"""
        if not self.objective_scores or not self.weights:
            return self.total_score
        
        weighted_sum = sum(
            score * self.weights.get(objective, 1.0)
            for objective, score in self.objective_scores.items()
        )
        
        total_weight = sum(self.weights.values()) or 1.0
        return weighted_sum / total_weight
    
    def get_pareto_dominance(self, other: 'FitnessScore') -> int:
        """
        Compare Pareto dominance with another fitness score.
        Returns: 1 if this dominates other, -1 if other dominates this, 0 if non-dominated
        """
        better_count = 0
        worse_count = 0
        
        for objective in FitnessObjective:
            self_score = self.objective_scores.get(objective, 0.0)
            other_score = other.objective_scores.get(objective, 0.0)
            
            if self_score > other_score:
                better_count += 1
            elif self_score < other_score:
                worse_count += 1
        
        if better_count > 0 and worse_count == 0:
            return 1  # This dominates other
        elif worse_count > 0 and better_count == 0:
            return -1  # Other dominates this
        else:
            return 0  # Non-dominated


class FitnessEvaluator:
    """
    Multi-objective fitness evaluator for recipe evolution.
    
    Evaluates recipes across multiple dimensions:
    - Success rate and reliability
    - Performance and resource efficiency
    - Code quality and maintainability
    - Innovation and complexity
    - Compatibility and integration
    """
    
    def __init__(self, 
                 objectives: Optional[List[FitnessObjective]] = None,
                 weights: Optional[Dict[FitnessObjective, float]] = None):
        self.objectives = objectives or [
            FitnessObjective.SUCCESS_RATE,
            FitnessObjective.PERFORMANCE,
            FitnessObjective.MAINTAINABILITY,
            FitnessObjective.RESOURCE_EFFICIENCY
        ]
        
        self.weights = weights or {
            FitnessObjective.SUCCESS_RATE: 0.3,
            FitnessObjective.PERFORMANCE: 0.25,
            FitnessObjective.MAINTAINABILITY: 0.2,
            FitnessObjective.RESOURCE_EFFICIENCY: 0.15,
            FitnessObjective.RELIABILITY: 0.1
        }
        
        # Historical data for normalization
        self.historical_metrics = {
            'execution_times': [],
            'memory_usage': [],
            'success_rates': [],
            'complexity_scores': []
        }
    
    async def calculate_fitness(self, recipe: RecipeDefinition, 
                              test_results: Optional[List[RecipeTestResult]] = None) -> FitnessScore:
        """
        Calculate comprehensive fitness score for a recipe.
        
        Args:
            recipe: Recipe definition to evaluate
            test_results: Optional test results for the recipe
            
        Returns:
            Comprehensive fitness score
        """
        start_time = time.time()
        
        try:
            # Initialize fitness score
            fitness_score = FitnessScore(
                total_score=0.0,
                weights=self.weights.copy()
            )
            
            # Calculate individual objective scores
            objective_scores = {}
            
            for objective in self.objectives:
                score = await self._evaluate_objective(objective, recipe, test_results)
                objective_scores[objective] = score
                fitness_score.raw_metrics[f"{objective.value}_score"] = score
            
            fitness_score.objective_scores = objective_scores
            
            # Calculate weighted total score
            fitness_score.total_score = fitness_score.get_weighted_score()
            
            # Calculate confidence based on available data
            fitness_score.confidence = self._calculate_confidence(recipe, test_results)
            
            # Record evaluation time
            fitness_score.evaluation_time_ms = (time.time() - start_time) * 1000
            
            logger.debug(f"Calculated fitness for recipe '{recipe.name}': {fitness_score.total_score:.3f}")
            return fitness_score
            
        except Exception as e:
            logger.error(f"Failed to calculate fitness for recipe '{recipe.name}': {e}")
            return FitnessScore(total_score=0.0, confidence=0.0)
    
    async def _evaluate_objective(self, objective: FitnessObjective, 
                                recipe: RecipeDefinition,
                                test_results: Optional[List[RecipeTestResult]]) -> float:
        """Evaluate a specific fitness objective"""
        
        if objective == FitnessObjective.SUCCESS_RATE:
            return self._evaluate_success_rate(recipe, test_results)
        
        elif objective == FitnessObjective.PERFORMANCE:
            return self._evaluate_performance(recipe, test_results)
        
        elif objective == FitnessObjective.MAINTAINABILITY:
            return self._evaluate_maintainability(recipe)
        
        elif objective == FitnessObjective.RESOURCE_EFFICIENCY:
            return self._evaluate_resource_efficiency(recipe, test_results)
        
        elif objective == FitnessObjective.RELIABILITY:
            return self._evaluate_reliability(recipe, test_results)
        
        elif objective == FitnessObjective.COMPLEXITY:
            return self._evaluate_complexity(recipe)
        
        elif objective == FitnessObjective.INNOVATION:
            return self._evaluate_innovation(recipe)
        
        elif objective == FitnessObjective.COMPATIBILITY:
            return self._evaluate_compatibility(recipe)
        
        else:
            logger.warning(f"Unknown fitness objective: {objective}")
            return 0.5  # Neutral score
    
    def _evaluate_success_rate(self, recipe: RecipeDefinition, 
                             test_results: Optional[List[RecipeTestResult]]) -> float:
        """Evaluate recipe success rate"""
        if not test_results:
            # Estimate based on recipe characteristics
            complexity = self._calculate_recipe_complexity(recipe)
            estimated_success = max(0.1, 1.0 - complexity * 0.3)
            return estimated_success
        
        # Calculate actual success rate
        successful_tests = sum(1 for result in test_results if result.success)
        success_rate = successful_tests / len(test_results)
        
        # Update historical data
        self.historical_metrics['success_rates'].append(success_rate)
        
        return success_rate
    
    def _evaluate_performance(self, recipe: RecipeDefinition,
                            test_results: Optional[List[RecipeTestResult]]) -> float:
        """Evaluate recipe performance"""
        if not test_results:
            # Estimate based on recipe structure
            step_count = len(getattr(recipe, 'steps', []))
            agent_count = len(getattr(recipe, 'agent_requirements', []))
            mcp_count = len(getattr(recipe, 'mcp_requirements', []))
            
            # Simple heuristic: more components = potentially slower
            complexity_penalty = (step_count + agent_count + mcp_count) * 0.05
            estimated_performance = max(0.1, 1.0 - complexity_penalty)
            return estimated_performance
        
        # Calculate performance score from test results
        avg_execution_time = np.mean([r.execution_time_ms for r in test_results])
        
        # Normalize against historical data or use heuristic
        if self.historical_metrics['execution_times']:
            historical_avg = np.mean(self.historical_metrics['execution_times'])
            performance_score = max(0.1, min(1.0, historical_avg / avg_execution_time))
        else:
            # Heuristic: 5 seconds is baseline (score 0.5)
            baseline_time = 5000  # ms
            performance_score = max(0.1, min(1.0, baseline_time / avg_execution_time))
        
        # Update historical data
        self.historical_metrics['execution_times'].append(avg_execution_time)
        
        return performance_score
    
    def _evaluate_maintainability(self, recipe: RecipeDefinition) -> float:
        """Evaluate recipe maintainability"""
        score = 1.0
        
        # Check for clear naming
        name_quality = min(1.0, len(recipe.name) / 50) if recipe.name else 0.0
        description_quality = min(1.0, len(recipe.description) / 200) if recipe.description else 0.0
        
        # Check for reasonable complexity
        complexity = self._calculate_recipe_complexity(recipe)
        complexity_score = max(0.0, 1.0 - complexity)
        
        # Check for modular structure
        steps = getattr(recipe, 'steps', [])
        modularity_score = min(1.0, len(steps) / 10) if steps else 0.5
        
        # Weighted combination
        maintainability = (
            name_quality * 0.2 +
            description_quality * 0.3 +
            complexity_score * 0.3 +
            modularity_score * 0.2
        )
        
        return max(0.1, maintainability)
    
    def _evaluate_resource_efficiency(self, recipe: RecipeDefinition,
                                    test_results: Optional[List[RecipeTestResult]]) -> float:
        """Evaluate resource efficiency"""
        if not test_results:
            # Estimate based on recipe requirements
            agent_count = len(getattr(recipe, 'agent_requirements', []))
            mcp_count = len(getattr(recipe, 'mcp_requirements', []))
            
            # More resources = lower efficiency score
            resource_penalty = (agent_count + mcp_count) * 0.1
            estimated_efficiency = max(0.1, 1.0 - resource_penalty)
            return estimated_efficiency
        
        # Calculate efficiency from test results
        avg_memory = np.mean([r.memory_usage_mb for r in test_results])
        
        # Normalize against historical data or use heuristic
        if self.historical_metrics['memory_usage']:
            historical_avg = np.mean(self.historical_metrics['memory_usage'])
            efficiency_score = max(0.1, min(1.0, historical_avg / avg_memory))
        else:
            # Heuristic: 512MB is baseline (score 0.5)
            baseline_memory = 512  # MB
            efficiency_score = max(0.1, min(1.0, baseline_memory / avg_memory))
        
        # Update historical data
        self.historical_metrics['memory_usage'].append(avg_memory)
        
        return efficiency_score

    def _evaluate_reliability(self, recipe: RecipeDefinition,
                            test_results: Optional[List[RecipeTestResult]]) -> float:
        """Evaluate recipe reliability"""
        if not test_results:
            # Estimate based on error handling and retry mechanisms
            steps = getattr(recipe, 'steps', [])
            retry_steps = sum(1 for step in steps if getattr(step, 'retry_on_failure', False))
            reliability_score = min(1.0, retry_steps / max(1, len(steps)))
            return max(0.3, reliability_score)

        # Calculate reliability from test results variance
        success_rates = [1.0 if r.success else 0.0 for r in test_results]
        if len(success_rates) > 1:
            variance = np.var(success_rates)
            reliability_score = max(0.1, 1.0 - variance)
        else:
            reliability_score = success_rates[0] if success_rates else 0.5

        return reliability_score

    def _evaluate_complexity(self, recipe: RecipeDefinition) -> float:
        """Evaluate recipe complexity (lower complexity = higher score)"""
        complexity = self._calculate_recipe_complexity(recipe)
        # Invert complexity for scoring (simpler = better)
        complexity_score = max(0.1, 1.0 - complexity)
        return complexity_score

    def _evaluate_innovation(self, recipe: RecipeDefinition) -> float:
        """Evaluate recipe innovation"""
        # Check for unique combinations of agents and MCP tools
        agent_types = set()
        mcp_servers = set()

        for agent_req in getattr(recipe, 'agent_requirements', []):
            agent_types.add(getattr(agent_req, 'agent_type', 'default'))

        for mcp_req in getattr(recipe, 'mcp_requirements', []):
            mcp_servers.add(getattr(mcp_req, 'server_name', 'default'))

        # Innovation score based on diversity
        diversity_score = min(1.0, (len(agent_types) + len(mcp_servers)) / 10)

        # Check for advanced features
        steps = getattr(recipe, 'steps', [])
        advanced_features = sum(1 for step in steps if getattr(step, 'parallel_execution', False))
        feature_score = min(1.0, advanced_features / max(1, len(steps)))

        innovation_score = (diversity_score + feature_score) / 2
        return max(0.1, innovation_score)

    def _evaluate_compatibility(self, recipe: RecipeDefinition) -> float:
        """Evaluate recipe compatibility"""
        # Check for known compatible combinations
        mcp_servers = [getattr(req, 'server_name', '') for req in getattr(recipe, 'mcp_requirements', [])]

        # Simple compatibility heuristic
        known_compatible = {'filesystem', 'database', 'api', 'tool'}
        compatible_count = sum(1 for server in mcp_servers if server in known_compatible)

        if mcp_servers:
            compatibility_score = compatible_count / len(mcp_servers)
        else:
            compatibility_score = 1.0  # No MCP requirements = fully compatible

        return max(0.1, compatibility_score)

    def _calculate_recipe_complexity(self, recipe: RecipeDefinition) -> float:
        """Calculate normalized recipe complexity (0-1)"""
        complexity = 0.0

        # Step complexity
        steps = getattr(recipe, 'steps', [])
        complexity += len(steps) * 0.1

        # Agent complexity
        agent_reqs = getattr(recipe, 'agent_requirements', [])
        complexity += len(agent_reqs) * 0.15

        # MCP complexity
        mcp_reqs = getattr(recipe, 'mcp_requirements', [])
        complexity += len(mcp_reqs) * 0.1

        # Dependency complexity
        for step in steps:
            dependencies = getattr(step, 'dependencies', [])
            complexity += len(dependencies) * 0.05

        # Normalize to 0-1 range
        return min(1.0, complexity)

    def _calculate_confidence(self, recipe: RecipeDefinition,
                            test_results: Optional[List[RecipeTestResult]]) -> float:
        """Calculate confidence in fitness evaluation"""
        confidence = 0.5  # Base confidence

        # Increase confidence with test results
        if test_results:
            confidence += min(0.4, len(test_results) * 0.1)

        # Increase confidence with recipe completeness
        if hasattr(recipe, 'description') and recipe.description:
            confidence += 0.1

        if hasattr(recipe, 'steps') and getattr(recipe, 'steps', []):
            confidence += 0.1

        return min(1.0, confidence)

    def update_historical_data(self, test_results: List[RecipeTestResult]):
        """Update historical metrics for normalization"""
        for result in test_results:
            self.historical_metrics['execution_times'].append(result.execution_time_ms)
            self.historical_metrics['memory_usage'].append(result.memory_usage_mb)
            self.historical_metrics['success_rates'].append(1.0 if result.success else 0.0)

        # Keep only recent data (last 1000 entries)
        for key in self.historical_metrics:
            if len(self.historical_metrics[key]) > 1000:
                self.historical_metrics[key] = self.historical_metrics[key][-1000:]

    def get_fitness_statistics(self) -> Dict[str, Any]:
        """Get statistics about fitness evaluations"""
        stats = {
            'objectives': [obj.value for obj in self.objectives],
            'weights': {obj.value: weight for obj, weight in self.weights.items()},
            'historical_data_size': {
                key: len(values) for key, values in self.historical_metrics.items()
            }
        }

        # Calculate historical averages if data available
        if self.historical_metrics['execution_times']:
            stats['avg_execution_time'] = np.mean(self.historical_metrics['execution_times'])

        if self.historical_metrics['memory_usage']:
            stats['avg_memory_usage'] = np.mean(self.historical_metrics['memory_usage'])

        if self.historical_metrics['success_rates']:
            stats['avg_success_rate'] = np.mean(self.historical_metrics['success_rates'])

        return stats

    def compare_fitness_scores(self, scores: List[FitnessScore]) -> Dict[str, Any]:
        """Compare multiple fitness scores and provide analysis"""
        if not scores:
            return {'error': 'No fitness scores provided'}

        analysis = {
            'count': len(scores),
            'best_score': max(score.total_score for score in scores),
            'worst_score': min(score.total_score for score in scores),
            'avg_score': np.mean([score.total_score for score in scores]),
            'score_variance': np.var([score.total_score for score in scores])
        }

        # Objective-wise analysis
        objective_analysis = {}
        for objective in self.objectives:
            obj_scores = [score.objective_scores.get(objective, 0.0) for score in scores]
            objective_analysis[objective.value] = {
                'best': max(obj_scores),
                'worst': min(obj_scores),
                'avg': np.mean(obj_scores),
                'variance': np.var(obj_scores)
            }

        analysis['objective_analysis'] = objective_analysis

        # Pareto frontier analysis
        pareto_frontier = self._find_pareto_frontier(scores)
        analysis['pareto_frontier_size'] = len(pareto_frontier)
        analysis['pareto_frontier_scores'] = [score.total_score for score in pareto_frontier]

        return analysis

    def _find_pareto_frontier(self, scores: List[FitnessScore]) -> List[FitnessScore]:
        """Find Pareto frontier from fitness scores"""
        pareto_frontier = []

        for candidate in scores:
            is_dominated = False

            for other in scores:
                if candidate != other and other.get_pareto_dominance(candidate) == 1:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_frontier.append(candidate)

        return pareto_frontier
