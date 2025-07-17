"""
Recipe Optimizer

This module provides ML-powered optimization for Agent + MCP recipes using
advanced optimization algorithms, hyperparameter tuning, and performance modeling.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

# ML/Optimization imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
import lightgbm as lgb

from ..core.framework import RecipeTestResult
from ..recipes.schema import RecipeDefinition, RecipeStep, AgentRequirement, MCPToolRequirement
from ..ml.predictor import RecipeSuccessPredictor, PredictionResult


logger = logging.getLogger(__name__)


@dataclass
class OptimizationTarget:
    """Optimization target specification"""
    metric: str  # success_rate, execution_time, memory_usage, composite_score
    direction: str  # maximize, minimize
    weight: float = 1.0
    constraint_min: Optional[float] = None
    constraint_max: Optional[float] = None


@dataclass
class OptimizationParameter:
    """Parameter to optimize"""
    name: str
    param_type: str  # int, float, categorical, boolean
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None
    log_scale: bool = False


@dataclass
class OptimizationResult:
    """Result of recipe optimization"""
    original_recipe: RecipeDefinition
    optimized_recipe: RecipeDefinition
    optimization_history: List[Dict[str, Any]]
    best_score: float
    improvement_percentage: float
    optimization_time_seconds: float
    parameters_changed: Dict[str, Any]
    validation_results: List[RecipeTestResult]
    recommendations: List[str]


class RecipeOptimizer:
    """
    Advanced ML-powered recipe optimizer.
    
    Uses multiple optimization algorithms including Bayesian optimization,
    genetic algorithms, and gradient-based methods to improve recipe performance.
    """
    
    def __init__(self, 
                 predictor: Optional[RecipeSuccessPredictor] = None,
                 data_dir: str = "./data/optimization"):
        self.predictor = predictor
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization models
        self.performance_model: Optional[xgb.XGBRegressor] = None
        self.success_model: Optional[lgb.LGBMClassifier] = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        # Optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_recipes: Dict[str, RecipeDefinition] = {}
        
        # Default optimization parameters
        self.default_parameters = self._get_default_optimization_parameters()
        
        # Optimization algorithms
        self.algorithms = {
            "bayesian": self._bayesian_optimization,
            "genetic": self._genetic_optimization,
            "grid_search": self._grid_search_optimization,
            "random_search": self._random_search_optimization
        }
    
    async def initialize(self) -> None:
        """Initialize the optimizer with historical data"""
        try:
            # Load historical optimization data
            await self._load_optimization_history()
            
            # Train performance models if we have enough data
            if len(self.optimization_history) >= 50:
                await self._train_performance_models()
            
            logger.info("Recipe optimizer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            raise
    
    async def optimize_recipe(self,
                            recipe: RecipeDefinition,
                            targets: List[OptimizationTarget],
                            parameters: Optional[List[OptimizationParameter]] = None,
                            algorithm: str = "bayesian",
                            n_trials: int = 100,
                            timeout_seconds: int = 3600,
                            validation_function: Optional[Callable] = None) -> OptimizationResult:
        """
        Optimize a recipe for specified targets.
        
        Args:
            recipe: Recipe to optimize
            targets: Optimization targets
            parameters: Parameters to optimize (uses defaults if None)
            algorithm: Optimization algorithm to use
            n_trials: Number of optimization trials
            timeout_seconds: Optimization timeout
            validation_function: Function to validate optimized recipes
            
        Returns:
            Optimization result
        """
        start_time = datetime.utcnow()
        
        logger.info(f"Starting optimization for recipe: {recipe.name}")
        logger.info(f"Targets: {[f'{t.metric} ({t.direction})' for t in targets]}")
        logger.info(f"Algorithm: {algorithm}, Trials: {n_trials}")
        
        # Use default parameters if none provided
        if parameters is None:
            parameters = self.default_parameters
        
        # Validate inputs
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(self.algorithms.keys())}")
        
        try:
            # Run optimization
            optimization_func = self.algorithms[algorithm]
            best_params, optimization_history = await optimization_func(
                recipe, targets, parameters, n_trials, timeout_seconds
            )
            
            # Create optimized recipe
            optimized_recipe = await self._apply_optimization_parameters(recipe, best_params)
            
            # Calculate improvement
            original_score = await self._evaluate_recipe(recipe, targets)
            optimized_score = await self._evaluate_recipe(optimized_recipe, targets)
            improvement = ((optimized_score - original_score) / original_score * 100) if original_score > 0 else 0
            
            # Validate optimized recipe if validation function provided
            validation_results = []
            if validation_function:
                try:
                    validation_results = await validation_function(optimized_recipe)
                except Exception as e:
                    logger.warning(f"Validation failed: {e}")
            
            # Generate recommendations
            recommendations = await self._generate_optimization_recommendations(
                recipe, optimized_recipe, best_params, optimization_history
            )
            
            # Create result
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = OptimizationResult(
                original_recipe=recipe,
                optimized_recipe=optimized_recipe,
                optimization_history=optimization_history,
                best_score=optimized_score,
                improvement_percentage=improvement,
                optimization_time_seconds=execution_time,
                parameters_changed=best_params,
                validation_results=validation_results,
                recommendations=recommendations
            )
            
            # Store optimization result
            await self._store_optimization_result(result)
            
            logger.info(f"Optimization completed: {improvement:.1f}% improvement in {execution_time:.1f}s")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed for recipe {recipe.name}: {e}")
            raise
    
    async def _bayesian_optimization(self,
                                   recipe: RecipeDefinition,
                                   targets: List[OptimizationTarget],
                                   parameters: List[OptimizationParameter],
                                   n_trials: int,
                                   timeout_seconds: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Bayesian optimization using Optuna"""
        
        def objective(trial):
            # Sample parameters
            params = {}
            for param in parameters:
                if param.param_type == "int":
                    params[param.name] = trial.suggest_int(
                        param.name, int(param.min_value), int(param.max_value), step=int(param.step or 1)
                    )
                elif param.param_type == "float":
                    if param.log_scale:
                        params[param.name] = trial.suggest_loguniform(
                            param.name, param.min_value, param.max_value
                        )
                    else:
                        params[param.name] = trial.suggest_uniform(
                            param.name, param.min_value, param.max_value
                        )
                elif param.param_type == "categorical":
                    params[param.name] = trial.suggest_categorical(param.name, param.choices)
                elif param.param_type == "boolean":
                    params[param.name] = trial.suggest_categorical(param.name, [True, False])
            
            # Evaluate recipe with these parameters
            try:
                modified_recipe = asyncio.run(self._apply_optimization_parameters(recipe, params))
                score = asyncio.run(self._evaluate_recipe(modified_recipe, targets))
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)
        
        # Extract history
        history = []
        for trial in study.trials:
            history.append({
                "trial_number": trial.number,
                "params": trial.params,
                "value": trial.value,
                "state": trial.state.name
            })
        
        return study.best_params, history
    
    async def _genetic_optimization(self,
                                  recipe: RecipeDefinition,
                                  targets: List[OptimizationTarget],
                                  parameters: List[OptimizationParameter],
                                  n_trials: int,
                                  timeout_seconds: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Genetic algorithm optimization (simplified implementation)"""
        
        population_size = min(50, n_trials // 4)
        generations = n_trials // population_size
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for param in parameters:
                if param.param_type == "int":
                    individual[param.name] = np.random.randint(param.min_value, param.max_value + 1)
                elif param.param_type == "float":
                    if param.log_scale:
                        individual[param.name] = np.random.lognormal(
                            np.log(param.min_value), np.log(param.max_value / param.min_value)
                        )
                    else:
                        individual[param.name] = np.random.uniform(param.min_value, param.max_value)
                elif param.param_type == "categorical":
                    individual[param.name] = np.random.choice(param.choices)
                elif param.param_type == "boolean":
                    individual[param.name] = np.random.choice([True, False])
            population.append(individual)
        
        history = []
        best_individual = None
        best_score = -float('inf')
        
        for generation in range(generations):
            # Evaluate population
            scores = []
            for individual in population:
                try:
                    modified_recipe = await self._apply_optimization_parameters(recipe, individual)
                    score = await self._evaluate_recipe(modified_recipe, targets)
                    scores.append(score)
                    
                    # Track best
                    if score > best_score:
                        best_score = score
                        best_individual = individual.copy()
                    
                    # Add to history
                    history.append({
                        "generation": generation,
                        "individual": individual.copy(),
                        "score": score
                    })
                    
                except Exception as e:
                    scores.append(0.0)
            
            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                tournament_size = 3
                tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
                tournament_scores = [scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_scores)]
                new_population.append(population[winner_idx].copy())
            
            # Crossover and mutation
            for i in range(0, len(new_population) - 1, 2):
                if np.random.random() < crossover_rate:
                    # Simple crossover
                    parent1, parent2 = new_population[i], new_population[i + 1]
                    for param in parameters:
                        if np.random.random() < 0.5:
                            parent1[param.name], parent2[param.name] = parent2[param.name], parent1[param.name]
                
                # Mutation
                for individual in [new_population[i], new_population[i + 1]]:
                    if np.random.random() < mutation_rate:
                        param = np.random.choice(parameters)
                        if param.param_type == "int":
                            individual[param.name] = np.random.randint(param.min_value, param.max_value + 1)
                        elif param.param_type == "float":
                            individual[param.name] = np.random.uniform(param.min_value, param.max_value)
                        elif param.param_type == "categorical":
                            individual[param.name] = np.random.choice(param.choices)
                        elif param.param_type == "boolean":
                            individual[param.name] = np.random.choice([True, False])
            
            population = new_population
        
        return best_individual, history
    
    async def _grid_search_optimization(self,
                                      recipe: RecipeDefinition,
                                      targets: List[OptimizationTarget],
                                      parameters: List[OptimizationParameter],
                                      n_trials: int,
                                      timeout_seconds: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Grid search optimization"""
        
        # Create parameter grid
        param_grid = {}
        for param in parameters:
            if param.param_type == "int":
                param_grid[param.name] = np.linspace(
                    param.min_value, param.max_value, 
                    min(10, int(param.max_value - param.min_value + 1))
                ).astype(int).tolist()
            elif param.param_type == "float":
                param_grid[param.name] = np.linspace(param.min_value, param.max_value, 10).tolist()
            elif param.param_type == "categorical":
                param_grid[param.name] = param.choices
            elif param.param_type == "boolean":
                param_grid[param.name] = [True, False]
        
        # Generate all combinations (limited by n_trials)
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = list(product(*param_values))
        if len(combinations) > n_trials:
            combinations = np.random.choice(len(combinations), n_trials, replace=False)
            combinations = [list(product(*param_values))[i] for i in combinations]
        
        # Evaluate combinations
        history = []
        best_params = None
        best_score = -float('inf')
        
        for i, combination in enumerate(combinations):
            params = dict(zip(param_names, combination))
            
            try:
                modified_recipe = await self._apply_optimization_parameters(recipe, params)
                score = await self._evaluate_recipe(modified_recipe, targets)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                history.append({
                    "iteration": i,
                    "params": params.copy(),
                    "score": score
                })
                
            except Exception as e:
                logger.warning(f"Grid search iteration {i} failed: {e}")
        
        return best_params, history
    
    async def _random_search_optimization(self,
                                        recipe: RecipeDefinition,
                                        targets: List[OptimizationTarget],
                                        parameters: List[OptimizationParameter],
                                        n_trials: int,
                                        timeout_seconds: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Random search optimization"""
        
        history = []
        best_params = None
        best_score = -float('inf')
        
        for trial in range(n_trials):
            # Random parameter sampling
            params = {}
            for param in parameters:
                if param.param_type == "int":
                    params[param.name] = np.random.randint(param.min_value, param.max_value + 1)
                elif param.param_type == "float":
                    if param.log_scale:
                        params[param.name] = np.random.lognormal(
                            np.log(param.min_value), np.log(param.max_value / param.min_value)
                        )
                    else:
                        params[param.name] = np.random.uniform(param.min_value, param.max_value)
                elif param.param_type == "categorical":
                    params[param.name] = np.random.choice(param.choices)
                elif param.param_type == "boolean":
                    params[param.name] = np.random.choice([True, False])
            
            try:
                modified_recipe = await self._apply_optimization_parameters(recipe, params)
                score = await self._evaluate_recipe(modified_recipe, targets)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
                history.append({
                    "trial": trial,
                    "params": params.copy(),
                    "score": score
                })
                
            except Exception as e:
                logger.warning(f"Random search trial {trial} failed: {e}")
        
        return best_params, history

    async def _apply_optimization_parameters(self,
                                           recipe: RecipeDefinition,
                                           params: Dict[str, Any]) -> RecipeDefinition:
        """Apply optimization parameters to create modified recipe"""
        # Create a copy of the recipe
        modified_recipe = RecipeDefinition(
            id=f"{recipe.id}_optimized",
            name=f"{recipe.name} (Optimized)",
            description=f"Optimized version of {recipe.description}",
            category=recipe.category,
            difficulty=recipe.difficulty,
            agent_requirements=recipe.agent_requirements.copy(),
            mcp_requirements=recipe.mcp_requirements.copy(),
            steps=recipe.steps.copy(),
            expected_outputs=recipe.expected_outputs.copy(),
            success_criteria=recipe.success_criteria.copy(),
            tags=recipe.tags.copy(),
            metadata=recipe.metadata.copy()
        )

        # Apply parameter modifications
        for param_name, param_value in params.items():
            if param_name == "agent_timeout":
                for agent_req in modified_recipe.agent_requirements:
                    agent_req.timeout_seconds = int(param_value)

            elif param_name == "agent_memory_limit":
                for agent_req in modified_recipe.agent_requirements:
                    if "memory_limit" not in agent_req.config:
                        agent_req.config["memory_limit"] = int(param_value)

            elif param_name == "mcp_timeout":
                for mcp_req in modified_recipe.mcp_requirements:
                    mcp_req.timeout_seconds = int(param_value)

            elif param_name == "parallel_execution":
                modified_recipe.metadata["parallel_execution"] = param_value

            elif param_name == "retry_count":
                modified_recipe.metadata["retry_count"] = int(param_value)

            elif param_name == "step_timeout":
                for step in modified_recipe.steps:
                    step.timeout_seconds = int(param_value)

            elif param_name == "batch_size":
                modified_recipe.metadata["batch_size"] = int(param_value)

            elif param_name == "optimization_level":
                modified_recipe.metadata["optimization_level"] = param_value

        return modified_recipe

    async def _evaluate_recipe(self,
                             recipe: RecipeDefinition,
                             targets: List[OptimizationTarget]) -> float:
        """Evaluate recipe performance for optimization targets"""

        # Use predictor if available
        if self.predictor:
            try:
                prediction = await self.predictor.predict_success(recipe)

                # Calculate composite score based on targets
                total_score = 0.0
                total_weight = 0.0

                for target in targets:
                    score = 0.0

                    if target.metric == "success_rate":
                        score = prediction.success_probability
                    elif target.metric == "execution_time":
                        # Convert to score (lower is better)
                        max_time = 300000  # 5 minutes in ms
                        score = max(0, 1 - (prediction.execution_time_prediction / max_time))
                    elif target.metric == "memory_usage":
                        # Convert to score (lower is better)
                        max_memory = 2048  # 2GB in MB
                        score = max(0, 1 - (prediction.memory_usage_prediction / max_memory))
                    elif target.metric == "composite_score":
                        score = (prediction.success_probability +
                                max(0, 1 - prediction.execution_time_prediction / 300000) +
                                max(0, 1 - prediction.memory_usage_prediction / 2048)) / 3

                    # Apply direction
                    if target.direction == "minimize":
                        score = 1 - score

                    # Apply constraints
                    if target.constraint_min is not None and score < target.constraint_min:
                        score = 0.0
                    if target.constraint_max is not None and score > target.constraint_max:
                        score = target.constraint_max

                    total_score += score * target.weight
                    total_weight += target.weight

                return total_score / total_weight if total_weight > 0 else 0.0

            except Exception as e:
                logger.warning(f"Prediction failed during evaluation: {e}")

        # Fallback: simple heuristic evaluation
        return self._heuristic_evaluation(recipe, targets)

    def _heuristic_evaluation(self,
                            recipe: RecipeDefinition,
                            targets: List[OptimizationTarget]) -> float:
        """Simple heuristic evaluation when predictor is not available"""

        total_score = 0.0
        total_weight = 0.0

        for target in targets:
            score = 0.5  # Default neutral score

            if target.metric == "success_rate":
                # Heuristic based on recipe complexity
                complexity = len(recipe.steps) + len(recipe.agent_requirements) + len(recipe.mcp_requirements)
                score = max(0.1, 1.0 - (complexity * 0.05))

            elif target.metric == "execution_time":
                # Heuristic based on step count and timeouts
                total_timeout = sum(step.timeout_seconds or 60 for step in recipe.steps)
                score = max(0.1, 1.0 - (total_timeout / 3600))  # Normalize by 1 hour

            elif target.metric == "memory_usage":
                # Heuristic based on agent requirements
                agent_count = len(recipe.agent_requirements)
                score = max(0.1, 1.0 - (agent_count * 0.1))

            # Apply direction
            if target.direction == "minimize":
                score = 1 - score

            total_score += score * target.weight
            total_weight += target.weight

        return total_score / total_weight if total_weight > 0 else 0.5

    def _get_default_optimization_parameters(self) -> List[OptimizationParameter]:
        """Get default optimization parameters"""
        return [
            OptimizationParameter(
                name="agent_timeout",
                param_type="int",
                min_value=30,
                max_value=600,
                step=30
            ),
            OptimizationParameter(
                name="agent_memory_limit",
                param_type="int",
                min_value=512,
                max_value=4096,
                step=256
            ),
            OptimizationParameter(
                name="mcp_timeout",
                param_type="int",
                min_value=10,
                max_value=120,
                step=10
            ),
            OptimizationParameter(
                name="parallel_execution",
                param_type="boolean"
            ),
            OptimizationParameter(
                name="retry_count",
                param_type="int",
                min_value=0,
                max_value=5,
                step=1
            ),
            OptimizationParameter(
                name="step_timeout",
                param_type="int",
                min_value=30,
                max_value=300,
                step=30
            ),
            OptimizationParameter(
                name="batch_size",
                param_type="int",
                min_value=1,
                max_value=10,
                step=1
            ),
            OptimizationParameter(
                name="optimization_level",
                param_type="categorical",
                choices=["basic", "standard", "aggressive"]
            )
        ]

    async def _generate_optimization_recommendations(self,
                                                   original_recipe: RecipeDefinition,
                                                   optimized_recipe: RecipeDefinition,
                                                   best_params: Dict[str, Any],
                                                   history: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Analyze parameter changes
        significant_changes = []
        for param_name, param_value in best_params.items():
            if param_name == "agent_timeout":
                original_timeout = original_recipe.agent_requirements[0].timeout_seconds if original_recipe.agent_requirements else 300
                if abs(param_value - original_timeout) > 60:
                    significant_changes.append(f"Agent timeout changed from {original_timeout}s to {param_value}s")

            elif param_name == "parallel_execution" and param_value:
                recommendations.append("Enable parallel execution for better performance")

            elif param_name == "retry_count" and param_value > 0:
                recommendations.append(f"Set retry count to {param_value} for better reliability")

        # Analyze optimization history
        if history:
            scores = [h.get("score", h.get("value", 0)) for h in history if h.get("score") or h.get("value")]
            if scores:
                improvement_trend = np.diff(scores[-10:]) if len(scores) >= 10 else np.diff(scores)
                if np.mean(improvement_trend) > 0:
                    recommendations.append("Optimization showed consistent improvement - consider longer optimization runs")
                elif np.std(scores) < 0.1:
                    recommendations.append("Low variance in scores - consider expanding parameter search space")

        # Recipe-specific recommendations
        if len(optimized_recipe.steps) > 5:
            recommendations.append("Consider breaking down complex recipes into smaller sub-recipes")

        if len(optimized_recipe.agent_requirements) > 3:
            recommendations.append("Multiple agents detected - ensure proper coordination and resource management")

        return recommendations

    async def _store_optimization_result(self, result: OptimizationResult) -> None:
        """Store optimization result for future analysis"""
        try:
            result_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "original_recipe_id": result.original_recipe.id,
                "optimized_recipe_id": result.optimized_recipe.id,
                "best_score": result.best_score,
                "improvement_percentage": result.improvement_percentage,
                "optimization_time_seconds": result.optimization_time_seconds,
                "parameters_changed": result.parameters_changed,
                "recommendations": result.recommendations
            }

            # Save to file
            result_file = self.data_dir / f"optimization_result_{result.original_recipe.id}_{int(datetime.utcnow().timestamp())}.json"
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)

            # Add to history
            self.optimization_history.append(result_data)

            # Update best recipes
            if (result.original_recipe.id not in self.best_recipes or
                result.best_score > getattr(self.best_recipes[result.original_recipe.id], 'best_score', 0)):
                self.best_recipes[result.original_recipe.id] = result.optimized_recipe

        except Exception as e:
            logger.error(f"Failed to store optimization result: {e}")

    async def _load_optimization_history(self) -> None:
        """Load historical optimization data"""
        try:
            history_files = list(self.data_dir.glob("optimization_result_*.json"))

            for file_path in history_files:
                try:
                    with open(file_path, 'r') as f:
                        result_data = json.load(f)
                    self.optimization_history.append(result_data)
                except Exception as e:
                    logger.warning(f"Failed to load optimization history from {file_path}: {e}")

            logger.info(f"Loaded {len(self.optimization_history)} optimization history records")

        except Exception as e:
            logger.error(f"Failed to load optimization history: {e}")

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.optimization_history:
            return {"message": "No optimization history available"}

        improvements = [r["improvement_percentage"] for r in self.optimization_history]
        times = [r["optimization_time_seconds"] for r in self.optimization_history]
        scores = [r["best_score"] for r in self.optimization_history]

        return {
            "total_optimizations": len(self.optimization_history),
            "average_improvement": np.mean(improvements),
            "max_improvement": np.max(improvements),
            "average_optimization_time": np.mean(times),
            "average_score": np.mean(scores),
            "best_recipes_count": len(self.best_recipes),
            "model_trained": self.performance_model is not None
        }
