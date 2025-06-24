"""
Advanced Recipe Evolution System

Combines genetic algorithms, Tree of Thought reasoning, s3 RAG retrieval,
and GPU vector search for intelligent recipe optimization and evolution.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from ..ai.reasoning.unified_pipeline import UnifiedReasoningPipeline, UnifiedConfig, ReasoningMode
from ..ai.reasoning.tot.tasks.recipe_optimization import RecipeOptimizationTask

logger = logging.getLogger(__name__)


class EvolutionStrategy(Enum):
    """Strategy for recipe evolution"""
    GENETIC_ONLY = "genetic_only"           # Pure genetic algorithm
    TOT_GUIDED = "tot_guided"               # ToT-guided evolution
    RAG_INFORMED = "rag_informed"           # RAG-informed evolution
    HYBRID = "hybrid"                       # All techniques combined
    ADAPTIVE = "adaptive"                   # Adaptive strategy selection


class FitnessMetric(Enum):
    """Metrics for evaluating recipe fitness"""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    EFFICIENCY = "efficiency"
    INNOVATION = "innovation"
    COMPOSITE = "composite"


@dataclass
class Recipe:
    """Represents a recipe in the evolution system"""
    id: str
    name: str
    description: str
    steps: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Evolution metadata
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    execution_time: float = 0.0
    success_rate: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    complexity_score: float = 0.0
    innovation_score: float = 0.0
    maintainability_score: float = 0.0
    
    def get_composite_fitness(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate composite fitness score"""
        if not self.fitness_scores:
            return 0.0
        
        default_weights = {
            'performance': 0.3,
            'reliability': 0.25,
            'maintainability': 0.2,
            'efficiency': 0.15,
            'innovation': 0.1
        }
        
        weights = weights or default_weights
        
        composite = 0.0
        total_weight = 0.0
        
        for metric, score in self.fitness_scores.items():
            weight = weights.get(metric, 0.0)
            composite += score * weight
            total_weight += weight
        
        return composite / max(total_weight, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'steps': self.steps,
            'parameters': self.parameters,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'fitness_scores': self.fitness_scores,
            'execution_time': self.execution_time,
            'success_rate': self.success_rate,
            'resource_usage': self.resource_usage,
            'complexity_score': self.complexity_score,
            'innovation_score': self.innovation_score,
            'maintainability_score': self.maintainability_score,
            'composite_fitness': self.get_composite_fitness()
        }


@dataclass
class EvolutionConfig:
    """Configuration for recipe evolution"""
    # Evolution parameters
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_rate: float = 0.1
    
    # Strategy settings
    evolution_strategy: EvolutionStrategy = EvolutionStrategy.HYBRID
    fitness_metric: FitnessMetric = FitnessMetric.COMPOSITE
    
    # Reasoning integration
    use_tot_reasoning: bool = True
    use_rag_retrieval: bool = True
    use_vector_search: bool = True
    
    # Quality thresholds
    min_fitness_threshold: float = 0.6
    convergence_threshold: float = 0.01
    max_stagnation_generations: int = 10
    
    # Performance settings
    max_evolution_time: float = 3600.0  # 1 hour max
    parallel_evaluation: bool = True
    max_concurrent_evaluations: int = 10


class AdvancedRecipeEvolution:
    """
    Advanced Recipe Evolution System
    
    Uses genetic algorithms enhanced with Tree of Thought reasoning,
    s3 RAG retrieval, and GPU vector search for intelligent recipe optimization.
    """
    
    def __init__(self, config: EvolutionConfig, reasoning_pipeline: UnifiedReasoningPipeline):
        self.config = config
        self.reasoning_pipeline = reasoning_pipeline
        
        # Evolution state
        self.current_generation = 0
        self.population: List[Recipe] = []
        self.fitness_history: List[Dict[str, float]] = []
        self.best_recipes: List[Recipe] = []
        
        # Performance tracking
        self.evolution_start_time = 0.0
        self.generation_times: List[float] = []
        self.evaluation_count = 0
        
        logger.info(f"Advanced recipe evolution initialized with strategy: {config.evolution_strategy.value}")
    
    async def evolve_recipes(self, initial_recipes: List[Recipe], 
                           target_objectives: List[str],
                           constraints: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evolve recipes using advanced AI techniques
        
        Args:
            initial_recipes: Starting recipe population
            target_objectives: Optimization objectives
            constraints: Evolution constraints
            
        Returns:
            Evolution results with best recipes and metrics
        """
        logger.info(f"Starting recipe evolution with {len(initial_recipes)} initial recipes")
        
        self.evolution_start_time = time.time()
        
        try:
            # Initialize population
            await self._initialize_population(initial_recipes, target_objectives)
            
            # Evolution loop
            for generation in range(self.config.max_generations):
                self.current_generation = generation
                generation_start = time.time()
                
                logger.info(f"Generation {generation + 1}/{self.config.max_generations}")
                
                # Evaluate population fitness
                await self._evaluate_population(target_objectives, constraints)
                
                # Check convergence
                if await self._check_convergence():
                    logger.info(f"Convergence reached at generation {generation + 1}")
                    break
                
                # Create next generation
                await self._create_next_generation(target_objectives, constraints)
                
                # Track generation time
                generation_time = time.time() - generation_start
                self.generation_times.append(generation_time)
                
                # Log progress
                best_fitness = max(recipe.get_composite_fitness() for recipe in self.population)
                avg_fitness = sum(recipe.get_composite_fitness() for recipe in self.population) / len(self.population)
                
                logger.info(f"Generation {generation + 1} completed in {generation_time:.2f}s")
                logger.info(f"Best fitness: {best_fitness:.3f}, Average: {avg_fitness:.3f}")
                
                # Check time limit
                if time.time() - self.evolution_start_time > self.config.max_evolution_time:
                    logger.warning("Evolution time limit reached")
                    break
            
            # Finalize results
            return await self._finalize_evolution_results()
            
        except Exception as e:
            logger.error(f"Recipe evolution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'evolution_time': time.time() - self.evolution_start_time
            }
    
    async def _initialize_population(self, initial_recipes: List[Recipe], 
                                   target_objectives: List[str]):
        """Initialize the evolution population"""
        
        # Start with provided recipes
        self.population = initial_recipes.copy()
        
        # Generate additional recipes if needed
        while len(self.population) < self.config.population_size:
            # Use reasoning to generate new recipe variants
            if self.config.use_tot_reasoning and len(self.population) > 0:
                base_recipe = np.random.choice(self.population)
                new_recipe = await self._generate_recipe_variant(base_recipe, target_objectives)
                if new_recipe:
                    self.population.append(new_recipe)
            else:
                # Create random variations
                if initial_recipes:
                    base_recipe = np.random.choice(initial_recipes)
                    variant = await self._create_random_variant(base_recipe)
                    self.population.append(variant)
        
        # Ensure population size
        self.population = self.population[:self.config.population_size]
        
        logger.info(f"Population initialized with {len(self.population)} recipes")
    
    async def _generate_recipe_variant(self, base_recipe: Recipe, 
                                     target_objectives: List[str]) -> Optional[Recipe]:
        """Generate a recipe variant using AI reasoning"""
        
        try:
            # Create optimization query
            objectives_text = ", ".join(target_objectives)
            query = f"""
            Optimize this recipe for: {objectives_text}
            
            Current recipe: {base_recipe.name}
            Description: {base_recipe.description}
            Steps: {'; '.join(base_recipe.steps)}
            
            Generate an improved variant that addresses the optimization objectives.
            """
            
            # Use reasoning pipeline to generate variant
            result = await self.reasoning_pipeline.reason(
                query,
                mode=ReasoningMode.TOT_ENHANCED_RAG,
                context={'base_recipe': base_recipe.to_dict()}
            )
            
            if result.success and result.confidence_score > 0.6:
                # Parse the result into a new recipe
                variant = Recipe(
                    id=f"{base_recipe.id}_variant_{int(time.time())}",
                    name=f"{base_recipe.name} (Optimized)",
                    description=result.response,
                    steps=base_recipe.steps.copy(),  # Would parse from result in real implementation
                    parameters=base_recipe.parameters.copy(),
                    generation=self.current_generation,
                    parent_ids=[base_recipe.id]
                )
                
                return variant
            
        except Exception as e:
            logger.error(f"Failed to generate recipe variant: {e}")
        
        return None
    
    async def _create_random_variant(self, base_recipe: Recipe) -> Recipe:
        """Create a random variant of a recipe"""
        
        variant = Recipe(
            id=f"{base_recipe.id}_random_{int(time.time())}",
            name=f"{base_recipe.name} (Variant)",
            description=base_recipe.description,
            steps=base_recipe.steps.copy(),
            parameters=base_recipe.parameters.copy(),
            generation=self.current_generation,
            parent_ids=[base_recipe.id]
        )
        
        # Apply random mutations
        if np.random.random() < self.config.mutation_rate:
            # Mutate parameters
            for key, value in variant.parameters.items():
                if isinstance(value, (int, float)):
                    mutation_factor = 1.0 + np.random.normal(0, 0.1)
                    variant.parameters[key] = value * mutation_factor
        
        return variant
    
    async def _evaluate_population(self, target_objectives: List[str], 
                                 constraints: Optional[List[str]]):
        """Evaluate fitness of all recipes in population"""
        
        if self.config.parallel_evaluation:
            # Parallel evaluation
            semaphore = asyncio.Semaphore(self.config.max_concurrent_evaluations)
            
            async def evaluate_recipe(recipe):
                async with semaphore:
                    return await self._evaluate_recipe_fitness(recipe, target_objectives, constraints)
            
            tasks = [evaluate_recipe(recipe) for recipe in self.population]
            await asyncio.gather(*tasks)
        else:
            # Sequential evaluation
            for recipe in self.population:
                await self._evaluate_recipe_fitness(recipe, target_objectives, constraints)
        
        # Update fitness history
        generation_fitness = {
            'generation': self.current_generation,
            'best_fitness': max(recipe.get_composite_fitness() for recipe in self.population),
            'average_fitness': sum(recipe.get_composite_fitness() for recipe in self.population) / len(self.population),
            'worst_fitness': min(recipe.get_composite_fitness() for recipe in self.population)
        }
        self.fitness_history.append(generation_fitness)
    
    async def _evaluate_recipe_fitness(self, recipe: Recipe, target_objectives: List[str],
                                     constraints: Optional[List[str]]):
        """Evaluate fitness of a single recipe"""
        
        self.evaluation_count += 1
        
        try:
            # Use reasoning to evaluate recipe quality
            objectives_text = ", ".join(target_objectives)
            constraints_text = ", ".join(constraints) if constraints else "None"
            
            evaluation_query = f"""
            Evaluate this recipe against the following criteria:
            
            Objectives: {objectives_text}
            Constraints: {constraints_text}
            
            Recipe: {recipe.name}
            Description: {recipe.description}
            Steps: {'; '.join(recipe.steps)}
            
            Rate the recipe on:
            1. Performance (0.0-1.0)
            2. Reliability (0.0-1.0)
            3. Maintainability (0.0-1.0)
            4. Efficiency (0.0-1.0)
            5. Innovation (0.0-1.0)
            
            Provide scores for each metric.
            """
            
            result = await self.reasoning_pipeline.reason(
                evaluation_query,
                mode=ReasoningMode.TOT_ENHANCED_RAG,
                context={'recipe': recipe.to_dict()}
            )
            
            if result.success:
                # Parse fitness scores from result (simplified)
                recipe.fitness_scores = self._parse_fitness_scores(result.response)
            else:
                # Default scores if evaluation fails
                recipe.fitness_scores = {
                    'performance': 0.5,
                    'reliability': 0.5,
                    'maintainability': 0.5,
                    'efficiency': 0.5,
                    'innovation': 0.5
                }
            
        except Exception as e:
            logger.error(f"Recipe evaluation failed: {e}")
            # Default scores on error
            recipe.fitness_scores = {metric.value: 0.3 for metric in FitnessMetric if metric != FitnessMetric.COMPOSITE}
    
    def _parse_fitness_scores(self, evaluation_text: str) -> Dict[str, float]:
        """Parse fitness scores from evaluation text (simplified)"""
        
        # Simple parsing - in practice would use more sophisticated NLP
        scores = {}
        metrics = ['performance', 'reliability', 'maintainability', 'efficiency', 'innovation']
        
        for metric in metrics:
            # Look for patterns like "Performance: 0.8" or "Performance (0.8)"
            import re
            pattern = rf"{metric}[:\s\(]*([0-9]*\.?[0-9]+)"
            match = re.search(pattern, evaluation_text.lower())
            
            if match:
                try:
                    score = float(match.group(1))
                    scores[metric] = max(0.0, min(1.0, score))
                except ValueError:
                    scores[metric] = 0.5
            else:
                scores[metric] = 0.5
        
        return scores
    
    async def _check_convergence(self) -> bool:
        """Check if evolution has converged"""
        
        if len(self.fitness_history) < 5:
            return False
        
        # Check fitness improvement over last few generations
        recent_fitness = [gen['best_fitness'] for gen in self.fitness_history[-5:]]
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        if fitness_improvement < self.config.convergence_threshold:
            return True
        
        # Check stagnation
        if len(self.fitness_history) >= self.config.max_stagnation_generations:
            recent_best = [gen['best_fitness'] for gen in self.fitness_history[-self.config.max_stagnation_generations:]]
            if max(recent_best) - min(recent_best) < self.config.convergence_threshold:
                return True
        
        return False
    
    async def _create_next_generation(self, target_objectives: List[str], 
                                    constraints: Optional[List[str]]):
        """Create the next generation of recipes"""
        
        # Sort population by fitness
        self.population.sort(key=lambda r: r.get_composite_fitness(), reverse=True)
        
        # Keep elite recipes
        elite_count = int(self.config.population_size * self.config.elitism_rate)
        next_generation = self.population[:elite_count].copy()
        
        # Generate offspring
        while len(next_generation) < self.config.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                offspring = await self._crossover(parent1, parent2, target_objectives)
            else:
                offspring = parent1 if np.random.random() < 0.5 else parent2
            
            # Mutation
            if np.random.random() < self.config.mutation_rate:
                offspring = await self._mutate(offspring, target_objectives)
            
            next_generation.append(offspring)
        
        # Update population
        self.population = next_generation[:self.config.population_size]
        
        # Update generation counter
        for recipe in self.population[elite_count:]:
            recipe.generation = self.current_generation + 1
    
    def _tournament_selection(self, tournament_size: int = 3) -> Recipe:
        """Select a recipe using tournament selection"""
        
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        return max(tournament, key=lambda r: r.get_composite_fitness())
    
    async def _crossover(self, parent1: Recipe, parent2: Recipe, 
                        target_objectives: List[str]) -> Recipe:
        """Create offspring through crossover"""
        
        # Use reasoning to intelligently combine parents
        if self.config.use_tot_reasoning:
            try:
                crossover_query = f"""
                Combine the best features of these two recipes to create an improved version:
                
                Recipe 1: {parent1.name}
                Description: {parent1.description}
                Steps: {'; '.join(parent1.steps)}
                
                Recipe 2: {parent2.name}
                Description: {parent2.description}
                Steps: {'; '.join(parent2.steps)}
                
                Objectives: {', '.join(target_objectives)}
                
                Create a hybrid recipe that combines the strengths of both parents.
                """
                
                result = await self.reasoning_pipeline.reason(
                    crossover_query,
                    mode=ReasoningMode.TOT_ONLY,
                    context={'parent1': parent1.to_dict(), 'parent2': parent2.to_dict()}
                )
                
                if result.success and result.confidence_score > 0.6:
                    offspring = Recipe(
                        id=f"crossover_{int(time.time())}",
                        name=f"Hybrid of {parent1.name} and {parent2.name}",
                        description=result.response,
                        steps=parent1.steps + parent2.steps,  # Simplified
                        parameters={**parent1.parameters, **parent2.parameters},
                        generation=self.current_generation + 1,
                        parent_ids=[parent1.id, parent2.id]
                    )
                    return offspring
                    
            except Exception as e:
                logger.error(f"Intelligent crossover failed: {e}")
        
        # Fallback to simple crossover
        offspring = Recipe(
            id=f"crossover_{int(time.time())}",
            name=f"Hybrid of {parent1.name} and {parent2.name}",
            description=f"Combination of {parent1.name} and {parent2.name}",
            steps=parent1.steps[:len(parent1.steps)//2] + parent2.steps[len(parent2.steps)//2:],
            parameters={**parent1.parameters, **parent2.parameters},
            generation=self.current_generation + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return offspring
    
    async def _mutate(self, recipe: Recipe, target_objectives: List[str]) -> Recipe:
        """Mutate a recipe"""
        
        # Use reasoning for intelligent mutation
        if self.config.use_tot_reasoning:
            try:
                mutation_query = f"""
                Improve this recipe with a small modification:
                
                Recipe: {recipe.name}
                Description: {recipe.description}
                Steps: {'; '.join(recipe.steps)}
                
                Objectives: {', '.join(target_objectives)}
                
                Suggest a minor improvement or modification.
                """
                
                result = await self.reasoning_pipeline.reason(
                    mutation_query,
                    mode=ReasoningMode.TOT_ONLY,
                    context={'recipe': recipe.to_dict()}
                )
                
                if result.success:
                    mutated = Recipe(
                        id=f"mutated_{int(time.time())}",
                        name=f"{recipe.name} (Improved)",
                        description=result.response,
                        steps=recipe.steps.copy(),
                        parameters=recipe.parameters.copy(),
                        generation=recipe.generation,
                        parent_ids=[recipe.id]
                    )
                    return mutated
                    
            except Exception as e:
                logger.error(f"Intelligent mutation failed: {e}")
        
        # Fallback to random mutation
        mutated = Recipe(
            id=f"mutated_{int(time.time())}",
            name=f"{recipe.name} (Mutated)",
            description=recipe.description,
            steps=recipe.steps.copy(),
            parameters=recipe.parameters.copy(),
            generation=recipe.generation,
            parent_ids=[recipe.id]
        )
        
        # Random parameter mutation
        for key, value in mutated.parameters.items():
            if isinstance(value, (int, float)) and np.random.random() < 0.3:
                mutation_factor = 1.0 + np.random.normal(0, 0.2)
                mutated.parameters[key] = value * mutation_factor
        
        return mutated
    
    async def _finalize_evolution_results(self) -> Dict[str, Any]:
        """Finalize and return evolution results"""
        
        # Sort final population by fitness
        self.population.sort(key=lambda r: r.get_composite_fitness(), reverse=True)
        
        # Get best recipes
        best_recipes = self.population[:10]  # Top 10
        
        # Calculate statistics
        total_time = time.time() - self.evolution_start_time
        avg_generation_time = sum(self.generation_times) / max(len(self.generation_times), 1)
        
        results = {
            'success': True,
            'total_time': total_time,
            'generations_completed': self.current_generation + 1,
            'evaluations_performed': self.evaluation_count,
            'average_generation_time': avg_generation_time,
            'best_recipes': [recipe.to_dict() for recipe in best_recipes],
            'fitness_history': self.fitness_history,
            'final_population_size': len(self.population),
            'convergence_achieved': len(self.fitness_history) < self.config.max_generations
        }
        
        logger.info(f"Evolution completed: {self.current_generation + 1} generations, "
                   f"best fitness: {best_recipes[0].get_composite_fitness():.3f}")
        
        return results
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics"""
        
        if not self.population:
            return {}
        
        fitness_scores = [recipe.get_composite_fitness() for recipe in self.population]
        
        return {
            'current_generation': self.current_generation,
            'population_size': len(self.population),
            'evaluations_performed': self.evaluation_count,
            'best_fitness': max(fitness_scores),
            'average_fitness': sum(fitness_scores) / len(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'fitness_std': np.std(fitness_scores),
            'generations_completed': len(self.fitness_history),
            'evolution_time': time.time() - self.evolution_start_time if self.evolution_start_time > 0 else 0,
            'strategy': self.config.evolution_strategy.value
        }
