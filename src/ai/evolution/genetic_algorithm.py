"""
Genetic Algorithm for Recipe Evolution

Main orchestrator for evolutionary optimization of PyGent Factory recipes.
Combines all evolution components to evolve high-quality recipe populations.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .recipe_genome import RecipeGenome, GenomeConfig
from .fitness_functions import FitnessEvaluator, FitnessScore, FitnessObjective
from .crossover_operators import CrossoverOperatorFactory, CrossoverType, CrossoverConfig
from .mutation_operators import MutationOperatorFactory, MutationType, MutationConfig
from .selection_strategies import SelectionStrategyFactory, SelectionType, SelectionConfig, Individual
from .population_manager import PopulationManager, PopulationConfig

# Use local RecipeDefinition to avoid circular imports
from dataclasses import dataclass

@dataclass
class RecipeDefinition:
    """Local recipe definition to avoid circular imports"""
    name: str = ""
    description: str = ""
    steps: list = None
    parameters: dict = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = []
        if self.parameters is None:
            self.parameters = {}

logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    """Configuration for genetic algorithm"""
    # Population settings
    population_size: int = 50
    max_generations: int = 100
    
    # Evolution operators
    crossover_type: CrossoverType = CrossoverType.ADAPTIVE
    mutation_type: MutationType = MutationType.ADAPTIVE
    selection_type: SelectionType = SelectionType.TOURNAMENT
    
    # Evolution parameters
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_rate: float = 0.1
    
    # Fitness objectives
    fitness_objectives: List[FitnessObjective] = field(default_factory=lambda: [
        FitnessObjective.SUCCESS_RATE,
        FitnessObjective.PERFORMANCE,
        FitnessObjective.MAINTAINABILITY
    ])
    
    # Termination criteria
    target_fitness: float = 0.95
    stagnation_threshold: int = 20
    max_evaluations: int = 5000
    
    # Advanced settings
    adaptive_parameters: bool = True
    diversity_preservation: bool = True
    parallel_evaluation: bool = True
    
    # Logging and monitoring
    log_interval: int = 10
    checkpoint_interval: int = 50


@dataclass
class EvolutionResult:
    """Result of evolutionary optimization"""
    best_recipe: RecipeDefinition
    best_fitness: FitnessScore
    best_genome: List[float]
    final_population: List[Individual]
    generations_completed: int
    total_evaluations: int
    evolution_time_seconds: float
    termination_reason: str
    convergence_history: List[float]
    diversity_history: List[float]
    elite_archive: List[Individual]


class GeneticAlgorithm:
    """
    Main genetic algorithm for recipe evolution.
    
    Orchestrates the evolutionary process using configurable operators
    and strategies to evolve high-quality recipe populations.
    """
    
    def __init__(self, config: Optional[GAConfig] = None):
        self.config = config or GAConfig()
        
        # Initialize components
        self.genome_handler = RecipeGenome()
        self.fitness_evaluator = FitnessEvaluator(
            objectives=self.config.fitness_objectives
        )
        
        # Initialize evolution operators
        self.crossover_operator = CrossoverOperatorFactory.create_operator(
            self.config.crossover_type,
            CrossoverConfig(crossover_rate=self.config.crossover_rate)
        )
        
        self.mutation_operator = MutationOperatorFactory.create_operator(
            self.config.mutation_type,
            MutationConfig(mutation_rate=self.config.mutation_rate)
        )
        
        self.selection_strategy = SelectionStrategyFactory.create_strategy(
            self.config.selection_type,
            SelectionConfig(elitism_rate=self.config.elitism_rate)
        )
        
        # Initialize population manager
        self.population_manager = PopulationManager(
            PopulationConfig(
                population_size=self.config.population_size,
                max_generations=self.config.max_generations,
                stagnation_threshold=self.config.stagnation_threshold
            )
        )
        
        # Evolution state
        self.is_running = False
        self.start_time = None
        self.callbacks: List[Callable] = []
    
    async def evolve_population(self, 
                               initial_recipes: Optional[List[RecipeDefinition]] = None,
                               progress_callback: Optional[Callable] = None) -> EvolutionResult:
        """
        Evolve a population of recipes to optimize fitness.
        
        Args:
            initial_recipes: Optional seed recipes to include in initial population
            progress_callback: Optional callback for progress updates
            
        Returns:
            Evolution result with best recipe and statistics
        """
        self.start_time = time.time()
        self.is_running = True
        
        try:
            logger.info(f"Starting recipe evolution with {self.config.population_size} individuals")
            
            # Initialize population
            population = await self._initialize_population(initial_recipes)
            
            # Evolution loop
            generation = 0
            while generation < self.config.max_generations:
                # Evaluate population
                population = await self.population_manager.evaluate_population(population)
                
                # Update population manager
                self.population_manager.advance_generation(population)
                
                # Check termination criteria
                should_terminate, reason = self._check_termination_criteria()
                if should_terminate:
                    logger.info(f"Evolution terminated: {reason}")
                    break
                
                # Generate next generation
                population = await self._generate_next_generation(population)
                
                # Progress callback
                if progress_callback and generation % self.config.log_interval == 0:
                    await self._call_progress_callback(progress_callback, generation)
                
                # Adaptive parameter adjustment
                if self.config.adaptive_parameters:
                    self._adjust_parameters(generation)
                
                generation += 1
            
            # Final evaluation
            population = await self.population_manager.evaluate_population(population)
            
            # Create result
            result = self._create_evolution_result(population, generation, reason)
            
            logger.info(f"Evolution completed: {generation} generations, "
                       f"best fitness: {result.best_fitness.total_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            raise
        finally:
            self.is_running = False
    
    async def _initialize_population(self, 
                                   initial_recipes: Optional[List[RecipeDefinition]]) -> List[Individual]:
        """Initialize the population with optional seed recipes"""
        population = self.population_manager.initialize_population()
        
        # Include seed recipes if provided
        if initial_recipes:
            for i, recipe in enumerate(initial_recipes[:len(population)]):
                # Encode recipe to genome
                genome = self.genome_handler.encode_recipe(recipe)
                
                # Replace random individual with seed
                population[i] = Individual(
                    genome=genome,
                    fitness=FitnessScore(total_score=0.0),
                    age=0,
                    id=f"seed_{i}"
                )
            
            logger.info(f"Included {len(initial_recipes)} seed recipes in population")
        
        return population
    
    async def _generate_next_generation(self, population: List[Individual]) -> List[Individual]:
        """Generate the next generation through selection, crossover, and mutation"""
        
        # Select parents for reproduction
        num_offspring = self.config.population_size - int(self.config.population_size * self.config.elitism_rate)
        parents = self.selection_strategy.select_parents(population, num_offspring)
        
        # Generate offspring through crossover and mutation
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            # Crossover
            child1_genome, child2_genome = self.crossover_operator.crossover(
                parent1.genome, parent2.genome
            )
            
            # Mutation
            child1_genome = self.mutation_operator.mutate(child1_genome)
            child2_genome = self.mutation_operator.mutate(child2_genome)
            
            # Create offspring individuals
            child1 = Individual(
                genome=child1_genome,
                fitness=FitnessScore(total_score=0.0),
                age=0,
                parent_ids=[parent1.id, parent2.id],
                id=f"gen{self.population_manager.generation + 1}_off{len(offspring)}"
            )
            
            child2 = Individual(
                genome=child2_genome,
                fitness=FitnessScore(total_score=0.0),
                age=0,
                parent_ids=[parent1.id, parent2.id],
                id=f"gen{self.population_manager.generation + 1}_off{len(offspring) + 1}"
            )
            
            offspring.extend([child1, child2])
        
        # Trim to exact size needed
        offspring = offspring[:num_offspring]
        
        # Select survivors for next generation
        next_generation = self.selection_strategy.select_survivors(
            population, offspring, self.config.population_size
        )
        
        logger.debug(f"Generated {len(offspring)} offspring, "
                    f"selected {len(next_generation)} survivors")
        
        return next_generation
    
    def _check_termination_criteria(self) -> Tuple[bool, str]:
        """Check if evolution should terminate"""
        # Check population manager termination
        should_terminate, reason = self.population_manager.should_terminate()
        if should_terminate:
            return True, reason
        
        # Check target fitness
        if self.population_manager.current_population:
            best_fitness = max(ind.fitness.total_score 
                             for ind in self.population_manager.current_population)
            if best_fitness >= self.config.target_fitness:
                return True, "target_fitness_reached"
        
        # Check maximum evaluations
        if self.population_manager.total_evaluations >= self.config.max_evaluations:
            return True, "max_evaluations_reached"
        
        return False, "continue"
    
    def _adjust_parameters(self, generation: int):
        """Adjust evolution parameters adaptively"""
        # Adjust mutation rate based on diversity
        if hasattr(self.population_manager, 'diversity_history') and self.population_manager.diversity_history:
            current_diversity = self.population_manager.diversity_history[-1]
            
            if current_diversity < 0.2:  # Low diversity
                # Increase mutation rate
                if hasattr(self.mutation_operator.config, 'mutation_rate'):
                    self.mutation_operator.config.mutation_rate = min(0.3, 
                        self.mutation_operator.config.mutation_rate * 1.1)
            elif current_diversity > 0.8:  # High diversity
                # Decrease mutation rate
                if hasattr(self.mutation_operator.config, 'mutation_rate'):
                    self.mutation_operator.config.mutation_rate = max(0.01,
                        self.mutation_operator.config.mutation_rate * 0.9)
        
        # Adjust selection pressure over time
        if hasattr(self.selection_strategy.config, 'tournament_size'):
            # Increase selection pressure in later generations
            progress = generation / self.config.max_generations
            base_size = 3
            max_size = 7
            self.selection_strategy.config.tournament_size = int(
                base_size + (max_size - base_size) * progress
            )
    
    async def _call_progress_callback(self, callback: Callable, generation: int):
        """Call progress callback with current statistics"""
        try:
            stats = self.population_manager.get_population_summary()
            stats['generation'] = generation
            stats['elapsed_time'] = time.time() - self.start_time
            
            if asyncio.iscoroutinefunction(callback):
                await callback(stats)
            else:
                callback(stats)
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")
    
    def _create_evolution_result(self, final_population: List[Individual], 
                               generations: int, termination_reason: str) -> EvolutionResult:
        """Create evolution result from final state"""
        # Get best individual
        best_individual = self.population_manager.get_best_individual(final_population)
        
        # Decode best genome to recipe
        best_recipe = self.genome_handler.decode_genome(best_individual.genome)
        
        # Extract history data
        convergence_history = [stats.best_fitness for stats in self.population_manager.population_history]
        diversity_history = self.population_manager.diversity_history.copy()
        
        result = EvolutionResult(
            best_recipe=best_recipe,
            best_fitness=best_individual.fitness,
            best_genome=best_individual.genome,
            final_population=final_population,
            generations_completed=generations,
            total_evaluations=self.population_manager.total_evaluations,
            evolution_time_seconds=time.time() - self.start_time,
            termination_reason=termination_reason,
            convergence_history=convergence_history,
            diversity_history=diversity_history,
            elite_archive=self.population_manager.elite_archive.copy()
        )
        
        return result
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get current evolution status"""
        if not self.is_running:
            return {"status": "not_running"}
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        status = {
            "status": "running",
            "generation": self.population_manager.generation,
            "elapsed_time": elapsed_time,
            "total_evaluations": self.population_manager.total_evaluations,
            "population_summary": self.population_manager.get_population_summary()
        }
        
        return status
    
    def stop_evolution(self):
        """Stop the evolution process"""
        self.is_running = False
        logger.info("Evolution stop requested")
    
    def add_progress_callback(self, callback: Callable):
        """Add a progress callback function"""
        self.callbacks.append(callback)
    
    def export_evolution_data(self, filepath: str):
        """Export evolution data for analysis"""
        self.population_manager.export_population_data(filepath)


async def test_genetic_algorithm():
    """Test function for genetic algorithm"""
    # Create test configuration
    config = GAConfig(
        population_size=20,
        max_generations=10,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    # Create genetic algorithm
    ga = GeneticAlgorithm(config)
    
    # Progress callback
    async def progress_callback(stats):
        print(f"Generation {stats['generation']}: "
              f"Best={stats['current_stats']['best_fitness']:.3f}, "
              f"Avg={stats['current_stats']['average_fitness']:.3f}")
    
    # Run evolution
    result = await ga.evolve_population(progress_callback=progress_callback)
    
    print(f"\nEvolution completed:")
    print(f"Best fitness: {result.best_fitness.total_score:.3f}")
    print(f"Generations: {result.generations_completed}")
    print(f"Evaluations: {result.total_evaluations}")
    print(f"Time: {result.evolution_time_seconds:.1f}s")
    print(f"Termination: {result.termination_reason}")


if __name__ == "__main__":
    asyncio.run(test_genetic_algorithm())
