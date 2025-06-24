"""
Population Manager for Recipe Evolution

Manages the population lifecycle, diversity, and statistics
for genetic algorithm optimization of recipes.
"""

import logging
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from .recipe_genome import RecipeGenome
from .fitness_functions import FitnessEvaluator, FitnessScore
from .selection_strategies import Individual

logger = logging.getLogger(__name__)


@dataclass
class PopulationStats:
    """Statistics about the population"""
    generation: int
    population_size: int
    best_fitness: float
    worst_fitness: float
    average_fitness: float
    fitness_variance: float
    diversity_score: float
    convergence_rate: float
    stagnation_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PopulationConfig:
    """Configuration for population management"""
    population_size: int = 50
    max_generations: int = 100
    diversity_threshold: float = 0.1
    stagnation_threshold: int = 10
    elite_preservation: bool = True
    diversity_preservation: bool = True
    migration_rate: float = 0.05
    archive_size: int = 100


class PopulationManager:
    """
    Manages the population of recipe genomes throughout evolution.
    
    Handles population initialization, diversity monitoring, statistics tracking,
    and population-level operations like migration and archiving.
    """
    
    def __init__(self, config: Optional[PopulationConfig] = None):
        self.config = config or PopulationConfig()
        self.genome_handler = RecipeGenome()
        self.fitness_evaluator = FitnessEvaluator()
        
        # Population state
        self.current_population: List[Individual] = []
        self.generation = 0
        self.population_history: List[PopulationStats] = []
        self.elite_archive: List[Individual] = []
        
        # Diversity tracking
        self.diversity_history: List[float] = []
        self.stagnation_count = 0
        self.last_best_fitness = 0.0
        
        # Statistics
        self.total_evaluations = 0
        self.convergence_data = []
    
    def initialize_population(self, population_size: Optional[int] = None) -> List[Individual]:
        """
        Initialize a random population of recipe genomes.
        
        Args:
            population_size: Size of population to create
            
        Returns:
            Initialized population
        """
        size = population_size or self.config.population_size
        
        population = []
        for i in range(size):
            # Generate random genome
            genome = [random.random() for _ in range(self.genome_handler.genome_length)]
            
            # Create individual with placeholder fitness
            individual = Individual(
                genome=genome,
                fitness=FitnessScore(total_score=0.0),
                age=0,
                id=f"gen0_ind{i}"
            )
            
            population.append(individual)
        
        self.current_population = population
        self.generation = 0
        
        logger.info(f"Initialized population with {len(population)} individuals")
        return population
    
    async def evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """
        Evaluate fitness for all individuals in the population.
        
        Args:
            population: Population to evaluate
            
        Returns:
            Population with updated fitness scores
        """
        for individual in population:
            if individual.fitness.total_score == 0.0:  # Not yet evaluated
                # Decode genome to recipe for evaluation
                recipe = self.genome_handler.decode_genome(individual.genome)
                
                # Calculate fitness
                fitness = await self.fitness_evaluator.calculate_fitness(recipe)
                individual.fitness = fitness
                
                self.total_evaluations += 1
        
        logger.debug(f"Evaluated {len(population)} individuals")
        return population
    
    def calculate_population_diversity(self, population: List[Individual]) -> float:
        """
        Calculate population diversity based on genome similarity.
        
        Args:
            population: Population to analyze
            
        Returns:
            Diversity score (0-1, higher = more diverse)
        """
        if len(population) < 2:
            return 1.0
        
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                similarity = self.genome_handler.calculate_genome_similarity(
                    population[i].genome, population[j].genome
                )
                distance = 1.0 - similarity
                total_distance += distance
                comparisons += 1
        
        if comparisons == 0:
            return 1.0
        
        average_distance = total_distance / comparisons
        return average_distance
    
    def calculate_population_stats(self, population: List[Individual]) -> PopulationStats:
        """
        Calculate comprehensive statistics for the population.
        
        Args:
            population: Population to analyze
            
        Returns:
            Population statistics
        """
        fitness_scores = [ind.fitness.total_score for ind in population]
        
        best_fitness = max(fitness_scores) if fitness_scores else 0.0
        worst_fitness = min(fitness_scores) if fitness_scores else 0.0
        average_fitness = np.mean(fitness_scores) if fitness_scores else 0.0
        fitness_variance = np.var(fitness_scores) if fitness_scores else 0.0
        
        diversity_score = self.calculate_population_diversity(population)
        
        # Calculate convergence rate
        convergence_rate = 0.0
        if len(self.population_history) > 0:
            prev_avg = self.population_history[-1].average_fitness
            convergence_rate = (average_fitness - prev_avg) / max(abs(prev_avg), 1e-6)
        
        # Update stagnation count
        if abs(best_fitness - self.last_best_fitness) < 1e-6:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            self.last_best_fitness = best_fitness
        
        stats = PopulationStats(
            generation=self.generation,
            population_size=len(population),
            best_fitness=best_fitness,
            worst_fitness=worst_fitness,
            average_fitness=average_fitness,
            fitness_variance=fitness_variance,
            diversity_score=diversity_score,
            convergence_rate=convergence_rate,
            stagnation_count=self.stagnation_count
        )
        
        return stats
    
    def update_population_history(self, population: List[Individual]):
        """Update population history and statistics"""
        stats = self.calculate_population_stats(population)
        self.population_history.append(stats)
        
        # Update diversity history
        self.diversity_history.append(stats.diversity_score)
        
        # Keep limited history
        if len(self.population_history) > 1000:
            self.population_history = self.population_history[-1000:]
        
        if len(self.diversity_history) > 1000:
            self.diversity_history = self.diversity_history[-1000:]
        
        logger.debug(f"Generation {self.generation}: "
                    f"Best={stats.best_fitness:.3f}, "
                    f"Avg={stats.average_fitness:.3f}, "
                    f"Diversity={stats.diversity_score:.3f}")
    
    def update_elite_archive(self, population: List[Individual]):
        """Update elite archive with best individuals"""
        if not self.config.elite_preservation:
            return
        
        # Add current best individuals to archive
        sorted_population = sorted(population, 
                                 key=lambda ind: ind.fitness.total_score, 
                                 reverse=True)
        
        # Add top individuals to archive
        for individual in sorted_population[:5]:  # Top 5
            # Check if already in archive
            if not any(ind.id == individual.id for ind in self.elite_archive):
                self.elite_archive.append(individual)
        
        # Sort archive by fitness
        self.elite_archive.sort(key=lambda ind: ind.fitness.total_score, reverse=True)
        
        # Limit archive size
        if len(self.elite_archive) > self.config.archive_size:
            self.elite_archive = self.elite_archive[:self.config.archive_size]
        
        logger.debug(f"Elite archive updated: {len(self.elite_archive)} individuals")
    
    def check_diversity_crisis(self, population: List[Individual]) -> bool:
        """Check if population is experiencing diversity crisis"""
        diversity = self.calculate_population_diversity(population)
        return diversity < self.config.diversity_threshold
    
    def inject_diversity(self, population: List[Individual], 
                        injection_rate: float = 0.2) -> List[Individual]:
        """Inject diversity into population to combat convergence"""
        if not self.config.diversity_preservation:
            return population
        
        injection_count = int(len(population) * injection_rate)
        
        # Replace worst individuals with random ones
        sorted_population = sorted(population, 
                                 key=lambda ind: ind.fitness.total_score)
        
        for i in range(min(injection_count, len(sorted_population))):
            # Generate new random genome
            new_genome = [random.random() for _ in range(self.genome_handler.genome_length)]
            
            # Create new individual
            new_individual = Individual(
                genome=new_genome,
                fitness=FitnessScore(total_score=0.0),  # Will be evaluated later
                age=0,
                id=f"gen{self.generation}_diverse{i}"
            )
            
            # Replace worst individual
            sorted_population[i] = new_individual
        
        logger.info(f"Injected {injection_count} diverse individuals")
        return sorted_population
    
    def migrate_individuals(self, population: List[Individual]) -> List[Individual]:
        """Perform migration between populations (for future multi-population support)"""
        # Placeholder for migration logic
        # In a multi-population setup, this would exchange individuals between populations
        return population
    
    def advance_generation(self, new_population: List[Individual]):
        """Advance to the next generation"""
        self.current_population = new_population
        self.generation += 1
        
        # Age all individuals
        for individual in self.current_population:
            individual.age += 1
        
        # Update statistics
        self.update_population_history(self.current_population)
        self.update_elite_archive(self.current_population)
        
        # Check for diversity crisis
        if self.check_diversity_crisis(self.current_population):
            logger.warning(f"Diversity crisis detected at generation {self.generation}")
            self.current_population = self.inject_diversity(self.current_population)
    
    def get_best_individual(self, population: Optional[List[Individual]] = None) -> Individual:
        """Get the best individual from population or current population"""
        pop = population or self.current_population
        if not pop:
            raise ValueError("No population available")
        
        return max(pop, key=lambda ind: ind.fitness.total_score)
    
    def get_population_summary(self) -> Dict[str, Any]:
        """Get comprehensive population summary"""
        if not self.population_history:
            return {"error": "No population history available"}
        
        latest_stats = self.population_history[-1]
        best_individual = self.get_best_individual()
        
        summary = {
            "generation": self.generation,
            "population_size": len(self.current_population),
            "total_evaluations": self.total_evaluations,
            "current_stats": {
                "best_fitness": latest_stats.best_fitness,
                "average_fitness": latest_stats.average_fitness,
                "diversity": latest_stats.diversity_score,
                "stagnation_count": latest_stats.stagnation_count
            },
            "best_individual": {
                "id": best_individual.id,
                "fitness": best_individual.fitness.total_score,
                "age": best_individual.age
            },
            "elite_archive_size": len(self.elite_archive),
            "convergence_trend": self._calculate_convergence_trend()
        }
        
        return summary
    
    def _calculate_convergence_trend(self) -> str:
        """Calculate convergence trend from recent history"""
        if len(self.population_history) < 5:
            return "insufficient_data"
        
        recent_fitness = [stats.average_fitness for stats in self.population_history[-5:]]
        
        # Calculate trend
        improvements = sum(1 for i in range(1, len(recent_fitness)) 
                         if recent_fitness[i] > recent_fitness[i-1])
        
        if improvements >= 3:
            return "improving"
        elif improvements <= 1:
            return "stagnating"
        else:
            return "stable"
    
    def export_population_data(self, filepath: str):
        """Export population data to file"""
        data = {
            "generation": self.generation,
            "population_size": len(self.current_population),
            "total_evaluations": self.total_evaluations,
            "population_history": [
                {
                    "generation": stats.generation,
                    "best_fitness": stats.best_fitness,
                    "average_fitness": stats.average_fitness,
                    "diversity": stats.diversity_score,
                    "timestamp": stats.timestamp.isoformat()
                }
                for stats in self.population_history
            ],
            "elite_archive": [
                {
                    "id": ind.id,
                    "fitness": ind.fitness.total_score,
                    "age": ind.age
                }
                for ind in self.elite_archive
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Population data exported to {filepath}")
    
    def should_terminate(self) -> Tuple[bool, str]:
        """Check if evolution should terminate"""
        # Maximum generations reached
        if self.generation >= self.config.max_generations:
            return True, "max_generations_reached"
        
        # Stagnation threshold reached
        if self.stagnation_count >= self.config.stagnation_threshold:
            return True, "stagnation_threshold_reached"
        
        # Perfect fitness achieved
        if self.current_population:
            best_fitness = max(ind.fitness.total_score for ind in self.current_population)
            if best_fitness >= 0.999:  # Near-perfect fitness
                return True, "optimal_solution_found"
        
        return False, "continue"
