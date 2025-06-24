"""
Recipe Evolution System

RL-based system for evolving and optimizing test recipes through
reinforcement learning, genetic algorithms, and adaptive strategies.
"""

import logging
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics
import json

logger = logging.getLogger(__name__)


class EvolutionStrategy(Enum):
    """Evolution strategies"""
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    HYBRID = "hybrid"
    RANDOM_SEARCH = "random_search"
    GRADIENT_BASED = "gradient_based"


@dataclass
class RecipeGenome:
    """Genetic representation of a test recipe"""
    recipe_id: str
    genes: Dict[str, Any]  # Recipe parameters as genes
    fitness_score: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvolutionResult:
    """Result of evolution process"""
    best_recipe: RecipeGenome
    generation: int
    population_size: int
    average_fitness: float
    fitness_improvement: float
    convergence_achieved: bool
    evolution_time: float
    total_evaluations: int


@dataclass
class EvolutionConfig:
    """Configuration for evolution process"""
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_pressure: float = 2.0
    convergence_threshold: float = 0.001
    elite_size: int = 5
    tournament_size: int = 3


class RecipeEvolutionSystem:
    """
    RL-based Recipe Evolution System.
    
    Evolves test recipes using reinforcement learning and genetic algorithms
    to optimize performance, reliability, and effectiveness.
    """
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        
        # Evolution state
        self.current_population: List[RecipeGenome] = []
        self.generation = 0
        self.evolution_history: List[EvolutionResult] = []
        self.best_ever_recipe: Optional[RecipeGenome] = None
        
        # Learning components
        self.fitness_predictor = None  # Would be ML model in production
        self.parameter_importance: Dict[str, float] = {}
        self.successful_mutations: Dict[str, int] = {}
        
        # Performance tracking
        self.total_evaluations = 0
        self.convergence_history: List[float] = []
        self.diversity_history: List[float] = []
        
        # Recipe parameter space
        self.parameter_space = self._initialize_parameter_space()
    
    def _initialize_parameter_space(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the parameter space for recipe evolution"""
        return {
            "timeout": {
                "type": "float",
                "min": 1.0,
                "max": 300.0,
                "default": 30.0
            },
            "retry_count": {
                "type": "int",
                "min": 0,
                "max": 5,
                "default": 1
            },
            "parallel_workers": {
                "type": "int",
                "min": 1,
                "max": 8,
                "default": 2
            },
            "memory_limit": {
                "type": "float",
                "min": 100.0,
                "max": 2048.0,
                "default": 512.0
            },
            "optimization_level": {
                "type": "categorical",
                "values": ["none", "basic", "aggressive"],
                "default": "basic"
            },
            "validation_strictness": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5
            },
            "caching_enabled": {
                "type": "boolean",
                "default": True
            },
            "logging_level": {
                "type": "categorical",
                "values": ["debug", "info", "warning", "error"],
                "default": "info"
            }
        }
    
    def initialize_population(self, base_recipes: Optional[List[Dict[str, Any]]] = None) -> List[RecipeGenome]:
        """Initialize the evolution population"""
        self.current_population = []
        
        if base_recipes:
            # Use provided base recipes
            for i, recipe_data in enumerate(base_recipes[:self.config.population_size]):
                genome = RecipeGenome(
                    recipe_id=f"base_{i}",
                    genes=recipe_data.copy(),
                    generation=0
                )
                self.current_population.append(genome)
        
        # Fill remaining population with random recipes
        while len(self.current_population) < self.config.population_size:
            genome = self._create_random_genome(f"random_{len(self.current_population)}")
            self.current_population.append(genome)
        
        logger.info(f"Initialized population with {len(self.current_population)} recipes")
        return self.current_population
    
    def _create_random_genome(self, recipe_id: str) -> RecipeGenome:
        """Create a random recipe genome"""
        genes = {}
        
        for param_name, param_config in self.parameter_space.items():
            if param_config["type"] == "float":
                genes[param_name] = random.uniform(param_config["min"], param_config["max"])
            elif param_config["type"] == "int":
                genes[param_name] = random.randint(param_config["min"], param_config["max"])
            elif param_config["type"] == "boolean":
                genes[param_name] = random.choice([True, False])
            elif param_config["type"] == "categorical":
                genes[param_name] = random.choice(param_config["values"])
            else:
                genes[param_name] = param_config["default"]
        
        return RecipeGenome(
            recipe_id=recipe_id,
            genes=genes,
            generation=self.generation
        )
    
    async def evolve_recipes(self, fitness_evaluator, max_generations: Optional[int] = None) -> EvolutionResult:
        """
        Evolve recipes using the specified strategy.
        
        Args:
            fitness_evaluator: Function to evaluate recipe fitness
            max_generations: Maximum generations to evolve
            
        Returns:
            EvolutionResult with best evolved recipe
        """
        start_time = datetime.utcnow()
        max_gens = max_generations or self.config.max_generations
        
        if not self.current_population:
            self.initialize_population()
        
        # Evaluate initial population
        await self._evaluate_population(fitness_evaluator)
        
        best_fitness_history = []
        
        for generation in range(max_gens):
            self.generation = generation
            
            # Create next generation
            new_population = await self._create_next_generation()
            
            # Evaluate new population
            self.current_population = new_population
            await self._evaluate_population(fitness_evaluator)
            
            # Track progress
            current_best = max(self.current_population, key=lambda x: x.fitness_score)
            best_fitness_history.append(current_best.fitness_score)
            
            # Update best ever recipe
            if not self.best_ever_recipe or current_best.fitness_score > self.best_ever_recipe.fitness_score:
                self.best_ever_recipe = current_best
            
            # Check convergence
            if len(best_fitness_history) >= 10:
                recent_improvement = best_fitness_history[-1] - best_fitness_history[-10]
                if recent_improvement < self.config.convergence_threshold:
                    logger.info(f"Convergence achieved at generation {generation}")
                    break
            
            # Log progress
            avg_fitness = statistics.mean([g.fitness_score for g in self.current_population])
            logger.info(f"Generation {generation}: Best={current_best.fitness_score:.4f}, "
                       f"Avg={avg_fitness:.4f}")
        
        # Create result
        evolution_time = (datetime.utcnow() - start_time).total_seconds()
        avg_fitness = statistics.mean([g.fitness_score for g in self.current_population])
        
        result = EvolutionResult(
            best_recipe=self.best_ever_recipe,
            generation=self.generation,
            population_size=len(self.current_population),
            average_fitness=avg_fitness,
            fitness_improvement=best_fitness_history[-1] - best_fitness_history[0] if best_fitness_history else 0.0,
            convergence_achieved=len(best_fitness_history) >= 10,
            evolution_time=evolution_time,
            total_evaluations=self.total_evaluations
        )
        
        self.evolution_history.append(result)
        return result
    
    async def _evaluate_population(self, fitness_evaluator):
        """Evaluate fitness of all recipes in population"""
        evaluation_tasks = []
        
        for genome in self.current_population:
            if genome.fitness_score == 0.0:  # Only evaluate if not already evaluated
                task = self._evaluate_single_recipe(genome, fitness_evaluator)
                evaluation_tasks.append(task)
        
        if evaluation_tasks:
            await asyncio.gather(*evaluation_tasks)
    
    async def _evaluate_single_recipe(self, genome: RecipeGenome, fitness_evaluator):
        """Evaluate fitness of a single recipe"""
        try:
            # Convert genome to recipe format
            recipe_config = genome.genes.copy()
            recipe_config["recipe_id"] = genome.recipe_id
            
            # Evaluate fitness
            fitness_result = await fitness_evaluator(recipe_config)
            
            if isinstance(fitness_result, dict):
                genome.fitness_score = fitness_result.get("fitness", 0.0)
                genome.performance_metrics = fitness_result.get("metrics", {})
            else:
                genome.fitness_score = float(fitness_result)
            
            self.total_evaluations += 1
            
        except Exception as e:
            logger.error(f"Failed to evaluate recipe {genome.recipe_id}: {e}")
            genome.fitness_score = 0.0
    
    async def _create_next_generation(self) -> List[RecipeGenome]:
        """Create the next generation using genetic operations"""
        new_population = []
        
        # Elitism: Keep best recipes
        elite_recipes = sorted(self.current_population, key=lambda x: x.fitness_score, reverse=True)
        for i in range(min(self.config.elite_size, len(elite_recipes))):
            elite_copy = self._copy_genome(elite_recipes[i])
            elite_copy.generation = self.generation + 1
            new_population.append(elite_copy)
        
        # Generate rest of population through crossover and mutation
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
            else:
                # Clone and mutate
                parent = self._tournament_selection()
                child = self._copy_genome(parent)
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                self._mutate(child)
            
            child.generation = self.generation + 1
            child.recipe_id = f"gen{self.generation + 1}_{len(new_population)}"
            new_population.append(child)
        
        return new_population
    
    def _tournament_selection(self) -> RecipeGenome:
        """Select parent using tournament selection"""
        tournament = random.sample(self.current_population, 
                                 min(self.config.tournament_size, len(self.current_population)))
        return max(tournament, key=lambda x: x.fitness_score)
    
    def _crossover(self, parent1: RecipeGenome, parent2: RecipeGenome) -> RecipeGenome:
        """Create child through crossover of two parents"""
        child_genes = {}
        
        for param_name in self.parameter_space.keys():
            # Randomly choose gene from either parent
            if random.random() < 0.5:
                child_genes[param_name] = parent1.genes.get(param_name)
            else:
                child_genes[param_name] = parent2.genes.get(param_name)
        
        child = RecipeGenome(
            recipe_id="temp_child",
            genes=child_genes,
            parent_ids=[parent1.recipe_id, parent2.recipe_id]
        )
        
        return child
    
    def _mutate(self, genome: RecipeGenome):
        """Mutate a genome"""
        mutation_applied = False
        
        for param_name, param_config in self.parameter_space.items():
            if random.random() < 0.1:  # 10% chance to mutate each parameter
                old_value = genome.genes.get(param_name)
                
                if param_config["type"] == "float":
                    # Gaussian mutation
                    current_value = genome.genes[param_name]
                    mutation_range = (param_config["max"] - param_config["min"]) * 0.1
                    new_value = current_value + random.gauss(0, mutation_range)
                    new_value = max(param_config["min"], min(param_config["max"], new_value))
                    genome.genes[param_name] = new_value
                
                elif param_config["type"] == "int":
                    # Random integer mutation
                    genome.genes[param_name] = random.randint(param_config["min"], param_config["max"])
                
                elif param_config["type"] == "boolean":
                    # Flip boolean
                    genome.genes[param_name] = not genome.genes[param_name]
                
                elif param_config["type"] == "categorical":
                    # Random categorical choice
                    genome.genes[param_name] = random.choice(param_config["values"])
                
                # Track mutation
                genome.mutation_history.append(f"{param_name}: {old_value} -> {genome.genes[param_name]}")
                mutation_applied = True
        
        if mutation_applied:
            logger.debug(f"Applied mutations to {genome.recipe_id}: {genome.mutation_history[-1:]}")
    
    def _copy_genome(self, genome: RecipeGenome) -> RecipeGenome:
        """Create a deep copy of a genome"""
        return RecipeGenome(
            recipe_id=genome.recipe_id,
            genes=genome.genes.copy(),
            fitness_score=genome.fitness_score,
            generation=genome.generation,
            parent_ids=genome.parent_ids.copy(),
            mutation_history=genome.mutation_history.copy(),
            performance_metrics=genome.performance_metrics.copy()
        )
    
    def get_best_recipes(self, top_k: int = 5) -> List[RecipeGenome]:
        """Get the top K best recipes from current population"""
        if not self.current_population:
            return []
        
        sorted_population = sorted(self.current_population, key=lambda x: x.fitness_score, reverse=True)
        return sorted_population[:top_k]
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution process"""
        if not self.evolution_history:
            return {"status": "no_evolution_history"}
        
        latest_result = self.evolution_history[-1]
        
        # Calculate diversity
        current_diversity = self._calculate_population_diversity()
        
        # Parameter importance analysis
        param_importance = self._analyze_parameter_importance()
        
        return {
            "total_generations": self.generation,
            "total_evaluations": self.total_evaluations,
            "best_fitness": latest_result.best_recipe.fitness_score if latest_result.best_recipe else 0.0,
            "average_fitness": latest_result.average_fitness,
            "population_diversity": current_diversity,
            "convergence_achieved": latest_result.convergence_achieved,
            "evolution_time": latest_result.evolution_time,
            "parameter_importance": param_importance,
            "successful_mutations": dict(self.successful_mutations),
            "population_size": len(self.current_population)
        }
    
    def _calculate_population_diversity(self) -> float:
        """Calculate diversity of current population"""
        if len(self.current_population) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(self.current_population)):
            for j in range(i + 1, len(self.current_population)):
                distance = self._calculate_genome_distance(
                    self.current_population[i], 
                    self.current_population[j]
                )
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _calculate_genome_distance(self, genome1: RecipeGenome, genome2: RecipeGenome) -> float:
        """Calculate distance between two genomes"""
        distance = 0.0
        param_count = 0
        
        for param_name, param_config in self.parameter_space.items():
            if param_name in genome1.genes and param_name in genome2.genes:
                value1 = genome1.genes[param_name]
                value2 = genome2.genes[param_name]
                
                if param_config["type"] in ["float", "int"]:
                    # Normalized distance for numeric parameters
                    param_range = param_config["max"] - param_config["min"]
                    if param_range > 0:
                        distance += abs(value1 - value2) / param_range
                elif param_config["type"] in ["boolean", "categorical"]:
                    # Binary distance for categorical parameters
                    distance += 0.0 if value1 == value2 else 1.0
                
                param_count += 1
        
        return distance / param_count if param_count > 0 else 0.0
    
    def _analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze importance of different parameters"""
        if len(self.current_population) < 10:
            return {}
        
        importance = {}
        
        for param_name in self.parameter_space.keys():
            # Calculate correlation between parameter value and fitness
            param_values = []
            fitness_values = []
            
            for genome in self.current_population:
                if param_name in genome.genes:
                    param_values.append(genome.genes[param_name])
                    fitness_values.append(genome.fitness_score)
            
            if len(param_values) > 5:
                # Simple correlation calculation
                correlation = self._calculate_correlation(param_values, fitness_values)
                importance[param_name] = abs(correlation)
        
        return importance
    
    def _calculate_correlation(self, x_values: List, y_values: List) -> float:
        """Calculate correlation between two lists of values"""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        # Convert categorical values to numeric
        numeric_x = []
        for val in x_values:
            if isinstance(val, bool):
                numeric_x.append(1.0 if val else 0.0)
            elif isinstance(val, str):
                numeric_x.append(hash(val) % 100)  # Simple hash to numeric
            else:
                numeric_x.append(float(val))
        
        # Calculate Pearson correlation
        n = len(numeric_x)
        sum_x = sum(numeric_x)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(numeric_x, y_values))
        sum_x2 = sum(x * x for x in numeric_x)
        sum_y2 = sum(y * y for y in y_values)
        
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        correlation = (n * sum_xy - sum_x * sum_y) / denominator
        return correlation
    
    def export_best_recipe(self, filepath: str):
        """Export the best recipe to a file"""
        if not self.best_ever_recipe:
            logger.warning("No best recipe available to export")
            return
        
        export_data = {
            "recipe_id": self.best_ever_recipe.recipe_id,
            "genes": self.best_ever_recipe.genes,
            "fitness_score": self.best_ever_recipe.fitness_score,
            "generation": self.best_ever_recipe.generation,
            "performance_metrics": self.best_ever_recipe.performance_metrics,
            "evolution_summary": self.get_evolution_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported best recipe to {filepath}")
    
    def import_recipe(self, filepath: str) -> RecipeGenome:
        """Import a recipe from a file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        genome = RecipeGenome(
            recipe_id=data["recipe_id"],
            genes=data["genes"],
            fitness_score=data.get("fitness_score", 0.0),
            generation=data.get("generation", 0),
            performance_metrics=data.get("performance_metrics", {})
        )
        
        logger.info(f"Imported recipe {genome.recipe_id} from {filepath}")
        return genome
