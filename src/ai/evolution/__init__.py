"""
AI Evolution Module

Genetic algorithms and evolutionary computation for recipe optimization.
Includes genome encoding, fitness evaluation, crossover, mutation, and selection.
"""

from .recipe_genome import RecipeGenome
from .genetic_algorithm import GeneticAlgorithm
from .fitness_functions import FitnessEvaluator, FitnessScore
from .crossover_operators import CrossoverOperator
from .mutation_operators import MutationOperator
from .selection_strategies import SelectionStrategy
from .population_manager import PopulationManager

# Observer-approved evolution system
try:
    from .evo_loop_fixed import ObserverEvolutionLoop
    OBSERVER_EVOLUTION_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Observer evolution loop not available: {e}")
    OBSERVER_EVOLUTION_AVAILABLE = False

__all__ = [
    'RecipeGenome',
    'GeneticAlgorithm',
    'FitnessEvaluator',
    'FitnessScore',
    'CrossoverOperator',
    'MutationOperator',
    'SelectionStrategy',
    'PopulationManager'
]

# Add Observer evolution loop if available
if OBSERVER_EVOLUTION_AVAILABLE:
    __all__.append('ObserverEvolutionLoop')
