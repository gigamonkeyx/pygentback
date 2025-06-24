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
