"""
Selection Strategies for Recipe Evolution

Implements various selection methods for choosing parents and survivors
in genetic algorithm optimization of recipes.
"""

import logging
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from .fitness_functions import FitnessScore

logger = logging.getLogger(__name__)


class SelectionType(Enum):
    """Types of selection strategies"""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    ELITIST = "elitist"
    NSGA2 = "nsga2"  # Non-dominated Sorting Genetic Algorithm II
    SPEA2 = "spea2"  # Strength Pareto Evolutionary Algorithm 2


@dataclass
class SelectionConfig:
    """Configuration for selection operations"""
    selection_type: SelectionType = SelectionType.TOURNAMENT
    tournament_size: int = 3
    elitism_rate: float = 0.1
    selection_pressure: float = 1.5
    crowding_distance_weight: float = 0.5
    diversity_preservation: bool = True


@dataclass
class Individual:
    """Individual in the population"""
    genome: List[float]
    fitness: FitnessScore
    age: int = 0
    parent_ids: List[str] = None
    id: str = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"ind_{random.randint(10000, 99999)}"
        if self.parent_ids is None:
            self.parent_ids = []


class SelectionStrategy(ABC):
    """Abstract base class for selection strategies"""
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()
    
    @abstractmethod
    def select_parents(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """
        Select parents for reproduction.
        
        Args:
            population: Current population
            num_parents: Number of parents to select
            
        Returns:
            Selected parents
        """
        pass
    
    @abstractmethod
    def select_survivors(self, population: List[Individual], 
                        offspring: List[Individual], 
                        target_size: int) -> List[Individual]:
        """
        Select survivors for the next generation.
        
        Args:
            population: Current population
            offspring: Generated offspring
            target_size: Target population size
            
        Returns:
            Selected survivors
        """
        pass
    
    def calculate_selection_probabilities(self, fitness_scores: List[float]) -> List[float]:
        """Calculate selection probabilities from fitness scores"""
        if not fitness_scores:
            return []
        
        # Handle negative fitness scores
        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            adjusted_scores = [score - min_fitness + 1e-6 for score in fitness_scores]
        else:
            adjusted_scores = [score + 1e-6 for score in fitness_scores]
        
        # Calculate probabilities
        total_fitness = sum(adjusted_scores)
        probabilities = [score / total_fitness for score in adjusted_scores]
        
        return probabilities


class TournamentSelection(SelectionStrategy):
    """Tournament selection strategy"""
    
    def select_parents(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """Select parents using tournament selection"""
        parents = []
        
        for _ in range(num_parents):
            # Select tournament participants
            tournament = random.sample(population, 
                                     min(self.config.tournament_size, len(population)))
            
            # Select best individual from tournament
            winner = max(tournament, key=lambda ind: ind.fitness.total_score)
            parents.append(winner)
        
        logger.debug(f"Tournament selection: selected {len(parents)} parents")
        return parents
    
    def select_survivors(self, population: List[Individual], 
                        offspring: List[Individual], 
                        target_size: int) -> List[Individual]:
        """Select survivors using tournament selection with elitism"""
        combined = population + offspring
        
        # Apply elitism - keep best individuals
        elite_count = int(target_size * self.config.elitism_rate)
        elite = sorted(combined, key=lambda ind: ind.fitness.total_score, reverse=True)[:elite_count]
        
        # Tournament selection for remaining slots
        remaining_slots = target_size - elite_count
        remaining_population = [ind for ind in combined if ind not in elite]
        
        survivors = elite.copy()
        
        for _ in range(remaining_slots):
            if not remaining_population:
                break
            
            tournament = random.sample(remaining_population, 
                                     min(self.config.tournament_size, len(remaining_population)))
            winner = max(tournament, key=lambda ind: ind.fitness.total_score)
            survivors.append(winner)
            remaining_population.remove(winner)
        
        logger.debug(f"Tournament survivor selection: {len(survivors)} survivors")
        return survivors


class RouletteWheelSelection(SelectionStrategy):
    """Roulette wheel (fitness proportionate) selection strategy"""
    
    def select_parents(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """Select parents using roulette wheel selection"""
        fitness_scores = [ind.fitness.total_score for ind in population]
        probabilities = self.calculate_selection_probabilities(fitness_scores)
        
        parents = []
        for _ in range(num_parents):
            # Spin the roulette wheel
            r = random.random()
            cumulative_prob = 0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    parents.append(population[i])
                    break
            else:
                # Fallback - select last individual
                parents.append(population[-1])
        
        logger.debug(f"Roulette wheel selection: selected {len(parents)} parents")
        return parents
    
    def select_survivors(self, population: List[Individual], 
                        offspring: List[Individual], 
                        target_size: int) -> List[Individual]:
        """Select survivors using roulette wheel with elitism"""
        combined = population + offspring
        
        # Apply elitism
        elite_count = int(target_size * self.config.elitism_rate)
        elite = sorted(combined, key=lambda ind: ind.fitness.total_score, reverse=True)[:elite_count]
        
        # Roulette wheel for remaining
        remaining_slots = target_size - elite_count
        remaining_population = [ind for ind in combined if ind not in elite]
        
        if not remaining_population:
            return elite
        
        fitness_scores = [ind.fitness.total_score for ind in remaining_population]
        probabilities = self.calculate_selection_probabilities(fitness_scores)
        
        survivors = elite.copy()
        
        for _ in range(remaining_slots):
            r = random.random()
            cumulative_prob = 0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    survivors.append(remaining_population[i])
                    break
            else:
                survivors.append(remaining_population[-1])
        
        logger.debug(f"Roulette wheel survivor selection: {len(survivors)} survivors")
        return survivors


class RankBasedSelection(SelectionStrategy):
    """Rank-based selection strategy"""
    
    def select_parents(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """Select parents using rank-based selection"""
        # Sort population by fitness
        sorted_population = sorted(population, key=lambda ind: ind.fitness.total_score)
        
        # Calculate rank-based probabilities
        n = len(population)
        probabilities = []
        
        for i in range(n):
            rank = i + 1  # Rank from 1 to n
            prob = (2 - self.config.selection_pressure + 
                   2 * (self.config.selection_pressure - 1) * (rank - 1) / (n - 1)) / n
            probabilities.append(prob)
        
        parents = []
        for _ in range(num_parents):
            r = random.random()
            cumulative_prob = 0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    parents.append(sorted_population[i])
                    break
            else:
                parents.append(sorted_population[-1])
        
        logger.debug(f"Rank-based selection: selected {len(parents)} parents")
        return parents
    
    def select_survivors(self, population: List[Individual], 
                        offspring: List[Individual], 
                        target_size: int) -> List[Individual]:
        """Select survivors using rank-based selection with elitism"""
        combined = population + offspring
        
        # Apply elitism
        elite_count = int(target_size * self.config.elitism_rate)
        elite = sorted(combined, key=lambda ind: ind.fitness.total_score, reverse=True)[:elite_count]
        
        # Rank-based selection for remaining
        remaining_slots = target_size - elite_count
        remaining_population = [ind for ind in combined if ind not in elite]
        
        if not remaining_population:
            return elite
        
        # Use rank-based selection on remaining population
        rank_selector = RankBasedSelection(self.config)
        additional_survivors = rank_selector.select_parents(remaining_population, remaining_slots)
        
        survivors = elite + additional_survivors
        
        logger.debug(f"Rank-based survivor selection: {len(survivors)} survivors")
        return survivors


class ElitistSelection(SelectionStrategy):
    """Elitist selection strategy"""
    
    def select_parents(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """Select parents using elitist selection"""
        # Select best individuals as parents
        sorted_population = sorted(population, key=lambda ind: ind.fitness.total_score, reverse=True)
        
        # Select top individuals, with some randomness
        elite_count = min(num_parents // 2, len(population))
        elite_parents = sorted_population[:elite_count]
        
        # Add some random selection for diversity
        remaining_count = num_parents - elite_count
        if remaining_count > 0:
            remaining_population = sorted_population[elite_count:]
            if remaining_population:
                random_parents = random.sample(remaining_population, 
                                             min(remaining_count, len(remaining_population)))
                elite_parents.extend(random_parents)
        
        # Fill remaining slots if needed
        while len(elite_parents) < num_parents:
            elite_parents.append(random.choice(population))
        
        logger.debug(f"Elitist selection: selected {len(elite_parents)} parents")
        return elite_parents
    
    def select_survivors(self, population: List[Individual], 
                        offspring: List[Individual], 
                        target_size: int) -> List[Individual]:
        """Select survivors using pure elitist selection"""
        combined = population + offspring
        
        # Select best individuals
        survivors = sorted(combined, key=lambda ind: ind.fitness.total_score, reverse=True)[:target_size]
        
        logger.debug(f"Elitist survivor selection: {len(survivors)} survivors")
        return survivors


class NSGA2Selection(SelectionStrategy):
    """NSGA-II (Non-dominated Sorting Genetic Algorithm II) selection"""
    
    def select_parents(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """Select parents using tournament selection based on NSGA-II ranking"""
        # Calculate NSGA-II ranking
        fronts = self._fast_non_dominated_sort(population)
        crowding_distances = self._calculate_crowding_distance(population, fronts)
        
        parents = []
        for _ in range(num_parents):
            # Tournament selection based on NSGA-II criteria
            tournament = random.sample(population, 
                                     min(self.config.tournament_size, len(population)))
            
            winner = self._nsga2_compare(tournament, fronts, crowding_distances)
            parents.append(winner)
        
        logger.debug(f"NSGA-II selection: selected {len(parents)} parents")
        return parents
    
    def select_survivors(self, population: List[Individual], 
                        offspring: List[Individual], 
                        target_size: int) -> List[Individual]:
        """Select survivors using NSGA-II environmental selection"""
        combined = population + offspring
        
        # Fast non-dominated sorting
        fronts = self._fast_non_dominated_sort(combined)
        
        survivors = []
        front_index = 0
        
        # Add complete fronts
        while front_index < len(fronts) and len(survivors) + len(fronts[front_index]) <= target_size:
            survivors.extend(fronts[front_index])
            front_index += 1
        
        # Add partial front if needed
        if front_index < len(fronts) and len(survivors) < target_size:
            remaining_slots = target_size - len(survivors)
            last_front = fronts[front_index]
            
            # Calculate crowding distance for last front
            crowding_distances = self._calculate_crowding_distance(last_front, [last_front])
            
            # Sort by crowding distance (descending)
            last_front_sorted = sorted(last_front, 
                                     key=lambda ind: crowding_distances[ind.id], 
                                     reverse=True)
            
            survivors.extend(last_front_sorted[:remaining_slots])
        
        logger.debug(f"NSGA-II survivor selection: {len(survivors)} survivors")
        return survivors
    
    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """Fast non-dominated sorting algorithm"""
        fronts = []
        domination_count = {}
        dominated_solutions = {}
        
        # Initialize
        for ind in population:
            domination_count[ind.id] = 0
            dominated_solutions[ind.id] = []
        
        # Calculate domination relationships
        for i, ind1 in enumerate(population):
            for j, ind2 in enumerate(population):
                if i != j:
                    dominance = ind1.fitness.get_pareto_dominance(ind2.fitness)
                    if dominance == 1:  # ind1 dominates ind2
                        dominated_solutions[ind1.id].append(ind2)
                    elif dominance == -1:  # ind2 dominates ind1
                        domination_count[ind1.id] += 1
        
        # Find first front
        current_front = []
        for ind in population:
            if domination_count[ind.id] == 0:
                current_front.append(ind)
        
        fronts.append(current_front)
        
        # Find subsequent fronts
        while current_front:
            next_front = []
            for ind in current_front:
                for dominated_ind in dominated_solutions[ind.id]:
                    domination_count[dominated_ind.id] -= 1
                    if domination_count[dominated_ind.id] == 0:
                        next_front.append(dominated_ind)
            
            if next_front:
                fronts.append(next_front)
            current_front = next_front
        
        return fronts
    
    def _calculate_crowding_distance(self, population: List[Individual], 
                                   fronts: List[List[Individual]]) -> Dict[str, float]:
        """Calculate crowding distance for individuals"""
        crowding_distances = {ind.id: 0.0 for ind in population}
        
        for front in fronts:
            if len(front) <= 2:
                # Set infinite distance for boundary solutions
                for ind in front:
                    crowding_distances[ind.id] = float('inf')
                continue
            
            # Calculate crowding distance for each objective
            for objective in ['total_score']:  # Can be extended for multi-objective
                # Sort by objective value
                front_sorted = sorted(front, 
                                    key=lambda ind: getattr(ind.fitness, objective, 0))
                
                # Set boundary solutions to infinite distance
                crowding_distances[front_sorted[0].id] = float('inf')
                crowding_distances[front_sorted[-1].id] = float('inf')
                
                # Calculate distance for intermediate solutions
                obj_range = (getattr(front_sorted[-1].fitness, objective, 0) - 
                           getattr(front_sorted[0].fitness, objective, 0))
                
                if obj_range > 0:
                    for i in range(1, len(front_sorted) - 1):
                        distance = (getattr(front_sorted[i + 1].fitness, objective, 0) - 
                                  getattr(front_sorted[i - 1].fitness, objective, 0)) / obj_range
                        crowding_distances[front_sorted[i].id] += distance
        
        return crowding_distances
    
    def _nsga2_compare(self, individuals: List[Individual], 
                      fronts: List[List[Individual]], 
                      crowding_distances: Dict[str, float]) -> Individual:
        """Compare individuals using NSGA-II criteria"""
        # Find front ranks
        front_ranks = {}
        for rank, front in enumerate(fronts):
            for ind in front:
                front_ranks[ind.id] = rank
        
        # Sort by rank first, then by crowding distance
        def nsga2_key(ind):
            rank = front_ranks.get(ind.id, len(fronts))
            distance = crowding_distances.get(ind.id, 0)
            return (rank, -distance)  # Negative distance for descending order
        
        return min(individuals, key=nsga2_key)


class SelectionStrategyFactory:
    """Factory for creating selection strategies"""
    
    @staticmethod
    def create_strategy(selection_type: SelectionType, 
                       config: Optional[SelectionConfig] = None) -> SelectionStrategy:
        """Create a selection strategy of the specified type"""
        
        if selection_type == SelectionType.TOURNAMENT:
            return TournamentSelection(config)
        elif selection_type == SelectionType.ROULETTE_WHEEL:
            return RouletteWheelSelection(config)
        elif selection_type == SelectionType.RANK_BASED:
            return RankBasedSelection(config)
        elif selection_type == SelectionType.ELITIST:
            return ElitistSelection(config)
        elif selection_type == SelectionType.NSGA2:
            return NSGA2Selection(config)
        else:
            raise ValueError(f"Unknown selection type: {selection_type}")
    
    @staticmethod
    def get_available_types() -> List[SelectionType]:
        """Get list of available selection types"""
        return list(SelectionType)
