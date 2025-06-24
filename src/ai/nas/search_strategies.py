"""
Search Strategies for Recipe NAS

Implements various search algorithms for neural architecture search,
including random search, Bayesian optimization, and evolutionary search.
"""

import logging
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import asyncio

from .architecture_encoder import RecipeArchitecture
from .search_space import SearchSpace
from .performance_predictor import PerformancePredictor, PerformancePrediction

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy/sklearn not available for Bayesian optimization")

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of architecture search"""
    best_architecture: RecipeArchitecture
    best_performance: PerformancePrediction
    search_history: List[Tuple[RecipeArchitecture, PerformancePrediction]]
    total_evaluations: int
    search_time_seconds: float
    convergence_history: List[float]
    search_strategy: str
    termination_reason: str


@dataclass
class SearchConfig:
    """Configuration for architecture search"""
    max_evaluations: int = 1000
    max_time_seconds: float = 3600
    target_performance: float = 0.95
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    early_stopping_patience: int = 50
    diversity_threshold: float = 0.1


class SearchStrategy(ABC):
    """Abstract base class for architecture search strategies"""
    
    def __init__(self, search_space: SearchSpace, 
                 predictor: PerformancePredictor,
                 config: Optional[SearchConfig] = None):
        self.search_space = search_space
        self.predictor = predictor
        self.config = config or SearchConfig()
        
        # Search state
        self.evaluations = 0
        self.start_time = None
        self.best_architecture = None
        self.best_performance = None
        self.search_history = []
        self.convergence_history = []
    
    @abstractmethod
    async def search(self, initial_architectures: Optional[List[RecipeArchitecture]] = None) -> SearchResult:
        """
        Perform architecture search.
        
        Args:
            initial_architectures: Optional starting architectures
            
        Returns:
            Search result with best architecture found
        """
        pass
    
    async def evaluate_architecture(self, architecture: RecipeArchitecture) -> PerformancePrediction:
        """Evaluate an architecture using the performance predictor"""
        self.evaluations += 1
        prediction = await self.predictor.predict_performance(architecture)
        
        # Update best if this is better
        if (self.best_performance is None or 
            prediction.success_probability > self.best_performance.success_probability):
            self.best_architecture = architecture
            self.best_performance = prediction
        
        # Record in history
        self.search_history.append((architecture, prediction))
        self.convergence_history.append(prediction.success_probability)
        
        return prediction
    
    def should_terminate(self) -> Tuple[bool, str]:
        """Check if search should terminate"""
        # Max evaluations
        if self.evaluations >= self.config.max_evaluations:
            return True, "max_evaluations_reached"
        
        # Max time
        if self.start_time and (datetime.utcnow() - self.start_time).total_seconds() > self.config.max_time_seconds:
            return True, "max_time_reached"
        
        # Target performance
        if (self.best_performance and 
            self.best_performance.success_probability >= self.config.target_performance):
            return True, "target_performance_reached"
        
        # Early stopping (no improvement)
        if len(self.convergence_history) >= self.config.early_stopping_patience:
            recent_best = max(self.convergence_history[-self.config.early_stopping_patience:])
            if recent_best <= self.best_performance.success_probability:
                return True, "early_stopping"
        
        return False, "continue"
    
    def create_search_result(self, termination_reason: str) -> SearchResult:
        """Create search result from current state"""
        search_time = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        
        return SearchResult(
            best_architecture=self.best_architecture,
            best_performance=self.best_performance,
            search_history=self.search_history.copy(),
            total_evaluations=self.evaluations,
            search_time_seconds=search_time,
            convergence_history=self.convergence_history.copy(),
            search_strategy=self.__class__.__name__,
            termination_reason=termination_reason
        )


class RandomSearch(SearchStrategy):
    """Random search strategy for architecture optimization"""
    
    async def search(self, initial_architectures: Optional[List[RecipeArchitecture]] = None) -> SearchResult:
        """Perform random search"""
        self.start_time = datetime.utcnow()
        logger.info(f"Starting random search with {self.config.max_evaluations} evaluations")
        
        # Evaluate initial architectures if provided
        if initial_architectures:
            for arch in initial_architectures:
                await self.evaluate_architecture(arch)
                
                should_stop, reason = self.should_terminate()
                if should_stop:
                    return self.create_search_result(reason)
        
        # Random search loop
        while True:
            # Generate random architecture
            architecture = self.search_space.generate_random_architecture()
            
            # Evaluate architecture
            await self.evaluate_architecture(architecture)
            
            # Check termination
            should_stop, reason = self.should_terminate()
            if should_stop:
                break
            
            # Log progress
            if self.evaluations % 100 == 0:
                logger.info(f"Random search: {self.evaluations} evaluations, "
                           f"best performance: {self.best_performance.success_probability:.3f}")
        
        logger.info(f"Random search completed: {self.evaluations} evaluations, "
                   f"best performance: {self.best_performance.success_probability:.3f}")
        
        return self.create_search_result(reason)


class BayesianSearch(SearchStrategy):
    """Bayesian optimization search strategy"""
    
    def __init__(self, search_space: SearchSpace, 
                 predictor: PerformancePredictor,
                 config: Optional[SearchConfig] = None):
        super().__init__(search_space, predictor, config)
        
        if not SCIPY_AVAILABLE:
            raise ImportError("Bayesian search requires scipy and sklearn")
        
        # Bayesian optimization components
        self.gp_model = None
        self.acquisition_function = "expected_improvement"
        self.exploration_weight = 0.1
        
        # Feature extraction for GP
        self.architecture_features = []
        self.performance_values = []
    
    async def search(self, initial_architectures: Optional[List[RecipeArchitecture]] = None) -> SearchResult:
        """Perform Bayesian optimization search"""
        self.start_time = datetime.utcnow()
        logger.info(f"Starting Bayesian search with {self.config.max_evaluations} evaluations")
        
        # Initial random sampling
        initial_samples = 10
        if initial_architectures:
            for arch in initial_architectures[:initial_samples]:
                await self._evaluate_and_update_gp(arch)
        
        # Fill remaining initial samples with random architectures
        while len(self.architecture_features) < initial_samples:
            arch = self.search_space.generate_random_architecture()
            await self._evaluate_and_update_gp(arch)
            
            should_stop, reason = self.should_terminate()
            if should_stop:
                return self.create_search_result(reason)
        
        # Bayesian optimization loop
        while True:
            # Fit GP model
            self._fit_gp_model()
            
            # Find next architecture to evaluate
            next_architecture = self._acquire_next_architecture()
            
            # Evaluate architecture
            await self._evaluate_and_update_gp(next_architecture)
            
            # Check termination
            should_stop, reason = self.should_terminate()
            if should_stop:
                break
            
            # Log progress
            if self.evaluations % 50 == 0:
                logger.info(f"Bayesian search: {self.evaluations} evaluations, "
                           f"best performance: {self.best_performance.success_probability:.3f}")
        
        logger.info(f"Bayesian search completed: {self.evaluations} evaluations, "
                   f"best performance: {self.best_performance.success_probability:.3f}")
        
        return self.create_search_result(reason)
    
    async def _evaluate_and_update_gp(self, architecture: RecipeArchitecture):
        """Evaluate architecture and update GP model data"""
        prediction = await self.evaluate_architecture(architecture)
        
        # Extract features for GP
        features = self.predictor.feature_extractor.extract_features(architecture)
        self.architecture_features.append(features)
        self.performance_values.append(prediction.success_probability)
    
    def _fit_gp_model(self):
        """Fit Gaussian Process model to current data"""
        if len(self.architecture_features) < 2:
            return
        
        X = np.array(self.architecture_features)
        y = np.array(self.performance_values)
        
        # Initialize GP with Matern kernel
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        
        try:
            self.gp_model.fit(X, y)
        except Exception as e:
            logger.warning(f"GP fitting failed: {e}")
            self.gp_model = None
    
    def _acquire_next_architecture(self) -> RecipeArchitecture:
        """Find next architecture to evaluate using acquisition function"""
        if self.gp_model is None:
            return self.search_space.generate_random_architecture()
        
        best_architecture = None
        best_acquisition_value = -np.inf
        
        # Sample candidate architectures
        num_candidates = 100
        for _ in range(num_candidates):
            candidate = self.search_space.generate_random_architecture()
            features = self.predictor.feature_extractor.extract_features(candidate)
            
            # Calculate acquisition function value
            acquisition_value = self._calculate_acquisition(features)
            
            if acquisition_value > best_acquisition_value:
                best_acquisition_value = acquisition_value
                best_architecture = candidate
        
        return best_architecture or self.search_space.generate_random_architecture()
    
    def _calculate_acquisition(self, features: np.ndarray) -> float:
        """Calculate acquisition function value"""
        if self.gp_model is None:
            return random.random()
        
        features = features.reshape(1, -1)
        
        try:
            # Get GP prediction
            mean, std = self.gp_model.predict(features, return_std=True)
            mean, std = mean[0], std[0]
            
            # Expected improvement acquisition function
            if self.acquisition_function == "expected_improvement":
                best_value = max(self.performance_values)
                z = (mean - best_value) / (std + 1e-9)
                
                from scipy.stats import norm
                ei = (mean - best_value) * norm.cdf(z) + std * norm.pdf(z)
                return ei
            
            # Upper confidence bound
            elif self.acquisition_function == "upper_confidence_bound":
                return mean + self.exploration_weight * std
            
            else:
                return mean + self.exploration_weight * std
                
        except Exception as e:
            logger.warning(f"Acquisition calculation failed: {e}")
            return random.random()


class EvolutionarySearch(SearchStrategy):
    """Evolutionary search strategy using genetic algorithms"""
    
    async def search(self, initial_architectures: Optional[List[RecipeArchitecture]] = None) -> SearchResult:
        """Perform evolutionary search"""
        self.start_time = datetime.utcnow()
        logger.info(f"Starting evolutionary search with population size {self.config.population_size}")
        
        # Initialize population
        population = []
        
        # Add initial architectures
        if initial_architectures:
            population.extend(initial_architectures[:self.config.population_size])
        
        # Fill population with random architectures
        while len(population) < self.config.population_size:
            population.append(self.search_space.generate_random_architecture())
        
        # Evaluate initial population
        population_fitness = []
        for arch in population:
            prediction = await self.evaluate_architecture(arch)
            population_fitness.append(prediction.success_probability)
            
            should_stop, reason = self.should_terminate()
            if should_stop:
                return self.create_search_result(reason)
        
        generation = 0
        
        # Evolution loop
        while True:
            generation += 1
            
            # Selection
            parents = self._select_parents(population, population_fitness)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                
                # Crossover
                if random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutation
                if random.random() < self.config.mutation_rate:
                    child1 = self._mutate(child1)
                if random.random() < self.config.mutation_rate:
                    child2 = self._mutate(child2)
                
                offspring.extend([child1, child2])
            
            # Evaluate offspring
            offspring_fitness = []
            for arch in offspring:
                prediction = await self.evaluate_architecture(arch)
                offspring_fitness.append(prediction.success_probability)
                
                should_stop, reason = self.should_terminate()
                if should_stop:
                    return self.create_search_result(reason)
            
            # Survival selection
            population, population_fitness = self._select_survivors(
                population + offspring, 
                population_fitness + offspring_fitness
            )
            
            # Log progress
            if generation % 10 == 0:
                best_fitness = max(population_fitness)
                avg_fitness = np.mean(population_fitness)
                logger.info(f"Generation {generation}: best={best_fitness:.3f}, avg={avg_fitness:.3f}")
        
        logger.info(f"Evolutionary search completed: {generation} generations, "
                   f"best performance: {self.best_performance.success_probability:.3f}")
        
        return self.create_search_result(reason)
    
    def _select_parents(self, population: List[RecipeArchitecture], 
                       fitness: List[float]) -> List[RecipeArchitecture]:
        """Select parents for reproduction using tournament selection"""
        parents = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Tournament selection
            tournament_indices = random.sample(range(len(population)), 
                                             min(tournament_size, len(population)))
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        return parents
    
    def _crossover(self, parent1: RecipeArchitecture, 
                  parent2: RecipeArchitecture) -> Tuple[RecipeArchitecture, RecipeArchitecture]:
        """Perform crossover between two parent architectures"""
        # Simple crossover: randomly mix nodes and edges
        all_nodes = parent1.nodes + parent2.nodes
        all_edges = parent1.edges + parent2.edges
        
        # Randomly select nodes and edges for each child
        num_nodes = min(len(all_nodes), self.search_space.constraints.max_nodes)
        num_edges = min(len(all_edges), self.search_space.constraints.max_edges)
        
        child1_nodes = random.sample(all_nodes, min(num_nodes, len(all_nodes)))
        child2_nodes = [n for n in all_nodes if n not in child1_nodes][:num_nodes]
        
        child1_edges = random.sample(all_edges, min(num_edges, len(all_edges)))
        child2_edges = [e for e in all_edges if e not in child1_edges][:num_edges]
        
        child1 = RecipeArchitecture(
            nodes=child1_nodes,
            edges=child1_edges,
            metadata={"crossover": True, "generation": "offspring"}
        )
        
        child2 = RecipeArchitecture(
            nodes=child2_nodes,
            edges=child2_edges,
            metadata={"crossover": True, "generation": "offspring"}
        )
        
        return child1, child2
    
    def _mutate(self, architecture: RecipeArchitecture) -> RecipeArchitecture:
        """Mutate an architecture"""
        return self.search_space.mutate_architecture(architecture, self.config.mutation_rate)
    
    def _select_survivors(self, combined_population: List[RecipeArchitecture],
                         combined_fitness: List[float]) -> Tuple[List[RecipeArchitecture], List[float]]:
        """Select survivors for next generation"""
        # Sort by fitness (descending)
        sorted_indices = np.argsort(combined_fitness)[::-1]
        
        # Select top individuals
        survivors = []
        survivor_fitness = []
        
        for i in range(min(self.config.population_size, len(combined_population))):
            idx = sorted_indices[i]
            survivors.append(combined_population[idx])
            survivor_fitness.append(combined_fitness[idx])
        
        return survivors, survivor_fitness


class SearchStrategyFactory:
    """Factory for creating search strategies"""
    
    @staticmethod
    def create_strategy(strategy_name: str, 
                       search_space: SearchSpace,
                       predictor: PerformancePredictor,
                       config: Optional[SearchConfig] = None) -> SearchStrategy:
        """Create a search strategy by name"""
        
        if strategy_name.lower() == "random":
            return RandomSearch(search_space, predictor, config)
        elif strategy_name.lower() == "bayesian":
            return BayesianSearch(search_space, predictor, config)
        elif strategy_name.lower() == "evolutionary":
            return EvolutionarySearch(search_space, predictor, config)
        else:
            raise ValueError(f"Unknown search strategy: {strategy_name}")
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available search strategies"""
        strategies = ["random", "evolutionary"]
        if SCIPY_AVAILABLE:
            strategies.append("bayesian")
        return strategies


async def test_search_strategies():
    """Test function for search strategies"""
    # Create components
    search_space = SearchSpace()
    predictor = PerformancePredictor(search_space)
    
    # Test each available strategy
    for strategy_name in SearchStrategyFactory.get_available_strategies():
        print(f"\nTesting {strategy_name} search...")
        
        config = SearchConfig(max_evaluations=50, max_time_seconds=30)
        strategy = SearchStrategyFactory.create_strategy(
            strategy_name, search_space, predictor, config
        )
        
        result = await strategy.search()
        
        print(f"  Best performance: {result.best_performance.success_probability:.3f}")
        print(f"  Evaluations: {result.total_evaluations}")
        print(f"  Time: {result.search_time_seconds:.1f}s")
        print(f"  Termination: {result.termination_reason}")


if __name__ == "__main__":
    asyncio.run(test_search_strategies())
