"""
Optimization Algorithms

Concrete implementations of optimization algorithms for recipe and parameter optimization.
"""

import logging
import asyncio
import numpy as np
import random
from typing import List, Dict, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    """
    
    def __init__(self, optimizer_name: str, optimizer_type: str):
        self.optimizer_name = optimizer_name
        self.optimizer_type = optimizer_type
        
        # Optimization configuration
        self.config = {
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'population_size': 50,
            'timeout_seconds': 300,
            'verbose': False
        }
        
        # Optimization state
        self.current_iteration = 0
        self.best_value = float('-inf')
        self.best_parameters = {}
        self.convergence_history = []
        
        # Statistics
        self.function_evaluations = 0
        self.start_time: Optional[datetime] = None
    
    @abstractmethod
    async def optimize(self, objective_function: Callable, parameter_space: Dict[str, Any],
                      constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize the objective function"""
        pass
    
    def _validate_parameter_space(self, parameter_space: Dict[str, Any]) -> bool:
        """Validate parameter space definition"""
        for param_name, param_config in parameter_space.items():
            if not isinstance(param_config, dict):
                return False
            
            if 'type' not in param_config:
                return False
            
            param_type = param_config['type']
            if param_type == 'continuous':
                if 'min' not in param_config or 'max' not in param_config:
                    return False
            elif param_type == 'discrete':
                if 'values' not in param_config:
                    return False
            elif param_type == 'integer':
                if 'min' not in param_config or 'max' not in param_config:
                    return False
        
        return True
    
    def _generate_random_parameters(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Generate random parameters within the parameter space"""
        parameters = {}
        
        for param_name, param_config in parameter_space.items():
            param_type = param_config['type']
            
            if param_type == 'continuous':
                min_val = param_config['min']
                max_val = param_config['max']
                parameters[param_name] = random.uniform(min_val, max_val)
            elif param_type == 'integer':
                min_val = int(param_config['min'])
                max_val = int(param_config['max'])
                parameters[param_name] = random.randint(min_val, max_val)
            elif param_type == 'discrete':
                values = param_config['values']
                parameters[param_name] = random.choice(values)
        
        return parameters
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged"""
        if len(self.convergence_history) < 10:
            return False
        
        # Check if improvement is below threshold for last few iterations
        recent_values = self.convergence_history[-10:]
        improvement = max(recent_values) - min(recent_values)
        
        return improvement < self.config['convergence_threshold']
    
    async def _evaluate_objective(self, objective_function: Callable, parameters: Dict[str, Any]) -> float:
        """Evaluate objective function with given parameters"""
        self.function_evaluations += 1
        
        try:
            if asyncio.iscoroutinefunction(objective_function):
                result = await objective_function(parameters)
            else:
                result = objective_function(parameters)
            
            return float(result)
        except Exception as e:
            logger.warning(f"Objective function evaluation failed: {e}")
            return float('-inf')


class GeneticOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer for global optimization.
    """
    
    def __init__(self):
        super().__init__("genetic_optimizer", "metaheuristic")
        
        # GA-specific configuration
        self.config.update({
            'population_size': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'selection_method': 'tournament',
            'tournament_size': 3,
            'elitism_ratio': 0.1
        })
    
    async def optimize(self, objective_function: Callable, parameter_space: Dict[str, Any],
                      constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize using genetic algorithm"""
        if not self._validate_parameter_space(parameter_space):
            raise ValueError("Invalid parameter space")
        
        self.start_time = datetime.utcnow()
        self.current_iteration = 0
        self.function_evaluations = 0
        self.convergence_history = []
        
        # Initialize population
        population = []
        fitness_values = []
        
        for _ in range(self.config['population_size']):
            individual = self._generate_random_parameters(parameter_space)
            fitness = await self._evaluate_objective(objective_function, individual)
            population.append(individual)
            fitness_values.append(fitness)
        
        # Track best individual
        best_idx = np.argmax(fitness_values)
        self.best_parameters = population[best_idx].copy()
        self.best_value = fitness_values[best_idx]
        
        # Evolution loop
        for generation in range(self.config['max_iterations']):
            self.current_iteration = generation
            
            # Selection
            selected_population = await self._selection(population, fitness_values)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[min(i + 1, len(selected_population) - 1)]
                
                # Crossover
                if random.random() < self.config['crossover_rate']:
                    child1, child2 = await self._crossover(parent1, parent2, parameter_space)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = await self._mutation(child1, parameter_space)
                child2 = await self._mutation(child2, parameter_space)
                
                new_population.extend([child1, child2])
            
            # Evaluate new population
            new_fitness_values = []
            for individual in new_population:
                fitness = await self._evaluate_objective(objective_function, individual)
                new_fitness_values.append(fitness)
            
            # Elitism: keep best individuals from previous generation
            elite_count = int(self.config['elitism_ratio'] * self.config['population_size'])
            if elite_count > 0:
                # Get indices of best individuals from previous generation
                elite_indices = np.argsort(fitness_values)[-elite_count:]
                
                # Replace worst individuals in new population with elites
                worst_indices = np.argsort(new_fitness_values)[:elite_count]
                
                for i, elite_idx in enumerate(elite_indices):
                    worst_idx = worst_indices[i]
                    new_population[worst_idx] = population[elite_idx].copy()
                    new_fitness_values[worst_idx] = fitness_values[elite_idx]
            
            # Update population
            population = new_population
            fitness_values = new_fitness_values
            
            # Update best solution
            current_best_idx = np.argmax(fitness_values)
            current_best_value = fitness_values[current_best_idx]
            
            if current_best_value > self.best_value:
                self.best_value = current_best_value
                self.best_parameters = population[current_best_idx].copy()
            
            # Track convergence
            self.convergence_history.append(self.best_value)
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"GA converged at generation {generation}")
                break
            
            # Yield control periodically
            if generation % 10 == 0:
                await asyncio.sleep(0.001)
        
        optimization_time = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'optimal_parameters': self.best_parameters,
            'optimal_value': self.best_value,
            'iterations': self.current_iteration + 1,
            'function_evaluations': self.function_evaluations,
            'converged': self._check_convergence(),
            'optimization_time_seconds': optimization_time,
            'improvement': self.best_value,
            'metadata': {
                'algorithm': 'genetic_algorithm',
                'population_size': self.config['population_size'],
                'convergence_history': self.convergence_history[-50:]  # Last 50 values
            }
        }
    
    async def _selection(self, population: List[Dict[str, Any]], fitness_values: List[float]) -> List[Dict[str, Any]]:
        """Select individuals for reproduction"""
        selected = []
        
        if self.config['selection_method'] == 'tournament':
            for _ in range(len(population)):
                # Tournament selection
                tournament_indices = random.sample(range(len(population)), 
                                                 min(self.config['tournament_size'], len(population)))
                tournament_fitness = [fitness_values[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                selected.append(population[winner_idx].copy())
        
        return selected
    
    async def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], 
                        parameter_space: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between two parents"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Uniform crossover
        for param_name in parameter_space.keys():
            if random.random() < 0.5:
                # Swap parameter values
                child1[param_name], child2[param_name] = child2[param_name], child1[param_name]
        
        return child1, child2
    
    async def _mutation(self, individual: Dict[str, Any], parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Perform mutation on an individual"""
        mutated = individual.copy()
        
        for param_name, param_config in parameter_space.items():
            if random.random() < self.config['mutation_rate']:
                param_type = param_config['type']
                
                if param_type == 'continuous':
                    # Gaussian mutation
                    current_value = mutated[param_name]
                    min_val = param_config['min']
                    max_val = param_config['max']
                    
                    # Mutation strength as 10% of parameter range
                    mutation_strength = (max_val - min_val) * 0.1
                    new_value = current_value + random.gauss(0, mutation_strength)
                    
                    # Ensure bounds
                    mutated[param_name] = max(min_val, min(max_val, new_value))
                
                elif param_type == 'integer':
                    # Random integer mutation
                    min_val = int(param_config['min'])
                    max_val = int(param_config['max'])
                    mutated[param_name] = random.randint(min_val, max_val)
                
                elif param_type == 'discrete':
                    # Random choice mutation
                    values = param_config['values']
                    mutated[param_name] = random.choice(values)
        
        return mutated


class GradientOptimizer(BaseOptimizer):
    """
    Gradient-based optimizer for continuous optimization.
    """
    
    def __init__(self):
        super().__init__("gradient_optimizer", "gradient_based")
        
        # Gradient-specific configuration
        self.config.update({
            'learning_rate': 0.01,
            'momentum': 0.9,
            'adaptive_learning_rate': True,
            'gradient_estimation_delta': 1e-6,
            'max_line_search_iterations': 20
        })
    
    async def optimize(self, objective_function: Callable, parameter_space: Dict[str, Any],
                      constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize using gradient descent"""
        if not self._validate_parameter_space(parameter_space):
            raise ValueError("Invalid parameter space")
        
        # Check if all parameters are continuous
        for param_config in parameter_space.values():
            if param_config['type'] != 'continuous':
                raise ValueError("Gradient optimizer only supports continuous parameters")
        
        self.start_time = datetime.utcnow()
        self.current_iteration = 0
        self.function_evaluations = 0
        self.convergence_history = []
        
        # Initialize parameters
        current_params = self._generate_random_parameters(parameter_space)
        current_value = await self._evaluate_objective(objective_function, current_params)
        
        self.best_parameters = current_params.copy()
        self.best_value = current_value
        
        # Initialize momentum
        momentum_terms = {param: 0.0 for param in current_params.keys()}
        learning_rate = self.config['learning_rate']
        
        # Optimization loop
        for iteration in range(self.config['max_iterations']):
            self.current_iteration = iteration
            
            # Estimate gradient
            gradient = await self._estimate_gradient(objective_function, current_params, parameter_space)
            
            # Update parameters with momentum
            for param_name in current_params.keys():
                # Momentum update
                momentum_terms[param_name] = (self.config['momentum'] * momentum_terms[param_name] + 
                                            learning_rate * gradient[param_name])
                
                # Parameter update
                new_value = current_params[param_name] + momentum_terms[param_name]
                
                # Enforce bounds
                param_config = parameter_space[param_name]
                min_val = param_config['min']
                max_val = param_config['max']
                current_params[param_name] = max(min_val, min(max_val, new_value))
            
            # Evaluate new parameters
            new_value = await self._evaluate_objective(objective_function, current_params)
            
            # Update best solution
            if new_value > self.best_value:
                self.best_value = new_value
                self.best_parameters = current_params.copy()
                
                # Adaptive learning rate: increase if improving
                if self.config['adaptive_learning_rate']:
                    learning_rate *= 1.05
            else:
                # Adaptive learning rate: decrease if not improving
                if self.config['adaptive_learning_rate']:
                    learning_rate *= 0.95
            
            current_value = new_value
            
            # Track convergence
            self.convergence_history.append(self.best_value)
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Gradient optimizer converged at iteration {iteration}")
                break
            
            # Yield control periodically
            if iteration % 10 == 0:
                await asyncio.sleep(0.001)
        
        optimization_time = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'optimal_parameters': self.best_parameters,
            'optimal_value': self.best_value,
            'iterations': self.current_iteration + 1,
            'function_evaluations': self.function_evaluations,
            'converged': self._check_convergence(),
            'optimization_time_seconds': optimization_time,
            'improvement': self.best_value,
            'metadata': {
                'algorithm': 'gradient_descent',
                'final_learning_rate': learning_rate,
                'convergence_history': self.convergence_history[-50:]
            }
        }
    
    async def _estimate_gradient(self, objective_function: Callable, parameters: Dict[str, Any],
                               parameter_space: Dict[str, Any]) -> Dict[str, float]:
        """Estimate gradient using finite differences"""
        gradient = {}
        delta = self.config['gradient_estimation_delta']
        
        # Current function value
        f_current = await self._evaluate_objective(objective_function, parameters)
        
        for param_name in parameters.keys():
            # Create perturbed parameters
            params_plus = parameters.copy()
            params_plus[param_name] += delta
            
            # Ensure bounds
            param_config = parameter_space[param_name]
            params_plus[param_name] = max(param_config['min'], 
                                        min(param_config['max'], params_plus[param_name]))
            
            # Evaluate perturbed function
            f_plus = await self._evaluate_objective(objective_function, params_plus)
            
            # Estimate partial derivative
            gradient[param_name] = (f_plus - f_current) / delta
        
        return gradient


class BayesianOptimizer(BaseOptimizer):
    """
    Bayesian optimization for expensive function optimization.
    """
    
    def __init__(self):
        super().__init__("bayesian_optimizer", "model_based")
        
        # Bayesian optimization configuration
        self.config.update({
            'initial_samples': 10,
            'acquisition_function': 'expected_improvement',
            'exploration_weight': 0.1,
            'kernel_type': 'rbf'
        })
        
        # Store evaluation history
        self.evaluation_history = []
    
    async def optimize(self, objective_function: Callable, parameter_space: Dict[str, Any],
                      constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize using Bayesian optimization"""
        if not self._validate_parameter_space(parameter_space):
            raise ValueError("Invalid parameter space")
        
        self.start_time = datetime.utcnow()
        self.current_iteration = 0
        self.function_evaluations = 0
        self.convergence_history = []
        self.evaluation_history = []
        
        # Initial random sampling
        for _ in range(self.config['initial_samples']):
            params = self._generate_random_parameters(parameter_space)
            value = await self._evaluate_objective(objective_function, params)
            
            self.evaluation_history.append((params, value))
            
            if value > self.best_value:
                self.best_value = value
                self.best_parameters = params.copy()
        
        # Bayesian optimization loop
        for iteration in range(self.config['max_iterations'] - self.config['initial_samples']):
            self.current_iteration = iteration + self.config['initial_samples']
            
            # Find next point to evaluate using acquisition function
            next_params = await self._optimize_acquisition_function(parameter_space)
            
            # Evaluate objective function
            next_value = await self._evaluate_objective(objective_function, next_params)
            
            # Update history
            self.evaluation_history.append((next_params, next_value))
            
            # Update best solution
            if next_value > self.best_value:
                self.best_value = next_value
                self.best_parameters = next_params.copy()
            
            # Track convergence
            self.convergence_history.append(self.best_value)
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Bayesian optimizer converged at iteration {iteration}")
                break
            
            # Yield control
            await asyncio.sleep(0.001)
        
        optimization_time = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            'optimal_parameters': self.best_parameters,
            'optimal_value': self.best_value,
            'iterations': self.current_iteration + 1,
            'function_evaluations': self.function_evaluations,
            'converged': self._check_convergence(),
            'optimization_time_seconds': optimization_time,
            'improvement': self.best_value,
            'metadata': {
                'algorithm': 'bayesian_optimization',
                'acquisition_function': self.config['acquisition_function'],
                'convergence_history': self.convergence_history[-50:]
            }
        }
    
    async def _optimize_acquisition_function(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize acquisition function to find next evaluation point"""
        # Simplified acquisition function optimization using random search
        # In practice, would use more sophisticated methods
        
        best_acquisition_value = float('-inf')
        best_params = None
        
        # Random search for acquisition function maximum
        for _ in range(1000):
            candidate_params = self._generate_random_parameters(parameter_space)
            acquisition_value = self._evaluate_acquisition_function(candidate_params)
            
            if acquisition_value > best_acquisition_value:
                best_acquisition_value = acquisition_value
                best_params = candidate_params
        
        return best_params or self._generate_random_parameters(parameter_space)
    
    def _evaluate_acquisition_function(self, parameters: Dict[str, Any]) -> float:
        """Evaluate acquisition function (simplified expected improvement)"""
        if not self.evaluation_history:
            return 1.0  # High acquisition for first point
        
        # Simple distance-based acquisition function
        # In practice, would use Gaussian Process predictions
        
        min_distance = float('inf')
        for eval_params, eval_value in self.evaluation_history:
            distance = self._calculate_parameter_distance(parameters, eval_params)
            min_distance = min(min_distance, distance)
        
        # Encourage exploration of distant points
        exploration_bonus = min_distance * self.config['exploration_weight']
        
        # Simple expected improvement approximation
        if self.evaluation_history:
            values = [eval_value for _, eval_value in self.evaluation_history]
            mean_value = np.mean(values)
            std_value = np.std(values) if len(values) > 1 else 1.0
            
            # Simplified expected improvement
            improvement = max(0, mean_value - self.best_value)
            expected_improvement = improvement + std_value
        else:
            expected_improvement = 1.0
        
        return expected_improvement + exploration_bonus
    
    def _calculate_parameter_distance(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Calculate normalized distance between parameter sets"""
        distance = 0.0
        count = 0
        
        for param_name in params1.keys():
            if param_name in params2:
                # Normalize difference by parameter range (simplified)
                diff = abs(float(params1[param_name]) - float(params2[param_name]))
                distance += diff
                count += 1
        
        return distance / max(count, 1)


class MultiObjectiveOptimizer(BaseOptimizer):
    """
    Multi-objective optimizer using NSGA-II algorithm.
    """

    def __init__(self):
        super().__init__("multi_objective_optimizer", "multi_objective")

        # Multi-objective specific configuration
        self.config.update({
            'population_size': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.9,
            'tournament_size': 2,
            'crowding_distance_weight': 1.0
        })

        # Multi-objective state
        self.pareto_front = []
        self.objective_functions = []
        self.objective_weights = []

    async def optimize(self, objective_function: Callable, parameter_space: Dict[str, Any],
                      constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize multiple objectives using NSGA-II"""
        if not self._validate_parameter_space(parameter_space):
            raise ValueError("Invalid parameter space")

        # Handle multiple objectives
        if isinstance(objective_function, list):
            self.objective_functions = objective_function
        else:
            self.objective_functions = [objective_function]

        # Set equal weights if not provided
        if constraints and 'objective_weights' in constraints:
            self.objective_weights = constraints['objective_weights']
        else:
            self.objective_weights = [1.0] * len(self.objective_functions)

        self.start_time = datetime.utcnow()
        self.current_iteration = 0
        self.function_evaluations = 0
        self.convergence_history = []
        self.pareto_front = []

        # Initialize population
        population = []
        objective_values = []

        for _ in range(self.config['population_size']):
            individual = self._generate_random_parameters(parameter_space)
            objectives = await self._evaluate_multiple_objectives(individual)
            population.append(individual)
            objective_values.append(objectives)

        # Evolution loop
        for generation in range(self.config['max_iterations']):
            self.current_iteration = generation

            # Non-dominated sorting
            fronts = self._non_dominated_sort(population, objective_values)

            # Calculate crowding distance
            for front in fronts:
                self._calculate_crowding_distance(front, objective_values)

            # Selection for next generation
            new_population = []
            new_objective_values = []

            # Add individuals from fronts until population is full
            for front in fronts:
                if len(new_population) + len(front) <= self.config['population_size']:
                    # Add entire front
                    for idx in front:
                        new_population.append(population[idx])
                        new_objective_values.append(objective_values[idx])
                else:
                    # Sort by crowding distance and add best individuals
                    remaining_slots = self.config['population_size'] - len(new_population)
                    front_with_distance = [(idx, getattr(population[idx], 'crowding_distance', 0))
                                         for idx in front]
                    front_with_distance.sort(key=lambda x: x[1], reverse=True)

                    for i in range(remaining_slots):
                        idx = front_with_distance[i][0]
                        new_population.append(population[idx])
                        new_objective_values.append(objective_values[idx])
                    break

            # Generate offspring
            offspring_population = []
            offspring_objectives = []

            for _ in range(self.config['population_size']):
                # Tournament selection
                parent1 = self._tournament_selection(new_population, new_objective_values)
                parent2 = self._tournament_selection(new_population, new_objective_values)

                # Crossover
                if random.random() < self.config['crossover_rate']:
                    child = await self._crossover_multi_objective(parent1, parent2, parameter_space)
                else:
                    child = parent1.copy()

                # Mutation
                child = await self._mutation_multi_objective(child, parameter_space)

                # Evaluate offspring
                child_objectives = await self._evaluate_multiple_objectives(child)

                offspring_population.append(child)
                offspring_objectives.append(child_objectives)

            # Combine parent and offspring populations
            combined_population = new_population + offspring_population
            combined_objectives = new_objective_values + offspring_objectives

            # Select next generation
            population, objective_values = self._environmental_selection(
                combined_population, combined_objectives
            )

            # Update Pareto front
            self._update_pareto_front(population, objective_values)

            # Track convergence (using hypervolume or diversity metric)
            diversity = self._calculate_population_diversity(objective_values)
            self.convergence_history.append(diversity)

            # Check convergence
            if self._check_multi_objective_convergence():
                logger.info(f"Multi-objective optimizer converged at generation {generation}")
                break

            # Yield control periodically
            if generation % 10 == 0:
                await asyncio.sleep(0.001)

        optimization_time = (datetime.utcnow() - self.start_time).total_seconds()

        # Calculate best compromise solution
        best_compromise = self._find_best_compromise_solution()

        return {
            'optimal_parameters': best_compromise['parameters'],
            'optimal_value': best_compromise['weighted_score'],
            'pareto_front': self.pareto_front,
            'pareto_front_size': len(self.pareto_front),
            'iterations': self.current_iteration + 1,
            'function_evaluations': self.function_evaluations,
            'converged': self._check_multi_objective_convergence(),
            'optimization_time_seconds': optimization_time,
            'metadata': {
                'algorithm': 'nsga_ii',
                'num_objectives': len(self.objective_functions),
                'objective_weights': self.objective_weights,
                'population_diversity': self.convergence_history[-1] if self.convergence_history else 0,
                'convergence_history': self.convergence_history[-50:]
            }
        }

    async def _evaluate_multiple_objectives(self, parameters: Dict[str, Any]) -> List[float]:
        """Evaluate all objective functions"""
        objectives = []

        for obj_func in self.objective_functions:
            value = await self._evaluate_objective(obj_func, parameters)
            objectives.append(value)

        return objectives

    def _non_dominated_sort(self, population: List[Dict[str, Any]],
                           objective_values: List[List[float]]) -> List[List[int]]:
        """Perform non-dominated sorting"""
        n = len(population)
        domination_count = [0] * n  # Number of solutions that dominate solution i
        dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by solution i
        fronts = [[]]

        # Calculate domination relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objective_values[i], objective_values[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objective_values[j], objective_values[i]):
                        domination_count[i] += 1

            # If solution i is not dominated by any other solution
            if domination_count[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        front_index = 0
        while len(fronts[front_index]) > 0:
            next_front = []

            for i in fronts[front_index]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            if next_front:
                fronts.append(next_front)
            front_index += 1

        return fronts[:-1] if not fronts[-1] else fronts

    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2 (assuming maximization)"""
        at_least_one_better = False

        for i in range(len(obj1)):
            if obj1[i] < obj2[i]:  # obj1 is worse in this objective
                return False
            elif obj1[i] > obj2[i]:  # obj1 is better in this objective
                at_least_one_better = True

        return at_least_one_better

    def _calculate_crowding_distance(self, front: List[int], objective_values: List[List[float]]):
        """Calculate crowding distance for individuals in a front"""
        if len(front) <= 2:
            # Boundary solutions get infinite distance
            for idx in front:
                setattr(objective_values[idx], 'crowding_distance', float('inf'))
            return

        num_objectives = len(objective_values[0])

        # Initialize distances
        for idx in front:
            setattr(objective_values[idx], 'crowding_distance', 0)

        # Calculate distance for each objective
        for obj_idx in range(num_objectives):
            # Sort front by objective value
            front_sorted = sorted(front, key=lambda x: objective_values[x][obj_idx])

            # Boundary solutions get infinite distance
            setattr(objective_values[front_sorted[0]], 'crowding_distance', float('inf'))
            setattr(objective_values[front_sorted[-1]], 'crowding_distance', float('inf'))

            # Calculate range
            obj_range = (objective_values[front_sorted[-1]][obj_idx] -
                        objective_values[front_sorted[0]][obj_idx])

            if obj_range == 0:
                continue

            # Calculate distances for intermediate solutions
            for i in range(1, len(front_sorted) - 1):
                idx = front_sorted[i]
                current_distance = getattr(objective_values[idx], 'crowding_distance', 0)

                distance_increment = ((objective_values[front_sorted[i + 1]][obj_idx] -
                                     objective_values[front_sorted[i - 1]][obj_idx]) / obj_range)

                setattr(objective_values[idx], 'crowding_distance',
                       current_distance + distance_increment)

    def _tournament_selection(self, population: List[Dict[str, Any]],
                            objective_values: List[List[float]]) -> Dict[str, Any]:
        """Tournament selection for multi-objective optimization"""
        tournament_size = min(self.config['tournament_size'], len(population))
        tournament_indices = random.sample(range(len(population)), tournament_size)

        # Select best individual based on dominance and crowding distance
        best_idx = tournament_indices[0]

        for idx in tournament_indices[1:]:
            if self._is_better_multi_objective(idx, best_idx, objective_values):
                best_idx = idx

        return population[best_idx].copy()

    def _is_better_multi_objective(self, idx1: int, idx2: int,
                                 objective_values: List[List[float]]) -> bool:
        """Compare two solutions for multi-objective optimization"""
        obj1 = objective_values[idx1]
        obj2 = objective_values[idx2]

        # Check dominance
        if self._dominates(obj1, obj2):
            return True
        elif self._dominates(obj2, obj1):
            return False
        else:
            # If non-dominated, prefer higher crowding distance
            dist1 = getattr(obj1, 'crowding_distance', 0)
            dist2 = getattr(obj2, 'crowding_distance', 0)
            return dist1 > dist2

    async def _crossover_multi_objective(self, parent1: Dict[str, Any], parent2: Dict[str, Any],
                                       parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover for multi-objective optimization"""
        child = {}

        for param_name, param_config in parameter_space.items():
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]

        return child

    async def _mutation_multi_objective(self, individual: Dict[str, Any],
                                      parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation for multi-objective optimization"""
        mutated = individual.copy()

        for param_name, param_config in parameter_space.items():
            if random.random() < self.config['mutation_rate']:
                param_type = param_config['type']

                if param_type == 'continuous':
                    min_val = param_config['min']
                    max_val = param_config['max']
                    mutation_strength = (max_val - min_val) * 0.1

                    current_value = mutated[param_name]
                    new_value = current_value + random.gauss(0, mutation_strength)
                    mutated[param_name] = max(min_val, min(max_val, new_value))

                elif param_type == 'integer':
                    min_val = int(param_config['min'])
                    max_val = int(param_config['max'])
                    mutated[param_name] = random.randint(min_val, max_val)

                elif param_type == 'discrete':
                    values = param_config['values']
                    mutated[param_name] = random.choice(values)

        return mutated

    def _environmental_selection(self, population: List[Dict[str, Any]],
                               objective_values: List[List[float]]) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """Environmental selection for next generation"""
        # Perform non-dominated sorting
        fronts = self._non_dominated_sort(population, objective_values)

        # Calculate crowding distance for each front
        for front in fronts:
            self._calculate_crowding_distance(front, objective_values)

        # Select individuals for next generation
        selected_population = []
        selected_objectives = []

        for front in fronts:
            if len(selected_population) + len(front) <= self.config['population_size']:
                # Add entire front
                for idx in front:
                    selected_population.append(population[idx])
                    selected_objectives.append(objective_values[idx])
            else:
                # Sort by crowding distance and select best
                remaining_slots = self.config['population_size'] - len(selected_population)
                front_with_distance = [(idx, getattr(objective_values[idx], 'crowding_distance', 0))
                                     for idx in front]
                front_with_distance.sort(key=lambda x: x[1], reverse=True)

                for i in range(remaining_slots):
                    idx = front_with_distance[i][0]
                    selected_population.append(population[idx])
                    selected_objectives.append(objective_values[idx])
                break

        return selected_population, selected_objectives

    def _update_pareto_front(self, population: List[Dict[str, Any]],
                           objective_values: List[List[float]]):
        """Update the Pareto front"""
        # Get first front (non-dominated solutions)
        fronts = self._non_dominated_sort(population, objective_values)

        if fronts:
            first_front = fronts[0]
            self.pareto_front = []

            for idx in first_front:
                self.pareto_front.append({
                    'parameters': population[idx].copy(),
                    'objectives': objective_values[idx].copy()
                })

    def _calculate_population_diversity(self, objective_values: List[List[float]]) -> float:
        """Calculate population diversity metric"""
        if len(objective_values) < 2:
            return 0.0

        total_distance = 0.0
        count = 0

        for i in range(len(objective_values)):
            for j in range(i + 1, len(objective_values)):
                distance = sum((objective_values[i][k] - objective_values[j][k]) ** 2
                             for k in range(len(objective_values[i])))
                total_distance += distance ** 0.5
                count += 1

        return total_distance / max(count, 1)

    def _check_multi_objective_convergence(self) -> bool:
        """Check convergence for multi-objective optimization"""
        if len(self.convergence_history) < 20:
            return False

        # Check if diversity has stabilized
        recent_diversity = self.convergence_history[-10:]
        diversity_change = max(recent_diversity) - min(recent_diversity)

        return diversity_change < self.config['convergence_threshold']

    def _find_best_compromise_solution(self) -> Dict[str, Any]:
        """Find best compromise solution from Pareto front"""
        if not self.pareto_front:
            return {'parameters': {}, 'weighted_score': 0.0}

        best_solution = None
        best_score = float('-inf')

        for solution in self.pareto_front:
            # Calculate weighted sum
            weighted_score = sum(obj * weight for obj, weight in
                               zip(solution['objectives'], self.objective_weights))

            if weighted_score > best_score:
                best_score = weighted_score
                best_solution = solution

        return {
            'parameters': best_solution['parameters'],
            'weighted_score': best_score,
            'objectives': best_solution['objectives']
        }
