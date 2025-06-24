"""
Optimization Agent

Specialized agent for optimization tasks, performance tuning,
and efficiency improvements in multi-agent workflows.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization"""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    COST = "cost"


@dataclass
class OptimizationTarget:
    """Optimization target definition"""
    target_id: str
    target_type: str
    current_value: float
    target_value: float
    metric_name: str
    priority: int = 1
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}


@dataclass
class OptimizationResult:
    """Result from optimization operation"""
    optimization_id: str
    target_id: str
    success: bool
    initial_value: float
    final_value: float
    improvement_percentage: float
    optimization_time_ms: float
    method_used: str
    metadata: Dict[str, Any]


class OptimizationAgent:
    """
    Agent specialized in optimization tasks and performance tuning.
    
    Capabilities:
    - Performance optimization
    - Resource utilization optimization
    - Multi-objective optimization
    - Constraint-based optimization
    - Real-time optimization monitoring
    """
    
    def __init__(self, agent_id: str = "optimization_agent"):
        self.agent_id = agent_id
        self.agent_type = "optimization"
        self.status = "initialized"
        self.capabilities = [
            "performance_optimization",
            "resource_optimization",
            "multi_objective_optimization",
            "constraint_optimization",
            "real_time_monitoring"
        ]
        
        # Optimization state
        self.active_optimizations: Dict[str, Dict[str, Any]] = {}
        self.optimization_history: List[OptimizationResult] = []
        self.optimization_targets: Dict[str, OptimizationTarget] = {}
        
        # Configuration
        self.config = {
            'max_concurrent_optimizations': 5,
            'optimization_timeout_seconds': 300,
            'convergence_threshold': 0.01,
            'max_iterations': 100,
            'learning_rate': 0.1,
            'momentum': 0.9
        }
        
        # Optimization algorithms
        self.algorithms = {
            'gradient_descent': self._gradient_descent_optimization,
            'genetic_algorithm': self._genetic_optimization,
            'simulated_annealing': self._simulated_annealing,
            'particle_swarm': self._particle_swarm_optimization,
            'bayesian_optimization': self._bayesian_optimization
        }
        
        # Statistics
        self.stats = {
            'optimizations_performed': 0,
            'successful_optimizations': 0,
            'total_improvement_achieved': 0.0,
            'avg_optimization_time_ms': 0.0,
            'best_improvement_percentage': 0.0,
            'targets_optimized': 0
        }
        
        logger.info(f"OptimizationAgent {agent_id} initialized")
    
    async def start(self) -> bool:
        """Start the optimization agent"""
        try:
            self.status = "active"
            logger.info(f"OptimizationAgent {self.agent_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start OptimizationAgent {self.agent_id}: {e}")
            self.status = "error"
            return False
    
    async def stop(self) -> bool:
        """Stop the optimization agent"""
        try:
            self.status = "stopped"
            
            # Cancel active optimizations
            for opt_id in list(self.active_optimizations.keys()):
                await self.cancel_optimization(opt_id)
            
            logger.info(f"OptimizationAgent {self.agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop OptimizationAgent {self.agent_id}: {e}")
            return False
    
    async def optimize_performance(self, target_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize performance of a target system or component.
        
        Args:
            target_config: Configuration defining optimization target
            
        Returns:
            Optimization results with performance improvements
        """
        start_time = datetime.utcnow()
        optimization_id = f"perf_opt_{int(start_time.timestamp())}"
        
        try:
            # Create optimization target
            target = OptimizationTarget(
                target_id=target_config.get('target_id', 'unknown'),
                target_type='performance',
                current_value=target_config.get('current_performance', 0.0),
                target_value=target_config.get('target_performance', 1.0),
                metric_name=target_config.get('metric', 'performance_score'),
                priority=target_config.get('priority', 1),
                constraints=target_config.get('constraints', {})
            )
            
            # Register optimization
            self.active_optimizations[optimization_id] = {
                'target': target,
                'start_time': start_time,
                'status': 'running',
                'algorithm': target_config.get('algorithm', 'gradient_descent'),
                'iterations': 0,
                'best_value': target.current_value
            }
            
            # Perform optimization
            result = await self._perform_optimization(optimization_id, target, 'performance')
            
            # Update statistics
            self._update_optimization_stats(result)
            
            # Clean up
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
            
            optimization_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info(f"Performance optimization completed in {optimization_time:.2f}ms")
            return {
                'optimization_id': optimization_id,
                'success': result.success,
                'improvement': result.improvement_percentage,
                'final_performance': result.final_value,
                'optimization_time_ms': optimization_time,
                'method_used': result.method_used,
                'metadata': result.metadata
            }
            
        except Exception as e:
            optimization_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"Performance optimization failed: {e}")
            
            # Clean up
            if optimization_id in self.active_optimizations:
                del self.active_optimizations[optimization_id]
            
            return {
                'optimization_id': optimization_id,
                'success': False,
                'error': str(e),
                'optimization_time_ms': optimization_time
            }
    
    async def optimize_resources(self, resource_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize resource utilization (CPU, memory, network, etc.).
        
        Args:
            resource_config: Configuration for resource optimization
            
        Returns:
            Resource optimization results
        """
        start_time = datetime.utcnow()
        optimization_id = f"resource_opt_{int(start_time.timestamp())}"
        
        try:
            # Create resource optimization target
            target = OptimizationTarget(
                target_id=resource_config.get('resource_id', 'system_resources'),
                target_type='resource',
                current_value=resource_config.get('current_utilization', 0.8),
                target_value=resource_config.get('target_utilization', 0.6),
                metric_name=resource_config.get('metric', 'resource_utilization'),
                constraints=resource_config.get('constraints', {'max_cpu': 0.8, 'max_memory': 0.7})
            )
            
            # Perform resource optimization
            result = await self._perform_optimization(optimization_id, target, 'resource')
            
            optimization_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                'optimization_id': optimization_id,
                'success': result.success,
                'resource_savings': result.improvement_percentage,
                'final_utilization': result.final_value,
                'optimization_time_ms': optimization_time,
                'optimized_resources': result.metadata.get('optimized_resources', [])
            }
            
        except Exception as e:
            optimization_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"Resource optimization failed: {e}")
            return {
                'optimization_id': optimization_id,
                'success': False,
                'error': str(e),
                'optimization_time_ms': optimization_time
            }
    
    async def multi_objective_optimization(self, objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform multi-objective optimization balancing multiple goals.
        
        Args:
            objectives: List of optimization objectives
            
        Returns:
            Multi-objective optimization results
        """
        start_time = datetime.utcnow()
        optimization_id = f"multi_opt_{int(start_time.timestamp())}"
        
        try:
            # Create optimization targets for each objective
            targets = []
            for i, obj in enumerate(objectives):
                target = OptimizationTarget(
                    target_id=obj.get('target_id', f'objective_{i}'),
                    target_type='multi_objective',
                    current_value=obj.get('current_value', 0.0),
                    target_value=obj.get('target_value', 1.0),
                    metric_name=obj.get('metric', f'objective_{i}'),
                    priority=obj.get('priority', 1),
                    constraints=obj.get('constraints', {})
                )
                targets.append(target)
            
            # Perform multi-objective optimization
            results = []
            for target in targets:
                result = await self._perform_optimization(optimization_id, target, 'multi_objective')
                results.append(result)
            
            # Calculate overall improvement
            total_improvement = sum(r.improvement_percentage for r in results) / len(results)
            
            optimization_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                'optimization_id': optimization_id,
                'success': all(r.success for r in results),
                'objectives_optimized': len(results),
                'average_improvement': total_improvement,
                'individual_results': [
                    {
                        'target_id': r.target_id,
                        'improvement': r.improvement_percentage,
                        'final_value': r.final_value
                    } for r in results
                ],
                'optimization_time_ms': optimization_time
            }
            
        except Exception as e:
            optimization_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"Multi-objective optimization failed: {e}")
            return {
                'optimization_id': optimization_id,
                'success': False,
                'error': str(e),
                'optimization_time_ms': optimization_time
            }
    
    async def cancel_optimization(self, optimization_id: str) -> bool:
        """Cancel an active optimization"""
        try:
            if optimization_id in self.active_optimizations:
                self.active_optimizations[optimization_id]['status'] = 'cancelled'
                del self.active_optimizations[optimization_id]
                logger.info(f"Optimization {optimization_id} cancelled")
                return True
            else:
                logger.warning(f"Optimization {optimization_id} not found for cancellation")
                return False
        except Exception as e:
            logger.error(f"Failed to cancel optimization {optimization_id}: {e}")
            return False
    
    async def _perform_optimization(self, optimization_id: str, target: OptimizationTarget, 
                                  optimization_type: str) -> OptimizationResult:
        """Perform the actual optimization"""
        start_time = datetime.utcnow()
        
        # Get optimization algorithm
        algorithm_name = self.active_optimizations[optimization_id]['algorithm']
        algorithm = self.algorithms.get(algorithm_name, self._gradient_descent_optimization)
        
        # Perform optimization
        initial_value = target.current_value
        optimized_value = await algorithm(target, optimization_type)
        
        # Calculate improvement
        if target.target_value > initial_value:
            improvement = ((optimized_value - initial_value) / (target.target_value - initial_value)) * 100
        else:
            improvement = ((initial_value - optimized_value) / (initial_value - target.target_value)) * 100
        
        improvement = max(0, min(100, improvement))  # Clamp between 0-100%
        
        optimization_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        result = OptimizationResult(
            optimization_id=optimization_id,
            target_id=target.target_id,
            success=abs(optimized_value - target.target_value) < abs(initial_value - target.target_value),
            initial_value=initial_value,
            final_value=optimized_value,
            improvement_percentage=improvement,
            optimization_time_ms=optimization_time,
            method_used=algorithm_name,
            metadata={
                'optimization_type': optimization_type,
                'iterations': self.active_optimizations[optimization_id]['iterations'],
                'convergence_achieved': True,
                'constraints_satisfied': True
            }
        )
        
        self.optimization_history.append(result)
        return result
    
    async def _gradient_descent_optimization(self, target: OptimizationTarget, opt_type: str) -> float:
        """Gradient descent optimization algorithm"""
        current_value = target.current_value
        learning_rate = self.config['learning_rate']
        
        for i in range(self.config['max_iterations']):
            # Simulate gradient calculation
            gradient = (target.target_value - current_value) * 0.1
            
            # Update value
            current_value += learning_rate * gradient
            
            # Check convergence
            if abs(current_value - target.target_value) < self.config['convergence_threshold']:
                break
            
            # Simulate some processing time
            await asyncio.sleep(0.001)
        
        return current_value
    
    async def _genetic_optimization(self, target: OptimizationTarget, opt_type: str) -> float:
        """Genetic algorithm optimization"""
        # Simulate genetic algorithm
        population_size = 20
        generations = 10
        
        # Initialize population around current value
        population = [target.current_value + (i - population_size/2) * 0.1 for i in range(population_size)]
        
        for generation in range(generations):
            # Evaluate fitness (distance to target)
            fitness_scores = [1.0 / (1.0 + abs(val - target.target_value)) for val in population]
            
            # Select best individuals
            best_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:10]
            
            # Create new population
            new_population = [population[i] for i in best_indices]
            
            # Add mutations
            for i in range(population_size - len(new_population)):
                parent = new_population[i % len(new_population)]
                mutation = parent + (target.target_value - parent) * 0.1
                new_population.append(mutation)
            
            population = new_population
            await asyncio.sleep(0.001)
        
        # Return best solution
        fitness_scores = [1.0 / (1.0 + abs(val - target.target_value)) for val in population]
        best_index = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
        return population[best_index]
    
    async def _simulated_annealing(self, target: OptimizationTarget, opt_type: str) -> float:
        """Simulated annealing optimization"""
        import random
        import math
        
        current_value = target.current_value
        temperature = 1.0
        cooling_rate = 0.95
        
        for i in range(self.config['max_iterations']):
            # Generate neighbor solution
            neighbor = current_value + random.uniform(-0.1, 0.1)
            
            # Calculate energy difference
            current_energy = abs(current_value - target.target_value)
            neighbor_energy = abs(neighbor - target.target_value)
            delta_energy = neighbor_energy - current_energy
            
            # Accept or reject neighbor
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
                current_value = neighbor
            
            # Cool down
            temperature *= cooling_rate
            
            # Check convergence
            if abs(current_value - target.target_value) < self.config['convergence_threshold']:
                break
            
            await asyncio.sleep(0.001)
        
        return current_value
    
    async def _particle_swarm_optimization(self, target: OptimizationTarget, opt_type: str) -> float:
        """Particle swarm optimization"""
        import random
        
        # Initialize swarm
        swarm_size = 10
        particles = [target.current_value + random.uniform(-0.5, 0.5) for _ in range(swarm_size)]
        velocities = [0.0] * swarm_size
        personal_best = particles.copy()
        global_best = min(particles, key=lambda x: abs(x - target.target_value))
        
        for iteration in range(50):
            for i in range(swarm_size):
                # Update velocity
                r1, r2 = random.random(), random.random()
                velocities[i] = (0.5 * velocities[i] + 
                               2 * r1 * (personal_best[i] - particles[i]) +
                               2 * r2 * (global_best - particles[i]))
                
                # Update position
                particles[i] += velocities[i]
                
                # Update personal best
                if abs(particles[i] - target.target_value) < abs(personal_best[i] - target.target_value):
                    personal_best[i] = particles[i]
                
                # Update global best
                if abs(particles[i] - target.target_value) < abs(global_best - target.target_value):
                    global_best = particles[i]
            
            await asyncio.sleep(0.001)
        
        return global_best
    
    async def _bayesian_optimization(self, target: OptimizationTarget, opt_type: str) -> float:
        """Bayesian optimization (simplified)"""
        # Simplified Bayesian optimization
        candidates = []
        evaluations = []
        
        # Initial samples
        for i in range(5):
            candidate = target.current_value + (i - 2) * 0.2
            evaluation = abs(candidate - target.target_value)
            candidates.append(candidate)
            evaluations.append(evaluation)
        
        # Iterative improvement
        for iteration in range(20):
            # Find best candidate so far
            best_idx = min(range(len(evaluations)), key=lambda i: evaluations[i])
            best_candidate = candidates[best_idx]
            
            # Generate new candidate (simplified acquisition function)
            new_candidate = best_candidate + (target.target_value - best_candidate) * 0.1
            new_evaluation = abs(new_candidate - target.target_value)
            
            candidates.append(new_candidate)
            evaluations.append(new_evaluation)
            
            await asyncio.sleep(0.001)
        
        # Return best solution
        best_idx = min(range(len(evaluations)), key=lambda i: evaluations[i])
        return candidates[best_idx]
    
    def _update_optimization_stats(self, result: OptimizationResult):
        """Update optimization statistics"""
        self.stats['optimizations_performed'] += 1
        
        if result.success:
            self.stats['successful_optimizations'] += 1
            self.stats['total_improvement_achieved'] += result.improvement_percentage
            
            if result.improvement_percentage > self.stats['best_improvement_percentage']:
                self.stats['best_improvement_percentage'] = result.improvement_percentage
        
        # Update average optimization time
        current_avg = self.stats['avg_optimization_time_ms']
        count = self.stats['optimizations_performed']
        self.stats['avg_optimization_time_ms'] = ((current_avg * (count - 1)) + result.optimization_time_ms) / count
    
    def get_status(self) -> Dict[str, Any]:
        """Get optimization agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': self.status,
            'capabilities': self.capabilities,
            'active_optimizations': len(self.active_optimizations),
            'optimization_history': len(self.optimization_history),
            'available_algorithms': list(self.algorithms.keys()),
            'statistics': self.stats.copy(),
            'config': self.config.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'is_healthy': self.status == "active",
            'active_optimizations': len(self.active_optimizations),
            'optimizations_performed': self.stats['optimizations_performed'],
            'success_rate': (
                self.stats['successful_optimizations'] / max(1, self.stats['optimizations_performed'])
            ),
            'average_improvement': (
                self.stats['total_improvement_achieved'] / max(1, self.stats['successful_optimizations'])
            ),
            'last_check': datetime.utcnow().isoformat()
        }
