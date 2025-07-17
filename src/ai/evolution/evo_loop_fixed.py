#!/usr/bin/env python3
"""
Observer-Approved Evolution Loop with Bloat Penalties and Termination Conditions
Fixes hang-prone evolution loops with systematic safeguards and performance optimization
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import torch

# Set up robust logger with fallback
try:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        # Add a handler if none exists
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
except Exception:
    # Fallback logger if setup fails
    import sys
    logger = logging.getLogger('evo_loop_fixed')
    logger.addHandler(logging.StreamHandler(sys.stdout))

class ObserverEvolutionLoop:
    """Observer-approved evolution loop with bloat penalties and termination safeguards"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generation = 0
        self.start_time = None
        self.last_improvement_time = None
        self.best_fitness = float('-inf')
        self.stagnation_count = 0
        
        # Observer-approved termination conditions
        self.termination_config = {
            "max_generations": config.get("max_generations", 100),
            "max_runtime_seconds": config.get("max_runtime_seconds", 3600),  # 1 hour max
            "stagnation_threshold": config.get("stagnation_threshold", 20),  # 20 gens without improvement
            "fitness_threshold": config.get("fitness_threshold", 0.95),  # Stop at 95% fitness
            "resource_threshold": config.get("resource_threshold", 0.9),  # Stop at 90% resource usage
            "bloat_threshold": config.get("bloat_threshold", 1000),  # Max code length
            "emergency_timeout": config.get("emergency_timeout", 30)  # 30 sec per generation max
        }
        
        # Bloat penalty configuration
        self.bloat_config = {
            "enabled": config.get("bloat_penalty_enabled", True),
            "base_penalty": config.get("bloat_base_penalty", 0.1),
            "length_threshold": config.get("bloat_length_threshold", 500),
            "complexity_threshold": config.get("bloat_complexity_threshold", 10),
            "penalty_scaling": config.get("bloat_penalty_scaling", 0.001)
        }
        
        # GPU optimization settings
        self.gpu_config = {
            "memory_fraction": config.get("gpu_memory_fraction", 0.8),
            "batch_size": config.get("gpu_batch_size", 32),
            "gradient_accumulation": config.get("gpu_gradient_accumulation", 4)
        }

        # Observer Enhanced Learning Configuration
        self.learning_phases = config.get('learning_phases', {
            'exploration_phase': {'generations': 3, 'mutation_rate': 0.3, 'diversity_bonus': 0.1},
            'exploitation_phase': {'generations': 7, 'crossover_rate': 0.8, 'elite_preservation': 0.2}
        })
        self.domain_adaptation_enabled = config.get('domain_adaptation_enabled', True)
        self.seeded_evolution_weights = config.get('seeded_evolution_weights', {
            'cooperation': 0.3,
            'sustainability': 0.2,
            'adaptation': 0.5
        })
        self.shift_adaptation_rate = config.get('shift_adaptation_rate', 0.1)
        
    async def run_evolution(self, 
                          population: List[Any], 
                          fitness_function: Callable,
                          mutation_function: Callable,
                          crossover_function: Callable,
                          progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run Observer-approved evolution loop with comprehensive safeguards
        
        Returns:
            Dict containing evolution results, metrics, and termination reason
        """
        try:
            self.start_time = time.time()
            self.last_improvement_time = self.start_time
            
            logger.info("Starting Observer-approved evolution loop")
            logger.info(f"Population size: {len(population)}")
            logger.info(f"Termination conditions: {self.termination_config}")
            
            # Configure GPU optimization
            await self._configure_gpu_optimization()
            
            evolution_results = {
                "generations_completed": 0,
                "best_fitness": float('-inf'),
                "best_individual": None,
                "termination_reason": "unknown",
                "runtime_seconds": 0,
                "bloat_penalties_applied": 0,
                "resource_usage_history": [],
                "fitness_history": [],
                "success": False
            }
            
            # Main evolution loop with Observer safeguards
            while self.generation < self.termination_config["max_generations"]:
                generation_start = time.time()
                
                # Check termination conditions before each generation
                should_terminate, termination_reason = await self._check_termination_conditions()
                if should_terminate:
                    evolution_results["termination_reason"] = termination_reason
                    logger.info(f"Evolution terminated: {termination_reason}")
                    break
                
                # Generation execution with timeout protection
                try:
                    generation_result = await asyncio.wait_for(
                        self._execute_generation(population, fitness_function, mutation_function, crossover_function),
                        timeout=self.termination_config["emergency_timeout"]
                    )
                    
                    population = generation_result["population"]
                    generation_fitness = generation_result["best_fitness"]
                    bloat_penalties = generation_result["bloat_penalties"]
                    
                    # Update evolution tracking
                    if generation_fitness > self.best_fitness:
                        self.best_fitness = generation_fitness
                        self.last_improvement_time = time.time()
                        self.stagnation_count = 0
                        evolution_results["best_individual"] = generation_result["best_individual"]
                    else:
                        self.stagnation_count += 1
                    
                    # Record metrics
                    evolution_results["bloat_penalties_applied"] += bloat_penalties
                    evolution_results["fitness_history"].append(generation_fitness)
                    
                    # Resource monitoring
                    resource_usage = await self._monitor_resources()
                    evolution_results["resource_usage_history"].append(resource_usage)
                    
                    # Progress callback
                    if progress_callback:
                        await progress_callback(self.generation, generation_fitness, resource_usage)
                    
                    generation_time = time.time() - generation_start
                    logger.info(f"Generation {self.generation}: fitness={generation_fitness:.4f}, "
                              f"time={generation_time:.2f}s, bloat_penalties={bloat_penalties}")
                    
                except asyncio.TimeoutError:
                    logger.error(f"Generation {self.generation} timed out after {self.termination_config['emergency_timeout']}s")
                    evolution_results["termination_reason"] = "generation_timeout"
                    break
                except Exception as e:
                    logger.error(f"Generation {self.generation} failed: {e}")
                    evolution_results["termination_reason"] = f"generation_error: {e}"
                    break
                
                self.generation += 1
                
                # Async yield to prevent blocking
                await asyncio.sleep(0.01)
            
            # Finalize results
            evolution_results["generations_completed"] = self.generation
            evolution_results["best_fitness"] = self.best_fitness
            evolution_results["runtime_seconds"] = time.time() - self.start_time
            evolution_results["success"] = self.best_fitness > 0  # Basic success criteria
            
            if evolution_results["termination_reason"] == "unknown":
                evolution_results["termination_reason"] = "max_generations_reached"
            
            logger.info(f"Evolution completed: {evolution_results['termination_reason']}")
            logger.info(f"Best fitness: {self.best_fitness:.4f} in {self.generation} generations")
            
            return evolution_results
            
        except Exception as e:
            logger.error(f"Evolution loop failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "generations_completed": self.generation,
                "runtime_seconds": time.time() - self.start_time if self.start_time else 0
            }

    async def _configure_gpu_optimization(self):
        """Configure GPU optimization settings"""
        try:
            if torch.cuda.is_available():
                # Set memory fraction to prevent GPU overload
                torch.cuda.set_per_process_memory_fraction(self.gpu_config["memory_fraction"])
                logger.info(f"GPU optimization configured: {self.gpu_config}")
            else:
                logger.warning("CUDA not available - using CPU only")
        except Exception as e:
            logger.error(f"GPU configuration failed: {e}")

    async def _check_termination_conditions(self) -> tuple[bool, str]:
        """Check all Observer-approved termination conditions"""
        current_time = time.time()

        # Runtime limit
        if current_time - self.start_time > self.termination_config["max_runtime_seconds"]:
            return True, "max_runtime_exceeded"

        # Fitness threshold
        if self.best_fitness >= self.termination_config["fitness_threshold"]:
            return True, "fitness_threshold_reached"

        # Stagnation check
        if self.stagnation_count >= self.termination_config["stagnation_threshold"]:
            return True, "stagnation_detected"

        # Resource usage check
        try:
            memory_usage = psutil.virtual_memory().percent / 100.0
            if memory_usage > self.termination_config["resource_threshold"]:
                return True, "resource_threshold_exceeded"
        except Exception as e:
            logger.warning(f"Resource monitoring failed: {e}")

        return False, "continue"

    async def _execute_generation(self, population, fitness_function, mutation_function, crossover_function) -> Dict[str, Any]:
        """Execute a single generation with bloat penalties"""
        try:
            # Evaluate fitness with bloat penalties
            fitness_scores = []
            bloat_penalties_applied = 0

            for individual in population:
                # Calculate base fitness
                base_fitness = await fitness_function(individual)

                # Apply bloat penalty if enabled
                if self.bloat_config["enabled"]:
                    bloat_penalty = self._calculate_bloat_penalty(individual)
                    final_fitness = base_fitness - bloat_penalty
                    if bloat_penalty > 0:
                        bloat_penalties_applied += 1
                else:
                    final_fitness = base_fitness

                fitness_scores.append(final_fitness)

            # Find best individual
            best_index = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            best_fitness = fitness_scores[best_index]
            best_individual = population[best_index]

            # Selection and reproduction
            new_population = []

            # Elitism - keep best individuals
            elite_count = max(1, len(population) // 10)  # Top 10%
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
            for idx in elite_indices:
                new_population.append(population[idx])

            # Generate rest through crossover and mutation
            while len(new_population) < len(population):
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Crossover
                if torch.rand(1).item() < 0.8:  # 80% crossover rate
                    offspring = await crossover_function(parent1, parent2)
                else:
                    offspring = parent1 if torch.rand(1).item() < 0.5 else parent2

                # Mutation
                if torch.rand(1).item() < 0.1:  # 10% mutation rate
                    offspring = await mutation_function(offspring)

                new_population.append(offspring)

            return {
                "population": new_population[:len(population)],
                "best_fitness": best_fitness,
                "best_individual": best_individual,
                "bloat_penalties": bloat_penalties_applied
            }

        except Exception as e:
            logger.error(f"Generation execution failed: {e}")
            raise

    def _calculate_bloat_penalty(self, individual) -> float:
        """Calculate Observer-approved bloat penalty for individual"""
        try:
            # Estimate code length/complexity
            if hasattr(individual, '__len__'):
                code_length = len(individual)
            elif hasattr(individual, 'code') and hasattr(individual.code, '__len__'):
                code_length = len(individual.code)
            else:
                # Default complexity estimation
                code_length = len(str(individual))

            # Apply penalty if above threshold
            if code_length > self.bloat_config["length_threshold"]:
                excess_length = code_length - self.bloat_config["length_threshold"]
                penalty = self.bloat_config["base_penalty"] + (excess_length * self.bloat_config["penalty_scaling"])
                return min(penalty, 0.5)  # Cap penalty at 50% fitness reduction

            return 0.0

        except Exception as e:
            logger.warning(f"Bloat penalty calculation failed: {e}")
            return 0.0

    def _tournament_selection(self, population, fitness_scores, tournament_size: int = 3):
        """Tournament selection for parent selection"""
        try:
            # Select random individuals for tournament
            tournament_indices = torch.randint(0, len(population), (tournament_size,)).tolist()

            # Find best in tournament
            best_tournament_idx = max(tournament_indices, key=lambda i: fitness_scores[i])

            return population[best_tournament_idx]

        except Exception as e:
            logger.error(f"Tournament selection failed: {e}")
            return population[0]  # Fallback to first individual

    async def _monitor_resources(self) -> Dict[str, float]:
        """Monitor system resource usage"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # GPU if available
            gpu_memory = 0.0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0.0

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "gpu_memory_percent": gpu_memory * 100,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.warning(f"Resource monitoring failed: {e}")
            return {"cpu_percent": 0, "memory_percent": 0, "gpu_memory_percent": 0, "timestamp": time.time()}
