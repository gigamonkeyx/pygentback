"""
Evolution Agent - Recipe Optimization Implementation

Uses ToT reasoning combined with evolutionary algorithms for recipe optimization.
Replaces the mock "evolution agent" with real AI-powered optimization capabilities.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from core.agent import BaseAgent, AgentMessage, MessageType
from core.agent.config import AgentConfig
from ai.reasoning.tot.thought_generator import OllamaBackend

logger = logging.getLogger(__name__)


class EvolutionAgent(BaseAgent):
    """
    Real Evolution Agent using ToT + evolutionary optimization
    
    This agent combines Tree of Thought reasoning with evolutionary algorithms
    to optimize recipes, processes, and solutions iteratively.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Initialize Ollama backend for AI-powered optimization
        self.model_name = config.get_custom_config("model_name", "")
        self.ollama_url = config.get_custom_config("ollama_url", "http://localhost:11434")
        
        self.ollama_backend = OllamaBackend(
            model_name=self.model_name,
            base_url=self.ollama_url
        )
        
        # Evolution parameters
        self.population_size = config.get_custom_config("population_size", 5)
        self.max_generations = config.get_custom_config("max_generations", 3)
        self.mutation_rate = config.get_custom_config("mutation_rate", 0.3)
        
        # Agent state
        self.optimization_history = []
        self.total_optimization_time = 0.0
        
        logger.info(f"EvolutionAgent initialized (model: {self.model_name}, pop_size: {self.population_size})")

    async def _agent_initialize(self) -> None:
        """Agent-specific initialization logic"""
        # Test Ollama connection
        try:
            test_response = await self.ollama_backend.generate("Test", max_tokens=5)
            if test_response:
                logger.info("EvolutionAgent Ollama connection verified")
            else:
                logger.warning("EvolutionAgent Ollama connection test failed")
        except Exception as e:
            logger.error(f"EvolutionAgent Ollama initialization error: {e}")

    async def _agent_shutdown(self) -> None:
        """Agent-specific shutdown logic"""
        logger.info("EvolutionAgent shutting down")

    async def _handle_request(self, message: AgentMessage) -> AgentMessage:
        """Handle a request message"""
        return await self.process_message(message)

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process message using evolutionary optimization
        
        Args:
            message: Incoming message with optimization request
            
        Returns:
            AgentMessage: Response with optimization results
        """
        start_time = datetime.now()
        
        try:
            # Extract the optimization request from the message
            request = message.content.get("content", str(message.content))
            
            logger.info(f"Starting evolutionary optimization for: {request[:100]}...")
            
            # Parse the optimization request
            optimization_context = self._parse_optimization_request(request)
            
            # Run evolutionary optimization
            result = await self._run_evolutionary_optimization(optimization_context)
            
            # Calculate optimization time
            optimization_time = (datetime.now() - start_time).total_seconds()
            self.total_optimization_time += optimization_time
            
            # Format response
            response_content = self._format_optimization_response(result, optimization_context, optimization_time)
            
            # Store in optimization history
            self.optimization_history.append({
                "request": request,
                "best_solution": result.get("best_solution", ""),
                "optimization_time": optimization_time,
                "generations": result.get("generations_run", 0),
                "timestamp": start_time.isoformat()
            })
            
            logger.info(f"Evolutionary optimization completed in {optimization_time:.2f}s")
            
            # Create response message
            response = AgentMessage(
                type=MessageType.RESPONSE,
                sender=self.agent_id,
                recipient=message.sender,
                content=response_content,
                correlation_id=message.id
            )
            
            # Update agent activity
            self.last_activity = datetime.utcnow()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in EvolutionAgent.process_message: {e}")
            
            # Create error response
            error_response = AgentMessage(
                type=MessageType.RESPONSE,
                sender=self.agent_id,
                recipient=message.sender,
                content={
                    "type": "error",
                    "message": f"Optimization failed: {str(e)}",
                    "agent": "evolution"
                },
                correlation_id=message.id
            )
            
            return error_response
    
    def _parse_optimization_request(self, request: str) -> Dict[str, Any]:
        """Parse the optimization request to extract key components"""
        
        # Simple parsing - in a real implementation, this would be more sophisticated
        context = {
            "original_request": request,
            "optimization_type": "general",
            "constraints": [],
            "objectives": ["improve performance", "reduce complexity", "enhance reliability"]
        }
        
        # Detect optimization type based on keywords
        if any(keyword in request.lower() for keyword in ["recipe", "process", "workflow"]):
            context["optimization_type"] = "recipe"
        elif any(keyword in request.lower() for keyword in ["code", "algorithm", "function"]):
            context["optimization_type"] = "code"
        elif any(keyword in request.lower() for keyword in ["system", "architecture", "design"]):
            context["optimization_type"] = "system"
        
        return context
    
    async def _run_evolutionary_optimization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run the evolutionary optimization process"""
        
        # Initialize population
        population = await self._generate_initial_population(context)
        
        best_solution = None
        best_fitness = 0.0
        generation_history = []
        
        for generation in range(self.max_generations):
            logger.debug(f"Running generation {generation + 1}/{self.max_generations}")
            
            # Evaluate population
            evaluated_population = await self._evaluate_population(population, context)
            
            # Find best solution in this generation
            current_best = max(evaluated_population, key=lambda x: x["fitness"])
            
            if current_best["fitness"] > best_fitness:
                best_solution = current_best
                best_fitness = current_best["fitness"]
            
            # Store generation info
            generation_history.append({
                "generation": generation + 1,
                "best_fitness": current_best["fitness"],
                "average_fitness": sum(sol["fitness"] for sol in evaluated_population) / len(evaluated_population),
                "population_size": len(evaluated_population)
            })
            
            # Generate next generation (except for last iteration)
            if generation < self.max_generations - 1:
                population = await self._generate_next_generation(evaluated_population, context)
        
        return {
            "best_solution": best_solution["solution"] if best_solution else "No solution found",
            "best_fitness": best_fitness,
            "generations_run": len(generation_history),
            "generation_history": generation_history,
            "final_population_size": len(population)
        }
    
    async def _generate_initial_population(self, context: Dict[str, Any]) -> List[str]:
        """Generate initial population of solutions"""
        
        population = []
        
        for i in range(self.population_size):
            prompt = f"""Generate a solution for the following optimization problem:

Problem: {context['original_request']}
Type: {context['optimization_type']}

Please provide a creative and practical solution approach #{i+1}:"""
            
            try:
                solution = await self.ollama_backend.generate(
                    prompt,
                    temperature=0.8,  # Higher temperature for diversity
                    max_tokens=200
                )
                
                if solution:
                    population.append(solution.strip())
                
            except Exception as e:
                logger.warning(f"Failed to generate solution {i+1}: {e}")
        
        # Ensure we have at least one solution
        if not population:
            population.append("Basic solution: Apply standard optimization techniques and best practices.")
        
        return population
    
    async def _evaluate_population(self, population: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate the fitness of each solution in the population"""
        
        evaluated_solutions = []
        
        for solution in population:
            try:
                # Use AI to evaluate the solution
                evaluation_prompt = f"""Evaluate the following solution for the optimization problem:

Problem: {context['original_request']}
Solution: {solution}

Rate this solution on a scale of 0.0 to 1.0 based on:
- Feasibility and practicality
- Potential impact and effectiveness
- Innovation and creativity
- Implementation complexity (lower is better)

Provide only a numeric score between 0.0 and 1.0:"""
                
                score_response = await self.ollama_backend.generate(
                    evaluation_prompt,
                    temperature=0.3,  # Lower temperature for consistent evaluation
                    max_tokens=50
                )
                
                # Extract numeric score
                fitness = self._extract_fitness_score(score_response)
                
                evaluated_solutions.append({
                    "solution": solution,
                    "fitness": fitness
                })
                
            except Exception as e:
                logger.warning(f"Failed to evaluate solution: {e}")
                # Assign default fitness
                evaluated_solutions.append({
                    "solution": solution,
                    "fitness": 0.5
                })
        
        return evaluated_solutions
    
    def _extract_fitness_score(self, score_response: str) -> float:
        """Extract numeric fitness score from AI response"""
        try:
            # Look for a number between 0 and 1
            import re
            numbers = re.findall(r'0\.\d+|1\.0|0|1', score_response)
            if numbers:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except:
            pass
        
        # Default score if extraction fails
        return 0.5
    
    async def _generate_next_generation(self, evaluated_population: List[Dict[str, Any]], 
                                      context: Dict[str, Any]) -> List[str]:
        """Generate the next generation through selection, crossover, and mutation"""
        
        # Sort by fitness (best first)
        sorted_population = sorted(evaluated_population, key=lambda x: x["fitness"], reverse=True)
        
        # Select top performers for breeding
        elite_size = max(1, len(sorted_population) // 3)
        elite = sorted_population[:elite_size]
        
        next_generation = []
        
        # Keep elite solutions
        for solution in elite:
            next_generation.append(solution["solution"])
        
        # Generate new solutions through mutation and crossover
        while len(next_generation) < self.population_size:
            if len(elite) >= 2:
                # Crossover between two elite solutions
                parent1 = elite[0]["solution"]
                parent2 = elite[1]["solution"]
                child = await self._crossover_solutions(parent1, parent2, context)
            else:
                # Mutate the best solution
                child = await self._mutate_solution(elite[0]["solution"], context)
            
            if child:
                next_generation.append(child)
        
        return next_generation[:self.population_size]
    
    async def _crossover_solutions(self, parent1: str, parent2: str, context: Dict[str, Any]) -> str:
        """Create a child solution by combining two parent solutions"""
        
        crossover_prompt = f"""Combine the best aspects of these two solutions for the optimization problem:

Problem: {context['original_request']}

Solution A: {parent1}

Solution B: {parent2}

Create a new solution that combines the strengths of both approaches:"""
        
        try:
            child = await self.ollama_backend.generate(
                crossover_prompt,
                temperature=0.6,
                max_tokens=200
            )
            return child.strip() if child else parent1
            
        except Exception as e:
            logger.warning(f"Crossover failed: {e}")
            return parent1
    
    async def _mutate_solution(self, solution: str, context: Dict[str, Any]) -> str:
        """Create a mutated version of a solution"""
        
        mutation_prompt = f"""Improve and modify this solution for the optimization problem:

Problem: {context['original_request']}
Current Solution: {solution}

Create an improved variation that addresses potential weaknesses or adds new ideas:"""
        
        try:
            mutated = await self.ollama_backend.generate(
                mutation_prompt,
                temperature=0.7,
                max_tokens=200
            )
            return mutated.strip() if mutated else solution
            
        except Exception as e:
            logger.warning(f"Mutation failed: {e}")
            return solution
    
    def _format_optimization_response(self, result: Dict[str, Any], context: Dict[str, Any], 
                                    optimization_time: float) -> Dict[str, Any]:
        """Format the optimization result into a response"""
        
        return {
            "type": "evolution_response",
            "optimization_type": context["optimization_type"],
            "best_solution": result["best_solution"],
            "confidence": result["best_fitness"],
            "evolution_summary": {
                "generations_run": result["generations_run"],
                "final_fitness": result["best_fitness"],
                "population_size": self.population_size
            },
            "generation_history": result["generation_history"],
            "metrics": {
                "optimization_time": optimization_time,
                "generations": result["generations_run"],
                "population_size": result.get("final_population_size", self.population_size)
            },
            "agent": "evolution"
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics for this agent"""
        return {
            "total_optimizations": len(self.optimization_history),
            "total_optimization_time": self.total_optimization_time,
            "average_optimization_time": self.total_optimization_time / max(1, len(self.optimization_history)),
            "evolution_parameters": {
                "population_size": self.population_size,
                "max_generations": self.max_generations,
                "mutation_rate": self.mutation_rate
            },
            "recent_optimizations": self.optimization_history[-3:] if self.optimization_history else []
        }
