"""
Test Suite for Darwinian A2A Implementation - Phase 2.1: Genetic Algorithm Distribution via A2A
Tests distributed genetic algorithm implementation across A2A network.

Aligned with Sakana AI Darwin Gödel Machine (DGM) research:
https://sakana.ai/dgm/
"""

import pytest
import asyncio
import logging
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

# Import the modules under test
from src.orchestration.distributed_genetic_algorithm import (
    DistributedGeneticAlgorithm,
    Individual,
    Population,
    FitnessMetrics,
    SelectionMethod,
    MutationStrategy,
    CrossoverStrategy,
    DistributedGAMetrics,
    handle_crossover_request,
    handle_mutation_request,
    handle_selection_request,
    handle_fitness_evaluation_request,
    handle_migration_request
)
from src.a2a import AgentCard, A2AServer, AgentDiscoveryService

from src.orchestration.distributed_genetic_algorithm import (
    DistributedGeneticAlgorithm,
    Individual,
    GAEvolutionEvent,
    MigrationProtocol,
    DistributedFitnessEvaluator
)


class TestDistributedGeneticAlgorithm:
    """Test distributed genetic algorithm implementation."""
    
    @pytest.fixture
    def mock_a2a_server(self):
        """Mock A2A server for testing."""
        server = Mock()
        server.broadcast_evolution_event = AsyncMock()
        server.request_crossover = AsyncMock()
        server.request_fitness_evaluation = AsyncMock()
        server.get_population_stats = AsyncMock()
        return server
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Mock orchestrator for testing."""
        orchestrator = Mock()
        orchestrator.a2a_server = Mock()
        orchestrator.get_peer_agents = AsyncMock(return_value=[
            {"id": "agent1", "capabilities": ["evolution", "ga"]},
            {"id": "agent2", "capabilities": ["evolution", "ga"]}
        ])
        return orchestrator
    
    @pytest.fixture
    def dga(self, mock_orchestrator):
        """Create distributed genetic algorithm instance."""
        return DistributedGeneticAlgorithm(
            orchestrator=mock_orchestrator,
            population_size=50,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=5
        )
    
    def test_dga_initialization(self, dga):
        """Test DGA initialization."""
        assert dga.population_size == 50
        assert dga.mutation_rate == 0.1
        assert dga.crossover_rate == 0.8
        assert dga.elite_size == 5
        assert dga.generation == 0
        assert len(dga.population.individuals) == 0
        assert len(dga.evolution_history) == 0
    
    @pytest.mark.asyncio
    async def test_initialize_population(self, dga):
        """Test population initialization."""
        # Mock problem space
        problem_space = {
            'dimensions': 10,
            'bounds': [(-10, 10)] * 10,
            'fitness_function': lambda x: sum(x**2)  # Simple sphere function
        }
        
        await dga.initialize_population(problem_space)
        
        assert len(dga.population.individuals) == 50
        assert all(len(ind.genes) == 10 for ind in dga.population.individuals)
        assert all(-10 <= gene <= 10 for ind in dga.population.individuals for gene in ind.genes)
    
    @pytest.mark.asyncio
    async def test_distributed_fitness_evaluation(self, dga, mock_orchestrator):
        """Test distributed fitness evaluation."""
        # Create test population
        individuals = [
            Individual(id=f"ind_{i}", genes=np.random.rand(5).tolist())
            for i in range(10)
        ]
        
        # Mock A2A fitness evaluation
        mock_orchestrator.a2a_server.request_fitness_evaluation = AsyncMock(
            return_value={"fitness": 0.8, "evaluation_time": 0.1}
        )
        
        evaluator = DistributedFitnessEvaluator(mock_orchestrator)
        fitness_values = await evaluator.evaluate_population(individuals)
        
        assert len(fitness_values) == 10
        assert all(isinstance(f, float) for f in fitness_values)
        assert mock_orchestrator.a2a_server.request_fitness_evaluation.call_count == 10
    
    @pytest.mark.asyncio
    async def test_a2a_coordinated_crossover(self, dga, mock_orchestrator):
        """Test A2A-coordinated crossover operations."""
        # Create parent individuals
        parent1 = Individual(id="p1", genes=[1, 2, 3, 4, 5])
        parent2 = Individual(id="p2", genes=[6, 7, 8, 9, 10])
        
        # Mock A2A crossover response
        mock_orchestrator.a2a_server.request_crossover = AsyncMock(
            return_value={
                "offspring": [
                    {"genes": [1, 2, 8, 9, 10]},
                    {"genes": [6, 7, 3, 4, 5]}
                ]
            }
        )
        
        offspring = await dga.a2a_coordinated_crossover(parent1, parent2)
        
        assert len(offspring) == 2
        assert offspring[0].genes == [1, 2, 8, 9, 10]
        assert offspring[1].genes == [6, 7, 3, 4, 5]
        mock_orchestrator.a2a_server.request_crossover.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_distributed_mutation(self, dga):
        """Test distributed mutation strategies."""
        individual = Individual(id="test", genes=[1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test adaptive mutation
        mutated = await dga.distributed_mutation(individual, strategy="adaptive")
        assert len(mutated.genes) == len(individual.genes)
        
        # Test collaborative mutation
        mutated_collab = await dga.distributed_mutation(individual, strategy="collaborative")
        assert len(mutated_collab.genes) == len(individual.genes)
    
    @pytest.mark.asyncio
    async def test_selection_pressure_coordination(self, dga, mock_orchestrator):
        """Test A2A-enabled selection pressure coordination."""
        # Mock population stats from peers
        mock_orchestrator.a2a_server.get_population_stats = AsyncMock(
            return_value={
                "peer_stats": [
                    {"avg_fitness": 0.7, "best_fitness": 0.9, "diversity": 0.6},
                    {"avg_fitness": 0.8, "best_fitness": 0.95, "diversity": 0.5}
                ]
            }
        )
        
        pressure = await dga.coordinate_selection_pressure()
        
        assert 0.0 <= pressure <= 1.0
        mock_orchestrator.a2a_server.get_population_stats.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_evolutionary_migration(self, dga, mock_orchestrator):
        """Test A2A-coordinated evolutionary migration."""
        # Create test population
        individuals = [
            Individual(id=f"ind_{i}", genes=np.random.rand(5).tolist(), fitness=np.random.rand())
            for i in range(10)
        ]
        dga.population.individuals = individuals
        
        # Mock migration response
        mock_orchestrator.a2a_server.broadcast_evolution_event = AsyncMock()
        
        migrated = await dga.coordinate_migration(migration_rate=0.2)
        
        assert len(migrated) <= 2  # 20% of 10
        mock_orchestrator.a2a_server.broadcast_evolution_event.assert_called()
    
    @pytest.mark.asyncio
    async def test_evolution_cycle(self, dga, mock_orchestrator):
        """Test complete evolution cycle."""
        # Initialize with simple problem
        problem_space = {
            'dimensions': 5,
            'bounds': [(-5, 5)] * 5,
            'fitness_function': lambda x: -sum(x**2)  # Maximize negative sphere
        }
        
        await dga.initialize_population(problem_space)
        
        # Mock fitness evaluations
        mock_orchestrator.a2a_server.request_fitness_evaluation = AsyncMock(
            return_value={"fitness": -10.0, "evaluation_time": 0.1}
        )
        
        # Run one evolution cycle
        best_fitness = await dga.evolve_generation()
        
        assert isinstance(best_fitness, float)
        assert dga.generation == 1
        assert len(dga.evolution_history) == 1
    
    def test_migration_protocol(self):
        """Test migration protocol implementation."""
        protocol = MigrationProtocol()
        
        # Test migration rate calculation
        stats = {
            'diversity': 0.3,
            'stagnation_count': 5,
            'peer_diversity': 0.7
        }
        
        rate = protocol.calculate_migration_rate(stats)
        assert 0.0 <= rate <= 1.0
        
        # Test individual selection
        individuals = [
            Individual(id=f"ind_{i}", genes=[i] * 5, fitness=i/10)
            for i in range(10)
        ]
        
        selected = protocol.select_migrants(individuals, count=3)
        assert len(selected) == 3
        assert all(ind.fitness >= 0.5 for ind in selected)  # Should select high fitness
    
    @pytest.mark.asyncio
    async def test_ga_evolution_events(self, dga, mock_orchestrator):
        """Test GA evolution event broadcasting."""
        event = GAEvolutionEvent(
            event_type="generation_complete",
            generation=5,
            best_fitness=0.95,
            avg_fitness=0.75,
            diversity=0.6,
            population_size=50,
            timestamp=asyncio.get_event_loop().time()
        )
        
        await dga.broadcast_evolution_event(event)
        
        mock_orchestrator.a2a_server.broadcast_evolution_event.assert_called_once()
        
        # Verify event data
        call_args = mock_orchestrator.a2a_server.broadcast_evolution_event.call_args[0][0]
        assert call_args['event_type'] == "generation_complete"
        assert call_args['generation'] == 5
        assert call_args['best_fitness'] == 0.95
    
    @pytest.mark.asyncio
    async def test_distributed_population_management(self, dga, mock_orchestrator):
        """Test distributed population management across A2A network."""
        # Mock peer population data
        mock_orchestrator.get_peer_agents = AsyncMock(return_value=[
            {"id": "peer1", "population_size": 30, "best_fitness": 0.8},
            {"id": "peer2", "population_size": 40, "best_fitness": 0.9}
        ])
        
        # Test population synchronization
        await dga.synchronize_populations()
        
        # Verify coordination occurred
        mock_orchestrator.get_peer_agents.assert_called_once()
    
    def test_fitness_sharing(self, dga):
        """Test fitness sharing mechanisms."""
        individuals = [
            Individual(id=f"ind_{i}", genes=[i] * 3, fitness=i/10)
            for i in range(5)
        ]
        
        # Apply fitness sharing
        shared_fitness = dga.apply_fitness_sharing(individuals, sigma=0.5)
        
        assert len(shared_fitness) == 5
        assert all(isinstance(f, float) for f in shared_fitness)
        # Fitness sharing should reduce fitness for similar individuals
        assert shared_fitness[0] <= individuals[0].fitness


class TestDistributedFitnessEvaluator:
    """Test distributed fitness evaluator."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Mock orchestrator."""
        orchestrator = Mock()
        orchestrator.a2a_server = Mock()
        orchestrator.a2a_server.request_fitness_evaluation = AsyncMock()
        return orchestrator
    
    @pytest.fixture
    def evaluator(self, mock_orchestrator):
        """Create evaluator instance."""
        return DistributedFitnessEvaluator(mock_orchestrator)
    
    @pytest.mark.asyncio
    async def test_parallel_evaluation(self, evaluator, mock_orchestrator):
        """Test parallel fitness evaluation."""
        individuals = [
            Individual(id=f"ind_{i}", genes=[i] * 3)
            for i in range(5)
        ]
        
        # Mock varying evaluation results
        mock_orchestrator.a2a_server.request_fitness_evaluation = AsyncMock(
            side_effect=[
                {"fitness": 0.1 * i, "evaluation_time": 0.1}
                for i in range(5)
            ]
        )
        
        fitness_values = await evaluator.evaluate_population(individuals)
        
        assert len(fitness_values) == 5
        assert fitness_values == [0.0, 0.1, 0.2, 0.3, 0.4]
        assert mock_orchestrator.a2a_server.request_fitness_evaluation.call_count == 5
    
    @pytest.mark.asyncio
    async def test_evaluation_caching(self, evaluator):
        """Test fitness evaluation caching."""
        individual = Individual(id="test", genes=[1, 2, 3])
        
        # First evaluation
        evaluator.cache[individual.id] = 0.5
        fitness = await evaluator.evaluate_individual(individual)
        
        assert fitness == 0.5  # Should use cached value
    
    @pytest.mark.asyncio
    async def test_evaluation_failure_handling(self, evaluator, mock_orchestrator):
        """Test handling of evaluation failures."""
        individual = Individual(id="test", genes=[1, 2, 3])
        
        # Mock evaluation failure
        mock_orchestrator.a2a_server.request_fitness_evaluation = AsyncMock(
            side_effect=Exception("Evaluation failed")
        )
        
        fitness = await evaluator.evaluate_individual(individual)
        
        assert fitness == 0.0  # Should return default fitness on failure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
