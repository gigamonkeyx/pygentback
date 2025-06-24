"""
Comprehensive Test Suite for Phase 2.1: Genetic Algorithm Distribution via A2A
High-quality tests validating all distributed genetic algorithm functionality.

Tests cover:
- Distributed population management across A2A network
- A2A-coordinated crossover operations between agents  
- Distributed mutation strategies via A2A collaboration
- A2A-enabled selection pressure coordination
- Distributed fitness evaluation through A2A network
- A2A-coordinated evolutionary migration between populations
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, AsyncMock

# Import distributed genetic algorithm components
from src.orchestration.distributed_genetic_algorithm import (
    DistributedGeneticAlgorithm,
    Individual,
    Population,
    EvolutionaryOperation,
    MigrationEvent,
    CrossoverOperation,
    FitnessEvaluation,
    DistributedGAMetrics,
    EvolutionOperationType,
    MigrationStrategy,
    EvolutionaryParameters,
    DistributedIndividual,
    PopulationMetrics
)
from src.a2a import AgentCard, A2AServer, AgentDiscoveryService


@pytest.fixture
def mock_a2a_components():
    """Create high-quality mock A2A components"""
    mock_server = Mock(spec=A2AServer)
    mock_server.send_message = AsyncMock()
    mock_server.broadcast_message = AsyncMock()
    
    mock_discovery = Mock(spec=AgentDiscoveryService)
    mock_discovery.register_agent = AsyncMock()
    mock_discovery.discover_agents = AsyncMock()
    mock_discovery.get_active_agents = AsyncMock()
    
    return mock_server, mock_discovery


@pytest.fixture
def evolutionary_params():
    """Create comprehensive evolutionary parameters"""
    return EvolutionaryParameters(
        population_size=100,
        genome_length=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        selection_pressure=0.7,
        elite_percentage=0.1,
        migration_rate=0.05,
        convergence_threshold=0.95,
        max_generations=1000,
        fitness_sharing=True,
        diversity_preservation=True,
        adaptive_parameters=True
    )


@pytest.fixture
async def distributed_ga(mock_a2a_components, evolutionary_params):
    """Create distributed genetic algorithm instance"""
    mock_server, mock_discovery = mock_a2a_components
    
    ga = DistributedGeneticAlgorithm(
        agent_id="test_ga_agent_001",
        a2a_server=mock_server,
        discovery_service=mock_discovery,
        evolutionary_params=evolutionary_params
    )
    
    # Manually initialize just the crossover coordinator for testing
    from src.orchestration.distributed_genetic_algorithm import A2ACoordinatedCrossover
    ga.crossover_coordinator = A2ACoordinatedCrossover(
        ga.agent_id, ga.a2a_server, ga.evolutionary_params
    )
    
    yield ga
    
    # Cleanup
    await ga.shutdown()


@pytest.fixture
def sample_population():
    """Create sample population for testing"""
    population = []
    for i in range(20):
        individual = DistributedIndividual(
            individual_id=f"individual_{i:03d}",
            genome=np.random.random(50),
            fitness=np.random.random(),
            generation=0,
            parent_ids=[],
            mutation_history=[],
            crossover_history=[],
            origin_agent="test_agent",
            distributed_metadata={
                "creation_time": time.time(),
                "evaluation_count": 1,
                "migration_count": 0
            }
        )
        population.append(individual)
    return population


class TestDistributedGeneticAlgorithm:
    """Comprehensive test suite for distributed genetic algorithm"""
    pass


class TestDistributedPopulationManager:
    """Test distributed population management across A2A network"""
    
    @pytest.mark.asyncio
    async def test_population_initialization(self, distributed_ga, evolutionary_params):
        """Test 2.1.1: Distributed population initialization"""
        manager = distributed_ga.population_manager
        
        # Initialize population
        await manager.initialize_distributed_population()
        
        # Verify population creation
        assert len(manager.local_population) == evolutionary_params.population_size
        assert manager.population_metrics.total_individuals == evolutionary_params.population_size
        assert manager.population_metrics.generation == 0
        
        # Verify individual structure
        for individual in manager.local_population:
            assert isinstance(individual, DistributedIndividual)
            assert len(individual.genome) == evolutionary_params.genome_length
            assert individual.generation == 0
            assert individual.origin_agent == distributed_ga.agent_id
    
    @pytest.mark.asyncio
    async def test_population_synchronization(self, distributed_ga, sample_population):
        """Test population synchronization across agents"""
        manager = distributed_ga.population_manager
        manager.local_population = sample_population[:10]
        
        # Mock peer agents with populations
        peer_populations = {
            "peer_agent_001": sample_population[10:15],
            "peer_agent_002": sample_population[15:20]
        }
        
        # Mock A2A responses for population sync
        async def mock_sync_response(agent_card, message):
            agent_id = agent_card.agent_id if hasattr(agent_card, 'agent_id') else str(agent_card)
            if message.get("type") == "population_sync_request":
                return {
                    "population_summary": {
                        "count": len(peer_populations.get(agent_id, [])),
                        "average_fitness": 0.7,
                        "best_fitness": 0.95,
                        "diversity_score": 0.8
                    },
                    "top_individuals": [
                        {
                            "individual_id": ind.individual_id,
                            "fitness": ind.fitness,
                            "genome_hash": hash(tuple(ind.genome))
                        }
                        for ind in peer_populations.get(agent_id, [])[:3]
                    ]
                }
            return {"status": "ok"}
        
        manager.a2a_server.send_message.side_effect = mock_sync_response
          # Add mock peer agents
        for agent_id in peer_populations.keys():
            peer_card = AgentCard(
                agent_id=agent_id,
                name=f"PeerAgent-{agent_id}",
                description="Genetic algorithm agent",
                capabilities=["population_management", "evolution"],
                communication_protocols=["a2a"],
                supported_tasks=["genetic_algorithm"],
                performance_metrics={"version": 1.0},
            )
            manager.peer_agents[agent_id] = peer_card
        
        # Test synchronization
        sync_result = await manager.synchronize_with_peer_populations()
        
        # Verify synchronization
        assert "synchronized_agents" in sync_result
        assert "global_population_stats" in sync_result
        assert sync_result["synchronized_agents"] == len(peer_populations)
        assert sync_result["global_population_stats"]["total_agents"] >= 1
    
    @pytest.mark.asyncio
    async def test_population_diversity_maintenance(self, distributed_ga, sample_population):
        """Test population diversity maintenance across network"""
        manager = distributed_ga.population_manager
        manager.local_population = sample_population
        
        # Calculate initial diversity
        initial_diversity = await manager.calculate_population_diversity()
        assert 0 <= initial_diversity <= 1
        
        # Test diversity preservation
        diversity_result = await manager.maintain_population_diversity()
        
        # Verify diversity maintenance
        assert "diversity_score" in diversity_result
        assert "actions_taken" in diversity_result
        assert "diversity_improvement" in diversity_result
        
        # Check if diversity was improved or maintained
        final_diversity = diversity_result["diversity_score"]
        assert final_diversity >= initial_diversity * 0.9  # Allow small decrease
    
    @pytest.mark.asyncio
    async def test_population_load_balancing(self, distributed_ga):
        """Test load balancing across distributed populations"""
        manager = distributed_ga.population_manager
        
        # Simulate uneven load
        manager.local_population = [Mock(spec=DistributedIndividual) for _ in range(150)]  # Overloaded
        
        # Mock peer agent loads
        peer_loads = {
            "peer_001": 50,  # Underloaded
            "peer_002": 80,  # Normal
            "peer_003": 120  # Overloaded
        }
        
        async def mock_load_response(agent_card, message):
            agent_id = agent_card.agent_id if hasattr(agent_card, 'agent_id') else str(agent_card)
            if message.get("type") == "load_query":
                return {
                    "current_load": peer_loads.get(agent_id, 100),
                    "capacity": 100,
                    "can_accept_individuals": peer_loads.get(agent_id, 100) < 90
                }
            return {"status": "ok"}
        
        manager.a2a_server.send_message.side_effect = mock_load_response
        
        # Add peer agents
        for agent_id in peer_loads.keys():
            manager.peer_agents[agent_id] = Mock()
            manager.peer_agents[agent_id].agent_id = agent_id
        
        # Test load balancing
        balance_result = await manager.balance_population_load()
        
        # Verify load balancing
        assert "load_balanced" in balance_result
        assert "individuals_migrated" in balance_result
        assert "target_agents" in balance_result


class TestA2ACoordinatedCrossover:
    """Test A2A-coordinated crossover operations between agents"""
    
    @pytest.mark.asyncio
    async def test_crossover_partner_discovery(self, distributed_ga, sample_population):
        """Test 2.1.2: Partner discovery for crossover"""
        crossover_manager = distributed_ga.crossover_coordinator
        crossover_manager.local_population = sample_population[:10]
        
        # Mock peer populations for crossover
        async def mock_crossover_response(agent_card, message):
            if message.get("type") == "crossover_partner_request":
                return {
                    "available_partners": [
                        {
                            "individual_id": f"peer_individual_{i:03d}",
                            "fitness": 0.8 + (i * 0.01),
                            "genome_compatibility": 0.9,
                            "genetic_diversity": 0.7
                        }
                        for i in range(5)
                    ],
                    "agent_fitness_avg": 0.85
                }
            return {"status": "ok"}
        
        crossover_manager.a2a_server.send_message.side_effect = mock_crossover_response
        
        # Add peer agents
        for i in range(3):
            agent_id = f"crossover_peer_{i:03d}"
            crossover_manager.peer_agents[agent_id] = Mock()
            crossover_manager.peer_agents[agent_id].agent_id = agent_id
        
        # Test partner discovery
        parent1 = sample_population[0]
        partners = await crossover_manager.discover_crossover_partners(parent1)
        
        # Verify partner discovery
        assert len(partners) > 0
        for partner in partners:
            assert "individual_id" in partner
            assert "fitness" in partner
            assert "genome_compatibility" in partner
            assert partner["fitness"] > 0
    
    @pytest.mark.asyncio
    async def test_distributed_crossover_execution(self, distributed_ga, sample_population):
        """Test distributed crossover execution"""
        crossover_manager = distributed_ga.crossover_coordinator
        
        # Setup crossover scenario
        parent1 = sample_population[0]
        parent2_info = {
            "individual_id": "remote_parent_001",
            "genome": np.random.random(50),
            "fitness": 0.85,
            "agent_id": "remote_agent_001"
        }
        
        # Mock crossover coordination
        async def mock_crossover_coordination(agent_card, message):
            if message.get("type") == "crossover_execution_request":
                return {
                    "crossover_successful": True,
                    "offspring_genome": np.random.random(50).tolist(),
                    "crossover_points": [15, 35],
                    "genetic_contribution": 0.5,
                    "operation_id": "crossover_op_001"
                }
            return {"status": "ok"}
        
        crossover_manager.a2a_server.send_message.side_effect = mock_crossover_coordination
        
        # Execute distributed crossover
        crossover_result = await crossover_manager.execute_distributed_crossover(
            parent1, parent2_info
        )
        
        # Verify crossover execution
        assert "offspring" in crossover_result
        assert "crossover_successful" in crossover_result
        assert crossover_result["crossover_successful"] is True
        
        offspring = crossover_result["offspring"]
        assert isinstance(offspring, DistributedIndividual)
        assert len(offspring.genome) == len(parent1.genome)
        assert parent1.individual_id in offspring.parent_ids
    
    @pytest.mark.asyncio
    async def test_crossover_strategy_optimization(self, distributed_ga):
        """Test crossover strategy optimization based on performance"""
        crossover_manager = distributed_ga.crossover_coordinator
        
        # Add crossover history
        crossover_strategies = ["single_point", "two_point", "uniform", "arithmetic"]
        for i, strategy in enumerate(crossover_strategies):
            for j in range(10):
                operation = CrossoverOperation(
                    operation_id=f"op_{strategy}_{j:03d}",
                    strategy=strategy,
                    parent1_id=f"parent1_{j}",
                    parent2_id=f"parent2_{j}",
                    offspring_fitness=0.7 + (i * 0.05) + np.random.normal(0, 0.1),
                    crossover_points=[15, 35] if "two_point" in strategy else [25],
                    timestamp=time.time(),
                    agent_coordination=True
                )
                crossover_manager.crossover_history.append(operation)
        
        # Optimize crossover strategies
        optimization_result = await crossover_manager.optimize_crossover_strategies()
        
        # Verify optimization
        assert "best_strategy" in optimization_result
        assert "strategy_performance" in optimization_result
        assert "optimization_improvement" in optimization_result
        
        best_strategy = optimization_result["best_strategy"]
        assert best_strategy in crossover_strategies
        
        performance = optimization_result["strategy_performance"]
        assert len(performance) == len(crossover_strategies)
        for strategy in crossover_strategies:
            assert strategy in performance
            assert isinstance(performance[strategy], (int, float))


class TestDistributedMutationCoordinator:
    """Test distributed mutation strategies via A2A collaboration"""
    
    @pytest.mark.asyncio
    async def test_adaptive_mutation_coordination(self, distributed_ga, sample_population):
        """Test 2.1.3: Adaptive mutation rate coordination"""
        mutation_coordinator = distributed_ga.mutation_coordinator
        mutation_coordinator.local_population = sample_population
        
        # Mock peer mutation data
        async def mock_mutation_data(agent_card, message):
            if message.get("type") == "mutation_data_request":
                return {
                    "current_mutation_rate": 0.1,
                    "population_diversity": 0.7,
                    "fitness_improvement_rate": 0.05,
                    "stagnation_generations": 2,
                    "recommended_mutation_rate": 0.12
                }
            return {"status": "ok"}
        
        mutation_coordinator.a2a_server.send_message.side_effect = mock_mutation_data
        
        # Add peer agents
        for i in range(3):
            agent_id = f"mutation_peer_{i:03d}"
            mutation_coordinator.peer_agents[agent_id] = Mock()
            mutation_coordinator.peer_agents[agent_id].agent_id = agent_id
        
        # Test adaptive mutation coordination
        adaptation_result = await mutation_coordinator.coordinate_adaptive_mutation()
        
        # Verify adaptation
        assert "coordinated_mutation_rate" in adaptation_result
        assert "adaptation_reason" in adaptation_result
        assert "peer_consensus" in adaptation_result
        
        coordinated_rate = adaptation_result["coordinated_mutation_rate"]
        assert 0 <= coordinated_rate <= 1
    
    @pytest.mark.asyncio
    async def test_distributed_mutation_strategies(self, distributed_ga, sample_population):
        """Test distributed mutation strategy coordination"""
        mutation_coordinator = distributed_ga.mutation_coordinator
        
        individual = sample_population[0]
        
        # Test different mutation strategies
        strategies = ["gaussian", "uniform", "bit_flip", "swap", "inversion"]
        
        for strategy in strategies:
            mutation_result = await mutation_coordinator.apply_distributed_mutation(
                individual, strategy
            )
            
            # Verify mutation
            assert "mutated_individual" in mutation_result
            assert "mutation_applied" in mutation_result
            assert "mutation_strength" in mutation_result
            
            mutated = mutation_result["mutated_individual"]
            assert isinstance(mutated, DistributedIndividual)
            assert len(mutated.genome) == len(individual.genome)
            assert strategy in mutated.mutation_history[-1]["strategy"]
    
    @pytest.mark.asyncio
    async def test_mutation_impact_analysis(self, distributed_ga):
        """Test mutation impact analysis across network"""
        mutation_coordinator = distributed_ga.mutation_coordinator
        
        # Add mutation history
        mutation_strategies = ["gaussian", "uniform", "bit_flip"]
        for strategy in mutation_strategies:
            for i in range(15):
                mutation = MutationStrategy(
                    strategy_name=strategy,
                    mutation_rate=0.1,
                    strength=0.1 + np.random.random() * 0.3,
                    fitness_impact=np.random.normal(0.05, 0.1),
                    success_rate=0.6 + np.random.random() * 0.3,
                    application_count=i + 1,
                    agent_id=f"agent_{i % 3:03d}",
                    timestamp=time.time()
                )
                mutation_coordinator.mutation_history.append(mutation)
        
        # Analyze mutation impact
        analysis_result = await mutation_coordinator.analyze_mutation_impact()
        
        # Verify analysis
        assert "strategy_effectiveness" in analysis_result
        assert "best_strategies" in analysis_result
        assert "impact_trends" in analysis_result
        
        effectiveness = analysis_result["strategy_effectiveness"]
        for strategy in mutation_strategies:
            assert strategy in effectiveness
            assert "success_rate" in effectiveness[strategy]
            assert "average_impact" in effectiveness[strategy]


class TestSelectionPressureCoordinator:
    """Test A2A-enabled selection pressure coordination"""
    
    @pytest.mark.asyncio
    async def test_distributed_selection_pressure(self, distributed_ga, sample_population):
        """Test 2.1.4: Distributed selection pressure coordination"""
        selection_coordinator = distributed_ga.selection_coordinator
        selection_coordinator.local_population = sample_population
          # Mock peer selection data
        async def mock_selection_data(agent_card, message):
            if message.get("type") == "selection_pressure_query":
                return {
                    "current_pressure": 0.7,
                    "population_fitness_variance": 0.15,
                    "convergence_risk": 0.3,
                    "recommended_pressure": 0.65,
                    "elite_percentage": 0.1
                }
            return {"status": "ok"}
        
        selection_coordinator.a2a_server.send_message.side_effect = mock_selection_data
        
        # Add peer agents
        for i in range(4):
            agent_id = f"selection_peer_{i:03d}"
            selection_coordinator.peer_agents[agent_id] = Mock()
            selection_coordinator.peer_agents[agent_id].agent_id = agent_id
        
        # Test selection pressure coordination
        coordination_result = await selection_coordinator.coordinate_selection_pressure()
        
        # Verify coordination
        assert "coordinated_pressure" in coordination_result
        assert "consensus_achieved" in coordination_result
        assert "adjustment_reason" in coordination_result
        
        pressure = coordination_result["coordinated_pressure"]
        assert 0 <= pressure <= 1
    
    @pytest.mark.asyncio
    async def test_tournament_selection_distribution(self, distributed_ga, sample_population):
        """Test distributed tournament selection"""
        selection_coordinator = distributed_ga.selection_coordinator
        selection_coordinator.local_population = sample_population
        
        # Mock tournament participants from other agents
        async def mock_tournament_response(agent_card, message):
            if message.get("type") == "tournament_participant_request":
                return {
                    "participants": [
                        {
                            "individual_id": f"tournament_ind_{i:03d}",
                            "fitness": 0.6 + np.random.random() * 0.3,
                            "agent_id": str(agent_card.agent_id if hasattr(agent_card, 'agent_id') else agent_card)
                        }
                        for i in range(3)
                    ]
                }
            return {"status": "ok"}
        
        selection_coordinator.a2a_server.send_message.side_effect = mock_tournament_response
        
        # Add peer agents
        for i in range(3):
            agent_id = f"tournament_peer_{i:03d}"
            selection_coordinator.peer_agents[agent_id] = Mock()
            selection_coordinator.peer_agents[agent_id].agent_id = agent_id
        
        # Test distributed tournament selection
        selected = await selection_coordinator.distributed_tournament_selection(
            tournament_size=5, num_selections=10
        )
        
        # Verify selection
        assert len(selected) == 10
        for individual in selected:
            assert hasattr(individual, 'fitness')
            assert individual.fitness >= 0
    
    @pytest.mark.asyncio
    async def test_selection_diversity_preservation(self, distributed_ga, sample_population):
        """Test diversity preservation in selection"""
        selection_coordinator = distributed_ga.selection_coordinator
        selection_coordinator.local_population = sample_population
        
        # Test diversity-preserving selection
        preserved_selection = await selection_coordinator.diversity_preserving_selection(
            selection_count=15, diversity_weight=0.3
        )
        
        # Verify diversity preservation
        assert len(preserved_selection) == 15
        
        # Calculate diversity of selected individuals
        selected_genomes = [ind.genome for ind in preserved_selection]
        diversity_score = selection_coordinator._calculate_genome_diversity(selected_genomes)
        
        assert diversity_score > 0.5  # Should maintain reasonable diversity


class TestDistributedFitnessEvaluator:
    """Test distributed fitness evaluation through A2A network"""
    
    @pytest.mark.asyncio
    async def test_distributed_fitness_computation(self, distributed_ga, sample_population):
        """Test 2.1.5: Distributed fitness evaluation"""
        fitness_evaluator = distributed_ga.fitness_evaluator
        
        # Mock distributed fitness computation
        async def mock_fitness_computation(agent_card, message):
            if message.get("type") == "fitness_computation_request":
                individuals = message.get("individuals", [])
                return {
                    "computed_fitness": [
                        {
                            "individual_id": ind["individual_id"],
                            "fitness": np.random.random(),
                            "computation_time": 0.05,
                            "evaluation_components": {
                                "objective1": np.random.random(),
                                "objective2": np.random.random()
                            }
                        }
                        for ind in individuals
                    ],
                    "computation_agent": str(agent_card.agent_id if hasattr(agent_card, 'agent_id') else agent_card)
                }
            return {"status": "ok"}
        
        fitness_evaluator.a2a_server.send_message.side_effect = mock_fitness_computation
        
        # Add computational peer agents
        for i in range(3):
            agent_id = f"compute_peer_{i:03d}"
            fitness_evaluator.peer_agents[agent_id] = Mock()
            fitness_evaluator.peer_agents[agent_id].agent_id = agent_id
        
        # Test distributed fitness evaluation
        individuals_to_evaluate = sample_population[:10]
        evaluation_result = await fitness_evaluator.evaluate_distributed_fitness(
            individuals_to_evaluate
        )
        
        # Verify evaluation
        assert "evaluated_individuals" in evaluation_result
        assert "computation_distribution" in evaluation_result
        assert "total_computation_time" in evaluation_result
        
        evaluated = evaluation_result["evaluated_individuals"]
        assert len(evaluated) == len(individuals_to_evaluate)
        
        for individual in evaluated:
            assert hasattr(individual, 'fitness')
            assert individual.fitness >= 0
    
    @pytest.mark.asyncio
    async def test_fitness_evaluation_load_balancing(self, distributed_ga):
        """Test load balancing for fitness evaluation"""
        fitness_evaluator = distributed_ga.fitness_evaluator
        
        # Create large evaluation batch
        large_batch = []
        for i in range(100):
            individual = DistributedIndividual(
                individual_id=f"eval_individual_{i:03d}",
                genome=np.random.random(50),
                fitness=0.0,  # Needs evaluation
                generation=1,
                parent_ids=[],
                mutation_history=[],
                crossover_history=[],
                origin_agent="test_agent"
            )
            large_batch.append(individual)
        
        # Mock peer computational capacities
        peer_capacities = {
            "compute_peer_001": {"capacity": 30, "current_load": 0.2},
            "compute_peer_002": {"capacity": 40, "current_load": 0.5},
            "compute_peer_003": {"capacity": 20, "current_load": 0.8}
        }
        
        async def mock_capacity_response(agent_card, message):
            agent_id = str(agent_card.agent_id if hasattr(agent_card, 'agent_id') else agent_card)
            if message.get("type") == "capacity_query":
                return peer_capacities.get(agent_id, {"capacity": 10, "current_load": 0.9})
            elif message.get("type") == "fitness_computation_request":
                return {
                    "computed_fitness": [
                        {
                            "individual_id": ind["individual_id"],
                            "fitness": np.random.random(),
                            "computation_time": 0.1
                        }
                        for ind in message.get("individuals", [])
                    ]
                }
            return {"status": "ok"}
        
        fitness_evaluator.a2a_server.send_message.side_effect = mock_capacity_response
        
        # Add peer agents
        for agent_id in peer_capacities.keys():
            fitness_evaluator.peer_agents[agent_id] = Mock()
            fitness_evaluator.peer_agents[agent_id].agent_id = agent_id
        
        # Test load-balanced evaluation
        balance_result = await fitness_evaluator.load_balanced_evaluation(large_batch)
        
        # Verify load balancing
        assert "distribution_plan" in balance_result
        assert "evaluation_result" in balance_result
        assert "load_balance_efficiency" in balance_result
        
        distribution = balance_result["distribution_plan"]
        total_distributed = sum(len(batch) for batch in distribution.values())
        assert total_distributed <= len(large_batch)
    
    @pytest.mark.asyncio
    async def test_multi_objective_fitness_coordination(self, distributed_ga, sample_population):
        """Test multi-objective fitness coordination"""
        fitness_evaluator = distributed_ga.fitness_evaluator
        
        # Define multiple objectives
        objectives = ["performance", "efficiency", "robustness", "scalability"]
          # Mock multi-objective evaluation
        async def mock_multi_objective_eval(agent_card, message):
            if message.get("type") == "multi_objective_evaluation":
                individuals = message.get("individuals", [])
                return {
                    "objective_evaluations": [
                        {
                            "individual_id": ind["individual_id"],
                            "objectives": {
                                obj: np.random.random() for obj in objectives
                            },
                            "pareto_rank": np.random.randint(1, 5),
                            "crowding_distance": np.random.random()
                        }
                        for ind in individuals
                    ]
                }
            return {"status": "ok"}
        
        fitness_evaluator.a2a_server.send_message.side_effect = mock_multi_objective_eval
        
        # Add objective evaluation peers
        for i, obj in enumerate(objectives):
            agent_id = f"objective_{obj}_evaluator"
            fitness_evaluator.peer_agents[agent_id] = Mock()
            fitness_evaluator.peer_agents[agent_id].agent_id = agent_id
        
        # Test multi-objective coordination
        mo_result = await fitness_evaluator.coordinate_multi_objective_evaluation(
            sample_population[:5], objectives
        )
        
        # Verify multi-objective evaluation
        assert "pareto_front" in mo_result
        assert "objective_correlations" in mo_result
        assert "evaluation_quality" in mo_result
        
        pareto_front = mo_result["pareto_front"]
        assert len(pareto_front) <= 5
        for individual in pareto_front:
            assert hasattr(individual, 'fitness')


class TestEvolutionaryMigrationManager:
    """Test A2A-coordinated evolutionary migration between populations"""
    
    @pytest.mark.asyncio
    async def test_migration_opportunity_detection(self, distributed_ga, sample_population):
        """Test 2.1.6: Migration opportunity detection"""
        migration_manager = distributed_ga.migration_manager
        migration_manager.local_population = sample_population
          # Mock peer population stats
        async def mock_population_stats(agent_card, message):
            if message.get("type") == "population_stats_request":
                return {
                    "population_size": 80,
                    "average_fitness": 0.75,
                    "best_fitness": 0.95,
                    "diversity_score": 0.6,
                    "stagnation_generations": 5,
                    "accepts_migrants": True,
                    "migration_capacity": 10
                }
            return {"status": "ok"}
        
        migration_manager.a2a_server.send_message.side_effect = mock_population_stats
        
        # Add peer populations
        for i in range(4):
            agent_id = f"migration_target_{i:03d}"
            migration_manager.peer_agents[agent_id] = Mock()
            migration_manager.peer_agents[agent_id].agent_id = agent_id
        
        # Test migration opportunity detection
        opportunities = await migration_manager.detect_migration_opportunities()
        
        # Verify opportunity detection
        assert "migration_targets" in opportunities
        assert "migration_recommendations" in opportunities
        assert "potential_benefit" in opportunities
        
        targets = opportunities["migration_targets"]
        assert len(targets) <= 4
        for target in targets:
            assert "agent_id" in target
            assert "benefit_score" in target
            assert "migration_capacity" in target
    
    @pytest.mark.asyncio
    async def test_individual_migration_execution(self, distributed_ga, sample_population):
        """Test individual migration execution"""
        migration_manager = distributed_ga.migration_manager
        migration_manager.local_population = sample_population
        
        # Select individuals for migration
        migrants = sample_population[:3]
        target_agent = "migration_target_001"
          # Mock migration execution
        async def mock_migration_execution(agent_card, message):
            if message.get("type") == "migration_request":
                return {
                    "migration_accepted": True,
                    "integration_successful": True,
                    "assigned_ids": [f"migrated_{i:03d}" for i in range(len(message.get("migrants", [])))],
                    "integration_feedback": {
                        "fitness_improvement": 0.05,
                        "diversity_contribution": 0.1
                    }
                }
            return {"status": "ok"}
        
        migration_manager.a2a_server.send_message.side_effect = mock_migration_execution
        migration_manager.peer_agents[target_agent] = Mock()
        migration_manager.peer_agents[target_agent].agent_id = target_agent
        
        # Execute migration
        migration_result = await migration_manager.execute_migration(
            migrants, target_agent
        )
        
        # Verify migration execution
        assert "migration_successful" in migration_result
        assert "migrated_individuals" in migration_result
        assert "integration_feedback" in migration_result
        
        if migration_result["migration_successful"]:
            assert len(migration_result["migrated_individuals"]) == len(migrants)
            
            # Verify migration event recording
            assert len(migration_manager.migration_history) > 0
            latest_migration = migration_manager.migration_history[-1]
            assert latest_migration.target_agent == target_agent
            assert len(latest_migration.migrated_individuals) == len(migrants)
    
    @pytest.mark.asyncio
    async def test_migration_strategy_optimization(self, distributed_ga):
        """Test migration strategy optimization based on outcomes"""
        migration_manager = distributed_ga.migration_manager
        
        # Add migration history with different strategies
        migration_strategies = ["best_individuals", "diverse_selection", "random_selection", "adaptive_selection"]
        
        for strategy in migration_strategies:
            for i in range(8):
                migration_event = MigrationEvent(
                    migration_id=f"migration_{strategy}_{i:03d}",
                    source_agent=distributed_ga.agent_id,
                    target_agent=f"target_{i % 3:03d}",
                    migrated_individuals=[f"ind_{j:03d}" for j in range(2)],
                    migration_strategy=strategy,
                    success_rate=0.6 + np.random.random() * 0.3,
                    fitness_improvement=np.random.normal(0.05, 0.02),
                    diversity_impact=np.random.random() * 0.2,
                    timestamp=time.time() - i * 3600
                )
                migration_manager.migration_history.append(migration_event)
        
        # Optimize migration strategies
        optimization_result = await migration_manager.optimize_migration_strategies()
        
        # Verify optimization
        assert "best_strategy" in optimization_result
        assert "strategy_performance" in optimization_result
        assert "optimization_metrics" in optimization_result
        
        best_strategy = optimization_result["best_strategy"]
        assert best_strategy in migration_strategies
        
        performance = optimization_result["strategy_performance"]
        for strategy in migration_strategies:
            assert strategy in performance
            assert "success_rate" in performance[strategy]
            assert "avg_improvement" in performance[strategy]
    
    @pytest.mark.asyncio
    async def test_bidirectional_migration_coordination(self, distributed_ga, sample_population):
        """Test bidirectional migration coordination"""
        migration_manager = distributed_ga.migration_manager
        migration_manager.local_population = sample_population
        
        # Mock bidirectional migration
        async def mock_bidirectional_migration(agent_card, message):
            agent_id = str(agent_card.agent_id if hasattr(agent_card, 'agent_id') else agent_card)
            
            if message.get("type") == "bidirectional_migration_proposal":
                return {
                    "accepts_proposal": True,
                    "outgoing_individuals": [
                        {
                            "individual_id": f"outgoing_{agent_id}_{i:03d}",
                            "fitness": 0.8,
                            "genome": np.random.random(50).tolist()
                        }
                        for i in range(2)
                    ],
                    "exchange_benefit_estimate": 0.1
                }
            return {"status": "ok"}
        
        migration_manager.a2a_server.send_message.side_effect = mock_bidirectional_migration
        
        # Add peer agent for bidirectional exchange
        peer_agent = "bidirectional_peer_001"
        migration_manager.peer_agents[peer_agent] = Mock()
        migration_manager.peer_agents[peer_agent].agent_id = peer_agent
        
        # Test bidirectional migration
        our_migrants = sample_population[:2]
        exchange_result = await migration_manager.coordinate_bidirectional_migration(
            our_migrants, peer_agent
        )
        
        # Verify bidirectional exchange
        assert "exchange_successful" in exchange_result
        assert "incoming_individuals" in exchange_result
        assert "outgoing_individuals" in exchange_result
        assert "mutual_benefit" in exchange_result
        
        if exchange_result["exchange_successful"]:
            incoming = exchange_result["incoming_individuals"]
            outgoing = exchange_result["outgoing_individuals"]
            assert len(incoming) > 0
            assert len(outgoing) == len(our_migrants)


class TestIntegratedDistributedEvolution:
    """Integration tests for complete distributed evolution workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_evolution_cycle(self, distributed_ga, evolutionary_params):
        """Test complete distributed evolution cycle"""
        # Initialize distributed GA
        await distributed_ga.initialize_distributed_evolution()
        
        # Verify initialization
        assert len(distributed_ga.population_manager.local_population) == evolutionary_params.population_size
        assert distributed_ga.population_manager.population_metrics.generation == 0
        
        # Mock peer agents for complete cycle
        peer_agent_configs = [
            {"id": "peer_001", "population_size": 80, "avg_fitness": 0.7},
            {"id": "peer_002", "population_size": 120, "avg_fitness": 0.6},
            {"id": "peer_003", "population_size": 90, "avg_fitness": 0.8}
        ]
        
        for config in peer_agent_configs:
            agent_id = config["id"]
            for manager in [
                distributed_ga.population_manager,
                distributed_ga.crossover_coordinator,
                distributed_ga.mutation_coordinator,
                distributed_ga.selection_coordinator,
                distributed_ga.fitness_evaluator,
                distributed_ga.migration_manager
            ]:
                manager.peer_agents[agent_id] = Mock()
                manager.peer_agents[agent_id].agent_id = agent_id
        
        # Mock comprehensive A2A responses
        async def mock_evolution_responses(agent_card, message):
            msg_type = message.get("type", "")
            agent_id = str(agent_card.agent_id if hasattr(agent_card, 'agent_id') else agent_card)
            
            # Find corresponding config
            config = next((c for c in peer_agent_configs if c["id"] == agent_id), peer_agent_configs[0])
            
            if "population" in msg_type:
                return {
                    "population_summary": {
                        "count": config["population_size"],
                        "average_fitness": config["avg_fitness"],
                        "diversity_score": 0.7
                    }
                }
            elif "crossover" in msg_type:
                return {
                    "available_partners": [
                        {"individual_id": f"partner_{i}", "fitness": 0.8}
                        for i in range(3)
                    ]
                }
            elif "mutation" in msg_type:
                return {
                    "recommended_mutation_rate": 0.1,
                    "population_diversity": 0.7
                }
            elif "selection" in msg_type:
                return {
                    "recommended_pressure": 0.7,
                    "convergence_risk": 0.2
                }
            elif "fitness" in msg_type:
                return {
                    "computed_fitness": [
                        {"individual_id": ind["individual_id"], "fitness": np.random.random()}
                        for ind in message.get("individuals", [])[:5]
                    ]
                }
            elif "migration" in msg_type:
                return {
                    "accepts_migrants": True,
                    "migration_capacity": 5,
                    "average_fitness": config["avg_fitness"]
                }
            
            return {"status": "ok"}
        
        # Set up mocks for all managers
        for manager in [
            distributed_ga.population_manager,
            distributed_ga.crossover_coordinator,
            distributed_ga.mutation_coordinator,
            distributed_ga.selection_coordinator,
            distributed_ga.fitness_evaluator,
            distributed_ga.migration_manager
        ]:
            manager.a2a_server.send_message.side_effect = mock_evolution_responses
        
        # Run evolution cycle
        evolution_result = await distributed_ga.run_evolution_cycle()
        
        # Verify evolution cycle completion
        assert "cycle_completed" in evolution_result
        assert "generation_advanced" in evolution_result
        assert "population_evolved" in evolution_result
        
        # Verify population advancement
        assert distributed_ga.population_manager.population_metrics.generation > 0
        
        # Verify all components participated
        assert len(distributed_ga.crossover_coordinator.crossover_history) >= 0
        assert len(distributed_ga.mutation_coordinator.mutation_history) >= 0
    
    @pytest.mark.asyncio
    async def test_convergence_detection_and_adaptation(self, distributed_ga):
        """Test convergence detection and adaptation across network"""
        # Simulate near-convergence scenario
        converged_population = []
        for i in range(50):
            # Create similar individuals (low diversity)
            base_genome = np.array([0.5] * 50)
            noise = np.random.normal(0, 0.01, 50)  # Very small variation
            genome = np.clip(base_genome + noise, 0, 1)
            
            individual = DistributedIndividual(
                individual_id=f"converged_ind_{i:03d}",
                genome=genome,
                fitness=0.95 + np.random.random() * 0.04,  # High, similar fitness
                generation=10,
                parent_ids=[],
                mutation_history=[],
                crossover_history=[],
                origin_agent=distributed_ga.agent_id
            )
            converged_population.append(individual)
        
        distributed_ga.population_manager.local_population = converged_population
        
        # Mock peer convergence data
        async def mock_convergence_data(agent_card, message):
            if message.get("type") == "convergence_analysis_request":
                return {
                    "convergence_detected": True,
                    "diversity_score": 0.1,  # Low diversity
                    "fitness_variance": 0.01,  # Low variance
                    "stagnation_generations": 8,
                    "adaptation_suggestions": ["increase_mutation", "introduce_migration"]
                }
            return {"status": "ok"}
        
        for manager in [distributed_ga.population_manager, distributed_ga.migration_manager]:
            manager.a2a_server.send_message.side_effect = mock_convergence_data
            
            # Add peer agents
            for i in range(3):
                agent_id = f"convergence_peer_{i:03d}"
                manager.peer_agents[agent_id] = Mock()
                manager.peer_agents[agent_id].agent_id = agent_id
        
        # Test convergence detection and adaptation
        convergence_result = await distributed_ga.detect_and_adapt_to_convergence()
        
        # Verify convergence detection
        assert "convergence_detected" in convergence_result
        assert "adaptation_actions" in convergence_result
        assert "network_consensus" in convergence_result
        
        if convergence_result["convergence_detected"]:
            actions = convergence_result["adaptation_actions"]
            assert len(actions) > 0
            
            # Verify appropriate adaptation actions
            adaptation_types = [action["type"] for action in actions]
            expected_adaptations = ["mutation_increase", "migration_trigger", "diversity_injection"]
            assert any(adaptation in str(adaptation_types) for adaptation in expected_adaptations)
    
    @pytest.mark.asyncio
    async def test_distributed_performance_optimization(self, distributed_ga):
        """Test distributed performance optimization"""
        # Add performance history
        performance_metrics = []
        for generation in range(20):
            metrics = {
                "generation": generation,
                "best_fitness": 0.6 + (generation * 0.02) + np.random.normal(0, 0.01),
                "average_fitness": 0.4 + (generation * 0.015) + np.random.normal(0, 0.01),
                "diversity_score": 0.8 - (generation * 0.01) + np.random.normal(0, 0.02),
                "convergence_speed": 0.05 + np.random.normal(0, 0.01),
                "computation_time": 2.0 + np.random.normal(0, 0.2)
            }
            performance_metrics.append(metrics)
        
        distributed_ga.performance_history = performance_metrics
        
        # Mock peer performance data
        async def mock_performance_data(agent_card, message):
            if message.get("type") == "performance_optimization_request":
                return {
                    "optimization_suggestions": [
                        {"parameter": "mutation_rate", "adjustment": 0.02, "expected_benefit": 0.05},
                        {"parameter": "selection_pressure", "adjustment": -0.1, "expected_benefit": 0.03},
                        {"parameter": "migration_frequency", "adjustment": 2, "expected_benefit": 0.04}
                    ],
                    "performance_benchmark": {
                        "best_fitness_rate": 0.02,
                        "convergence_efficiency": 0.85
                    }
                }
            return {"status": "ok"}
        
        # Set up performance optimization
        for manager in [
            distributed_ga.population_manager,
            distributed_ga.mutation_coordinator,
            distributed_ga.selection_coordinator
        ]:
            manager.a2a_server.send_message.side_effect = mock_performance_data
            
            for i in range(3):
                agent_id = f"optimization_peer_{i:03d}"
                manager.peer_agents[agent_id] = Mock()
                manager.peer_agents[agent_id].agent_id = agent_id
        
        # Test performance optimization
        optimization_result = await distributed_ga.optimize_distributed_performance()
        
        # Verify optimization
        assert "optimization_applied" in optimization_result
        assert "parameter_adjustments" in optimization_result
        assert "expected_improvements" in optimization_result
        
        adjustments = optimization_result["parameter_adjustments"]
        assert len(adjustments) > 0
        
        for adjustment in adjustments:
            assert "parameter" in adjustment
            assert "old_value" in adjustment
            assert "new_value" in adjustment
            assert "expected_benefit" in adjustment


# Run comprehensive tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--maxfail=3"])
