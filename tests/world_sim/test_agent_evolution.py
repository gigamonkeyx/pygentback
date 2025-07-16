#!/usr/bin/env python3
"""
Test suite for agent evolution system.

Tests evolution cycle convergence, fitness optimization, mutation/crossover operations,
and population management. RIPER-Ω Protocol compliant testing with observer supervision.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.sim_env import (
    SimulationEnvironment,
    AgentPopulationManager,
    WorldSimulationEvolution,
    DGMValidationIntegration
)


class TestAgentPopulationManager:
    """Test AgentPopulationManager functionality"""
    
    @pytest.fixture
    async def sim_env(self):
        """Create test simulation environment"""
        env = SimulationEnvironment({"test_mode": True})
        env.agent_factory = AsyncMock()
        env.redis_manager = AsyncMock()
        yield env
        await env.shutdown()
    
    @pytest.fixture
    def population_manager(self, sim_env):
        """Create test population manager"""
        return AgentPopulationManager(sim_env)
    
    def test_agent_specifications_definition(self, population_manager):
        """Test agent specification definitions"""
        specs = population_manager.agent_specs
        
        # Verify all 10 agents are defined
        assert len(specs) == 10
        
        # Verify role distribution
        role_counts = {}
        for spec in specs.values():
            role = spec["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
        
        assert role_counts["explorer"] == 2
        assert role_counts["builder"] == 2
        assert role_counts["harvester"] == 2
        assert role_counts["defender"] == 1
        assert role_counts["communicator"] == 1
        assert role_counts["adapter"] == 2
        
        # Verify required fields in each spec
        for agent_name, spec in specs.items():
            assert "type" in spec
            assert "role" in spec
            assert "capabilities" in spec
            assert "mcp_tools" in spec
            assert "traits" in spec
            assert isinstance(spec["capabilities"], list)
            assert isinstance(spec["mcp_tools"], list)
            assert isinstance(spec["traits"], dict)
    
    @pytest.mark.asyncio
    async def test_trait_randomization(self, population_manager):
        """Test agent trait randomization"""
        base_spec = {
            "type": "test",
            "role": "test",
            "capabilities": ["test"],
            "mcp_tools": ["test"],
            "traits": {"efficiency": 0.8, "adaptability": 0.6}
        }
        
        randomized_spec = await population_manager._randomize_agent_traits(base_spec)
        
        # Verify structure is preserved
        assert randomized_spec["type"] == base_spec["type"]
        assert randomized_spec["role"] == base_spec["role"]
        assert randomized_spec["capabilities"] == base_spec["capabilities"]
        assert randomized_spec["mcp_tools"] == base_spec["mcp_tools"]
        
        # Verify traits are randomized within bounds
        for trait_name, original_value in base_spec["traits"].items():
            new_value = randomized_spec["traits"][trait_name]
            assert 0.0 <= new_value <= 1.0
            assert abs(new_value - original_value) <= 0.1  # ±10% variation
        
        # Verify specialization is added
        assert "specialization" in randomized_spec
        specialization = randomized_spec["specialization"]
        assert "focus_area" in specialization
        assert "learning_style" in specialization
        assert "collaboration_preference" in specialization
    
    @pytest.mark.asyncio
    async def test_population_spawning_success(self, population_manager):
        """Test successful population spawning"""
        # Mock successful agent creation
        population_manager.simulation_env.add_agent = AsyncMock()
        population_manager.simulation_env.add_agent.return_value = "agent_123"
        
        spawn_results = await population_manager.spawn_population()
        
        assert spawn_results["total_agents"] == 10
        assert spawn_results["successful_spawns"] == 10
        assert len(spawn_results["spawn_results"]) == 10
        
        # Verify all spawns were successful
        for agent_name, result in spawn_results["spawn_results"].items():
            assert result["success"] == True
            assert "agent_id" in result
        
        # Verify population state
        assert len(population_manager.population) == 10
        
        # Verify agent creation was called for each agent
        assert population_manager.simulation_env.add_agent.call_count == 10
    
    @pytest.mark.asyncio
    async def test_population_spawning_partial_failure(self, population_manager):
        """Test population spawning with partial failures"""
        # Mock mixed success/failure
        call_count = 0
        async def mock_add_agent(config):
            nonlocal call_count
            call_count += 1
            return "agent_123" if call_count <= 7 else None  # 7 successes, 3 failures
        
        population_manager.simulation_env.add_agent = mock_add_agent
        
        spawn_results = await population_manager.spawn_population()
        
        assert spawn_results["total_agents"] == 10
        assert spawn_results["successful_spawns"] == 7
        
        # Verify mixed results
        successful_count = sum(1 for result in spawn_results["spawn_results"].values() if result["success"])
        failed_count = sum(1 for result in spawn_results["spawn_results"].values() if not result["success"])
        
        assert successful_count == 7
        assert failed_count == 3
    
    @pytest.mark.asyncio
    async def test_population_state_retrieval(self, population_manager):
        """Test population state retrieval"""
        # Add mock population data
        population_manager.population = {
            "explorer_1": {
                "agent_id": "agent_1",
                "spec": {"role": "explorer", "type": "research", "traits": {"curiosity": 0.9}},
                "spawn_time": datetime.now() - timedelta(minutes=5),
                "performance_metrics": {"tasks_completed": 3, "efficiency_score": 0.8}
            },
            "builder_1": {
                "agent_id": "agent_2", 
                "spec": {"role": "builder", "type": "coding", "traits": {"precision": 0.85}},
                "spawn_time": datetime.now() - timedelta(minutes=3),
                "performance_metrics": {"tasks_completed": 2, "efficiency_score": 0.75}
            }
        }
        
        pop_state = await population_manager.get_population_state()
        
        assert pop_state["population_size"] == 2
        assert len(pop_state["agents"]) == 2
        
        # Verify agent data structure
        for agent_name, agent_data in pop_state["agents"].items():
            assert "agent_id" in agent_data
            assert "role" in agent_data
            assert "type" in agent_data
            assert "traits" in agent_data
            assert "performance" in agent_data
            assert "uptime" in agent_data
            assert agent_data["uptime"] > 0  # Should have positive uptime
        
        # Verify role distribution
        role_dist = pop_state["role_distribution"]
        assert role_dist["explorer"] == 1
        assert role_dist["builder"] == 1
        
        # Verify average performance calculation
        expected_avg = (0.8 + 0.75) / 2
        assert abs(pop_state["average_performance"] - expected_avg) < 0.01
    
    @pytest.mark.asyncio
    async def test_performance_metrics_update(self, population_manager):
        """Test agent performance metrics updates"""
        # Add mock agent
        population_manager.population["test_agent"] = {
            "agent_id": "agent_1",
            "spec": {"role": "test"},
            "spawn_time": datetime.now(),
            "performance_metrics": {"tasks_completed": 0, "efficiency_score": 0.5}
        }
        
        # Update performance
        new_metrics = {"tasks_completed": 5, "efficiency_score": 0.85}
        result = await population_manager.update_agent_performance("test_agent", new_metrics)
        
        assert result == True
        
        # Verify update
        updated_metrics = population_manager.population["test_agent"]["performance_metrics"]
        assert updated_metrics["tasks_completed"] == 5
        assert updated_metrics["efficiency_score"] == 0.85
        
        # Test update for non-existent agent
        result = await population_manager.update_agent_performance("non_existent", new_metrics)
        assert result == False


class TestWorldSimulationEvolution:
    """Test WorldSimulationEvolution functionality"""
    
    @pytest.fixture
    async def sim_env(self):
        """Create test simulation environment"""
        env = SimulationEnvironment({"test_mode": True})
        env.agent_factory = AsyncMock()
        env.redis_manager = AsyncMock()
        yield env
        await env.shutdown()
    
    @pytest.fixture
    def population_manager(self, sim_env):
        """Create test population manager"""
        return AgentPopulationManager(sim_env)
    
    @pytest.fixture
    def evolution_system(self, sim_env, population_manager):
        """Create test evolution system"""
        return WorldSimulationEvolution(sim_env, population_manager)
    
    def test_evolution_system_initialization(self, evolution_system):
        """Test evolution system initialization"""
        assert evolution_system.generation == 0
        assert evolution_system.mutation_rate == 0.15
        assert evolution_system.crossover_rate == 0.7
        assert evolution_system.elite_ratio == 0.2
        assert evolution_system.bloat_penalty_rate == 0.1
        assert evolution_system.convergence_threshold == 0.05
        assert evolution_system.max_stagnation_generations == 3
        assert len(evolution_system.fitness_history) == 0
    
    @pytest.mark.asyncio
    async def test_fitness_evaluation(self, evolution_system):
        """Test population fitness evaluation"""
        # Mock environment state
        env_state = {
            "profile": {"tools_available": ["filesystem", "memory", "search", "analysis"]}
        }
        evolution_system.simulation_env.get_environment_state = AsyncMock(return_value=env_state)
        
        # Mock population state
        population_state = {
            "agents": {
                "agent_1": {
                    "mcp_tools": ["filesystem", "memory"],  # 50% coverage
                    "capabilities": ["test1", "test2"],
                    "performance": {"efficiency_score": 0.8}
                },
                "agent_2": {
                    "mcp_tools": ["search", "analysis", "extra1", "extra2"],  # 50% coverage but more tools
                    "capabilities": ["test1", "test2", "test3"],
                    "performance": {"efficiency_score": 0.9}
                }
            }
        }
        evolution_system.population_manager.get_population_state = AsyncMock(return_value=population_state)
        
        fitness_results = await evolution_system._evaluate_population_fitness()
        
        assert "agent_fitness" in fitness_results
        assert "population_stats" in fitness_results
        
        # Verify fitness calculations
        agent1_fitness = fitness_results["agent_fitness"]["agent_1"]
        agent2_fitness = fitness_results["agent_fitness"]["agent_2"]
        
        # Agent 1: coverage=0.5, efficiency=0.8, bloat_penalty=4*0.1=0.4
        # fitness = (0.5 * 0.8) - 0.4 = 0.4 - 0.4 = 0.0
        assert abs(agent1_fitness["fitness"] - 0.0) < 0.01
        assert agent1_fitness["coverage"] == 0.5
        assert agent1_fitness["efficiency"] == 0.8
        assert agent1_fitness["bloat_penalty"] == 0.4
        
        # Agent 2: coverage=0.5, efficiency=0.9, bloat_penalty=7*0.1=0.7
        # fitness = (0.5 * 0.9) - 0.7 = 0.45 - 0.7 = 0.0 (clamped to non-negative)
        assert agent2_fitness["fitness"] >= 0.0
        assert agent2_fitness["coverage"] == 0.5
        assert agent2_fitness["efficiency"] == 0.9
        
        # Verify population statistics
        pop_stats = fitness_results["population_stats"]
        assert "mean_fitness" in pop_stats
        assert "max_fitness" in pop_stats
        assert "min_fitness" in pop_stats
        assert "fitness_std" in pop_stats
    
    @pytest.mark.asyncio
    async def test_mutation_generation(self, evolution_system):
        """Test environment-based mutation generation"""
        # Mock environment state
        env_state = {
            "profile": {"tools_available": ["filesystem", "memory", "search", "new_tool"]}
        }
        evolution_system.simulation_env.get_environment_state = AsyncMock(return_value=env_state)
        
        # Test agent with missing tools (should add tool)
        agent_data = {"mcp_tools": ["filesystem", "memory"]}
        
        with patch('random.random', return_value=0.1):  # Below mutation rate
            mutation = await evolution_system._generate_environment_based_mutation(agent_data)
            
            assert mutation is not None
            assert mutation["type"] == "add_tool"
            assert mutation["tool"] in ["search", "new_tool"]
            assert mutation["reason"] == "improve_environment_coverage"
        
        # Test agent with too many tools (should remove tool)
        agent_data = {"mcp_tools": ["filesystem", "memory", "search", "tool1", "tool2"]}
        
        with patch('random.random', return_value=0.1):  # Below mutation rate
            mutation = await evolution_system._generate_environment_based_mutation(agent_data)
            
            if mutation and mutation["type"] == "remove_tool":
                assert mutation["tool"] in agent_data["mcp_tools"]
                assert mutation["reason"] == "reduce_bloat"
    
    @pytest.mark.asyncio
    async def test_crossover_generation(self, evolution_system):
        """Test crossover generation between agents"""
        parent1_data = {
            "type": "research",
            "mcp_tools": ["filesystem", "memory"],
            "capabilities": ["analysis"],
            "traits": {"efficiency": 0.8}
        }
        
        parent2_data = {
            "type": "coding", 
            "mcp_tools": ["search", "github"],
            "capabilities": ["building"],
            "traits": {"precision": 0.9}
        }
        
        with patch('random.random', return_value=0.5):  # Below crossover rate
            crossover = await evolution_system._generate_crossover(parent1_data, parent2_data)
            
            assert crossover is not None
            assert crossover["type"] == "tool_blend"
            assert "blend_ratio" in crossover
            assert crossover["inherit_traits"] == True
    
    @pytest.mark.asyncio
    async def test_convergence_detection(self, evolution_system):
        """Test evolution convergence detection"""
        # Test insufficient history
        assert await evolution_system._check_convergence({}) == False
        
        # Add fitness history with low improvement
        evolution_system.fitness_history = [
            {"mean_fitness": 0.50},
            {"mean_fitness": 0.51},
            {"mean_fitness": 0.52}
        ]
        
        # Should converge (improvement < 0.05)
        assert await evolution_system._check_convergence({}) == True
        
        # Add fitness history with high improvement
        evolution_system.fitness_history = [
            {"mean_fitness": 0.50},
            {"mean_fitness": 0.60},
            {"mean_fitness": 0.70}
        ]
        
        # Should not converge (improvement >= 0.05)
        assert await evolution_system._check_convergence({}) == False
    
    @pytest.mark.asyncio
    async def test_evolution_cycle_execution(self, evolution_system):
        """Test complete evolution cycle execution"""
        # Mock all dependencies
        evolution_system.simulation_env.get_environment_state = AsyncMock(return_value={
            "profile": {"tools_available": ["filesystem", "memory"]}
        })
        
        evolution_system.population_manager.get_population_state = AsyncMock(return_value={
            "agents": {
                "agent_1": {
                    "mcp_tools": ["filesystem"],
                    "capabilities": ["test"],
                    "performance": {"efficiency_score": 0.8}
                }
            }
        })
        
        evolution_system.simulation_env.add_agent = AsyncMock(return_value="new_agent_id")
        evolution_system.simulation_env.remove_agent = AsyncMock(return_value=True)
        
        # Run evolution cycle
        results = await evolution_system.run_evolution_cycle(generations=2)
        
        assert "generations_completed" in results
        assert "fitness_improvements" in results
        assert "population_changes" in results
        assert "convergence_achieved" in results
        assert "total_time" in results
        
        assert results["generations_completed"] <= 2
        assert results["total_time"] > 0
        assert isinstance(results["convergence_achieved"], bool)


class TestDGMValidationIntegration:
    """Test DGM validation integration"""
    
    @pytest.fixture
    async def sim_env(self):
        """Create test simulation environment"""
        env = SimulationEnvironment({"test_mode": True})
        env.population_manager = Mock()
        yield env
        await env.shutdown()
    
    @pytest.fixture
    def dgm_validator(self, sim_env):
        """Create test DGM validator"""
        return DGMValidationIntegration(sim_env)
    
    def test_dgm_validator_initialization(self, dgm_validator):
        """Test DGM validator initialization"""
        assert dgm_validator.min_performance_threshold == 0.6
        assert dgm_validator.max_complexity_threshold == 10
        assert dgm_validator.validation_timeout == 30.0
        assert len(dgm_validator.validation_history) == 0
    
    @pytest.mark.asyncio
    async def test_single_agent_validation_success(self, dgm_validator):
        """Test successful single agent validation"""
        agent_data = {
            "agent_id": "test_agent",
            "capabilities": ["test1", "test2"],
            "mcp_tools": ["filesystem", "memory"],
            "performance": {"efficiency_score": 0.8},
            "custom_config": {}
        }
        
        # Mock safety check
        dgm_validator._perform_safety_check = AsyncMock(return_value={
            "safe": True,
            "safety_score": 0.9,
            "violations": []
        })
        
        dgm_validator._identify_improvements = AsyncMock(return_value=[])
        
        validation_result = await dgm_validator._validate_single_agent("test_agent", agent_data)
        
        assert validation_result["approved"] == True
        assert validation_result["score"] > 0.0
        assert validation_result["reason"] == "Validation passed"
        assert len(validation_result["violations"]) == 0
    
    @pytest.mark.asyncio
    async def test_single_agent_validation_performance_failure(self, dgm_validator):
        """Test agent validation failure due to low performance"""
        agent_data = {
            "agent_id": "test_agent",
            "capabilities": ["test"],
            "mcp_tools": ["filesystem"],
            "performance": {"efficiency_score": 0.4},  # Below threshold
            "custom_config": {}
        }
        
        validation_result = await dgm_validator._validate_single_agent("test_agent", agent_data)
        
        assert validation_result["approved"] == False
        assert "Performance below threshold" in validation_result["reason"]
    
    @pytest.mark.asyncio
    async def test_single_agent_validation_complexity_failure(self, dgm_validator):
        """Test agent validation failure due to high complexity"""
        agent_data = {
            "agent_id": "test_agent",
            "capabilities": ["test1", "test2", "test3", "test4", "test5", "test6"],
            "mcp_tools": ["tool1", "tool2", "tool3", "tool4", "tool5", "tool6"],  # 12 total > 10
            "performance": {"efficiency_score": 0.8},
            "custom_config": {}
        }
        
        validation_result = await dgm_validator._validate_single_agent("test_agent", agent_data)
        
        assert validation_result["approved"] == False
        assert "Complexity exceeds threshold" in validation_result["reason"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
