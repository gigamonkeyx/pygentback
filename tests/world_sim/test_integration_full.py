#!/usr/bin/env python3
"""
Full integration test suite for world simulation system.

Tests end-to-end simulation runs with all phases integrated: simulation environment,
agent population & evolution, emergent behavior detection, and RIPER-Ω protocol.
RIPER-Ω Protocol compliant testing with observer supervision.
"""

import pytest
import asyncio
import json
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
    AgentInteractionSystem,
    EmergentBehaviorMonitor,
    DGMValidationIntegration,
    RIPEROmegaIntegration,
    create_simulation_environment
)


class TestFullSystemIntegration:
    """Test complete world simulation system integration"""
    
    @pytest.fixture
    async def complete_simulation_system(self):
        """Create complete simulation system with all components"""
        # Create simulation environment
        sim_env = SimulationEnvironment({"test_mode": True, "integration_test": True})
        
        # Mock external dependencies
        sim_env.redis_manager = AsyncMock()
        sim_env.mcp_discovery = AsyncMock()
        sim_env.agent_factory = AsyncMock()
        
        # Create all subsystems
        population_manager = AgentPopulationManager(sim_env)
        evolution_system = WorldSimulationEvolution(sim_env, population_manager)
        interaction_system = AgentInteractionSystem(sim_env)
        behavior_monitor = EmergentBehaviorMonitor(sim_env, interaction_system)
        dgm_validator = DGMValidationIntegration(sim_env)
        riperω_integration = RIPEROmegaIntegration(sim_env)
        
        # Initialize all systems
        await interaction_system.initialize()
        await behavior_monitor.initialize()
        await dgm_validator.initialize() if hasattr(dgm_validator, 'initialize') else None
        await riperω_integration.initialize()
        
        # Attach to simulation environment
        sim_env.population_manager = population_manager
        sim_env.evolution_system = evolution_system
        sim_env.interaction_system = interaction_system
        sim_env.behavior_monitor = behavior_monitor
        sim_env.dgm_validator = dgm_validator
        sim_env.riperω_integration = riperω_integration
        
        sim_env.status = "active"  # Set to active for testing
        
        yield sim_env
        
        # Cleanup
        await sim_env.shutdown()
    
    @pytest.mark.asyncio
    async def test_phase_1_simulation_environment(self, complete_simulation_system):
        """Test Phase 1: Simulation Environment functionality"""
        sim_env = complete_simulation_system
        
        # Test environment state retrieval
        env_state = await sim_env.get_environment_state()
        
        assert env_state["status"] == "active"
        assert "resources" in env_state
        assert "profile" in env_state
        assert env_state["agent_count"] == 0
        
        # Test resource decay
        initial_compute = sim_env.resources["compute"].available
        decay_results = await sim_env.apply_resource_decay()
        
        assert "compute" in decay_results
        assert sim_env.resources["compute"].available <= initial_compute
        
        # Test simulation cycle
        cycle_result = await sim_env.run_simulation_cycle()
        
        assert "cycle" in cycle_result
        assert "duration" in cycle_result
        assert "decay_results" in cycle_result
        assert "environment_state" in cycle_result
        assert sim_env.cycle_count == 1
    
    @pytest.mark.asyncio
    async def test_phase_2_agent_population_and_evolution(self, complete_simulation_system):
        """Test Phase 2: Agent Population & Evolution functionality"""
        sim_env = complete_simulation_system
        
        # Mock agent creation for population spawning
        sim_env.agent_factory.create_agent = AsyncMock()
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_123"
        mock_agent.shutdown = AsyncMock()
        sim_env.agent_factory.create_agent.return_value = mock_agent
        
        # Test population spawning
        spawn_results = await sim_env.population_manager.spawn_population()
        
        assert spawn_results["total_agents"] == 10
        assert spawn_results["successful_spawns"] > 0
        
        # Test population state
        pop_state = await sim_env.population_manager.get_population_state()
        
        assert "population_size" in pop_state
        assert "role_distribution" in pop_state
        assert "average_performance" in pop_state
        
        # Test evolution system
        # Mock environment and population states for evolution
        sim_env.get_environment_state = AsyncMock(return_value={
            "profile": {"tools_available": ["filesystem", "memory", "search"]}
        })
        
        sim_env.population_manager.get_population_state = AsyncMock(return_value={
            "agents": {
                "agent_1": {
                    "mcp_tools": ["filesystem"],
                    "capabilities": ["test"],
                    "performance": {"efficiency_score": 0.7}
                },
                "agent_2": {
                    "mcp_tools": ["memory", "search"],
                    "capabilities": ["test1", "test2"],
                    "performance": {"efficiency_score": 0.8}
                }
            }
        })
        
        # Test fitness evaluation
        fitness_results = await sim_env.evolution_system._evaluate_population_fitness()
        
        assert "agent_fitness" in fitness_results
        assert "population_stats" in fitness_results
        assert len(fitness_results["agent_fitness"]) == 2
        
        # Test DGM validation
        validation_results = await sim_env.dgm_validator.validate_evolved_agents({})
        
        assert "validated_agents" in validation_results
        assert "rejected_agents" in validation_results
    
    @pytest.mark.asyncio
    async def test_phase_3_emergence_and_orchestration(self, complete_simulation_system):
        """Test Phase 3: Emergence & Orchestration functionality"""
        sim_env = complete_simulation_system
        
        # Setup test data for interactions
        sim_env.get_environment_state = AsyncMock(return_value={
            "resources": {
                "compute": {"utilization": 0.8, "available": 200.0},
                "memory": {"utilization": 0.5, "available": 500.0}
            }
        })
        
        sim_env.population_manager.get_population_state = AsyncMock(return_value={
            "agents": {
                "explorer_1": {"role": "explorer", "type": "research", "performance": {"efficiency_score": 0.8}},
                "builder_1": {"role": "builder", "type": "coding", "performance": {"efficiency_score": 0.7}},
                "harvester_1": {"role": "harvester", "type": "analysis", "performance": {"efficiency_score": 0.6}}
            }
        })
        
        # Test agent interactions
        coordination_results = await sim_env.interaction_system.enable_swarm_coordination()
        
        assert "resource_sharing_events" in coordination_results
        assert "collaboration_tasks" in coordination_results
        assert "alliance_formations" in coordination_results
        
        # Test emergent behavior monitoring
        monitoring_results = await sim_env.behavior_monitor.monitor_emergent_behaviors()
        
        assert "spontaneous_cooperation" in monitoring_results
        assert "resource_optimization" in monitoring_results
        assert "tool_sharing_networks" in monitoring_results
        assert "adaptive_triggers" in monitoring_results
        assert "feedback_loops" in monitoring_results
        
        # Test RIPER-Ω workflow execution
        workflow_results = await sim_env.riperω_integration.execute_simulation_workflow()
        
        assert "modes_executed" in workflow_results
        assert "confidence_scores" in workflow_results
        assert "final_status" in workflow_results
        
        # Verify all modes were executed
        executed_modes = [mode[0] for mode in workflow_results["modes_executed"]]
        assert "RESEARCH" in executed_modes
        assert "PLAN" in executed_modes
        assert "EXECUTE" in executed_modes
        assert "REVIEW" in executed_modes
    
    @pytest.mark.asyncio
    async def test_end_to_end_simulation_run(self, complete_simulation_system):
        """Test complete end-to-end simulation run"""
        sim_env = complete_simulation_system
        
        # Mock all external dependencies for full run
        sim_env.agent_factory.create_agent = AsyncMock()
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        mock_agent.shutdown = AsyncMock()
        sim_env.agent_factory.create_agent.return_value = mock_agent
        
        sim_env.add_agent = AsyncMock(return_value="agent_123")
        sim_env.remove_agent = AsyncMock(return_value=True)
        
        # Step 1: Initialize simulation
        initial_state = await sim_env.get_environment_state()
        assert initial_state["status"] == "active"
        
        # Step 2: Spawn agent population
        spawn_results = await sim_env.population_manager.spawn_population()
        assert spawn_results["successful_spawns"] > 0
        
        # Step 3: Run simulation cycles with evolution
        for cycle in range(3):
            # Run simulation cycle
            cycle_result = await sim_env.run_simulation_cycle()
            assert cycle_result["cycle"] == cycle + 1
            
            # Enable agent interactions
            coordination_results = await sim_env.interaction_system.enable_swarm_coordination()
            assert "resource_sharing_events" in coordination_results
            
            # Monitor emergent behaviors
            monitoring_results = await sim_env.behavior_monitor.monitor_emergent_behaviors()
            assert "adaptive_triggers" in monitoring_results
        
        # Step 4: Execute RIPER-Ω workflow
        workflow_results = await sim_env.riperω_integration.execute_simulation_workflow()
        assert workflow_results["final_status"] in ["completed", "halted"]
        
        # Step 5: Validate final state
        final_state = await sim_env.get_environment_state()
        assert final_state["cycle_count"] == 3
        
        # Verify system health
        assert sim_env.status in ["active", "degraded"]  # Should not be critical or shutdown
    
    @pytest.mark.asyncio
    async def test_system_resilience_and_error_handling(self, complete_simulation_system):
        """Test system resilience and error handling"""
        sim_env = complete_simulation_system
        
        # Test handling of agent creation failures
        sim_env.agent_factory.create_agent = AsyncMock(side_effect=Exception("Agent creation failed"))
        
        spawn_results = await sim_env.population_manager.spawn_population()
        assert spawn_results["successful_spawns"] == 0
        assert all(not result["success"] for result in spawn_results["spawn_results"].values())
        
        # Test handling of evolution system failures
        sim_env.evolution_system.simulation_env.get_environment_state = AsyncMock(
            side_effect=Exception("Environment state error")
        )
        
        evolution_results = await sim_env.evolution_system.run_evolution_cycle(generations=1)
        assert "error" in evolution_results
        
        # Test handling of behavior monitoring failures
        sim_env.behavior_monitor.interaction_system.collaboration_history = None  # Cause error
        
        monitoring_results = await sim_env.behavior_monitor.monitor_emergent_behaviors()
        # Should handle gracefully and return partial results or error
        assert isinstance(monitoring_results, dict)
        
        # Test RIPER-Ω error handling
        sim_env.riperω_integration.simulation_env.get_environment_state = AsyncMock(
            side_effect=Exception("RIPER-Ω error")
        )
        
        workflow_results = await sim_env.riperω_integration.execute_simulation_workflow()
        # Should handle errors gracefully
        assert isinstance(workflow_results, dict)
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, complete_simulation_system):
        """Test performance benchmarking of simulation system"""
        sim_env = complete_simulation_system
        
        # Mock fast agent operations
        sim_env.add_agent = AsyncMock(return_value="agent_123")
        sim_env.remove_agent = AsyncMock(return_value=True)
        
        # Benchmark simulation cycle performance
        start_time = datetime.now()
        
        for _ in range(5):
            await sim_env.run_simulation_cycle()
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        avg_cycle_time = total_duration / 5
        
        # Performance assertions (adjust thresholds as needed)
        assert total_duration < 10.0  # Total time under 10 seconds
        assert avg_cycle_time < 2.0   # Average cycle under 2 seconds
        
        # Benchmark memory usage (basic check)
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory usage should be reasonable (adjust threshold as needed)
        assert memory_usage < 500  # Under 500MB
        
        # Benchmark evolution system performance
        sim_env.get_environment_state = AsyncMock(return_value={
            "profile": {"tools_available": ["filesystem", "memory"]}
        })
        
        sim_env.population_manager.get_population_state = AsyncMock(return_value={
            "agents": {f"agent_{i}": {
                "mcp_tools": ["filesystem"],
                "capabilities": ["test"],
                "performance": {"efficiency_score": 0.7}
            } for i in range(10)}
        })
        
        evolution_start = datetime.now()
        fitness_results = await sim_env.evolution_system._evaluate_population_fitness()
        evolution_end = datetime.now()
        
        evolution_duration = (evolution_end - evolution_start).total_seconds()
        assert evolution_duration < 5.0  # Fitness evaluation under 5 seconds
        assert len(fitness_results["agent_fitness"]) == 10
    
    @pytest.mark.asyncio
    async def test_data_persistence_and_recovery(self, complete_simulation_system):
        """Test data persistence and recovery mechanisms"""
        sim_env = complete_simulation_system
        
        # Test state saving
        save_result = await sim_env.save_state()
        assert save_result == True
        
        # Verify Redis save was called
        sim_env.redis_manager.set.assert_called()
        
        # Test state data structure
        call_args = sim_env.redis_manager.set.call_args
        saved_data_json = call_args[0][1]
        saved_data = json.loads(saved_data_json)
        
        assert "environment_id" in saved_data
        assert "status" in saved_data
        assert "resources" in saved_data
        assert "profile" in saved_data
        
        # Test interaction system state persistence
        interaction_summary = await sim_env.interaction_system.get_interaction_summary()
        assert "total_interactions" in interaction_summary
        assert "network_stats" in interaction_summary or sim_env.interaction_system.interaction_graph is None
        
        # Test behavior monitoring state persistence
        behavior_summary = await sim_env.behavior_monitor.get_behavior_summary()
        assert isinstance(behavior_summary, dict)
        
        # Test RIPER-Ω state persistence
        riperω_status = await sim_env.riperω_integration.get_riperω_status()
        assert "current_mode" in riperω_status
        assert "confidence_scores" in riperω_status
    
    @pytest.mark.asyncio
    async def test_system_scalability(self, complete_simulation_system):
        """Test system scalability with larger populations"""
        sim_env = complete_simulation_system
        
        # Test with larger agent population (simulate)
        large_population = {
            "agents": {f"agent_{i}": {
                "mcp_tools": ["filesystem", "memory"],
                "capabilities": ["test1", "test2"],
                "performance": {"efficiency_score": 0.7 + (i % 3) * 0.1}
            } for i in range(50)}  # 50 agents
        }
        
        sim_env.population_manager.get_population_state = AsyncMock(return_value=large_population)
        
        # Test fitness evaluation scalability
        start_time = datetime.now()
        fitness_results = await sim_env.evolution_system._evaluate_population_fitness()
        end_time = datetime.now()
        
        evaluation_time = (end_time - start_time).total_seconds()
        assert evaluation_time < 10.0  # Should handle 50 agents in under 10 seconds
        assert len(fitness_results["agent_fitness"]) == 50
        
        # Test interaction system scalability
        if sim_env.interaction_system.interaction_graph is not None:
            # Add many nodes to test graph operations
            for i in range(20):
                sim_env.interaction_system.interaction_graph.add_node(f"agent_{i}")
            
            # Test alliance formation with larger network
            alliance_results = await sim_env.interaction_system._update_alliance_formations()
            assert isinstance(alliance_results, list)


class TestSystemFactory:
    """Test simulation system factory functions"""
    
    @pytest.mark.asyncio
    async def test_create_simulation_environment_factory(self):
        """Test simulation environment factory function"""
        with patch('core.sim_env.SimulationEnvironment') as mock_env_class:
            mock_env = AsyncMock()
            mock_env.initialize.return_value = True
            mock_env_class.return_value = mock_env
            
            config = {"factory_test": True}
            result = await create_simulation_environment(config)
            
            assert result == mock_env
            mock_env_class.assert_called_once_with(config)
            mock_env.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_simulation_environment_factory_failure(self):
        """Test simulation environment factory failure handling"""
        with patch('core.sim_env.SimulationEnvironment') as mock_env_class:
            mock_env = AsyncMock()
            mock_env.initialize.return_value = False
            mock_env_class.return_value = mock_env
            
            with pytest.raises(RuntimeError, match="Failed to initialize simulation environment"):
                await create_simulation_environment()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
