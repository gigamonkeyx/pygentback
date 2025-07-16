#!/usr/bin/env python3
"""
Test suite for world simulation environment.

Tests environment decay, resource management, agent lifecycle, and core simulation functionality.
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
    ResourceState, 
    EnvironmentProfile, 
    EnvironmentStatus,
    create_simulation_environment
)


class TestResourceState:
    """Test ResourceState functionality"""
    
    def test_resource_state_initialization(self):
        """Test ResourceState initialization with default values"""
        resource = ResourceState(total=1000.0, available=800.0, reserved=200.0)
        
        assert resource.total == 1000.0
        assert resource.available == 800.0
        assert resource.reserved == 200.0
        assert resource.decay_rate == 0.05  # Default decay rate
        assert isinstance(resource.last_update, datetime)
    
    def test_resource_decay_calculation(self):
        """Test resource decay mathematics"""
        resource = ResourceState(total=1000.0, available=1000.0, reserved=0.0, decay_rate=0.1)
        
        # Test 1 hour decay (time_delta = 1.0)
        initial_available = resource.available
        decay_amount = resource.apply_decay(1.0)
        
        expected_decay = initial_available * 0.1 * 1.0  # 100.0
        assert abs(decay_amount - expected_decay) < 0.01
        assert resource.available == initial_available - expected_decay
        assert resource.available >= 0.0  # Never goes negative
    
    def test_resource_decay_prevents_negative(self):
        """Test that resource decay never goes below zero"""
        resource = ResourceState(total=100.0, available=10.0, reserved=0.0, decay_rate=0.5)
        
        # Apply massive decay that would go negative
        decay_amount = resource.apply_decay(10.0)  # 50% * 10 hours = 500% decay
        
        assert resource.available == 0.0
        assert decay_amount == 10.0  # Only decayed what was available


class TestEnvironmentProfile:
    """Test EnvironmentProfile functionality"""
    
    def test_environment_profile_initialization(self):
        """Test EnvironmentProfile initialization"""
        profile = EnvironmentProfile()
        
        assert profile.compute_available == False
        assert profile.gpu_available == False
        assert profile.storage_capacity == 1000.0
        assert profile.network_access == True
        assert profile.tools_available == []
        assert profile.models_available == []
    
    def test_environment_profile_to_dict(self):
        """Test EnvironmentProfile serialization"""
        profile = EnvironmentProfile(
            compute_available=True,
            gpu_available=True,
            storage_capacity=5000.0,
            tools_available=["filesystem", "memory"],
            models_available=["qwen2.5:7b"]
        )
        
        profile_dict = profile.to_dict()
        
        assert profile_dict["compute_available"] == True
        assert profile_dict["gpu_available"] == True
        assert profile_dict["storage_capacity"] == 5000.0
        assert "filesystem" in profile_dict["tools_available"]
        assert "qwen2.5:7b" in profile_dict["models_available"]


class TestSimulationEnvironment:
    """Test SimulationEnvironment core functionality"""
    
    @pytest.fixture
    async def sim_env(self):
        """Create test simulation environment"""
        config = {"test_mode": True}
        env = SimulationEnvironment(config)
        yield env
        # Cleanup
        if env.status != EnvironmentStatus.SHUTDOWN:
            await env.shutdown()
    
    def test_simulation_environment_initialization(self, sim_env):
        """Test SimulationEnvironment initialization"""
        assert sim_env.status == EnvironmentStatus.INITIALIZING
        assert sim_env.config == {"test_mode": True}
        assert len(sim_env.environment_id) > 0
        assert isinstance(sim_env.start_time, datetime)
        
        # Check resource initialization
        assert "compute" in sim_env.resources
        assert "memory" in sim_env.resources
        assert "storage" in sim_env.resources
        assert "network" in sim_env.resources
        
        # Check initial resource values
        compute_resource = sim_env.resources["compute"]
        assert compute_resource.total == 1000.0
        assert compute_resource.available == 1000.0
        assert compute_resource.reserved == 0.0
    
    @pytest.mark.asyncio
    async def test_simulation_environment_initialization_success(self):
        """Test successful simulation environment initialization"""
        with patch('core.sim_env.RedisManager') as mock_redis, \
             patch('core.sim_env.MCPAutoDiscovery') as mock_mcp, \
             patch('core.sim_env.AgentFactory') as mock_factory:
            
            # Mock successful initialization
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            mock_mcp_instance = AsyncMock()
            mock_mcp.return_value = mock_mcp_instance
            
            mock_factory_instance = AsyncMock()
            mock_factory.return_value = mock_factory_instance
            
            sim_env = SimulationEnvironment()
            result = await sim_env.initialize()
            
            assert result == True
            assert sim_env.status == EnvironmentStatus.ACTIVE
            assert sim_env.redis_manager is not None
            assert sim_env.mcp_discovery is not None
            assert sim_env.agent_factory is not None
    
    @pytest.mark.asyncio
    async def test_simulation_environment_initialization_failure(self):
        """Test simulation environment initialization failure handling"""
        with patch('core.sim_env.RedisManager') as mock_redis:
            # Mock initialization failure
            mock_redis.side_effect = Exception("Redis connection failed")
            
            sim_env = SimulationEnvironment()
            result = await sim_env.initialize()
            
            assert result == False
            assert sim_env.status == EnvironmentStatus.CRITICAL
    
    @pytest.mark.asyncio
    async def test_resource_decay_application(self, sim_env):
        """Test resource decay application across all resources"""
        # Set initial state
        sim_env.last_cycle_time = datetime.now() - timedelta(hours=1)  # 1 hour ago
        
        initial_resources = {}
        for name, resource in sim_env.resources.items():
            initial_resources[name] = resource.available
        
        # Apply decay
        decay_results = await sim_env.apply_resource_decay()
        
        # Verify decay was applied
        assert len(decay_results) == len(sim_env.resources)
        
        for name, decay_amount in decay_results.items():
            assert decay_amount > 0  # Some decay should have occurred
            assert sim_env.resources[name].available < initial_resources[name]
    
    @pytest.mark.asyncio
    async def test_environment_state_retrieval(self, sim_env):
        """Test environment state retrieval"""
        sim_env.cycle_count = 5
        sim_env.status = EnvironmentStatus.ACTIVE
        
        env_state = await sim_env.get_environment_state()
        
        assert env_state["environment_id"] == sim_env.environment_id
        assert env_state["status"] == "active"
        assert env_state["cycle_count"] == 5
        assert "uptime" in env_state
        assert "resources" in env_state
        assert "profile" in env_state
        assert env_state["agent_count"] == 0  # No agents added yet
        
        # Check resource structure
        for resource_name in ["compute", "memory", "storage", "network"]:
            assert resource_name in env_state["resources"]
            resource_data = env_state["resources"][resource_name]
            assert "available" in resource_data
            assert "total" in resource_data
            assert "utilization" in resource_data
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle_management(self, sim_env):
        """Test agent addition and removal"""
        # Mock agent factory
        sim_env.agent_factory = AsyncMock()
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent_123"
        mock_agent.shutdown = AsyncMock()
        sim_env.agent_factory.create_agent.return_value = mock_agent
        
        # Test agent addition
        agent_config = {
            "type": "test",
            "name": "test_agent",
            "capabilities": ["testing"]
        }
        
        agent_id = await sim_env.add_agent(agent_config)
        
        assert agent_id == "test_agent_123"
        assert agent_id in sim_env.agents
        assert sim_env.agents[agent_id] == mock_agent
        
        # Test agent removal
        removal_result = await sim_env.remove_agent(agent_id)
        
        assert removal_result == True
        assert agent_id not in sim_env.agents
        mock_agent.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_simulation_cycle_execution(self, sim_env):
        """Test simulation cycle execution"""
        sim_env.status = EnvironmentStatus.ACTIVE
        initial_cycle_count = sim_env.cycle_count
        
        cycle_result = await sim_env.run_simulation_cycle()
        
        assert "cycle" in cycle_result
        assert "duration" in cycle_result
        assert "decay_results" in cycle_result
        assert "environment_state" in cycle_result
        
        assert cycle_result["cycle"] == initial_cycle_count + 1
        assert sim_env.cycle_count == initial_cycle_count + 1
        assert cycle_result["duration"] > 0
    
    @pytest.mark.asyncio
    async def test_environment_status_updates(self, sim_env):
        """Test environment status updates based on resource utilization"""
        # Test critical status (>90% utilization)
        for resource in sim_env.resources.values():
            resource.available = resource.total * 0.05  # 95% utilization
        
        await sim_env._update_environment_status()
        assert sim_env.status == EnvironmentStatus.CRITICAL
        
        # Test degraded status (>70% utilization)
        for resource in sim_env.resources.values():
            resource.available = resource.total * 0.25  # 75% utilization
        
        await sim_env._update_environment_status()
        assert sim_env.status == EnvironmentStatus.DEGRADED
        
        # Test active status (<70% utilization)
        for resource in sim_env.resources.values():
            resource.available = resource.total * 0.5  # 50% utilization
        
        await sim_env._update_environment_status()
        assert sim_env.status == EnvironmentStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_state_persistence(self, sim_env):
        """Test simulation state persistence to Redis"""
        # Mock Redis manager
        sim_env.redis_manager = AsyncMock()
        sim_env.redis_manager.set.return_value = True
        
        result = await sim_env.save_state()
        
        assert result == True
        sim_env.redis_manager.set.assert_called_once()
        
        # Verify the saved data structure
        call_args = sim_env.redis_manager.set.call_args
        key = call_args[0][0]
        data_json = call_args[0][1]
        
        assert key.startswith("sim_env:")
        
        # Parse and verify saved data
        saved_data = json.loads(data_json)
        assert "environment_id" in saved_data
        assert "status" in saved_data
        assert "resources" in saved_data
        assert "profile" in saved_data
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, sim_env):
        """Test graceful simulation environment shutdown"""
        # Add mock agent
        sim_env.agents["test_agent"] = Mock()
        sim_env.agents["test_agent"].shutdown = AsyncMock()
        
        # Mock Redis manager
        sim_env.redis_manager = AsyncMock()
        
        await sim_env.shutdown()
        
        assert sim_env.status == EnvironmentStatus.SHUTDOWN
        assert len(sim_env.agents) == 0  # All agents removed
        sim_env.redis_manager.close.assert_called_once()


class TestSimulationEnvironmentFactory:
    """Test simulation environment factory function"""
    
    @pytest.mark.asyncio
    async def test_create_simulation_environment_success(self):
        """Test successful simulation environment creation"""
        with patch('core.sim_env.SimulationEnvironment') as mock_env_class:
            mock_env = AsyncMock()
            mock_env.initialize.return_value = True
            mock_env_class.return_value = mock_env
            
            config = {"test": True}
            result = await create_simulation_environment(config)
            
            assert result == mock_env
            mock_env_class.assert_called_once_with(config)
            mock_env.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_simulation_environment_failure(self):
        """Test simulation environment creation failure"""
        with patch('core.sim_env.SimulationEnvironment') as mock_env_class:
            mock_env = AsyncMock()
            mock_env.initialize.return_value = False
            mock_env_class.return_value = mock_env
            
            with pytest.raises(RuntimeError, match="Failed to initialize simulation environment"):
                await create_simulation_environment()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
