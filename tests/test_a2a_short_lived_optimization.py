#!/usr/bin/env python3
"""
Test A2A Short-lived Agent Optimization

Tests the A2A-compliant short-lived agent optimization implementation.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta

# Import the A2A short-lived optimization components
try:
    from src.a2a_protocol.short_lived_optimization import (
        ShortLivedAgentOptimizer, OptimizedAgent, OptimizationConfig,
        ResourceLimits, AgentMetrics, AgentLifecycleState
    )
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestShortLivedAgentOptimizer:
    """Test Short-lived Agent Optimizer"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = OptimizationConfig(
            enable_fast_startup=True,
            enable_resource_monitoring=False,  # Disable for testing
            enable_auto_shutdown=False,  # Disable for testing
            enable_task_pooling=True,
            enable_memory_optimization=True
        )
        self.optimizer = ShortLivedAgentOptimizer(self.config)
    
    def teardown_method(self):
        """Cleanup after tests"""
        asyncio.create_task(self.optimizer.shutdown())
    
    @pytest.mark.asyncio
    async def test_create_optimized_agent(self):
        """Test creating an optimized agent"""
        agent_id = "test-agent-123"
        agent_type = "research"
        
        # Create agent
        agent = await self.optimizer.create_optimized_agent(agent_id, agent_type)
        
        # Verify agent creation
        assert agent.agent_id == agent_id
        assert agent.agent_type == agent_type
        assert agent.state == AgentLifecycleState.READY
        assert agent_id in self.optimizer.active_agents
        assert agent.metrics.startup_time_ms > 0
        
        # Cleanup
        await self.optimizer.shutdown_agent(agent_id, pool_for_reuse=False)
    
    @pytest.mark.asyncio
    async def test_agent_pooling(self):
        """Test agent pooling for reuse"""
        agent_id_1 = "test-agent-1"
        agent_id_2 = "test-agent-2"
        
        # Create first agent
        agent1 = await self.optimizer.create_optimized_agent(agent_id_1, "research")
        
        # Shutdown and pool for reuse
        await self.optimizer.shutdown_agent(agent_id_1, pool_for_reuse=True)
        
        # Verify agent is pooled
        assert len(self.optimizer.agent_pool) == 1
        assert agent_id_1 not in self.optimizer.active_agents
        
        # Create second agent (should reuse from pool)
        agent2 = await self.optimizer.create_optimized_agent(agent_id_2, "analysis")
        
        # Verify reuse
        assert agent2.agent_id == agent_id_2  # ID should be updated
        assert agent2.agent_type == "analysis"  # Type should be updated
        assert len(self.optimizer.agent_pool) == 0  # Pool should be empty
        assert agent_id_2 in self.optimizer.active_agents
        
        # Cleanup
        await self.optimizer.shutdown_agent(agent_id_2, pool_for_reuse=False)
    
    @pytest.mark.asyncio
    async def test_execute_task_optimized(self):
        """Test optimized task execution"""
        agent_id = "test-agent-123"
        agent = await self.optimizer.create_optimized_agent(agent_id, "research")
        
        # Execute task
        task_data = {"id": "task-1", "type": "research", "query": "test query"}
        result = await self.optimizer.execute_task_optimized(agent_id, task_data)
        
        # Verify task execution
        assert result is not None
        assert agent.metrics.tasks_completed == 1
        assert agent.metrics.total_execution_time_ms > 0
        
        # Cleanup
        await self.optimizer.shutdown_agent(agent_id, pool_for_reuse=False)
    
    @pytest.mark.asyncio
    async def test_batch_task_processing(self):
        """Test batch task processing"""
        agent_id = "test-agent-123"
        agent = await self.optimizer.create_optimized_agent(agent_id, "research")
        
        # Execute multiple small tasks that should be batched
        tasks = []
        for i in range(3):
            task_data = {"id": f"task-{i}", "small": True}
            task = asyncio.create_task(
                self.optimizer.execute_task_optimized(agent_id, task_data)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Verify results
        assert len(results) == 3
        for result in results:
            assert result is not None
        
        # Cleanup
        await self.optimizer.shutdown_agent(agent_id, pool_for_reuse=False)


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestOptimizedAgent:
    """Test Optimized Agent"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = OptimizationConfig()
        self.optimizer = ShortLivedAgentOptimizer(self.config)
        self.resource_limits = ResourceLimits(
            max_memory_mb=128,
            max_cpu_percent=25.0,
            max_execution_time_seconds=60,
            max_idle_time_seconds=30
        )
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization"""
        agent = OptimizedAgent(
            agent_id="test-agent",
            agent_type="research",
            resource_limits=self.resource_limits,
            optimizer=self.optimizer
        )
        
        # Initialize agent
        await agent.initialize()
        
        # Verify initialization
        assert agent.state == AgentLifecycleState.READY
        assert agent.created_at is not None
        assert agent.last_activity is not None
        assert len(agent.active_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_execute_single_task(self):
        """Test single task execution"""
        agent = OptimizedAgent(
            agent_id="test-agent",
            agent_type="research",
            resource_limits=self.resource_limits,
            optimizer=self.optimizer
        )
        
        await agent.initialize()
        
        # Execute task
        task_data = {"id": "task-1", "type": "research"}
        result = await agent.execute_task(task_data)
        
        # Verify task execution
        assert result is not None
        assert agent.metrics.tasks_completed == 1
        assert agent.metrics.tasks_failed == 0
        assert agent.state == AgentLifecycleState.IDLE
        assert len(agent.active_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_execute_task_batch(self):
        """Test batch task execution"""
        agent = OptimizedAgent(
            agent_id="test-agent",
            agent_type="research",
            resource_limits=self.resource_limits,
            optimizer=self.optimizer
        )
        
        await agent.initialize()
        
        # Execute batch of tasks
        tasks = [
            {"id": "task-1", "type": "research"},
            {"id": "task-2", "type": "research"},
            {"id": "task-3", "type": "research"}
        ]
        
        results = await agent.execute_task_batch(tasks)
        
        # Verify batch execution
        assert len(results) == 3
        assert agent.metrics.tasks_completed == 3
        assert agent.metrics.tasks_failed == 0
        assert agent.state == AgentLifecycleState.IDLE
    
    @pytest.mark.asyncio
    async def test_metrics_update(self):
        """Test metrics updating"""
        agent = OptimizedAgent(
            agent_id="test-agent",
            agent_type="research",
            resource_limits=self.resource_limits,
            optimizer=self.optimizer
        )
        
        await agent.initialize()
        
        # Update metrics
        await agent.update_metrics()
        
        # Verify metrics
        assert agent.metrics.memory_usage_mb >= 0
        assert agent.metrics.cpu_usage_percent >= 0
        assert agent.metrics.last_activity is not None
    
    @pytest.mark.asyncio
    async def test_resource_limit_checking(self):
        """Test resource limit checking"""
        # Create agent with very low limits
        strict_limits = ResourceLimits(
            max_memory_mb=1,  # Very low memory limit
            max_cpu_percent=0.1,  # Very low CPU limit
            max_execution_time_seconds=1,  # Very short execution time
            max_idle_time_seconds=1
        )
        
        agent = OptimizedAgent(
            agent_id="test-agent",
            agent_type="research",
            resource_limits=strict_limits,
            optimizer=self.optimizer
        )
        
        await agent.initialize()
        
        # Wait a moment to exceed time limit
        await asyncio.sleep(1.1)
        
        # Check if limits are exceeded
        exceeds_limits = agent.exceeds_resource_limits()
        
        # Should exceed time limit
        assert exceeds_limits == True
    
    @pytest.mark.asyncio
    async def test_idle_detection(self):
        """Test idle state detection"""
        agent = OptimizedAgent(
            agent_id="test-agent",
            agent_type="research",
            resource_limits=self.resource_limits,
            optimizer=self.optimizer
        )
        
        await agent.initialize()
        
        # Agent should be idle when ready with no tasks
        assert agent.is_idle() == True
        
        # Simulate active task
        agent.active_tasks.add("task-1")
        agent.state = AgentLifecycleState.ACTIVE
        
        assert agent.is_idle() == False
        
        # Remove task and set to idle
        agent.active_tasks.clear()
        agent.state = AgentLifecycleState.IDLE
        
        assert agent.is_idle() == True
    
    @pytest.mark.asyncio
    async def test_shutdown_detection(self):
        """Test shutdown detection based on inactivity"""
        agent = OptimizedAgent(
            agent_id="test-agent",
            agent_type="research",
            resource_limits=self.resource_limits,
            optimizer=self.optimizer
        )
        
        await agent.initialize()
        
        # Set agent to idle
        agent.state = AgentLifecycleState.IDLE
        
        # Should not shutdown immediately
        current_time = datetime.utcnow()
        assert agent.should_shutdown(current_time) == False
        
        # Set last activity to past
        agent.last_activity = current_time - timedelta(seconds=35)
        
        # Should shutdown after idle timeout
        assert agent.should_shutdown(current_time) == True
    
    @pytest.mark.asyncio
    async def test_agent_reset_for_reuse(self):
        """Test agent reset for reuse"""
        agent = OptimizedAgent(
            agent_id="original-agent",
            agent_type="research",
            resource_limits=self.resource_limits,
            optimizer=self.optimizer
        )
        
        await agent.initialize()
        
        # Execute a task to change state
        task_data = {"id": "task-1", "type": "research"}
        await agent.execute_task(task_data)
        
        # Reset for reuse
        agent.reset_for_reuse("new-agent", "analysis")
        
        # Verify reset
        assert agent.agent_id == "new-agent"
        assert agent.agent_type == "analysis"
        assert agent.state == AgentLifecycleState.READY
        assert len(agent.active_tasks) == 0
        assert agent.metrics.tasks_completed == 0  # Metrics should be reset
    
    @pytest.mark.asyncio
    async def test_agent_health_check(self):
        """Test agent health checking"""
        agent = OptimizedAgent(
            agent_id="test-agent",
            agent_type="research",
            resource_limits=self.resource_limits,
            optimizer=self.optimizer
        )
        
        await agent.initialize()
        
        # Healthy agent should pass health check
        assert agent.is_healthy() == True
        
        # Simulate many failures
        agent.metrics.tasks_failed = 10
        
        # Should fail health check
        assert agent.is_healthy() == False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestOptimizationConfig:
    """Test Optimization Configuration"""
    
    def test_default_config(self):
        """Test default optimization configuration"""
        config = OptimizationConfig()
        
        assert config.enable_fast_startup == True
        assert config.enable_resource_monitoring == True
        assert config.enable_auto_shutdown == True
        assert config.enable_task_pooling == True
        assert config.enable_memory_optimization == True
        assert config.preload_common_modules == True
        assert config.cache_agent_cards == True
        assert config.batch_task_processing == True
    
    def test_custom_config(self):
        """Test custom optimization configuration"""
        config = OptimizationConfig(
            enable_fast_startup=False,
            enable_resource_monitoring=False,
            enable_auto_shutdown=False,
            enable_task_pooling=False
        )
        
        assert config.enable_fast_startup == False
        assert config.enable_resource_monitoring == False
        assert config.enable_auto_shutdown == False
        assert config.enable_task_pooling == False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestResourceLimits:
    """Test Resource Limits"""
    
    def test_default_limits(self):
        """Test default resource limits"""
        limits = ResourceLimits()
        
        assert limits.max_memory_mb == 256
        assert limits.max_cpu_percent == 50.0
        assert limits.max_execution_time_seconds == 300
        assert limits.max_idle_time_seconds == 60
        assert limits.max_concurrent_tasks == 5
    
    def test_custom_limits(self):
        """Test custom resource limits"""
        limits = ResourceLimits(
            max_memory_mb=128,
            max_cpu_percent=25.0,
            max_execution_time_seconds=120,
            max_idle_time_seconds=30,
            max_concurrent_tasks=3
        )
        
        assert limits.max_memory_mb == 128
        assert limits.max_cpu_percent == 25.0
        assert limits.max_execution_time_seconds == 120
        assert limits.max_idle_time_seconds == 30
        assert limits.max_concurrent_tasks == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
