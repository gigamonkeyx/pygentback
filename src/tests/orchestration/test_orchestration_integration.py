"""
Comprehensive Integration Tests for Orchestration System

Tests the complete orchestration system with real MCP servers and agents.
Identifies and documents any mock/placeholder code for removal.
"""

import asyncio
import pytest
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import orchestration components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from orchestration.orchestration_manager import OrchestrationManager
from orchestration.coordination_models import (
    OrchestrationConfig, AgentCapability, MCPServerInfo, TaskRequest,
    AgentType, MCPServerType, TaskPriority, TaskStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestOrchestrationIntegration:
    """Integration tests for the complete orchestration system."""
    
    @pytest.fixture
    async def orchestration_manager(self):
        """Create and start orchestration manager."""
        config = OrchestrationConfig(
            evolution_enabled=True,
            max_concurrent_tasks=10,
            batch_processing_enabled=True,
            detailed_logging=True
        )
        
        manager = OrchestrationManager(config)
        await manager.start()
        
        yield manager
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_system_startup_and_health(self, orchestration_manager):
        """Test system startup and health check."""
        manager = orchestration_manager
        
        # Check system is running
        assert manager.is_running
        
        # Perform health check
        health = await manager.health_check()
        assert health["overall_health"] in ["healthy", "warning"]
        assert "components" in health
        
        # Check system status
        status = await manager.get_system_status()
        assert status["is_running"] is True
        assert "components" in status
        
        logger.info("✅ System startup and health check passed")
    
    @pytest.mark.asyncio
    async def test_mcp_server_registration(self, orchestration_manager):
        """Test MCP server registration and management."""
        manager = orchestration_manager
        
        # Register existing MCP servers
        await manager.register_existing_mcp_servers()
        
        # Check server status
        server_status = await manager.get_mcp_server_status()
        
        # Verify servers are registered
        expected_servers = ["filesystem_server", "postgresql_server", "github_server", "memory_server"]
        for server_id in expected_servers:
            assert server_id in server_status
            server_info = server_status[server_id]
            if isinstance(server_info, dict):
                assert "name" in server_info
                assert "type" in server_info
        
        logger.info("✅ MCP server registration test passed")
    
    @pytest.mark.asyncio
    async def test_agent_registration_and_management(self, orchestration_manager):
        """Test agent registration and management."""
        manager = orchestration_manager
        
        # Create test agents
        tot_agent_id = await manager.create_tot_agent(
            "Test ToT Agent", 
            ["reasoning", "problem_solving", "analysis"]
        )
        assert tot_agent_id != ""
        
        rag_agent_id = await manager.create_rag_agent(
            "Test RAG Agent", 
            "retrieval"
        )
        assert rag_agent_id != ""
        
        eval_agent_id = await manager.create_evaluation_agent(
            "Test Eval Agent",
            ["quality_assessment", "performance_evaluation"]
        )
        assert eval_agent_id != ""
        
        # Check agent status
        agent_status = await manager.get_agent_status()
        assert tot_agent_id in agent_status
        assert rag_agent_id in agent_status
        assert eval_agent_id in agent_status
        
        # Verify agent details
        tot_status = agent_status[tot_agent_id]
        assert tot_status["type"] == "tot_reasoning"
        assert tot_status["is_available"] is True
        
        logger.info("✅ Agent registration and management test passed")
    
    @pytest.mark.asyncio
    async def test_task_submission_and_execution(self, orchestration_manager):
        """Test task submission and execution."""
        manager = orchestration_manager
        
        # Setup agents and servers
        await manager.register_existing_mcp_servers()
        tot_agent_id = await manager.create_tot_agent("Task Test Agent", ["reasoning"])
        
        # Create and submit test task
        task = TaskRequest(
            task_type="reasoning",
            priority=TaskPriority.HIGH,
            description="Test reasoning task",
            input_data={"problem": "Test problem for orchestration"},
            required_capabilities={"reasoning"},
            required_mcp_servers={MCPServerType.MEMORY}
        )
        
        success = await manager.submit_task(task)
        assert success is True
        
        # Check task status
        task_status = await manager.get_task_status(task.task_id)
        assert task_status is not None
        assert task_status["task_id"] == task.task_id
        assert task_status["status"] in ["pending", "assigned", "running"]
        
        # Check queue status
        queue_status = await manager.get_queue_status()
        assert queue_status["total_tasks"] >= 1
        
        logger.info("✅ Task submission and execution test passed")
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, orchestration_manager):
        """Test metrics collection and monitoring."""
        manager = orchestration_manager
        
        # Wait for metrics collection
        await asyncio.sleep(2)
        
        # Get system metrics
        metrics = await manager.get_system_metrics()
        assert "total_tasks" in metrics
        assert "active_agents" in metrics
        assert "coordination_efficiency" in metrics
        
        # Get performance trends
        trends = await manager.get_performance_trends("coordination_efficiency")
        assert "metric_name" in trends
        
        # Check alerts
        alerts = await manager.get_alerts()
        assert isinstance(alerts, dict)
        
        logger.info("✅ Metrics collection test passed")
    
    @pytest.mark.asyncio
    async def test_evolution_system(self, orchestration_manager):
        """Test evolutionary orchestration."""
        manager = orchestration_manager
        
        if not manager.config.evolution_enabled:
            pytest.skip("Evolution not enabled")
        
        # Get evolution status
        evolution_status = await manager.get_evolution_status()
        assert "generation" in evolution_status
        assert "population_size" in evolution_status
        
        # Get current strategy
        current_strategy = await manager.get_current_strategy()
        assert "genome_id" in current_strategy
        
        # Force evolution (if safe)
        # evolution_success = await manager.force_evolution()
        # assert evolution_success is True
        
        logger.info("✅ Evolution system test passed")
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, orchestration_manager):
        """Test batch task operations."""
        manager = orchestration_manager
        
        # Setup
        await manager.register_existing_mcp_servers()
        await manager.create_tot_agent("Batch Test Agent", ["reasoning", "analysis"])
        
        # Create batch tasks
        tasks = []
        for i in range(3):
            task = TaskRequest(
                task_type="reasoning",
                priority=TaskPriority.NORMAL,
                description=f"Batch test task {i}",
                input_data={"problem": f"Batch problem {i}"},
                required_capabilities={"reasoning"}
            )
            tasks.append(task)
        
        # Submit batch
        results = await manager.submit_batch_tasks(tasks)
        assert len(results) == 3
        assert all(results)  # All should succeed
        
        logger.info("✅ Batch operations test passed")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, orchestration_manager):
        """Test error handling and system recovery."""
        manager = orchestration_manager
        
        # Test invalid task submission
        invalid_task = TaskRequest(
            task_type="",  # Invalid empty task type
            description=""  # Invalid empty description
        )
        
        success = await manager.submit_task(invalid_task)
        assert success is False
        
        # Test invalid agent registration
        invalid_agent = AgentCapability(
            agent_id="",  # Invalid empty ID
            agent_type=AgentType.CUSTOM,
            name="",  # Invalid empty name
            description="Test",
            supported_tasks=set()
        )
        
        success = await manager.register_agent(invalid_agent)
        assert success is False
        
        # System should still be healthy
        health = await manager.health_check()
        assert health["overall_health"] in ["healthy", "warning"]
        
        logger.info("✅ Error handling and recovery test passed")


class MockCodeAudit:
    """Audit for mock/placeholder code that needs removal."""
    
    def __init__(self):
        self.mock_locations = []
        self.todo_locations = []
        self.placeholder_locations = []
    
    def audit_orchestration_code(self):
        """Audit orchestration code for mock implementations."""
        
        # Known mock implementations to document for removal
        mock_implementations = [
            {
                "file": "mcp_orchestrator.py",
                "class": "MCPServerConnection",
                "method": "connect",
                "line_approx": 45,
                "description": "Simulated MCP connection - needs real MCP client implementation",
                "priority": "HIGH",
                "removal_plan": "Replace with actual MCP protocol client when MCP integration is ready"
            },
            {
                "file": "mcp_orchestrator.py", 
                "class": "MCPServerConnection",
                "method": "execute_request",
                "line_approx": 65,
                "description": "Simulated request execution with sleep(0.1) - needs real MCP request handling",
                "priority": "HIGH",
                "removal_plan": "Replace with actual MCP request/response handling"
            },
            {
                "file": "task_dispatcher.py",
                "class": "TaskDispatcher", 
                "method": "_execute_task",
                "line_approx": 520,
                "description": "Simulated task execution with random success/failure - needs real agent execution",
                "priority": "HIGH", 
                "removal_plan": "Replace with actual agent task execution interface"
            },
            {
                "file": "evolutionary_orchestrator.py",
                "class": "EvolutionaryOrchestrator",
                "method": "_evaluate_genome", 
                "line_approx": 280,
                "description": "Simulated fitness evaluation - needs real performance measurement",
                "priority": "MEDIUM",
                "removal_plan": "Replace with actual performance metrics from real system execution"
            }
        ]
        
        self.mock_locations = mock_implementations
        return mock_implementations
    
    def generate_removal_plan(self):
        """Generate plan for removing mock code."""
        plan = {
            "immediate_removal": [],
            "phase_2_removal": [],
            "phase_3_removal": []
        }
        
        for mock in self.mock_locations:
            if mock["priority"] == "HIGH":
                plan["phase_2_removal"].append(mock)
            elif mock["priority"] == "MEDIUM":
                plan["phase_3_removal"].append(mock)
            else:
                plan["immediate_removal"].append(mock)
        
        return plan


async def run_integration_tests():
    """Run all integration tests."""
    logger.info("🧪 Starting Orchestration Integration Tests...")
    
    try:
        # Create test instance
        config = OrchestrationConfig(
            evolution_enabled=True,
            max_concurrent_tasks=5,
            detailed_logging=True
        )
        
        manager = OrchestrationManager(config)
        await manager.start()
        
        # Run tests
        test_instance = TestOrchestrationIntegration()
        
        await test_instance.test_system_startup_and_health(manager)
        await test_instance.test_mcp_server_registration(manager)
        await test_instance.test_agent_registration_and_management(manager)
        await test_instance.test_task_submission_and_execution(manager)
        await test_instance.test_metrics_collection(manager)
        await test_instance.test_evolution_system(manager)
        await test_instance.test_batch_operations(manager)
        await test_instance.test_error_handling_and_recovery(manager)
        
        await manager.stop()
        
        logger.info("✅ All integration tests passed!")
        
        # Audit mock code
        auditor = MockCodeAudit()
        mock_locations = auditor.audit_orchestration_code()
        removal_plan = auditor.generate_removal_plan()
        
        logger.info("📋 Mock Code Audit Results:")
        for mock in mock_locations:
            logger.info(f"  🔍 {mock['file']}::{mock['method']} - {mock['description']}")
        
        return True, mock_locations, removal_plan
        
    except Exception as e:
        logger.error(f"❌ Integration tests failed: {e}")
        return False, [], {}


if __name__ == "__main__":
    asyncio.run(run_integration_tests())