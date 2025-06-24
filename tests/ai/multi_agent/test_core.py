"""
Tests for Multi-Agent core components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.ai.multi_agent.core import AgentCoordinator, WorkflowManager, CommunicationHub
from src.ai.multi_agent.models import Agent, Task, Workflow, Message, CoordinationResult
from tests.utils.helpers import create_test_agent, create_test_workflow


class TestAgentCoordinator:
    """Test cases for AgentCoordinator."""
    
    @pytest.fixture
    def agent_coordinator(self):
        """Create agent coordinator instance."""
        return AgentCoordinator()
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_coordinator_initialization(self, agent_coordinator):
        """Test agent coordinator initialization."""
        assert agent_coordinator.is_running is False
        
        await agent_coordinator.start()
        assert agent_coordinator.is_running is True
        
        status = agent_coordinator.get_system_status()
        assert status["is_running"] is True
        assert "agent_count" in status
        assert "active_workflows" in status
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_agent_registration(self, agent_coordinator):
        """Test agent registration and management."""
        await agent_coordinator.start()
        
        # Create test agent
        agent_config = create_test_agent("testing", ["test_execution", "validation"])
        
        # Register agent
        agent_id = await agent_coordinator.register_agent(agent_config)
        assert agent_id is not None
        
        # Verify agent is registered
        agents = agent_coordinator.get_registered_agents()
        assert agent_id in agents
        assert agents[agent_id]["agent_type"] == "testing"
        assert "test_execution" in agents[agent_id]["capabilities"]
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_agent_coordination(self, agent_coordinator):
        """Test agent coordination functionality."""
        await agent_coordinator.start()
        
        # Register multiple agents
        agent_configs = [
            create_test_agent("testing", ["test_execution"]),
            create_test_agent("analysis", ["data_analysis"]),
            create_test_agent("validation", ["result_validation"])
        ]
        
        agent_ids = []
        for config in agent_configs:
            agent_id = await agent_coordinator.register_agent(config)
            agent_ids.append(agent_id)
        
        # Create coordination request
        coordination_request = {
            "coordination_id": "coord_001",
            "target_agents": agent_ids,
            "coordination_type": "sequential",
            "task": {
                "task_id": "test_task_001",
                "description": "Execute comprehensive testing",
                "parameters": {"test_suite": "full"}
            }
        }
        
        # Execute coordination
        result = await agent_coordinator.coordinate_agents(coordination_request)
        
        assert result["success"] is True
        assert result["coordination_id"] == "coord_001"
        assert "results" in result
        assert len(result["results"]) == len(agent_ids)
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_agent_health_monitoring(self, agent_coordinator):
        """Test agent health monitoring."""
        await agent_coordinator.start()
        
        # Register agent
        agent_config = create_test_agent("monitoring")
        agent_id = await agent_coordinator.register_agent(agent_config)
        
        # Check agent health
        health_status = await agent_coordinator.check_agent_health(agent_id)
        
        assert "agent_id" in health_status
        assert "status" in health_status
        assert "health_score" in health_status
        assert 0.0 <= health_status["health_score"] <= 1.0
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_agent_deregistration(self, agent_coordinator):
        """Test agent deregistration."""
        await agent_coordinator.start()
        
        # Register and then deregister agent
        agent_config = create_test_agent("temporary")
        agent_id = await agent_coordinator.register_agent(agent_config)
        
        # Verify registration
        agents = agent_coordinator.get_registered_agents()
        assert agent_id in agents
        
        # Deregister agent
        success = await agent_coordinator.deregister_agent(agent_id)
        assert success is True
        
        # Verify deregistration
        agents = agent_coordinator.get_registered_agents()
        assert agent_id not in agents
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_coordinator_shutdown(self, agent_coordinator):
        """Test coordinator shutdown."""
        await agent_coordinator.start()
        
        # Register some agents
        for i in range(3):
            config = create_test_agent(f"agent_{i}")
            await agent_coordinator.register_agent(config)
        
        # Shutdown coordinator
        await agent_coordinator.shutdown()
        assert agent_coordinator.is_running is False
        
        # Verify all agents are deregistered
        agents = agent_coordinator.get_registered_agents()
        assert len(agents) == 0


class TestWorkflowManager:
    """Test cases for WorkflowManager."""
    
    @pytest.fixture
    def workflow_manager(self):
        """Create workflow manager instance."""
        from src.ai.multi_agent.core import CommunicationHub
        communication_hub = CommunicationHub()
        return WorkflowManager(communication_hub)
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_workflow_creation(self, workflow_manager):
        """Test workflow creation and management."""
        await workflow_manager.initialize()
        
        # Create test workflow
        workflow_def = create_test_workflow("test_workflow", num_steps=3)
        
        # Create workflow
        workflow_id = await workflow_manager.create_workflow(workflow_def)
        assert workflow_id is not None
        
        # Verify workflow exists
        workflows = workflow_manager.get_active_workflows()
        assert workflow_id in workflows
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_workflow_execution(self, workflow_manager):
        """Test workflow execution."""
        await workflow_manager.initialize()
        
        # Create simple workflow
        workflow_def = {
            "name": "simple_test",
            "steps": [
                {
                    "step_id": "step_1",
                    "agent_type": "testing",
                    "action": "execute_test",
                    "parameters": {"test_name": "unit_test"}
                },
                {
                    "step_id": "step_2",
                    "agent_type": "validation",
                    "action": "validate_results",
                    "parameters": {"validation_type": "strict"},
                    "dependencies": ["step_1"]
                }
            ]
        }
        
        # Execute workflow
        workflow_id = await workflow_manager.create_workflow(workflow_def)
        execution_result = await workflow_manager.execute_workflow(workflow_id)
        
        assert execution_result["success"] is True
        assert execution_result["workflow_id"] == workflow_id
        assert "step_results" in execution_result
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_workflow_monitoring(self, workflow_manager):
        """Test workflow monitoring and status tracking."""
        await workflow_manager.initialize()
        
        # Create and start workflow
        workflow_def = create_test_workflow("monitored_workflow")
        workflow_id = await workflow_manager.create_workflow(workflow_def)
        
        # Start workflow execution (non-blocking)
        execution_task = asyncio.create_task(
            workflow_manager.execute_workflow(workflow_id)
        )
        
        # Monitor workflow status
        await asyncio.sleep(0.1)  # Let workflow start
        
        status = await workflow_manager.get_workflow_status(workflow_id)
        assert status["workflow_id"] == workflow_id
        assert status["status"] in ["running", "completed", "failed"]
        
        # Wait for completion
        await execution_task
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_workflow_error_handling(self, workflow_manager):
        """Test workflow error handling."""
        await workflow_manager.initialize()
        
        # Create workflow with failing step
        failing_workflow = {
            "name": "failing_workflow",
            "steps": [
                {
                    "step_id": "failing_step",
                    "agent_type": "nonexistent",
                    "action": "invalid_action",
                    "parameters": {}
                }
            ]
        }
        
        workflow_id = await workflow_manager.create_workflow(failing_workflow)
        execution_result = await workflow_manager.execute_workflow(workflow_id)
        
        assert execution_result["success"] is False
        assert "error_message" in execution_result
        assert "failed_steps" in execution_result


class TestCommunicationHub:
    """Test cases for CommunicationHub."""
    
    @pytest.fixture
    def communication_hub(self):
        """Create communication hub instance."""
        return CommunicationHub()
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_hub_initialization(self, communication_hub):
        """Test communication hub initialization."""
        await communication_hub.start()
        assert communication_hub.is_running is True
        
        status = communication_hub.get_hub_status()
        assert status["is_running"] is True
        assert "connected_agents" in status
        assert "message_queue_size" in status
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_agent_connection(self, communication_hub):
        """Test agent connection to communication hub."""
        await communication_hub.start()
        
        # Connect agents
        agent_ids = ["agent_001", "agent_002", "agent_003"]
        
        for agent_id in agent_ids:
            success = await communication_hub.connect_agent(agent_id)
            assert success is True
        
        # Verify connections
        connected = communication_hub.get_connected_agents()
        for agent_id in agent_ids:
            assert agent_id in connected
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_message_routing(self, communication_hub):
        """Test message routing between agents."""
        await communication_hub.start()
        
        # Connect agents
        sender_id = "agent_sender"
        receiver_id = "agent_receiver"
        
        await communication_hub.connect_agent(sender_id)
        await communication_hub.connect_agent(receiver_id)
        
        # Send message
        message = {
            "message_id": "msg_001",
            "sender": sender_id,
            "receiver": receiver_id,
            "message_type": "task_request",
            "content": {
                "task": "execute_test",
                "parameters": {"test_id": "test_001"}
            }
        }
        
        success = await communication_hub.send_message(message)
        assert success is True
        
        # Receive message
        received_messages = await communication_hub.get_messages(receiver_id)
        assert len(received_messages) > 0
        assert received_messages[0]["message_id"] == "msg_001"
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_broadcast_messaging(self, communication_hub):
        """Test broadcast messaging to multiple agents."""
        await communication_hub.start()
        
        # Connect multiple agents
        agent_ids = ["agent_001", "agent_002", "agent_003"]
        for agent_id in agent_ids:
            await communication_hub.connect_agent(agent_id)
        
        # Broadcast message
        broadcast_message = {
            "message_id": "broadcast_001",
            "sender": "coordinator",
            "message_type": "system_announcement",
            "content": {
                "announcement": "System maintenance in 5 minutes"
            }
        }
        
        success = await communication_hub.broadcast_message(broadcast_message)
        assert success is True
        
        # Verify all agents received the message
        for agent_id in agent_ids:
            messages = await communication_hub.get_messages(agent_id)
            assert any(msg["message_id"] == "broadcast_001" for msg in messages)
    
    @pytest.mark.asyncio
    @pytest.mark.multiagent
    async def test_message_filtering(self, communication_hub):
        """Test message filtering and querying."""
        await communication_hub.start()
        
        agent_id = "test_agent"
        await communication_hub.connect_agent(agent_id)
        
        # Send different types of messages
        messages = [
            {
                "message_id": "msg_001",
                "sender": "coordinator",
                "receiver": agent_id,
                "message_type": "task_request",
                "content": {"task": "test_1"}
            },
            {
                "message_id": "msg_002",
                "sender": "coordinator",
                "receiver": agent_id,
                "message_type": "status_update",
                "content": {"status": "running"}
            },
            {
                "message_id": "msg_003",
                "sender": "coordinator",
                "receiver": agent_id,
                "message_type": "task_request",
                "content": {"task": "test_2"}
            }
        ]
        
        for message in messages:
            await communication_hub.send_message(message)
        
        # Filter messages by type
        task_messages = await communication_hub.get_messages(
            agent_id, message_type="task_request"
        )
        assert len(task_messages) == 2
        
        status_messages = await communication_hub.get_messages(
            agent_id, message_type="status_update"
        )
        assert len(status_messages) == 1


@pytest.mark.multiagent
@pytest.mark.integration
class TestMultiAgentIntegration:
    """Integration tests for multi-agent components."""
    
    @pytest.mark.asyncio
    async def test_full_multi_agent_workflow(self):
        """Test complete multi-agent workflow execution."""
        # Initialize components
        coordinator = AgentCoordinator()
        workflow_manager = WorkflowManager()
        communication_hub = CommunicationHub()
        
        await coordinator.start()
        await workflow_manager.initialize()
        await communication_hub.start()
        
        # Register agents
        agent_configs = [
            create_test_agent("testing", ["test_execution", "unit_testing"]),
            create_test_agent("analysis", ["data_analysis", "performance_analysis"]),
            create_test_agent("validation", ["result_validation", "quality_assurance"])
        ]
        
        agent_ids = []
        for config in agent_configs:
            agent_id = await coordinator.register_agent(config)
            agent_ids.append(agent_id)
            await communication_hub.connect_agent(agent_id)
        
        # Create comprehensive workflow
        workflow_def = {
            "name": "comprehensive_testing_workflow",
            "description": "Multi-agent testing and validation workflow",
            "steps": [
                {
                    "step_id": "execute_tests",
                    "agent_type": "testing",
                    "action": "run_test_suite",
                    "parameters": {
                        "test_suite": "comprehensive",
                        "parallel": True
                    }
                },
                {
                    "step_id": "analyze_results",
                    "agent_type": "analysis",
                    "action": "analyze_test_results",
                    "parameters": {
                        "analysis_type": "performance"
                    },
                    "dependencies": ["execute_tests"]
                },
                {
                    "step_id": "validate_quality",
                    "agent_type": "validation",
                    "action": "validate_quality_metrics",
                    "parameters": {
                        "quality_threshold": 0.95
                    },
                    "dependencies": ["analyze_results"]
                }
            ]
        }
        
        # Execute workflow
        workflow_id = await workflow_manager.create_workflow(workflow_def)
        execution_result = await workflow_manager.execute_workflow(workflow_id)
        
        # Verify results
        assert execution_result["success"] is True
        assert len(execution_result["step_results"]) == 3
        
        # Verify agent coordination
        coordination_request = {
            "coordination_id": "final_coordination",
            "target_agents": agent_ids,
            "coordination_type": "parallel",
            "task": {
                "task_id": "final_validation",
                "description": "Final system validation"
            }
        }
        
        coordination_result = await coordinator.coordinate_agents(coordination_request)
        assert coordination_result["success"] is True
        
        # Cleanup
        await coordinator.shutdown()
        await workflow_manager.shutdown()
        await communication_hub.stop()
