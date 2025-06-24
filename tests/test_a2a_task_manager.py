#!/usr/bin/env python3
"""
Test A2A Task Management System

Tests the A2A-compliant task management implementation according to Google A2A specification.
"""

import pytest
import asyncio
from datetime import datetime, timedelta

# Import the A2A task management components
try:
    from src.a2a_protocol.task_manager import (
        A2ATaskManager, Task, TaskState, TaskStatus, Message, MessagePart, 
        Artifact, TaskContext
    )
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False


@pytest.mark.skipif(not A2A_AVAILABLE, reason="A2A protocol not available")
class TestA2ATaskManager:
    """Test A2A Task Manager"""
    
    def setup_method(self):
        """Setup test environment"""
        self.task_manager = A2ATaskManager(task_timeout_hours=1)  # Short timeout for testing
    
    @pytest.mark.asyncio
    async def test_create_task(self):
        """Test task creation"""
        # Create message
        message = Message(
            messageId="test-msg-123",
            role="user",
            parts=[MessagePart(kind="text", text="Hello, world!")],
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Create task
        task = await self.task_manager.create_task(message)
        
        # Verify task structure
        assert task.id is not None
        assert task.contextId is not None
        assert task.status.state == TaskState.SUBMITTED
        assert len(task.history) == 1
        assert task.history[0] == message
        assert len(task.artifacts) == 0
        assert task.kind == "task"
        assert task.created_at is not None
        assert task.updated_at is not None
        assert task.expires_at is not None
        
        # Verify task is stored
        retrieved_task = await self.task_manager.get_task(task.id)
        assert retrieved_task == task
        
        # Verify context is created
        context = await self.task_manager.get_task_context(task.contextId)
        assert context is not None
        assert task.id in context.tasks
    
    @pytest.mark.asyncio
    async def test_create_task_with_custom_ids(self):
        """Test task creation with custom IDs"""
        message = Message(
            messageId="test-msg-123",
            role="user",
            parts=[MessagePart(kind="text", text="Test message")],
            timestamp=datetime.utcnow().isoformat()
        )
        
        custom_task_id = "custom-task-123"
        custom_context_id = "custom-context-123"
        custom_metadata = {"test": "value"}
        
        task = await self.task_manager.create_task(
            message=message,
            context_id=custom_context_id,
            task_id=custom_task_id,
            metadata=custom_metadata
        )
        
        # Verify custom values
        assert task.id == custom_task_id
        assert task.contextId == custom_context_id
        assert task.metadata == custom_metadata
    
    @pytest.mark.asyncio
    async def test_update_task_status(self):
        """Test task status updates"""
        # Create task
        message = self.task_manager.create_message("user", "Test message")
        task = await self.task_manager.create_task(message)
        
        # Update to working
        success = await self.task_manager.update_task_status(
            task.id, 
            TaskState.WORKING, 
            message="Processing task",
            progress=0.5
        )
        
        assert success == True
        
        # Verify update
        updated_task = await self.task_manager.get_task(task.id)
        assert updated_task.status.state == TaskState.WORKING
        assert updated_task.status.message == "Processing task"
        assert updated_task.status.progress == 0.5
        assert updated_task.updated_at != task.updated_at
        
        # Update to completed
        success = await self.task_manager.update_task_status(
            task.id,
            TaskState.COMPLETED,
            message="Task completed",
            progress=1.0
        )
        
        assert success == True
        
        # Verify completion
        completed_task = await self.task_manager.get_task(task.id)
        assert completed_task.status.state == TaskState.COMPLETED
        assert completed_task.status.progress == 1.0
    
    @pytest.mark.asyncio
    async def test_invalid_state_transitions(self):
        """Test invalid state transitions are rejected"""
        # Create and complete task
        message = self.task_manager.create_message("user", "Test message")
        task = await self.task_manager.create_task(message)
        
        await self.task_manager.update_task_status(task.id, TaskState.COMPLETED)
        
        # Try to transition from completed to working (invalid)
        success = await self.task_manager.update_task_status(task.id, TaskState.WORKING)
        assert success == False
        
        # Verify task state unchanged
        unchanged_task = await self.task_manager.get_task(task.id)
        assert unchanged_task.status.state == TaskState.COMPLETED
    
    @pytest.mark.asyncio
    async def test_add_message_to_task(self):
        """Test adding messages to task history"""
        # Create task
        initial_message = self.task_manager.create_message("user", "Initial message")
        task = await self.task_manager.create_task(initial_message)
        
        # Add response message
        response_message = self.task_manager.create_message("agent", "Response message")
        success = await self.task_manager.add_message_to_task(task.id, response_message)
        
        assert success == True
        
        # Verify message added
        updated_task = await self.task_manager.get_task(task.id)
        assert len(updated_task.history) == 2
        assert updated_task.history[1] == response_message
    
    @pytest.mark.asyncio
    async def test_add_artifact_to_task(self):
        """Test adding artifacts to task"""
        # Create task
        message = self.task_manager.create_message("user", "Test message")
        task = await self.task_manager.create_task(message)
        
        # Add artifact
        artifact = self.task_manager.create_artifact(
            name="response",
            text="Generated response",
            metadata={"type": "text_response"}
        )
        
        success = await self.task_manager.add_artifact_to_task(task.id, artifact)
        assert success == True
        
        # Verify artifact added
        updated_task = await self.task_manager.get_task(task.id)
        assert len(updated_task.artifacts) == 1
        assert updated_task.artifacts[0] == artifact
        
        # Update existing artifact
        updated_artifact = self.task_manager.create_artifact(
            name="response",
            text="Updated response",
            artifact_id=artifact.artifactId,
            metadata={"type": "updated_response"}
        )
        
        success = await self.task_manager.add_artifact_to_task(task.id, updated_artifact)
        assert success == True
        
        # Verify artifact updated (not duplicated)
        final_task = await self.task_manager.get_task(task.id)
        assert len(final_task.artifacts) == 1
        assert final_task.artifacts[0].parts[0].text == "Updated response"
        assert final_task.artifacts[0].metadata["type"] == "updated_response"
    
    @pytest.mark.asyncio
    async def test_cancel_task(self):
        """Test task cancellation"""
        # Create task
        message = self.task_manager.create_message("user", "Test message")
        task = await self.task_manager.create_task(message)
        
        # Cancel task
        success = await self.task_manager.cancel_task(task.id, "Test cancellation")
        assert success == True
        
        # Verify cancellation
        canceled_task = await self.task_manager.get_task(task.id)
        assert canceled_task.status.state == TaskState.CANCELED
        assert canceled_task.status.message == "Test cancellation"
    
    @pytest.mark.asyncio
    async def test_list_tasks_in_context(self):
        """Test listing tasks in a context"""
        context_id = "test-context-123"
        
        # Create multiple tasks in same context
        message1 = self.task_manager.create_message("user", "Message 1")
        message2 = self.task_manager.create_message("user", "Message 2")
        
        task1 = await self.task_manager.create_task(message1, context_id=context_id)
        task2 = await self.task_manager.create_task(message2, context_id=context_id)
        
        # List tasks in context
        tasks = await self.task_manager.list_tasks_in_context(context_id)
        
        # Verify both tasks returned
        assert len(tasks) == 2
        task_ids = [task.id for task in tasks]
        assert task1.id in task_ids
        assert task2.id in task_ids
    
    @pytest.mark.asyncio
    async def test_list_tasks_by_state(self):
        """Test listing tasks by state"""
        # Create tasks with different states
        message1 = self.task_manager.create_message("user", "Message 1")
        message2 = self.task_manager.create_message("user", "Message 2")
        message3 = self.task_manager.create_message("user", "Message 3")
        
        task1 = await self.task_manager.create_task(message1)
        task2 = await self.task_manager.create_task(message2)
        task3 = await self.task_manager.create_task(message3)
        
        # Update states
        await self.task_manager.update_task_status(task2.id, TaskState.WORKING)
        await self.task_manager.update_task_status(task3.id, TaskState.COMPLETED)
        
        # List tasks by state
        submitted_tasks = await self.task_manager.list_tasks_by_state(TaskState.SUBMITTED)
        working_tasks = await self.task_manager.list_tasks_by_state(TaskState.WORKING)
        completed_tasks = await self.task_manager.list_tasks_by_state(TaskState.COMPLETED)
        
        # Verify results
        assert len(submitted_tasks) == 1
        assert submitted_tasks[0].id == task1.id
        
        assert len(working_tasks) == 1
        assert working_tasks[0].id == task2.id
        
        assert len(completed_tasks) == 1
        assert completed_tasks[0].id == task3.id
    
    @pytest.mark.asyncio
    async def test_to_dict_serialization(self):
        """Test task serialization to dictionary"""
        # Create task with message and artifact
        message = self.task_manager.create_message("user", "Test message")
        task = await self.task_manager.create_task(message)
        
        artifact = self.task_manager.create_artifact("response", "Test response")
        await self.task_manager.add_artifact_to_task(task.id, artifact)
        
        # Convert to dictionary
        task_dict = self.task_manager.to_dict(task)
        
        # Verify structure
        assert "id" in task_dict
        assert "contextId" in task_dict
        assert "status" in task_dict
        assert "history" in task_dict
        assert "artifacts" in task_dict
        assert "kind" in task_dict
        assert "metadata" in task_dict
        
        # Verify status structure
        status = task_dict["status"]
        assert "state" in status
        assert "timestamp" in status
        
        # Verify history structure
        assert len(task_dict["history"]) == 1
        history_msg = task_dict["history"][0]
        assert "messageId" in history_msg
        assert "role" in history_msg
        assert "parts" in history_msg
        
        # Verify artifacts structure
        assert len(task_dict["artifacts"]) == 1
        artifact_data = task_dict["artifacts"][0]
        assert "artifactId" in artifact_data
        assert "name" in artifact_data
        assert "parts" in artifact_data
    
    def test_helper_methods(self):
        """Test helper methods for creating messages and artifacts"""
        # Test create_message
        message = self.task_manager.create_message("user", "Test message")
        assert message.role == "user"
        assert len(message.parts) == 1
        assert message.parts[0].kind == "text"
        assert message.parts[0].text == "Test message"
        assert message.messageId is not None
        assert message.timestamp is not None
        
        # Test create_artifact
        artifact = self.task_manager.create_artifact(
            name="test_artifact",
            text="Test content",
            metadata={"type": "test"}
        )
        assert artifact.name == "test_artifact"
        assert len(artifact.parts) == 1
        assert artifact.parts[0].kind == "text"
        assert artifact.parts[0].text == "Test content"
        assert artifact.metadata["type"] == "test"
        assert artifact.artifactId is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
