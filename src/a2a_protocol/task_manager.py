#!/usr/bin/env python3
"""
A2A Task Management System

Implements proper task lifecycle management according to Google A2A specification.
Handles task states, context management, and coordination between agents.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """A2A Task States according to specification"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class TaskStatus:
    """A2A Task Status"""
    state: TaskState
    timestamp: str
    message: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    progress: Optional[float] = None  # 0.0 to 1.0


# Import message parts from dedicated module
try:
    from .message_parts import A2AMessage, TextPart, FilePart, DataPart, A2AMessageBuilder
    MESSAGE_PARTS_AVAILABLE = True
    # Use A2AMessage as Message for compatibility
    Message = A2AMessage
    MessagePart = Union[TextPart, FilePart, DataPart]
except ImportError:
    MESSAGE_PARTS_AVAILABLE = False
    # Fallback to simple message structure
    @dataclass
    class MessagePart:
        """A2A Message Part (fallback)"""
        kind: str  # "text", "file", "data"
        text: Optional[str] = None
        file: Optional[Dict[str, Any]] = None
        data: Optional[Dict[str, Any]] = None

    @dataclass
    class Message:
        """A2A Message (fallback)"""
        messageId: str
        role: str  # "user" or "agent"
        parts: List[MessagePart]
        timestamp: Optional[str] = None


@dataclass
class Artifact:
    """A2A Task Artifact"""
    artifactId: str
    name: str
    parts: List[MessagePart]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A2A Task"""
    id: str
    contextId: str
    status: TaskStatus
    history: List[Message] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    kind: str = "task"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    expires_at: Optional[str] = None


@dataclass
class TaskContext:
    """A2A Task Context for grouping related tasks"""
    contextId: str
    tasks: List[str] = field(default_factory=list)  # Task IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None


class A2ATaskManager:
    """Manages A2A tasks and their lifecycle"""
    
    def __init__(self, task_timeout_hours: int = 24):
        self.tasks: Dict[str, Task] = {}
        self.contexts: Dict[str, TaskContext] = {}
        self.task_timeout_hours = task_timeout_hours
        self._lock = None  # Will be created when needed
        self._cleanup_task = None

        # Don't start cleanup task in __init__ - start it when first needed
        self._cleanup_started = False

    async def _ensure_async_setup(self):
        """Ensure async components are set up"""
        if self._lock is None:
            self._lock = asyncio.Lock()

        if not self._cleanup_started:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_tasks())
            self._cleanup_started = True
    
    async def create_task(self, message: Message, context_id: Optional[str] = None,
                         task_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Task:
        """Create a new A2A task"""
        await self._ensure_async_setup()
        async with self._lock:
            # Generate IDs if not provided
            if not task_id:
                task_id = str(uuid.uuid4())
            if not context_id:
                context_id = str(uuid.uuid4())
            
            # Create task status
            status = TaskStatus(
                state=TaskState.SUBMITTED,
                timestamp=datetime.utcnow().isoformat(),
                message="Task submitted"
            )
            
            # Calculate expiration
            expires_at = (datetime.utcnow() + timedelta(hours=self.task_timeout_hours)).isoformat()
            
            # Create task
            task = Task(
                id=task_id,
                contextId=context_id,
                status=status,
                history=[message],
                artifacts=[],
                metadata=metadata or {},
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
                expires_at=expires_at
            )
            
            # Store task
            self.tasks[task_id] = task
            
            # Create or update context
            if context_id not in self.contexts:
                self.contexts[context_id] = TaskContext(
                    contextId=context_id,
                    created_at=datetime.utcnow().isoformat()
                )
            
            self.contexts[context_id].tasks.append(task_id)
            
            logger.info(f"Created A2A task {task_id} in context {context_id}")
            return task
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    async def update_task_status(self, task_id: str, state: TaskState,
                               message: Optional[str] = None, error: Optional[Dict[str, Any]] = None,
                               progress: Optional[float] = None) -> bool:
        """Update task status"""
        await self._ensure_async_setup()
        async with self._lock:
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found for status update")
                return False
            
            task = self.tasks[task_id]
            
            # Validate state transition
            if not self._is_valid_state_transition(task.status.state, state):
                logger.warning(f"Invalid state transition for task {task_id}: {task.status.state} -> {state}")
                return False
            
            # Update status
            task.status = TaskStatus(
                state=state,
                timestamp=datetime.utcnow().isoformat(),
                message=message,
                error=error,
                progress=progress
            )
            
            task.updated_at = datetime.utcnow().isoformat()
            
            logger.info(f"Updated task {task_id} status to {state.value}")
            return True
    
    async def add_message_to_task(self, task_id: str, message: Message) -> bool:
        """Add message to task history"""
        await self._ensure_async_setup()
        async with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.history.append(message)
            task.updated_at = datetime.utcnow().isoformat()
            
            logger.debug(f"Added message to task {task_id}")
            return True
    
    async def add_artifact_to_task(self, task_id: str, artifact: Artifact) -> bool:
        """Add artifact to task"""
        await self._ensure_async_setup()
        async with self._lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            # Check if artifact already exists (update if so)
            existing_artifact = None
            for i, existing in enumerate(task.artifacts):
                if existing.artifactId == artifact.artifactId:
                    existing_artifact = i
                    break
            
            if existing_artifact is not None:
                task.artifacts[existing_artifact] = artifact
                logger.debug(f"Updated artifact {artifact.artifactId} in task {task_id}")
            else:
                task.artifacts.append(artifact)
                logger.debug(f"Added artifact {artifact.artifactId} to task {task_id}")
            
            task.updated_at = datetime.utcnow().isoformat()
            return True
    
    async def cancel_task(self, task_id: str, reason: Optional[str] = None) -> bool:
        """Cancel a task"""
        return await self.update_task_status(
            task_id, 
            TaskState.CANCELED, 
            message=reason or "Task canceled"
        )
    
    async def list_tasks_in_context(self, context_id: str) -> List[Task]:
        """List all tasks in a context"""
        if context_id not in self.contexts:
            return []
        
        context = self.contexts[context_id]
        tasks = []
        
        for task_id in context.tasks:
            if task_id in self.tasks:
                tasks.append(self.tasks[task_id])
        
        return tasks
    
    async def list_tasks_by_state(self, state: TaskState) -> List[Task]:
        """List all tasks with a specific state"""
        return [task for task in self.tasks.values() if task.status.state == state]
    
    async def get_task_context(self, context_id: str) -> Optional[TaskContext]:
        """Get task context by ID"""
        return self.contexts.get(context_id)
    
    def _is_valid_state_transition(self, current_state: TaskState, new_state: TaskState) -> bool:
        """Validate task state transitions according to A2A specification"""
        valid_transitions = {
            TaskState.SUBMITTED: [TaskState.WORKING, TaskState.CANCELED, TaskState.FAILED],
            TaskState.WORKING: [TaskState.INPUT_REQUIRED, TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED],
            TaskState.INPUT_REQUIRED: [TaskState.WORKING, TaskState.CANCELED, TaskState.FAILED],
            TaskState.COMPLETED: [],  # Terminal state
            TaskState.FAILED: [],     # Terminal state
            TaskState.CANCELED: []    # Terminal state
        }
        
        return new_state in valid_transitions.get(current_state, [])
    
    async def _cleanup_expired_tasks(self):
        """Background task to clean up expired tasks"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                current_time = datetime.utcnow()
                expired_tasks = []
                
                async with self._lock:
                    for task_id, task in self.tasks.items():
                        if task.expires_at:
                            expires_at = datetime.fromisoformat(task.expires_at.replace('Z', '+00:00'))
                            if current_time > expires_at.replace(tzinfo=None):
                                expired_tasks.append(task_id)
                    
                    # Remove expired tasks
                    for task_id in expired_tasks:
                        task = self.tasks[task_id]
                        
                        # Remove from context
                        if task.contextId in self.contexts:
                            context = self.contexts[task.contextId]
                            if task_id in context.tasks:
                                context.tasks.remove(task_id)
                            
                            # Remove empty contexts
                            if not context.tasks:
                                del self.contexts[task.contextId]
                        
                        # Remove task
                        del self.tasks[task_id]
                        logger.info(f"Cleaned up expired task {task_id}")
                
            except Exception as e:
                logger.error(f"Error in task cleanup: {e}")
    
    def to_dict(self, task: Task) -> Dict[str, Any]:
        """Convert task to dictionary for JSON serialization"""
        return {
            "id": task.id,
            "contextId": task.contextId,
            "status": {
                "state": task.status.state.value,
                "timestamp": task.status.timestamp,
                "message": task.status.message,
                "error": task.status.error,
                "progress": task.status.progress
            },
            "history": [
                {
                    "messageId": msg.messageId,
                    "role": msg.role,
                    "parts": [
                        {
                            "kind": part.kind,
                            "text": part.text,
                            "file": part.file,
                            "data": part.data
                        }
                        for part in msg.parts
                    ],
                    "timestamp": msg.timestamp
                }
                for msg in task.history
            ],
            "artifacts": [
                {
                    "artifactId": artifact.artifactId,
                    "name": artifact.name,
                    "parts": [
                        {
                            "kind": part.kind,
                            "text": part.text,
                            "file": part.file,
                            "data": part.data
                        }
                        for part in artifact.parts
                    ],
                    "metadata": artifact.metadata
                }
                for artifact in task.artifacts
            ],
            "kind": task.kind,
            "metadata": task.metadata
        }
    
    def create_message(self, role: str, text: str, message_id: Optional[str] = None) -> Message:
        """Helper to create a text message"""
        if MESSAGE_PARTS_AVAILABLE:
            return A2AMessageBuilder(role, message_id).add_text(text).build()
        else:
            # Fallback implementation
            return Message(
                messageId=message_id or str(uuid.uuid4()),
                role=role,
                parts=[MessagePart(kind="text", text=text)],
                timestamp=datetime.utcnow().isoformat()
            )
    
    def create_artifact(self, name: str, text: str, artifact_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Artifact:
        """Helper to create a text artifact"""
        return Artifact(
            artifactId=artifact_id or str(uuid.uuid4()),
            name=name,
            parts=[MessagePart(kind="text", text=text)],
            metadata=metadata or {}
        )


    # Synchronous methods for testing
    def create_task_sync(self, task_id: str, context_id: str, message_content: str) -> Task:
        """Synchronous version of create_task for testing"""
        try:
            # Create a simple message
            message = self.create_message("user", message_content)

            # Create task without async
            status = TaskStatus(
                state=TaskState.SUBMITTED,
                timestamp=datetime.utcnow().isoformat(),
                message="Task submitted"
            )

            expires_at = (datetime.utcnow() + timedelta(hours=self.task_timeout_hours)).isoformat()

            task = Task(
                id=task_id,
                contextId=context_id,
                status=status,
                history=[message],
                artifacts=[],
                metadata={},
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
                expires_at=expires_at
            )

            # Store task
            self.tasks[task_id] = task

            # Create or update context
            if context_id not in self.contexts:
                self.contexts[context_id] = TaskContext(
                    contextId=context_id,
                    created_at=datetime.utcnow().isoformat()
                )

            self.contexts[context_id].tasks.append(task_id)

            logger.info(f"Created A2A task {task_id} in context {context_id} (sync)")
            return task

        except Exception as e:
            logger.error(f"Failed to create task sync: {e}")
            raise

    def get_task_sync(self, task_id: str) -> Optional[Task]:
        """Synchronous version of get_task"""
        return self.tasks.get(task_id)

    def update_task_status_sync(self, task_id: str, state: TaskState,
                               message: Optional[str] = None) -> bool:
        """Synchronous version of update_task_status"""
        try:
            if task_id not in self.tasks:
                logger.warning(f"Task {task_id} not found for status update")
                return False

            task = self.tasks[task_id]

            # Validate state transition
            if not self._is_valid_state_transition(task.status.state, state):
                logger.warning(f"Invalid state transition for task {task_id}: {task.status.state} -> {state}")
                return False

            # Update status
            task.status = TaskStatus(
                state=state,
                timestamp=datetime.utcnow().isoformat(),
                message=message
            )

            task.updated_at = datetime.utcnow().isoformat()

            logger.info(f"Updated task {task_id} status to {state.value} (sync)")
            return True

        except Exception as e:
            logger.error(f"Failed to update task status sync: {e}")
            return False


# Global task manager instance
task_manager = A2ATaskManager()
