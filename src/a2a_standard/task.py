"""
Task Management Implementation

Implements the Task specification from Google A2A Protocol.
Tasks are the fundamental unit of work managed by A2A, with defined
lifecycle states and status tracking.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from .message import Message
from .artifact import Artifact


class TaskState(str, Enum):
    """Task lifecycle states as defined by A2A specification"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class TaskStatus:
    """Task status information"""
    state: TaskState
    timestamp: Optional[datetime] = None
    message: Optional[Message] = None
    error: Optional[str] = None
    progress: Optional[float] = None  # 0.0 to 1.0
    
    def __post_init__(self):
        """Set default timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "state": self.state.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
        
        if self.message:
            result["message"] = self.message.to_dict()
        if self.error:
            result["error"] = self.error
        if self.progress is not None:
            result["progress"] = self.progress
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskStatus":
        """Create TaskStatus from dictionary"""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
        
        message = None
        if "message" in data:
            from .message import Message
            message = Message.from_dict(data["message"])
        
        return cls(
            state=TaskState(data["state"]),
            timestamp=timestamp,
            message=message,
            error=data.get("error"),
            progress=data.get("progress")
        )


@dataclass
class Task:
    """
    Task - fundamental unit of work in A2A protocol
    
    Tasks are stateful and progress through a defined lifecycle.
    They can involve multiple message exchanges and produce artifacts.
    """
    
    # Required fields
    id: str
    status: TaskStatus
    
    # Optional fields
    context_id: Optional[str] = None
    history: List[Message] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Internal tracking
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Initialize task with unique ID if not provided"""
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.context_id:
            self.context_id = str(uuid.uuid4())
    
    def update_status(self, state: TaskState, message: Optional[Message] = None, 
                     error: Optional[str] = None, progress: Optional[float] = None) -> None:
        """Update task status"""
        self.status = TaskStatus(
            state=state,
            timestamp=datetime.now(timezone.utc),
            message=message,
            error=error,
            progress=progress
        )
        self.updated_at = datetime.now(timezone.utc)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the task history"""
        # Set task and context IDs on the message
        message.task_id = self.id
        message.context_id = self.context_id
        
        self.history.append(message)
        self.updated_at = datetime.now(timezone.utc)
    
    def add_artifact(self, artifact: Artifact) -> None:
        """Add an artifact to the task"""
        self.artifacts.append(artifact)
        self.updated_at = datetime.now(timezone.utc)
    
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state"""
        return self.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]
    
    def is_active(self) -> bool:
        """Check if task is in an active (non-terminal) state"""
        return not self.is_terminal()
    
    def requires_input(self) -> bool:
        """Check if task requires user input"""
        return self.status.state == TaskState.INPUT_REQUIRED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "id": self.id,
            "status": self.status.to_dict(),
            "kind": "task"
        }
        
        if self.context_id:
            result["contextId"] = self.context_id
        
        if self.history:
            result["history"] = [msg.to_dict() for msg in self.history]
        
        if self.artifacts:
            result["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create Task from dictionary"""
        from .message import Message
        from .artifact import Artifact
        
        # Parse status
        status = TaskStatus.from_dict(data["status"])
        
        # Parse history
        history = []
        if "history" in data:
            history = [Message.from_dict(msg_data) for msg_data in data["history"]]
        
        # Parse artifacts
        artifacts = []
        if "artifacts" in data:
            artifacts = [Artifact.from_dict(art_data) for art_data in data["artifacts"]]
        
        return cls(
            id=data["id"],
            status=status,
            context_id=data.get("contextId"),
            history=history,
            artifacts=artifacts,
            metadata=data.get("metadata", {})
        )


@dataclass
class TaskStatusUpdateEvent:
    """Event for task status updates in streaming"""
    task_id: str
    context_id: Optional[str]
    status: TaskStatus
    final: bool = False
    kind: str = "status-update"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "taskId": self.task_id,
            "status": self.status.to_dict(),
            "final": self.final,
            "kind": self.kind
        }
        
        if self.context_id:
            result["contextId"] = self.context_id
        
        return result


@dataclass
class TaskArtifactUpdateEvent:
    """Event for task artifact updates in streaming"""
    task_id: str
    context_id: Optional[str]
    artifact: Artifact
    append: bool = False
    last_chunk: bool = False
    kind: str = "artifact-update"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "taskId": self.task_id,
            "artifact": self.artifact.to_dict(),
            "append": self.append,
            "lastChunk": self.last_chunk,
            "kind": self.kind
        }
        
        if self.context_id:
            result["contextId"] = self.context_id
        
        return result


class TaskManager:
    """Task management utility for A2A servers"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.context_tasks: Dict[str, List[str]] = {}  # context_id -> task_ids
    
    def create_task(self, initial_message: Optional[Message] = None, 
                   context_id: Optional[str] = None) -> Task:
        """Create a new task"""
        task = Task(
            id=str(uuid.uuid4()),
            status=TaskStatus(state=TaskState.SUBMITTED),
            context_id=context_id or str(uuid.uuid4())
        )
        
        if initial_message:
            task.add_message(initial_message)
        
        self.tasks[task.id] = task
        
        # Track by context
        if task.context_id not in self.context_tasks:
            self.context_tasks[task.context_id] = []
        self.context_tasks[task.context_id].append(task.id)
        
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_tasks_by_context(self, context_id: str) -> List[Task]:
        """Get all tasks for a context"""
        task_ids = self.context_tasks.get(context_id, [])
        return [self.tasks[task_id] for task_id in task_ids if task_id in self.tasks]
    
    def update_task_status(self, task_id: str, state: TaskState, 
                          message: Optional[Message] = None, 
                          error: Optional[str] = None,
                          progress: Optional[float] = None) -> bool:
        """Update task status"""
        task = self.get_task(task_id)
        if not task:
            return False
        
        task.update_status(state, message, error, progress)
        return True
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if possible"""
        task = self.get_task(task_id)
        if not task or task.is_terminal():
            return False
        
        task.update_status(TaskState.CANCELED)
        return True
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """Clean up old completed tasks"""
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        removed_count = 0
        
        for task_id in list(self.tasks.keys()):
            task = self.tasks[task_id]
            if (task.is_terminal() and 
                task.updated_at.timestamp() < cutoff):
                
                # Remove from context tracking
                if task.context_id in self.context_tasks:
                    if task_id in self.context_tasks[task.context_id]:
                        self.context_tasks[task.context_id].remove(task_id)
                    
                    # Clean up empty contexts
                    if not self.context_tasks[task.context_id]:
                        del self.context_tasks[task.context_id]
                
                # Remove task
                del self.tasks[task_id]
                removed_count += 1
        
        return removed_count
