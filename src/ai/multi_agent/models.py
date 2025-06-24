"""
Multi-Agent Data Models

Data structures and models for the multi-agent coordination system.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

# Import core message system to avoid duplication
from core.agent import AgentMessage, MessageType


class Priority(Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ResourceType(Enum):
    """Resource types for agent requirements"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    CUSTOM = "custom"


class CoordinationMode(Enum):
    """Coordination modes for multi-agent workflows"""
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    PIPELINE = "pipeline"
    CONSENSUS = "consensus"
    COMPETITIVE = "competitive"
    COLLABORATIVE = "collaborative"


@dataclass
class Resource:
    """Resource requirement or allocation"""
    resource_type: ResourceType
    amount: float
    unit: str = ""
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'resource_type': self.resource_type.value,
            'amount': self.amount,
            'unit': self.unit,
            'description': self.description
        }


@dataclass
class TaskConstraint:
    """Constraint for task execution"""
    constraint_type: str
    value: Any
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'constraint_type': self.constraint_type,
            'value': self.value,
            'description': self.description
        }


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    success: bool
    result_data: Any = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'success': self.success,
            'result_data': self.result_data,
            'error_message': self.error_message,
            'execution_time_ms': self.execution_time_ms,
            'resource_usage': self.resource_usage,
            'metadata': self.metadata,
            'completed_at': self.completed_at.isoformat()
        }


@dataclass
class Task:
    """Task definition for agent execution"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    task_type: str = "generic"
    
    # Task data
    input_data: Any = None
    expected_output_type: str = "any"
    
    # Requirements
    required_capabilities: List[str] = field(default_factory=list)
    resource_requirements: List[Resource] = field(default_factory=list)
    constraints: List[TaskConstraint] = field(default_factory=list)
    
    # Execution properties
    priority: Priority = Priority.NORMAL
    timeout_seconds: float = 300.0
    max_retries: int = 3
    retry_count: int = 0
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    dependent_tasks: List[str] = field(default_factory=list)  # Task IDs
    
    # Status tracking
    status: 'TaskStatus' = None  # Will be imported from core
    assigned_agent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional[TaskResult] = None
    error_message: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.status is None:
            from .core import TaskStatus
            self.status = TaskStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type,
            'input_data': self.input_data,
            'expected_output_type': self.expected_output_type,
            'required_capabilities': self.required_capabilities,
            'resource_requirements': [r.to_dict() for r in self.resource_requirements],
            'constraints': [c.to_dict() for c in self.constraints],
            'priority': self.priority.value,
            'timeout_seconds': self.timeout_seconds,
            'max_retries': self.max_retries,
            'retry_count': self.retry_count,
            'dependencies': self.dependencies,
            'dependent_tasks': self.dependent_tasks,
            'status': self.status.value if self.status else 'pending',
            'assigned_agent_id': self.assigned_agent_id,
            'created_at': self.created_at.isoformat(),
            'assigned_at': self.assigned_at.isoformat() if self.assigned_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result.to_dict() if self.result else None,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'tags': self.tags
        }


# Use core AgentMessage instead of duplicate Message class
# Create alias for backward compatibility
Message = AgentMessage


@dataclass
class Agent:
    """Agent definition and state"""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    agent_type: str = "generic"
    description: str = ""
    
    # Capabilities
    capabilities: List[str] = field(default_factory=list)
    supported_task_types: List[str] = field(default_factory=list)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Dict[str, float] = field(default_factory=dict)
    
    # Status
    status: 'AgentStatus' = None  # Will be imported from core
    current_task_id: Optional[str] = None
    
    # Performance metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_execution_time_ms: float = 0.0
    success_rate: float = 0.0
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.status is None:
            from .core import AgentStatus
            self.status = AgentStatus.INITIALIZING
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'agent_type': self.agent_type,
            'description': self.description,
            'capabilities': self.capabilities,
            'supported_task_types': self.supported_task_types,
            'config': self.config,
            'resource_limits': self.resource_limits,
            'status': self.status.value if self.status else 'initializing',
            'current_task_id': self.current_task_id,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'avg_execution_time_ms': self.avg_execution_time_ms,
            'success_rate': self.success_rate,
            'created_at': self.created_at.isoformat(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'metadata': self.metadata,
            'tags': self.tags
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get the health status of this agent"""
        # Calculate health score based on success rate, uptime, and status
        uptime_seconds = (datetime.utcnow() - self.created_at).total_seconds()
        status_score = 1.0 if self.status not in ['failed', 'error', 'offline'] else 0.0
        success_score = self.success_rate if hasattr(self, 'success_rate') and self.success_rate is not None else 1.0
        uptime_score = min(1.0, uptime_seconds / 3600.0)  # Max score after 1 hour
        
        health_score = (status_score * 0.5) + (success_score * 0.3) + (uptime_score * 0.2)
        
        return {
            'agent_id': self.agent_id,
            'status': self.status.value if hasattr(self.status, 'value') else str(self.status),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'success_rate': self.success_rate,
            'current_task_id': self.current_task_id,
            'uptime_seconds': uptime_seconds,
            'health_score': health_score,
            'is_healthy': self.status not in ['failed', 'error', 'offline'] if hasattr(self.status, 'lower') else True
        }


@dataclass
class Workflow:
    """Workflow definition for coordinated task execution"""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    workflow_type: str = "generic"
    
    # Tasks
    tasks: List[Task] = field(default_factory=list)
    
    # Execution strategy
    execution_strategy: str = "sequential"  # sequential, parallel, pipeline, custom
    coordination_mode: CoordinationMode = CoordinationMode.HIERARCHICAL
    
    # Configuration
    max_parallel_tasks: int = 5
    timeout_seconds: float = 3600.0  # 1 hour default
    retry_failed_tasks: bool = True
    
    # Status
    status: 'TaskStatus' = None  # Will be imported from core
    progress_percentage: float = 0.0
    
    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    results: Dict[str, TaskResult] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.status is None:
            from .core import TaskStatus
            self.status = TaskStatus.PENDING
    
    def add_task(self, task: Task):
        """Add task to workflow"""
        self.tasks.append(task)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_completed_tasks(self) -> List[Task]:
        """Get completed tasks"""
        from .core import TaskStatus
        return [task for task in self.tasks if task.status == TaskStatus.COMPLETED]
    
    def get_failed_tasks(self) -> List[Task]:
        """Get failed tasks"""
        from .core import TaskStatus
        return [task for task in self.tasks if task.status == TaskStatus.FAILED]
    
    def calculate_progress(self) -> float:
        """Calculate workflow progress percentage"""
        if not self.tasks:
            return 0.0
        
        completed_count = len(self.get_completed_tasks())
        return (completed_count / len(self.tasks)) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'workflow_id': self.workflow_id,
            'name': self.name,
            'description': self.description,
            'workflow_type': self.workflow_type,
            'tasks': [task.to_dict() for task in self.tasks],
            'execution_strategy': self.execution_strategy,
            'coordination_mode': self.coordination_mode.value,
            'max_parallel_tasks': self.max_parallel_tasks,
            'timeout_seconds': self.timeout_seconds,
            'retry_failed_tasks': self.retry_failed_tasks,
            'status': self.status.value if self.status else 'pending',
            'progress_percentage': self.progress_percentage,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'error_message': self.error_message,
            'metadata': self.metadata,
            'tags': self.tags
        }


@dataclass
class CoordinationResult:
    """Result of coordination operation"""
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: str = ""
    success: bool = False
    
    # Results
    result_data: Any = None
    affected_agents: List[str] = field(default_factory=list)
    affected_tasks: List[str] = field(default_factory=list)
    
    # Performance
    execution_time_ms: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Status
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_id': self.operation_id,
            'operation_type': self.operation_type,
            'success': self.success,
            'result_data': self.result_data,
            'affected_agents': self.affected_agents,
            'affected_tasks': self.affected_tasks,
            'execution_time_ms': self.execution_time_ms,
            'resource_usage': self.resource_usage,
            'error_message': self.error_message,
            'warnings': self.warnings,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


# Utility functions for model validation
def validate_task(task: Task) -> bool:
    """Validate task structure"""
    if not task.task_id or not isinstance(task.task_id, str):
        return False
    
    if not task.name or not isinstance(task.name, str):
        return False
    
    if task.timeout_seconds <= 0:
        return False
    
    if task.max_retries < 0:
        return False
    
    return True


def validate_workflow(workflow: Workflow) -> bool:
    """Validate workflow structure"""
    if not workflow.workflow_id or not isinstance(workflow.workflow_id, str):
        return False
    
    if not workflow.name or not isinstance(workflow.name, str):
        return False
    
    if workflow.timeout_seconds <= 0:
        return False
    
    if workflow.max_parallel_tasks <= 0:
        return False
    
    # Validate all tasks
    for task in workflow.tasks:
        if not validate_task(task):
            return False
    
    return True


def validate_message(message: AgentMessage) -> bool:
    """Validate message structure"""
    if not message.id or not isinstance(message.id, str):
        return False

    if not message.sender or not isinstance(message.sender, str):
        return False

    if not message.recipient or not isinstance(message.recipient, str):
        return False

    # AgentMessage uses different structure, so basic validation
    return True
