"""
Integration Data Models

Data structures and models for the integration and orchestration system.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ExecutionMode(Enum):
    """Workflow execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"
    EVENT_DRIVEN = "event_driven"


class Priority(Enum):
    """Priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ResourceType(Enum):
    """Resource types for workflow execution"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    CUSTOM = "custom"


@dataclass
class ResourceRequirement:
    """Resource requirement specification"""
    resource_type: ResourceType
    amount: float
    unit: str = ""
    max_amount: Optional[float] = None
    priority: Priority = Priority.NORMAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'resource_type': self.resource_type.value,
            'amount': self.amount,
            'unit': self.unit,
            'max_amount': self.max_amount,
            'priority': self.priority.value
        }


@dataclass
class StepCondition:
    """Condition for conditional workflow execution"""
    condition_type: str  # 'success', 'failure', 'value_equals', 'value_greater', etc.
    target_step: str
    condition_value: Any = None
    operator: str = "equals"  # equals, not_equals, greater, less, contains, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'condition_type': self.condition_type,
            'target_step': self.target_step,
            'condition_value': self.condition_value,
            'operator': self.operator
        }


@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    step_id: str
    name: str
    component_type: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies and conditions
    dependencies: List[str] = field(default_factory=list)
    conditions: List[StepCondition] = field(default_factory=list)
    
    # Execution properties
    required: bool = True
    timeout_seconds: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    
    # Resource requirements
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    
    # Status and results
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'name': self.name,
            'component_type': self.component_type,
            'action': self.action,
            'parameters': self.parameters,
            'dependencies': self.dependencies,
            'conditions': [c.to_dict() for c in self.conditions],
            'required': self.required,
            'timeout_seconds': self.timeout_seconds,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'resource_requirements': [r.to_dict() for r in self.resource_requirements],
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'result': self.result,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'tags': self.tags
        }


@dataclass
class WorkflowDefinition:
    """Definition of a workflow"""
    name: str
    description: str = ""
    version: str = "1.0"
    
    # Workflow structure
    steps: List[Union[WorkflowStep, Dict[str, Any]]] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    
    # Configuration
    timeout_seconds: float = 3600.0
    max_parallel_steps: int = 5
    retry_failed_workflow: bool = False
    
    # Resource management
    total_resource_limits: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: Union[WorkflowStep, Dict[str, Any]]):
        """Add a step to the workflow"""
        self.steps.append(step)
    
    def get_step(self, step_id: str) -> Optional[Union[WorkflowStep, Dict[str, Any]]]:
        """Get a step by ID"""
        for step in self.steps:
            if isinstance(step, WorkflowStep):
                if step.step_id == step_id:
                    return step
            elif isinstance(step, dict):
                if step.get('step_id') == step_id:
                    return step
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        steps_data = []
        for step in self.steps:
            if isinstance(step, WorkflowStep):
                steps_data.append(step.to_dict())
            else:
                steps_data.append(step)
        
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'steps': steps_data,
            'execution_mode': self.execution_mode.value,
            'timeout_seconds': self.timeout_seconds,
            'max_parallel_steps': self.max_parallel_steps,
            'retry_failed_workflow': self.retry_failed_workflow,
            'total_resource_limits': self.total_resource_limits,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class ExecutionContext:
    """Context for workflow execution"""
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_name: str = ""
    
    # Execution environment
    environment: str = "default"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Input data and parameters
    input_data: Dict[str, Any] = field(default_factory=dict)
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution state
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    
    # Resource allocation
    allocated_resources: Dict[str, float] = field(default_factory=dict)
    resource_limits: Dict[str, float] = field(default_factory=dict)
    
    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    estimated_completion_time: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_id': self.execution_id,
            'workflow_name': self.workflow_name,
            'environment': self.environment,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'input_data': self.input_data,
            'global_parameters': self.global_parameters,
            'current_step': self.current_step,
            'completed_steps': self.completed_steps,
            'failed_steps': self.failed_steps,
            'allocated_resources': self.allocated_resources,
            'resource_limits': self.resource_limits,
            'start_time': self.start_time.isoformat(),
            'estimated_completion_time': self.estimated_completion_time.isoformat() if self.estimated_completion_time else None,
            'metadata': self.metadata
        }


@dataclass
class IntegrationResult:
    """Result of integration operation"""
    execution_id: str
    workflow_name: str
    success: bool
    
    # Results and data
    results: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    execution_time_seconds: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Error handling
    error_message: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    # Step-level results
    step_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    failed_steps: List[str] = field(default_factory=list)
    
    # Metadata
    completed_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution_id': self.execution_id,
            'workflow_name': self.workflow_name,
            'success': self.success,
            'results': self.results,
            'output_data': self.output_data,
            'execution_time_seconds': self.execution_time_seconds,
            'resource_usage': self.resource_usage,
            'error_message': self.error_message,
            'error_details': self.error_details,
            'warnings': self.warnings,
            'step_results': self.step_results,
            'failed_steps': self.failed_steps,
            'completed_at': self.completed_at.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class IntegrationConfig:
    """Configuration for integration system"""
    # System configuration
    max_concurrent_workflows: int = 10
    default_timeout_seconds: float = 3600.0
    health_check_interval_seconds: float = 30.0
    
    # Resource management
    default_resource_limits: Dict[str, float] = field(default_factory=lambda: {
        'cpu': 80.0,  # percentage
        'memory': 8192.0,  # MB
        'storage': 10240.0,  # MB
        'network': 1000.0  # Mbps
    })
    
    # Retry and error handling
    default_max_retries: int = 3
    retry_delay_seconds: float = 5.0
    enable_automatic_recovery: bool = True
    
    # Monitoring and logging
    enable_detailed_logging: bool = True
    enable_performance_monitoring: bool = True
    enable_resource_monitoring: bool = True
    
    # Security
    enable_authentication: bool = False
    enable_authorization: bool = False
    enable_encryption: bool = False
    
    # Feature flags
    enable_experimental_features: bool = False
    enable_advanced_optimization: bool = True
    enable_predictive_scaling: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'max_concurrent_workflows': self.max_concurrent_workflows,
            'default_timeout_seconds': self.default_timeout_seconds,
            'health_check_interval_seconds': self.health_check_interval_seconds,
            'default_resource_limits': self.default_resource_limits,
            'default_max_retries': self.default_max_retries,
            'retry_delay_seconds': self.retry_delay_seconds,
            'enable_automatic_recovery': self.enable_automatic_recovery,
            'enable_detailed_logging': self.enable_detailed_logging,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_resource_monitoring': self.enable_resource_monitoring,
            'enable_authentication': self.enable_authentication,
            'enable_authorization': self.enable_authorization,
            'enable_encryption': self.enable_encryption,
            'enable_experimental_features': self.enable_experimental_features,
            'enable_advanced_optimization': self.enable_advanced_optimization,
            'enable_predictive_scaling': self.enable_predictive_scaling
        }


@dataclass
class ComponentMetrics:
    """Metrics for integrated components"""
    component_id: str
    component_name: str
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    
    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    storage_usage_mb: float = 0.0
    network_usage_mbps: float = 0.0
    
    # Availability metrics
    uptime_seconds: float = 0.0
    downtime_seconds: float = 0.0
    availability_percent: float = 100.0
    
    # Usage statistics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.utcnow)
    measurement_period_seconds: float = 3600.0  # 1 hour
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'component_name': self.component_name,
            'avg_response_time_ms': self.avg_response_time_ms,
            'throughput_per_second': self.throughput_per_second,
            'success_rate': self.success_rate,
            'error_rate': self.error_rate,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'storage_usage_mb': self.storage_usage_mb,
            'network_usage_mbps': self.network_usage_mbps,
            'uptime_seconds': self.uptime_seconds,
            'downtime_seconds': self.downtime_seconds,
            'availability_percent': self.availability_percent,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'last_updated': self.last_updated.isoformat(),
            'measurement_period_seconds': self.measurement_period_seconds
        }


# Utility functions for model validation
def validate_workflow_definition(workflow: WorkflowDefinition) -> bool:
    """Validate workflow definition structure"""
    if not workflow.name or not isinstance(workflow.name, str):
        return False
    
    if not workflow.steps:
        return False
    
    # Validate step IDs are unique
    step_ids = []
    for step in workflow.steps:
        if isinstance(step, WorkflowStep):
            step_id = step.step_id
        elif isinstance(step, dict):
            step_id = step.get('step_id')
        else:
            return False
        
        if not step_id or step_id in step_ids:
            return False
        step_ids.append(step_id)
    
    return True


def validate_execution_context(context: ExecutionContext) -> bool:
    """Validate execution context structure"""
    if not context.execution_id or not isinstance(context.execution_id, str):
        return False
    
    if context.resource_limits:
        for resource_type, limit in context.resource_limits.items():
            if not isinstance(limit, (int, float)) or limit < 0:
                return False
    
    return True


def validate_integration_result(result: IntegrationResult) -> bool:
    """Validate integration result structure"""
    if not result.execution_id or not isinstance(result.execution_id, str):
        return False
    
    if not isinstance(result.success, bool):
        return False
    
    if result.execution_time_seconds < 0:
        return False
    
    return True
