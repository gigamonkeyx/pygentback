"""
Multi-Agent Core Components

Core classes and interfaces for multi-agent coordination system.
"""

import logging
import asyncio
import uuid
import time
from typing import List, Dict, Any, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent status states"""
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_DEPENDENCIES = "waiting_dependencies"


class MessageType(Enum):
    """Message types for agent communication"""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    STATUS_UPDATE = "status_update"
    COORDINATION_REQUEST = "coordination_request"
    RESOURCE_REQUEST = "resource_request"
    ERROR_NOTIFICATION = "error_notification"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"


@dataclass
class Message:
    """Message for agent communication"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: MessageType = MessageType.STATUS_UPDATE
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 0


@dataclass
class Task:
    """Task definition for agent execution"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    task_type: str = "generic"
    parameters: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    agent_id: str
    success: bool
    result_data: Any = None
    error_message: Optional[str] = None
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Workflow:
    """Workflow definition"""
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tasks: List[Task] = field(default_factory=list)
    execution_strategy: str = "sequential"  # sequential, parallel, pipeline, custom
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_execution_time_ms: float = 0.0
    success_rate: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    """
    
    def __init__(self, agent_id: str, name: str, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.status = AgentStatus.INITIALIZING
        self.metrics = AgentMetrics()
        
        # Communication
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.coordinator: Optional['AgentCoordinator'] = None
        
        # Task management
        self.current_task: Optional['Task'] = None
        self.task_history: List['Task'] = []
        
        # Configuration
        self.config: Dict[str, Any] = {}
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {}
        
        # Initialization timestamp
        self.created_at = datetime.utcnow()
    
    @abstractmethod
    async def execute_task(self, task: 'Task') -> 'TaskResult':
        """Execute a task and return result"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the agent gracefully"""
        pass
    
    async def start(self):
        """Start the agent"""
        try:
            success = await self.initialize()
            if success:
                self.status = AgentStatus.IDLE
                logger.info(f"Agent {self.name} started successfully")
                await self._start_message_loop()
            else:
                self.status = AgentStatus.ERROR
                logger.error(f"Agent {self.name} failed to initialize")
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Agent {self.name} startup failed: {e}")
    
    async def _start_message_loop(self):
        """Start the message processing loop"""
        while self.status != AgentStatus.OFFLINE:
            try:
                # Process messages with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
                await self._process_message(message)
            except asyncio.TimeoutError:
                # Periodic maintenance
                await self._periodic_maintenance()
            except Exception as e:
                logger.error(f"Agent {self.name} message loop error: {e}")
    
    async def _process_message(self, message: 'Message'):
        """Process incoming message"""
        try:
            if message.message_type == MessageType.TASK_ASSIGNMENT:
                await self._handle_task_assignment(message)
            elif message.message_type == MessageType.STATUS_UPDATE:
                await self._handle_status_update(message)
            elif message.message_type == MessageType.SHUTDOWN:
                await self._handle_shutdown(message)
            else:
                await self._handle_custom_message(message)
        except Exception as e:
            logger.error(f"Agent {self.name} failed to process message: {e}")
    
    async def _handle_task_assignment(self, message: 'Message'):
        """Handle task assignment message"""
        task = message.payload.get('task')
        if task and self.status == AgentStatus.IDLE:
            self.current_task = task
            self.status = AgentStatus.BUSY
            
            try:
                result = await self.execute_task(task)
                await self._send_task_result(result)
                self.metrics.tasks_completed += 1
            except Exception as e:
                await self._send_task_error(task, str(e))
                self.metrics.tasks_failed += 1
            finally:
                self.current_task = None
                self.status = AgentStatus.IDLE
    
    async def _handle_status_update(self, message: 'Message'):
        """Handle status update message"""
        # Update metrics or configuration based on message
        pass
    
    async def _handle_shutdown(self, message: 'Message'):
        """Handle shutdown message"""
        await self.shutdown()
        self.status = AgentStatus.OFFLINE
    
    async def _handle_custom_message(self, message: 'Message'):
        """Handle custom message types - override in subclasses"""
        pass
    
    async def _periodic_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Update metrics
        self.metrics.last_activity = datetime.utcnow()
        self.metrics.uptime_seconds = (datetime.utcnow() - self.created_at).total_seconds()
        
        # Calculate success rate
        total_tasks = self.metrics.tasks_completed + self.metrics.tasks_failed
        if total_tasks > 0:
            self.metrics.success_rate = self.metrics.tasks_completed / total_tasks
    
    async def _send_task_result(self, result: 'TaskResult'):
        """Send task result to coordinator"""
        if self.coordinator:
            await self.coordinator.receive_task_result(self.agent_id, result)
    
    async def _send_task_error(self, task: 'Task', error: str):
        """Send task error to coordinator"""
        if self.coordinator:
            await self.coordinator.receive_task_error(self.agent_id, task, error)
    
    def send_message(self, message: 'Message'):
        """Send message to agent"""
        try:
            self.message_queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning(f"Message queue full for agent {self.name}")
    
    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has specific capability"""
        return any(cap.name == capability_name for cap in self.capabilities)
    
    def get_capability(self, capability_name: str) -> Optional[AgentCapability]:
        """Get specific capability"""
        for cap in self.capabilities:
            if cap.name == capability_name:
                return cap
        return None
    
    def add_event_callback(self, event_type: str, callback: Callable):
        """Add event callback"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    async def emit_event(self, event_type: str, data: Any):
        """Emit event to callbacks"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Event callback error: {e}")


class CommunicationHub:
    """
    Central communication hub for agent messaging.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_history: List['Message'] = []
        self.broadcast_channels: Dict[str, Set[str]] = {}
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'broadcasts_sent': 0
        }
    
    async def start(self) -> bool:
        """Start the communication hub"""
        try:
            logger.info("Starting CommunicationHub")
            return True
        except Exception as e:
            logger.error(f"Failed to start CommunicationHub: {e}")
            return False
    
    def register_agent(self, agent: BaseAgent):
        """Register agent with communication hub"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.name} with communication hub")
    
    def unregister_agent(self, agent_id: str):
        """Unregister agent from communication hub"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent {agent_id} from communication hub")
    
    async def send_message(self, sender_id: str, recipient_id: str, message: 'Message'):
        """Send message between agents"""
        try:
            if recipient_id in self.agents:
                recipient = self.agents[recipient_id]
                recipient.send_message(message)
                
                self.message_history.append(message)
                self.stats['messages_sent'] += 1
                self.stats['messages_delivered'] += 1
                
                logger.debug(f"Message sent from {sender_id} to {recipient_id}")
            else:
                self.stats['messages_failed'] += 1
                logger.warning(f"Recipient {recipient_id} not found")
        except Exception as e:
            self.stats['messages_failed'] += 1
            logger.error(f"Failed to send message: {e}")
    
    async def broadcast_message(self, sender_id: str, message: 'Message', 
                              channel: Optional[str] = None):
        """Broadcast message to multiple agents"""
        try:
            recipients = set()
            
            if channel and channel in self.broadcast_channels:
                recipients = self.broadcast_channels[channel]
            else:
                recipients = set(self.agents.keys())
            
            # Remove sender from recipients
            recipients.discard(sender_id)
            
            for recipient_id in recipients:
                await self.send_message(sender_id, recipient_id, message)
            
            self.stats['broadcasts_sent'] += 1
            logger.debug(f"Broadcast sent from {sender_id} to {len(recipients)} agents")
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
    
    def create_broadcast_channel(self, channel_name: str, agent_ids: List[str]):
        """Create broadcast channel for specific agents"""
        self.broadcast_channels[channel_name] = set(agent_ids)
        logger.info(f"Created broadcast channel {channel_name} with {len(agent_ids)} agents")
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            'registered_agents': len(self.agents),
            'broadcast_channels': len(self.broadcast_channels),
            'message_history_size': len(self.message_history),
            'stats': self.stats.copy()
        }
    
    @property
    def is_running(self) -> bool:
        """Check if hub is running"""
        return True
        
    async def connect_agent(self, agent_id: str) -> bool:
        """Connect agent to the hub"""
        try:
            if agent_id in self.agents:
                logger.info(f"Agent {agent_id} connected to hub")
                return True
            else:
                logger.warning(f"Agent {agent_id} not registered")
                return False
        except Exception as e:
            logger.error(f"Failed to connect agent {agent_id}: {e}")
            return False
    
    async def route_message(self, message: 'Message', routing_rules: Optional[Dict] = None) -> bool:
        """Route message based on rules"""
        try:
            # Simple routing - just deliver to recipient
            if hasattr(message, 'recipient_id') and message.recipient_id:
                await self.send_message(message.sender_id, message.recipient_id, message)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to route message: {e}")
            return False
    
    def filter_messages(self, filter_criteria: Dict[str, Any]) -> List['Message']:
        """Filter messages based on criteria"""
        try:
            filtered = []
            for msg in self.message_history:
                matches = True
                for key, value in filter_criteria.items():
                    if not hasattr(msg, key) or getattr(msg, key) != value:
                        matches = False
                        break
                if matches:
                    filtered.append(msg)
            return filtered
        except Exception as e:
            logger.error(f"Failed to filter messages: {e}")
            return []


class WorkflowManager:
    """
    Manages complex workflows involving multiple agents.
    """
    
    def __init__(self, communication_hub: CommunicationHub):
        self.communication_hub = communication_hub
        self.active_workflows: Dict[str, 'Workflow'] = {}
        self.workflow_history: List['Workflow'] = []
        
        # Workflow templates
        self.workflow_templates: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            'workflows_started': 0,
            'workflows_completed': 0,
            'workflows_failed': 0,
            'avg_workflow_duration_ms': 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the workflow manager"""
        try:
            logger.info("Initializing WorkflowManager")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WorkflowManager: {e}")
            return False
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get list of active workflows"""
        return [
            {
                'workflow_id': wf.workflow_id,
                'name': wf.name,
                'status': wf.status.value if hasattr(wf.status, 'value') else str(wf.status),
                'started_at': wf.started_at.isoformat() if wf.started_at else None,
                'progress': getattr(wf, 'progress', 0)
            }
            for wf in self.active_workflows.values()
        ]
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow by ID"""
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        try:
            # Start workflow execution
            workflow.status = 'in_progress'
            result = await self._execute_workflow_steps(workflow)
            
            if result.get('success', False):
                workflow.status = 'completed'
                self.stats['workflows_completed'] += 1
            else:
                workflow.status = 'failed'
                self.stats['workflows_failed'] += 1
            
            # Move to history
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            workflow.status = 'failed'
            self.stats['workflows_failed'] += 1
            return {'success': False, 'error': str(e)}
    
    async def _execute_workflow_steps(self, workflow) -> Dict[str, Any]:
        """Execute workflow steps"""
        try:
            # Mock workflow execution
            steps_results = []
            for i, step in enumerate(getattr(workflow, 'steps', [])):
                step_result = {
                    'step_id': step.get('id', f'step_{i}'),
                    'status': 'completed',
                    'result': f"Step {i} completed successfully"
                }
                steps_results.append(step_result)
            
            return {
                'success': True,
                'workflow_id': workflow.workflow_id,
                'steps_completed': len(steps_results),
                'results': steps_results
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_workflow(self, name: str, description: str = "", steps: List[Dict] = None) -> str:
        """Create a new workflow"""
        workflow_id = str(uuid.uuid4())
        workflow_data = {
            'workflow_id': workflow_id,
            'name': name,
            'description': description,
            'steps': steps or [],
            'status': 'created',
            'created_at': datetime.utcnow()
        }
        
        # Create a simple workflow object
        from types import SimpleNamespace
        workflow = SimpleNamespace(**workflow_data)
        
        self.active_workflows[workflow_id] = workflow
        logger.info(f"Created workflow {workflow_id}: {name}")
        return workflow_id
    
    def monitor_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Monitor workflow progress"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return {
                'workflow_id': workflow_id,
                'status': getattr(workflow, 'status', 'unknown'),
                'progress': getattr(workflow, 'progress', 0),
                'current_step': getattr(workflow, 'current_step', None),
                'elapsed_time': (datetime.utcnow() - workflow.created_at).total_seconds() if hasattr(workflow, 'created_at') else 0
            }
        else:
            return {'workflow_id': workflow_id, 'status': 'not_found'}

    async def start_workflow(self, workflow: 'Workflow') -> str:
        """Start a new workflow"""
        workflow_id = str(uuid.uuid4())
        workflow.workflow_id = workflow_id
        workflow.status = 'in_progress'
        workflow.started_at = datetime.utcnow()
        
        self.active_workflows[workflow_id] = workflow
        self.stats['workflows_started'] += 1
        
        logger.info(f"Started workflow {workflow_id}: {workflow.name}")
        
        # Begin workflow execution
        asyncio.create_task(self._execute_workflow(workflow))
        
        return workflow_id
    
    async def _execute_workflow(self, workflow: 'Workflow'):
        """Execute workflow tasks"""
        try:
            # Execute tasks based on workflow strategy
            if getattr(workflow, 'execution_strategy', 'sequential') == 'sequential':
                await self._execute_sequential(workflow)
            elif getattr(workflow, 'execution_strategy', 'sequential') == 'parallel':
                await self._execute_parallel(workflow)
            elif getattr(workflow, 'execution_strategy', 'sequential') == 'pipeline':
                await self._execute_pipeline(workflow)
            else:
                await self._execute_custom(workflow)
            
            # Mark workflow as completed
            workflow.status = TaskStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()
            self.stats['workflows_completed'] += 1
            
        except Exception as e:
            workflow.status = TaskStatus.FAILED
            workflow.error_message = str(e)
            self.stats['workflows_failed'] += 1
            logger.error(f"Workflow {workflow.workflow_id} failed: {e}")
        
        finally:
            # Move to history
            self.workflow_history.append(workflow)
            if workflow.workflow_id in self.active_workflows:
                del self.active_workflows[workflow.workflow_id]
    
    async def _execute_sequential(self, workflow: 'Workflow'):
        """Execute tasks sequentially"""
        for task in workflow.tasks:
            await self._execute_task(task, workflow)
            if task.status == TaskStatus.FAILED:
                raise Exception(f"Task {task.task_id} failed")
    
    async def _execute_parallel(self, workflow: 'Workflow'):
        """Execute tasks in parallel"""
        tasks = [self._execute_task(task, workflow) for task in workflow.tasks]
        await asyncio.gather(*tasks)
    
    async def _execute_pipeline(self, workflow: 'Workflow'):
        """Execute tasks in pipeline fashion"""
        # Sort tasks by dependencies
        sorted_tasks = self._sort_tasks_by_dependencies(workflow.tasks)
        
        for task in sorted_tasks:
            await self._execute_task(task, workflow)
            if task.status == TaskStatus.FAILED:
                raise Exception(f"Pipeline task {task.task_id} failed")
    
    async def _execute_custom(self, workflow: 'Workflow'):
        """Execute tasks with custom strategy"""
        # Default to sequential for custom strategies
        await self._execute_sequential(workflow)
    
    async def _execute_task(self, task: 'Task', workflow: 'Workflow'):
        """Execute a single task"""
        # Find suitable agent
        suitable_agents = self._find_suitable_agents(task)
        
        if not suitable_agents:
            task.status = TaskStatus.FAILED
            task.error_message = "No suitable agents found"
            return
        
        # Select best agent (simple selection for now)
        selected_agent = suitable_agents[0]
        
        # Assign task
        task.assigned_agent_id = selected_agent.agent_id
        task.status = TaskStatus.ASSIGNED
        
        # Send task to agent
        message = self._create_task_message(task)
        selected_agent.send_message(message)
        
        # Wait for completion (simplified)
        await self._wait_for_task_completion(task)
    
    def _find_suitable_agents(self, task: 'Task') -> List[BaseAgent]:
        """Find agents suitable for task"""
        suitable_agents = []
        
        for agent in self.communication_hub.agents.values():
            if (agent.status == AgentStatus.IDLE and 
                self._agent_can_handle_task(agent, task)):
                suitable_agents.append(agent)
        
        # Sort by performance metrics
        suitable_agents.sort(key=lambda a: a.metrics.success_rate, reverse=True)
        
        return suitable_agents
    
    def _agent_can_handle_task(self, agent: BaseAgent, task: 'Task') -> bool:
        """Check if agent can handle task"""
        required_capabilities = task.required_capabilities
        
        for req_cap in required_capabilities:
            if not agent.has_capability(req_cap):
                return False
        
        return True
    
    def _create_task_message(self, task: 'Task') -> 'Message':
        """Create task assignment message"""
        from .models import Message  # Import here to avoid circular imports
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TASK_ASSIGNMENT,
            sender_id="workflow_manager",
            recipient_id=task.assigned_agent_id,
            payload={'task': task},
            timestamp=datetime.utcnow()
        )
    
    async def _wait_for_task_completion(self, task: 'Task'):
        """Wait for task completion"""
        timeout = 30.0  # seconds
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                break
            await asyncio.sleep(0.1)

        if task.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
            task.status = TaskStatus.FAILED
            task.error_message = "Task execution timeout"
    
    def _sort_tasks_by_dependencies(self, tasks: List['Task']) -> List['Task']:
        """Sort tasks by dependencies (topological sort)"""
        # Simple implementation - in practice would use proper topological sort
        return sorted(tasks, key=lambda t: len(t.dependencies))
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            # Check history
            for hist_workflow in self.workflow_history:
                if hist_workflow.workflow_id == workflow_id:
                    workflow = hist_workflow
                    break
        
        if workflow:
            return {
                'workflow_id': workflow.workflow_id,
                'name': workflow.name,
                'status': workflow.status.value,
                'progress': self._calculate_workflow_progress(workflow),
                'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
                'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
                'task_count': len(workflow.tasks),
                'completed_tasks': len([t for t in workflow.tasks if t.status == TaskStatus.COMPLETED])
            }
        
        return None
    
    def _calculate_workflow_progress(self, workflow: 'Workflow') -> float:
        """Calculate workflow progress percentage"""
        if not workflow.tasks:
            return 0.0
        
        completed_tasks = len([t for t in workflow.tasks if t.status == TaskStatus.COMPLETED])
        return (completed_tasks / len(workflow.tasks)) * 100.0
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        return {
            'active_workflows': len(self.active_workflows),
            'total_workflows': len(self.workflow_history) + len(self.active_workflows),
            'stats': self.stats.copy()
        }
    
    async def create_workflow(self, workflow_def: Dict[str, Any]) -> str:
        """Create a new workflow from definition."""
        workflow_id = str(uuid.uuid4())
        
        # Mock workflow creation
        workflow = {
            "id": workflow_id,
            "name": workflow_def.get("name", "unnamed_workflow"),
            "status": "created",
            "steps": workflow_def.get("steps", []),
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.active_workflows[workflow_id] = workflow
        self.stats['workflows_started'] += 1
        
        logger.info(f"Created workflow {workflow_id}: {workflow['name']}")
        return workflow_id


class AgentCoordinator:
    """
    Main coordinator for the multi-agent system.
    """
    
    def __init__(self):
        self.communication_hub = CommunicationHub()
        self.workflow_manager = WorkflowManager(self.communication_hub)
        
        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        
        # Coordination strategies
        self.coordination_strategies: Dict[str, 'CoordinationStrategy'] = {}
        
        # System state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = {}
    
    async def start(self):
        """Start the coordination system"""
        self.is_running = True
        self.start_time = datetime.utcnow()
        logger.info("Agent coordination system started")
    
    async def shutdown(self):
        """Shutdown the coordination system"""
        self.is_running = False
        
        # Shutdown all agents
        for agent in self.agents.values():
            if hasattr(agent, 'shutdown'):
                await agent.shutdown()
        
        # Clear the agents dictionary
        self.agents.clear()
        
        logger.info("Agent coordination system shutdown")
    
    async def register_agent(self, agent):
        """Register agent with coordinator - REAL IMPLEMENTATION ONLY"""
        # Handle both dict config and actual agent objects
        if isinstance(agent, dict):
            # Create REAL agent from dict config using AgentFactory
            from core.agent_factory import AgentFactory

            agent_factory = AgentFactory()
            agent_type = agent.get('agent_type', 'general')
            agent_name = agent.get('name', f'{agent_type}_agent')
            capabilities = agent.get('capabilities', [])

            # Create real agent instance
            real_agent = await agent_factory.create_agent(
                agent_type=agent_type,
                name=agent_name,
                capabilities=capabilities
            )

            # Set coordinator and register
            real_agent.coordinator = self
            self.agents[real_agent.agent_id] = real_agent
            self.communication_hub.register_agent(real_agent)

            logger.info(f"Registered REAL agent {real_agent.name} with coordinator")
            return real_agent.agent_id
        else:
            # Handle actual agent objects
            agent.coordinator = self
            self.agents[agent.agent_id] = agent
            self.communication_hub.register_agent(agent)
            
            logger.info(f"Registered agent {agent.name} with coordinator")
            return agent.agent_id
    
    def unregister_agent(self, agent_id: str):
        """Unregister agent from coordinator"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.communication_hub.unregister_agent(agent_id)
            logger.info(f"Unregistered agent {agent_id} from coordinator")
    
    async def receive_task_result(self, agent_id: str, result: 'TaskResult'):
        """Receive task result from agent"""
        logger.debug(f"Received task result from agent {agent_id}")
        # Handle task result - update workflow, notify other agents, etc.
    
    async def receive_task_error(self, agent_id: str, task: 'Task', error: str):
        """Receive task error from agent"""
        logger.warning(f"Received task error from agent {agent_id}: {error}")
        # Handle task error - retry, reassign, fail workflow, etc.
    
    async def coordinate_agents(self, coordination_request: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents to execute a task"""
        coordination_id = coordination_request.get("coordination_id", str(uuid.uuid4()))
        target_agents = coordination_request.get("target_agents", [])
        coordination_type = coordination_request.get("coordination_type", "sequential")
        task = coordination_request.get("task", {})
        
        logger.info(f"Starting coordination {coordination_id} with {len(target_agents)} agents")
        
        results = []
        success = True
        
        # REAL coordination logic - execute actual tasks
        for agent_id in target_agents:
            if agent_id in self.agents:
                # Execute REAL task with agent
                agent = self.agents[agent_id]

                # Create real task object
                from ai.multi_agent.models import Task, TaskStatus
                real_task = Task(
                    task_id=task.get("task_id", str(uuid.uuid4())),
                    description=task.get("description", "Coordination task"),
                    status=TaskStatus.PENDING
                )

                # Execute real task
                start_time = time.time()
                try:
                    task_result = await agent.execute_task(real_task)
                    execution_time = time.time() - start_time

                    agent_result = {
                        "agent_id": agent_id,
                        "task_id": real_task.task_id,
                        "status": "completed",
                        "result": task_result.get('result', f"Task completed by {agent.name}"),
                        "execution_time": execution_time
                    }
                except Exception as e:
                    execution_time = time.time() - start_time
                    agent_result = {
                        "agent_id": agent_id,
                        "task_id": real_task.task_id,
                        "status": "failed",
                        "error": str(e),
                        "execution_time": execution_time
                    }
                    success = False

                results.append(agent_result)
            else:
                success = False
                results.append({
                    "agent_id": agent_id,
                    "task_id": task.get("task_id", "unknown"),
                    "status": "failed",
                    "error": f"Agent {agent_id} not found"
                })
        
        return {
            "success": success,
            "coordination_id": coordination_id,
            "coordination_type": coordination_type,
            "results": results,
            "total_agents": len(target_agents),
            "completed_tasks": len([r for r in results if r.get("status") == "completed"])
        }
    
    async def check_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """Check the health status of a specific agent."""
        if agent_id not in self.agents:
            return {"status": "not_found", "message": f"Agent {agent_id} not found"}
        
        # Mock health check implementation
        return {
            "agent_id": agent_id,
            "status": "healthy",
            "uptime": 300,  # seconds
            "last_heartbeat": "2024-01-01T12:00:00Z",
            "tasks_completed": 5,
            "current_load": 0.3
        }
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent from the coordinator."""
        if agent_id not in self.agents:
            return False
        
        # Remove agent from registry
        del self.agents[agent_id]
        logger.info(f"Agent {agent_id} deregistered successfully")
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = {
                'name': agent.name,
                'status': agent.status.value,
                'capabilities': [cap.name for cap in agent.capabilities],
                'metrics': agent.metrics.__dict__
            }
        
        workflow_stats = self.workflow_manager.get_workflow_stats()
        
        return {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'agent_count': len(self.agents),
            'agents': agent_statuses,
            'active_workflows': workflow_stats.get('active_workflows', 0),
            'communication_stats': self.communication_hub.get_communication_stats(),
            'workflow_stats': workflow_stats
        }
    
    def get_registered_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get dictionary of registered agents with their details"""
        agents_info = {}
        for agent_id, agent in self.agents.items():
            agents_info[agent_id] = {
                'agent_type': getattr(agent, 'agent_type', 'unknown'),
                'name': getattr(agent, 'name', 'unknown'),
                'status': getattr(agent, 'status', AgentStatus.OFFLINE).value if hasattr(getattr(agent, 'status', None), 'value') else str(getattr(agent, 'status', 'unknown')),
                'capabilities': getattr(agent, 'capabilities', [])
            }
        return agents_info
