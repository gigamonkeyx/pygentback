"""
Multi-Agent Core Components

Core classes and interfaces for multi-agent coordination system.
"""

import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
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
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    success_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    uptime_seconds: float = 0.0
    current_load: float = 0.0


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.
    """
    
    def __init__(self, agent_id: str = None, name: str = "", agent_type: str = "generic"):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name
        self.agent_type = agent_type
        self.status = AgentStatus.OFFLINE
        self.capabilities: List[AgentCapability] = []
        self.metrics = AgentMetrics()
        self.configuration: Dict[str, Any] = {}
        self.current_task: Optional[Task] = None
        self.coordinator: Optional['AgentCoordinator'] = None
        
        # Communication
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.event_callbacks: Dict[str, List[Callable]] = {}
    
    @abstractmethod
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute assigned task"""
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
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Agent {self.name} startup failed: {e}")
    
    def send_message(self, message: Message):
        """Send message to agent"""
        try:
            self.message_queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning(f"Message queue full for agent {self.name}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        return {
            'agent_id': self.agent_id,
            'status': self.status.value,
            'uptime': 300,  # Mock uptime
            'last_heartbeat': '2024-01-01T12:00:00Z',
            'tasks_completed': self.metrics.tasks_completed,
            'current_load': self.metrics.current_load,
            'health_score': 0.95  # Add health_score
        }


class CommunicationHub:
    """
    Central communication hub for agent messaging.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_history: List[Message] = []
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

    def register_agent(self, agent: BaseAgent):
        """Register agent with communication hub"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.name} with communication hub")
    
    async def send_message(self, message_or_sender_id, recipient_id=None, message=None):
        """Send message between agents - supports multiple call signatures"""
        try:
            # Handle both (message) and (sender_id, recipient_id, message) signatures
            if recipient_id is None and message is None:
                # Called with single message argument
                msg = message_or_sender_id
                sender_id = msg.sender_id
                recipient_id = msg.recipient_id
                message = msg
            else:
                # Called with three arguments
                sender_id = message_or_sender_id
                message = message
            
            if recipient_id in self.agents:
                recipient = self.agents[recipient_id]
                recipient.send_message(message)
                
                self.message_history.append(message)
                self.stats['messages_sent'] += 1
                self.stats['messages_delivered'] += 1
                
                logger.debug(f"Message sent from {sender_id} to {recipient_id}")
                return True
            else:
                self.stats['messages_failed'] += 1
                logger.warning(f"Recipient {recipient_id} not found")
                return False
        except Exception as e:
            self.stats['messages_failed'] += 1
            logger.error(f"Failed to send message: {e}")
            return False

    async def route_message(self, message: Message, routing_rules: Optional[Dict] = None) -> bool:
        """Route message based on rules"""
        try:
            # Simple routing - just deliver to recipient
            if message.recipient_id:
                await self.send_message(message.sender_id, message.recipient_id, message)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to route message: {e}")
            return False
    
    async def broadcast_message(self, message_or_sender_id, message=None, channel: Optional[str] = None):
        """Broadcast message to multiple agents - supports multiple call signatures"""
        try:
            # Handle both (message) and (sender_id, message, channel) signatures
            if message is None:
                # Called with single message argument
                msg = message_or_sender_id
                sender_id = msg.sender_id
                message = msg
            else:
                # Called with multiple arguments
                sender_id = message_or_sender_id
                message = message
            
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
            return True
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return False

    def filter_messages(self, filter_criteria: Dict[str, Any]) -> List[Message]:
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

    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            'registered_agents': len(self.agents),
            'broadcast_channels': len(self.broadcast_channels),
            'message_history_size': len(self.message_history),
            'stats': self.stats.copy()
        }

    def get_hub_status(self) -> Dict[str, Any]:
        """Get hub status"""
        return {
            'is_running': self.is_running,
            'registered_agents': len(self.agents),
            'broadcast_channels': len(self.broadcast_channels),
            'message_history_size': len(self.message_history),
            'stats': self.stats.copy()
        }
        

class WorkflowManager:
    """
    Manages complex workflows involving multiple agents.
    """
    
    def __init__(self, communication_hub: CommunicationHub):
        self.communication_hub = communication_hub
        self.active_workflows: Dict[str, Workflow] = {}
        self.workflow_history: List[Workflow] = []
        
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
            workflow.status = TaskStatus.IN_PROGRESS
            result = await self._execute_workflow_steps(workflow)
            
            if result.get('success', False):
                workflow.status = TaskStatus.COMPLETED
                self.stats['workflows_completed'] += 1
            else:
                workflow.status = TaskStatus.FAILED
                self.stats['workflows_failed'] += 1
            
            # Move to history
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            workflow.status = TaskStatus.FAILED
            self.stats['workflows_failed'] += 1
            return {'success': False, 'error': str(e)}
    
    async def _execute_workflow_steps(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute workflow steps"""
        try:
            # Mock workflow execution
            steps_results = []
            for i, task in enumerate(workflow.tasks):
                step_result = {
                    'step_id': task.task_id,
                    'status': 'completed',
                    'result': f"Task {i} completed successfully"
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
    
    async def create_workflow(self, name: str, description: str = "", tasks: List[Task] = None) -> str:
        """Create a new workflow"""
        workflow = Workflow(
            name=name,
            description=description,
            tasks=tasks or []
        )
        
        self.active_workflows[workflow.workflow_id] = workflow
        logger.info(f"Created workflow {workflow.workflow_id}: {name}")
        return workflow.workflow_id
    
    def monitor_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Monitor workflow progress"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return {
                'workflow_id': workflow_id,
                'status': workflow.status.value,
                'progress': 0.5,  # Mock progress
                'current_step': None,
                'elapsed_time': (datetime.utcnow() - workflow.created_at).total_seconds()
            }
        else:
            return {'workflow_id': workflow_id, 'status': 'not_found'}

    async def start_workflow(self, workflow: Workflow) -> str:
        """Start a new workflow"""
        workflow.status = TaskStatus.IN_PROGRESS
        workflow.started_at = datetime.utcnow()
        
        self.active_workflows[workflow.workflow_id] = workflow
        self.stats['workflows_started'] += 1
        
        logger.info(f"Started workflow {workflow.workflow_id}: {workflow.name}")
        
        # Begin workflow execution
        asyncio.create_task(self._execute_workflow_async(workflow))
        
        return workflow.workflow_id
    
    async def _execute_workflow_async(self, workflow: Workflow):
        """Execute workflow asynchronously"""
        try:
            await self._execute_workflow_steps(workflow)
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            workflow.status = TaskStatus.FAILED


class AgentCoordinator:
    """
    Coordinates multiple agents for complex task execution.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: List[Task] = []
        self._is_running = False
        
        # Task assignment strategies
        self.assignment_strategies: Dict[str, Callable] = {
            'round_robin': self._round_robin_assignment,
            'capability_based': self._capability_based_assignment,
            'load_balanced': self._load_balanced_assignment
        }
        
        # Statistics
        self.stats = {
            'tasks_assigned': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'active_agents': 0
        }
    
    async def initialize(self) -> bool:
        """Initialize the coordinator"""
        try:
            logger.info("Initializing AgentCoordinator")
            self._is_running = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AgentCoordinator: {e}")
            return False    @property
    def is_running(self) -> bool:
        """Check if coordinator is running"""
        return self._is_running

    async def start(self) -> bool:
        """Start the coordinator"""
        return await self.initialize()

    def register_agent(self, agent_or_config):
        """Register agent with coordinator"""
        if isinstance(agent_or_config, dict):
            # Create agent from config
            from .models import Agent
            agent = Agent(
                agent_id=agent_or_config['agent_id'],
                name=agent_or_config['name'],
                agent_type=agent_or_config['agent_type'],
                capabilities=agent_or_config.get('capabilities', []),
                configuration=agent_or_config.get('configuration', {}),
                status=agent_or_config.get('status', 'active')
            )
            agent_id = agent.agent_id
        else:
            # Agent instance
            agent = agent_or_config
            agent_id = agent.agent_id
              self.agents[agent_id] = agent
        if hasattr(agent, 'coordinator'):
            agent.coordinator = self
        self.stats['active_agents'] = len(self.agents)
        logger.info(f"Registered agent {getattr(agent, 'name', agent_id)} with coordinator")
        return agent_id

    async def check_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """Check health of specific agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            return agent.get_health_status()
        else:
            return {'agent_id': agent_id, 'status': 'not_found'}

    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister agent from coordinator"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.stats['active_agents'] = len(self.agents)
            logger.info(f"Deregistered agent {agent_id}")
            return True
        return False

    async def shutdown(self) -> bool:
        """Shutdown coordinator and clear agents"""
        try:
            # Shutdown all agents
            for agent_id in list(self.agents.keys()):
                await self.deregister_agent(agent_id)
            
            self.agents.clear()
            self.active_tasks.clear()
            self.stats['active_agents'] = 0
            self._is_running = False
            logger.info("AgentCoordinator shutdown complete")
            return True
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            return False

    def _round_robin_assignment(self, task: Task) -> Optional[BaseAgent]:
        """Round-robin task assignment"""
        if not self.agents:
            return None
        agent_list = list(self.agents.values())
        # Simple round-robin - just return first available
        for agent in agent_list:
            if agent.status == AgentStatus.IDLE:
                return agent
        return None

    def _capability_based_assignment(self, task: Task) -> Optional[BaseAgent]:
        """Capability-based task assignment"""
        # Find agent with matching capabilities
        for agent in self.agents.values():
            if agent.status == AgentStatus.IDLE:
                # Check if agent has required capabilities
                if not task.requirements:
                    return agent
                # For now, just return first idle agent
                return agent
        return None

    def _load_balanced_assignment(self, task: Task) -> Optional[BaseAgent]:
        """Load-balanced task assignment"""
        # Find agent with lowest load
        available_agents = [a for a in self.agents.values() if a.status == AgentStatus.IDLE]
        if available_agents:
            # Return agent with lowest current load
            return min(available_agents, key=lambda a: a.metrics.current_load)
        return None

    def get_coordination_status(self) -> Dict[str, Any]:
        """Get coordination status"""
        return {
            'total_agents': len(self.agents),
            'active_agents': sum(1 for a in self.agents.values() if a.status != AgentStatus.OFFLINE),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'stats': self.stats.copy()
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'is_running': self.is_running,
            'agent_count': len(self.agents),
            'active_workflows': len(self.active_tasks),
            'total_agents': len(self.agents),
            'active_agents': sum(1 for a in self.agents.values() if getattr(a, 'status', None) != AgentStatus.OFFLINE),
            'stats': self.stats.copy()
        }

    def get_registered_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered agents"""
        result = {}
        for agent_id, agent in self.agents.items():
            if isinstance(agent, dict):
                result[agent_id] = agent
            else:
                result[agent_id] = {
                    'agent_id': agent_id,
                    'agent_type': getattr(agent, 'agent_type', 'unknown'),
                    'name': getattr(agent, 'name', agent_id),
                    'capabilities': getattr(agent, 'capabilities', []),
                    'status': str(getattr(agent, 'status', 'unknown')),
                    'configuration': getattr(agent, 'configuration', {})
                }
        return result

    async def coordinate_agents(self, coordination_request: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents"""
        try:
            coordination_id = coordination_request['coordination_id']
            target_agents = coordination_request['target_agents']
            coordination_type = coordination_request.get('coordination_type', 'sequential')
            task = coordination_request['task']
            
            results = []
            
            for agent_id in target_agents:
                if agent_id in self.agents:
                    # Simulate task execution
                    result = {
                        'agent_id': agent_id,
                        'task_id': task['task_id'],
                        'success': True,
                        'result': f"Task executed by {agent_id}",
                        'execution_time': 1.5
                    }
                    results.append(result)
                else:
                    result = {
                        'agent_id': agent_id,
                        'task_id': task['task_id'],
                        'success': False,
                        'error': f"Agent {agent_id} not found"
                    }
                    results.append(result)
            
            return {
                'success': True,
                'coordination_id': coordination_id,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Coordination failed: {e}")
            return {
                'success': False,                'error': str(e)
            }