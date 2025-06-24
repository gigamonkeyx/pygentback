#!/usr/bin/env python3
"""
Base Agent Architecture

Core agent architecture with lifecycle management, communication protocols,
and integration with PyGent Factory systems.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

try:
    from ..cache.cache_layers import cache_manager
except ImportError:
    cache_manager = None

try:
    from ..database.production_manager import db_manager
except ImportError:
    db_manager = None

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent status enumeration"""
    CREATED = "created"
    INITIALIZING = "initializing"
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentType(Enum):
    """Agent type enumeration"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    COORDINATION = "coordination"
    MONITORING = "monitoring"
    CUSTOM = "custom"


class MessageType(Enum):
    """Message type enumeration"""
    TASK = "task"
    RESPONSE = "response"
    STATUS = "status"
    ERROR = "error"
    COORDINATION = "coordination"
    BROADCAST = "broadcast"


@dataclass
class AgentMessage:
    """Agent communication message"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.TASK
    sender_id: str = ""
    recipient_id: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 1  # 1=low, 5=high
    requires_response: bool = False
    correlation_id: Optional[str] = None


@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    errors_encountered: int = 0
    last_activity: Optional[datetime] = None
    uptime_seconds: float = 0.0


class BaseAgent(ABC):
    """Base agent class with core functionality"""
    
    def __init__(self, agent_id: Optional[str] = None, agent_type: AgentType = AgentType.CUSTOM,
                 name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type
        self.name = name or f"{agent_type.value}_{self.agent_id[:8]}"
        self.config = config or {}
        
        # Agent state
        self.status = AgentStatus.CREATED
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.last_heartbeat = datetime.utcnow()
        
        # Communication
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_handlers: Dict[str, Callable] = {}
        self.subscriptions: List[str] = []
        
        # Capabilities and metrics
        self.capabilities: List[AgentCapability] = []
        self.metrics = AgentMetrics()
        
        # Task management
        self.current_task: Optional[Dict[str, Any]] = None
        self.task_history: List[Dict[str, Any]] = []
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Coordination
        self.coordinator_id: Optional[str] = None
        self.subordinates: List[str] = []
        
        logger.info(f"Agent {self.name} ({self.agent_id}) created")
    
    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            self.status = AgentStatus.INITIALIZING
            logger.info(f"Initializing agent {self.name}")
            
            # Register capabilities
            await self._register_capabilities()
            
            # Initialize agent-specific components
            success = await self._initialize_agent()
            
            if success:
                self.status = AgentStatus.IDLE
                self.started_at = datetime.utcnow()
                logger.info(f"Agent {self.name} initialized successfully")
                
                # Emit initialization event
                await self._emit_event("agent_initialized", {
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type.value,
                    "capabilities": [cap.name for cap in self.capabilities]
                })
                
                return True
            else:
                self.status = AgentStatus.ERROR
                logger.error(f"Agent {self.name} initialization failed")
                return False
                
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Agent {self.name} initialization error: {e}")
            return False
    
    @abstractmethod
    async def _initialize_agent(self) -> bool:
        """Agent-specific initialization logic"""
        pass
    
    @abstractmethod
    async def _register_capabilities(self):
        """Register agent capabilities"""
        pass
    
    async def start(self):
        """Start the agent"""
        try:
            if self.status != AgentStatus.IDLE:
                logger.warning(f"Agent {self.name} cannot start from status {self.status}")
                return False
            
            self.status = AgentStatus.RUNNING
            logger.info(f"Starting agent {self.name}")
            
            # Start message processing loop
            asyncio.create_task(self._message_processing_loop())
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat_loop())
            
            # Start agent-specific processing
            asyncio.create_task(self._agent_processing_loop())
            
            await self._emit_event("agent_started", {"agent_id": self.agent_id})
            
            return True
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            logger.error(f"Agent {self.name} start error: {e}")
            return False
    
    async def stop(self):
        """Stop the agent"""
        try:
            logger.info(f"Stopping agent {self.name}")
            
            # Cancel current task if any
            if self.current_task:
                await self._cancel_current_task()
            
            self.status = AgentStatus.TERMINATED
            
            await self._emit_event("agent_stopped", {"agent_id": self.agent_id})
            
            logger.info(f"Agent {self.name} stopped")
            
        except Exception as e:
            logger.error(f"Agent {self.name} stop error: {e}")
    
    async def pause(self):
        """Pause the agent"""
        if self.status == AgentStatus.RUNNING:
            self.status = AgentStatus.PAUSED
            await self._emit_event("agent_paused", {"agent_id": self.agent_id})
            logger.info(f"Agent {self.name} paused")
    
    async def resume(self):
        """Resume the agent"""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.RUNNING
            await self._emit_event("agent_resumed", {"agent_id": self.agent_id})
            logger.info(f"Agent {self.name} resumed")
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send message to another agent"""
        try:
            message.sender_id = self.agent_id
            message.timestamp = datetime.utcnow()
            
            # Cache message for delivery
            await cache_manager.cache_performance_metric(
                f"agent_message:{message.id}",
                {
                    "type": message.type.value,
                    "sender_id": message.sender_id,
                    "recipient_id": message.recipient_id,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat(),
                    "priority": message.priority,
                    "requires_response": message.requires_response,
                    "correlation_id": message.correlation_id
                },
                ttl=3600  # 1 hour
            )
            
            self.metrics.messages_sent += 1
            
            # Emit message sent event
            await self._emit_event("message_sent", {
                "message_id": message.id,
                "recipient_id": message.recipient_id,
                "type": message.type.value
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Agent {self.name} failed to send message: {e}")
            return False
    
    async def receive_message(self, message: AgentMessage):
        """Receive message from another agent"""
        try:
            await self.message_queue.put(message)
            self.metrics.messages_received += 1
            
        except Exception as e:
            logger.error(f"Agent {self.name} failed to receive message: {e}")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task"""
        try:
            task_id = task.get("id", str(uuid.uuid4()))
            task_start = datetime.utcnow()
            
            logger.info(f"Agent {self.name} executing task {task_id}")
            
            self.current_task = task
            self.metrics.last_activity = task_start
            
            # Execute agent-specific task logic
            result = await self._execute_task(task)
            
            # Update metrics
            execution_time = (datetime.utcnow() - task_start).total_seconds()
            self.metrics.tasks_completed += 1
            self.metrics.total_execution_time += execution_time
            self.metrics.average_execution_time = (
                self.metrics.total_execution_time / self.metrics.tasks_completed
            )
            
            # Add to task history
            self.task_history.append({
                "task_id": task_id,
                "started_at": task_start.isoformat(),
                "completed_at": datetime.utcnow().isoformat(),
                "execution_time": execution_time,
                "status": "completed",
                "result": result
            })
            
            self.current_task = None
            
            await self._emit_event("task_completed", {
                "task_id": task_id,
                "execution_time": execution_time,
                "result": result
            })
            
            return result
            
        except Exception as e:
            self.metrics.tasks_failed += 1
            self.metrics.errors_encountered += 1
            self.current_task = None
            
            logger.error(f"Agent {self.name} task execution failed: {e}")
            
            await self._emit_event("task_failed", {
                "task_id": task.get("id"),
                "error": str(e)
            })
            
            raise
    
    @abstractmethod
    async def _execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Agent-specific task execution logic"""
        pass
    
    async def _message_processing_loop(self):
        """Process incoming messages"""
        while self.status != AgentStatus.TERMINATED:
            try:
                if self.status == AgentStatus.PAUSED:
                    await asyncio.sleep(1)
                    continue
                
                # Get message with timeout
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                    await self._process_message(message)
                except asyncio.TimeoutError:
                    continue
                    
            except Exception as e:
                logger.error(f"Agent {self.name} message processing error: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: AgentMessage):
        """Process a received message"""
        try:
            logger.debug(f"Agent {self.name} processing message {message.id} from {message.sender_id}")
            
            if message.type == MessageType.TASK:
                # Execute task from message
                result = await self.execute_task(message.content)
                
                if message.requires_response:
                    response = AgentMessage(
                        type=MessageType.RESPONSE,
                        recipient_id=message.sender_id,
                        content={"result": result},
                        correlation_id=message.id
                    )
                    await self.send_message(response)
            
            elif message.type == MessageType.STATUS:
                # Handle status message
                await self._handle_status_message(message)
            
            elif message.type == MessageType.COORDINATION:
                # Handle coordination message
                await self._handle_coordination_message(message)
            
            elif message.type == MessageType.RESPONSE:
                # Handle response message
                await self._handle_response_message(message)
            
            elif message.type == MessageType.ERROR:
                # Handle error message
                await self._handle_error_message(message)
            
        except Exception as e:
            logger.error(f"Agent {self.name} message processing failed: {e}")
    
    async def _handle_status_message(self, message: AgentMessage):
        """Handle status message"""
        # Default implementation - can be overridden
        pass
    
    async def _handle_coordination_message(self, message: AgentMessage):
        """Handle coordination message"""
        # Default implementation - can be overridden
        pass
    
    async def _handle_response_message(self, message: AgentMessage):
        """Handle response message"""
        # Check if we have a response handler for this correlation ID
        if message.correlation_id and message.correlation_id in self.response_handlers:
            handler = self.response_handlers[message.correlation_id]
            await handler(message)
            del self.response_handlers[message.correlation_id]
    
    async def _handle_error_message(self, message: AgentMessage):
        """Handle error message"""
        logger.error(f"Agent {self.name} received error message: {message.content}")
        self.metrics.errors_encountered += 1
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat"""
        while self.status != AgentStatus.TERMINATED:
            try:
                self.last_heartbeat = datetime.utcnow()
                
                # Update uptime
                if self.started_at:
                    self.metrics.uptime_seconds = (
                        datetime.utcnow() - self.started_at
                    ).total_seconds()
                
                # Cache heartbeat
                await cache_manager.cache_performance_metric(
                    f"agent_heartbeat:{self.agent_id}",
                    {
                        "status": self.status.value,
                        "last_heartbeat": self.last_heartbeat.isoformat(),
                        "uptime_seconds": self.metrics.uptime_seconds,
                        "current_task": self.current_task.get("id") if self.current_task else None
                    },
                    ttl=300  # 5 minutes
                )
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Agent {self.name} heartbeat error: {e}")
                await asyncio.sleep(30)
    
    @abstractmethod
    async def _agent_processing_loop(self):
        """Agent-specific processing loop"""
        pass
    
    async def _cancel_current_task(self):
        """Cancel current task"""
        if self.current_task:
            logger.info(f"Agent {self.name} cancelling current task")
            self.current_task = None
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event"""
        try:
            # Cache event for event system
            if cache_manager:
                await cache_manager.cache_performance_metric(
                    f"agent_event:{self.agent_id}:{event_type}:{datetime.utcnow().timestamp()}",
                    {
                        "agent_id": self.agent_id,
                        "event_type": event_type,
                        "data": data,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    ttl=3600  # 1 hour
                )
            
            # Call registered event handlers
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        await handler(data)
                    except Exception as e:
                        logger.error(f"Event handler error: {e}")
                        
        except Exception as e:
            logger.error(f"Agent {self.name} event emission failed: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": self.agent_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "current_task": self.current_task.get("id") if self.current_task else None,
            "capabilities": [cap.name for cap in self.capabilities],
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "average_execution_time": self.metrics.average_execution_time,
                "messages_sent": self.metrics.messages_sent,
                "messages_received": self.metrics.messages_received,
                "errors_encountered": self.metrics.errors_encountered,
                "uptime_seconds": self.metrics.uptime_seconds
            }
        }
