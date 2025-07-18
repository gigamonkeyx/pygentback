#!/usr/bin/env python3
"""
A2A (Agent-to-Agent) Protocol Implementation

Implements Google's A2A protocol for PyGent Factory multi-agent communication.
Based on the official A2A specification with JSON-RPC messaging.
"""

import asyncio
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """A2A Task States"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    UNKNOWN = "unknown"


class PartType(Enum):
    """A2A Part Types"""
    TEXT = "text"
    FILE = "file"
    DATA = "data"


@dataclass
class TextPart:
    """Text content part"""
    type: str = "text"
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilePart:
    """File content part"""
    type: str = "file"
    file: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataPart:
    """Data content part"""
    type: str = "data"
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


Part = Union[TextPart, FilePart, DataPart]


@dataclass
class Message:
    """A2A Message"""
    role: str  # "user" or "agent"
    parts: List[Part] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Artifact:
    """A2A Artifact - results generated by an agent"""
    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[Part] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    index: int = 0
    append: bool = False
    lastChunk: bool = False


@dataclass
class TaskStatus:
    """A2A Task Status"""
    state: TaskState
    message: Optional[Message] = None
    timestamp: Optional[str] = None


@dataclass
class Task:
    """A2A Task - stateful entity for client-agent interaction"""
    id: str
    sessionId: str
    status: TaskStatus
    history: List[Message] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSkill:
    """A2A Agent Skill"""
    id: str
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    inputModes: List[str] = field(default_factory=list)
    outputModes: List[str] = field(default_factory=list)


@dataclass
class AgentCapabilities:
    """A2A Agent Capabilities"""
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


@dataclass
class AgentAuthentication:
    """A2A Agent Authentication"""
    schemes: List[str] = field(default_factory=list)
    credentials: Optional[str] = None


@dataclass
class AgentProvider:
    """A2A Agent Provider"""
    organization: str
    url: str


@dataclass
class AgentCard:
    """A2A Agent Card - describes agent capabilities"""
    name: str
    description: str
    url: str
    version: str
    authentication: AgentAuthentication
    defaultInputModes: List[str]
    defaultOutputModes: List[str]
    capabilities: AgentCapabilities
    skills: List[AgentSkill]
    provider: Optional[AgentProvider] = None
    documentationUrl: Optional[str] = None


class A2AProtocol:
    """A2A Protocol Implementation"""
    
    def __init__(self):
        self.agents: Dict[str, AgentCard] = {}
        self.tasks: Dict[str, Task] = {}
        self.sessions: Dict[str, List[str]] = {}  # session_id -> task_ids
        self.message_handlers: Dict[str, Callable] = {}
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default A2A protocol handlers"""
        self.message_handlers.update({
            "tasks/send": self._handle_task_send,
            "tasks/get": self._handle_task_get,
            "tasks/cancel": self._handle_task_cancel,
            "tasks/sendSubscribe": self._handle_task_send_subscribe,
            "tasks/resubscribe": self._handle_task_resubscribe,
            "agent/card": self._handle_agent_card,
        })
    
    async def register_agent(self, agent_card: AgentCard) -> bool:
        """Register an agent with the A2A protocol"""
        try:
            self.agents[agent_card.url] = agent_card
            logger.info(f"Registered A2A agent: {agent_card.name} at {agent_card.url}")
            return True
        except Exception as e:
            logger.error(f"Failed to register agent {agent_card.name}: {e}")
            return False
    
    async def discover_agents(self) -> List[AgentCard]:
        """Discover available A2A agents"""
        return list(self.agents.values())
    
    async def create_task(self, agent_url: str, message: Message, session_id: Optional[str] = None) -> Task:
        """Create a new A2A task"""
        task_id = str(uuid.uuid4())
        if not session_id:
            session_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            sessionId=session_id,
            status=TaskStatus(state=TaskState.SUBMITTED, timestamp=datetime.utcnow().isoformat()),
            history=[message],
            metadata={"agent_url": agent_url}
        )
        
        self.tasks[task_id] = task
        
        # Track session
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(task_id)
        
        logger.info(f"Created A2A task {task_id} for agent {agent_url}")
        return task
    
    async def send_message(self, task_id: str, message: Message) -> Task:
        """Send a message to an existing task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        task.history.append(message)
        task.status.timestamp = datetime.utcnow().isoformat()
        
        return task
    
    async def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update task status"""
        if task_id not in self.tasks:
            return False
        
        self.tasks[task_id].status = status
        logger.debug(f"Updated task {task_id} status to {status.state.value}")
        return True
    
    async def add_artifact(self, task_id: str, artifact: Artifact) -> bool:
        """Add artifact to task"""
        if task_id not in self.tasks:
            return False
        
        self.tasks[task_id].artifacts.append(artifact)
        logger.debug(f"Added artifact to task {task_id}")
        return True
    
    async def process_jsonrpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process A2A JSON-RPC request"""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method not in self.message_handlers:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
            
            handler = self.message_handlers[method]
            result = await handler(params)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"A2A JSON-RPC error: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def _handle_task_send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tasks/send request"""
        task_id = params.get("id")
        session_id = params.get("sessionId")
        message_data = params.get("message", {})
        
        # Convert message data to Message object
        message = Message(
            role=message_data.get("role", "user"),
            parts=[TextPart(text=part.get("text", "")) for part in message_data.get("parts", [])],
            metadata=message_data.get("metadata", {})
        )
        
        if task_id and task_id in self.tasks:
            # Continue existing task
            task = await self.send_message(task_id, message)
        else:
            # Create new task
            agent_url = params.get("agent_url", "default")
            task = await self.create_task(agent_url, message, session_id)
        
        return {
            "id": task.id,
            "sessionId": task.sessionId,
            "status": {
                "state": task.status.state.value,
                "timestamp": task.status.timestamp
            },
            "metadata": task.metadata
        }
    
    async def _handle_task_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tasks/get request"""
        task_id = params.get("id")
        
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        return {
            "id": task.id,
            "sessionId": task.sessionId,
            "status": {
                "state": task.status.state.value,
                "timestamp": task.status.timestamp
            },
            "artifacts": [
                {
                    "name": artifact.name,
                    "parts": [{"type": part.type, "text": getattr(part, "text", "")} for part in artifact.parts],
                    "metadata": artifact.metadata
                }
                for artifact in task.artifacts
            ],
            "history": [
                {
                    "role": msg.role,
                    "parts": [{"type": part.type, "text": getattr(part, "text", "")} for part in msg.parts]
                }
                for msg in task.history
            ],
            "metadata": task.metadata
        }
    
    async def _handle_task_cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tasks/cancel request"""
        task_id = params.get("id")
        
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        task.status.state = TaskState.CANCELED
        task.status.timestamp = datetime.utcnow().isoformat()
        
        return {
            "id": task.id,
            "sessionId": task.sessionId,
            "status": {
                "state": task.status.state.value,
                "timestamp": task.status.timestamp
            },
            "metadata": task.metadata
        }
    
    async def _handle_task_send_subscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tasks/sendSubscribe request (streaming)"""
        # For now, delegate to regular send
        return await self._handle_task_send(params)
    
    async def _handle_task_resubscribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tasks/resubscribe request"""
        task_id = params.get("id")
        
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        return {
            "id": task.id,
            "status": {
                "state": task.status.state.value,
                "timestamp": task.status.timestamp
            }
        }
    
    async def _handle_agent_card(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle agent card request"""
        agent_url = params.get("url", "default")
        
        if agent_url in self.agents:
            agent = self.agents[agent_url]
            return {
                "name": agent.name,
                "description": agent.description,
                "url": agent.url,
                "version": agent.version,
                "capabilities": {
                    "streaming": agent.capabilities.streaming,
                    "pushNotifications": agent.capabilities.pushNotifications
                },
                "skills": [
                    {
                        "id": skill.id,
                        "name": skill.name,
                        "description": skill.description,
                        "tags": skill.tags,
                        "examples": skill.examples
                    }
                    for skill in agent.skills
                ]
            }
        
        raise ValueError(f"Agent not found: {agent_url}")


# Global A2A protocol instance
a2a_protocol = A2AProtocol()
