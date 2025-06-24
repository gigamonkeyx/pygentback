"""
Core Agent System - Backward Compatibility Layer

This module provides backward compatibility for the modular agent system.
All core agent functionality has been moved to the agent submodule for better organization.
This file maintains the original interface while delegating to the new modular components.
"""

# Import all components from the modular agent system for backward compatibility
from .agent.base import BaseAgent as ModularBaseAgent, AgentError
from .agent.message import (
    AgentMessage as ModularAgentMessage, MessageType as ModularMessageType, MessagePriority,
    create_request_message, create_notification_message,
    create_tool_call_message, create_capability_request_message
)
from .agent.capability import AgentCapability as ModularAgentCapability, CapabilityType, CapabilityParameter
from .agent.config import AgentConfig as ModularAgentConfig
from .agent.status import AgentStatus as ModularAgentStatus, AgentStatusInfo, AgentStatusManager

# Legacy imports for backward compatibility
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

try:
    from mcp.types import Tool, Resource
except ImportError:
    # Fallback if mcp is not available
    Tool = Any
    Resource = Any


# Legacy compatibility classes that delegate to the modular components
class AgentStatus(Enum):
    """Agent status enumeration - Legacy compatibility"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    PAUSED = "paused"

    # Map to new status system
    @classmethod
    def from_modular_status(cls, modular_status: ModularAgentStatus) -> 'AgentStatus':
        """Convert from modular status to legacy status"""
        mapping = {
            ModularAgentStatus.INITIALIZING: cls.INACTIVE,
            ModularAgentStatus.ACTIVE: cls.ACTIVE,
            ModularAgentStatus.IDLE: cls.ACTIVE,
            ModularAgentStatus.BUSY: cls.BUSY,
            ModularAgentStatus.ERROR: cls.ERROR,
            ModularAgentStatus.STOPPING: cls.INACTIVE,
            ModularAgentStatus.STOPPED: cls.INACTIVE,
            ModularAgentStatus.MAINTENANCE: cls.PAUSED
        }
        return mapping.get(modular_status, cls.INACTIVE)


class MessageType(Enum):
    """Message type enumeration for agent communication - Legacy compatibility"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    NOTIFICATION = "notification"

    # Map to new message types
    @classmethod
    def from_modular_type(cls, modular_type: ModularMessageType) -> 'MessageType':
        """Convert from modular message type to legacy type"""
        mapping = {
            ModularMessageType.REQUEST: cls.REQUEST,
            ModularMessageType.RESPONSE: cls.RESPONSE,
            ModularMessageType.NOTIFICATION: cls.NOTIFICATION,
            ModularMessageType.ERROR: cls.RESPONSE,
            ModularMessageType.TOOL_CALL: cls.REQUEST,
            ModularMessageType.TOOL_RESULT: cls.RESPONSE,
            ModularMessageType.CAPABILITY_REQUEST: cls.REQUEST,
            ModularMessageType.CAPABILITY_RESPONSE: cls.RESPONSE
        }
        return mapping.get(modular_type, cls.REQUEST)


@dataclass
class AgentMessage:
    """Standard message format for agent communication - Legacy compatibility wrapper"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    sender: str = ""
    recipient: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None

    def to_modular_message(self) -> ModularAgentMessage:
        """Convert to modular message format"""
        # Map legacy type to modular type
        modular_type_mapping = {
            MessageType.REQUEST: ModularMessageType.REQUEST,
            MessageType.RESPONSE: ModularMessageType.RESPONSE,
            MessageType.EVENT: ModularMessageType.NOTIFICATION,
            MessageType.NOTIFICATION: ModularMessageType.NOTIFICATION
        }

        return ModularAgentMessage(
            id=self.id,
            type=modular_type_mapping.get(self.type, ModularMessageType.REQUEST),
            sender=self.sender,
            recipient=self.recipient,
            content=self.content,
            metadata=self.metadata,
            timestamp=self.timestamp,
            correlation_id=self.correlation_id
        )

    @classmethod
    def from_modular_message(cls, modular_msg: ModularAgentMessage) -> 'AgentMessage':
        """Create legacy message from modular message"""
        return cls(
            id=modular_msg.id,
            type=MessageType.from_modular_type(modular_msg.type),
            sender=modular_msg.sender,
            recipient=modular_msg.recipient,
            content=modular_msg.content,
            metadata=modular_msg.metadata,
            timestamp=modular_msg.timestamp,
            correlation_id=modular_msg.correlation_id
        )


@dataclass
class AgentCapability:
    """Represents an agent capability - Legacy compatibility wrapper"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_tools: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)

    def to_modular_capability(self) -> ModularAgentCapability:
        """Convert to modular capability format"""
        modular_cap = ModularAgentCapability(
            name=self.name,
            description=self.description,
            required_tools=self.required_tools,
            required_resources=self.required_resources
        )

        # Convert parameters to modular format
        for param_name, param_info in self.parameters.items():
            if isinstance(param_info, dict):
                modular_cap.add_parameter(
                    name=param_name,
                    param_type=param_info.get("type", "string"),
                    description=param_info.get("description", ""),
                    required=param_info.get("required", True),
                    default=param_info.get("default")
                )

        return modular_cap

    @classmethod
    def from_modular_capability(cls, modular_cap: ModularAgentCapability) -> 'AgentCapability':
        """Create legacy capability from modular capability"""
        parameters = {}
        for param in modular_cap.parameters:
            parameters[param.name] = {
                "type": param.type,
                "description": param.description,
                "required": param.required,
                "default": param.default
            }

        return cls(
            name=modular_cap.name,
            description=modular_cap.description,
            parameters=parameters,
            required_tools=modular_cap.required_tools,
            required_resources=modular_cap.required_resources
        )


@dataclass
class AgentConfig:
    """Agent configuration - Legacy compatibility wrapper"""
    agent_id: str
    name: str
    type: str
    capabilities: List[str] = field(default_factory=list)
    mcp_tools: List[str] = field(default_factory=list)
    memory_config: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 300
    max_retries: int = 3
    custom_config: Dict[str, Any] = field(default_factory=dict)

    def to_modular_config(self) -> ModularAgentConfig:
        """Convert to modular config format"""
        return ModularAgentConfig(
            agent_id=self.agent_id,
            name=self.name,
            agent_type=self.type,
            enabled_capabilities=self.capabilities,
            mcp_tools=self.mcp_tools,
            default_timeout=self.timeout,
            max_retries=self.max_retries,
            custom_config=self.custom_config,
            memory_enabled=bool(self.memory_config)
        )

    @classmethod
    def from_modular_config(cls, modular_config: ModularAgentConfig) -> 'AgentConfig':
        """Create legacy config from modular config"""
        return cls(
            agent_id=modular_config.agent_id,
            name=modular_config.name,
            type=modular_config.agent_type,
            capabilities=modular_config.enabled_capabilities,
            mcp_tools=modular_config.mcp_tools,
            timeout=modular_config.default_timeout,
            max_retries=modular_config.max_retries,
            custom_config=modular_config.custom_config,
            memory_config={"enabled": modular_config.memory_enabled}
        )


class BaseAgent(ABC):
    """
    Abstract base class for all agents in PyGent Factory - Legacy compatibility wrapper.

    This class provides backward compatibility while delegating to the new modular agent system.
    It maintains the original interface while using the enhanced modular components internally.
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the base agent.

        Args:
            config: Agent configuration object (legacy format)
        """
        # Convert legacy config to modular config
        self.config = config
        self._modular_config = config.to_modular_config()

        # Create internal modular agent (this will be implemented by subclasses)
        self._modular_agent: Optional[ModularBaseAgent] = None

        # Legacy properties for backward compatibility
        self.agent_id = config.agent_id
        self.name = config.name
        self.type = config.type
        self.capabilities = config.capabilities
        self.status = AgentStatus.INACTIVE
        self.mcp_tools: Dict[str, str] = {}  # tool_name -> server_id
        self.memory = None
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.error_count = 0
        self.execution_history: List[Dict[str, Any]] = []

    def _ensure_modular_agent(self) -> ModularBaseAgent:
        """Ensure modular agent is available (implemented by subclasses)"""
        if self._modular_agent is None:
            raise NotImplementedError("Modular agent not initialized")
        return self._modular_agent

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process an incoming message and return a response.

        Args:
            message: The incoming message to process (legacy format)

        Returns:
            AgentMessage: The response message (legacy format)
        """
        # Convert legacy message to modular format
        modular_message = message.to_modular_message()

        # Process with modular agent
        modular_agent = self._ensure_modular_agent()
        modular_response = await modular_agent.process_message(modular_message)

        # Convert response back to legacy format
        legacy_response = AgentMessage.from_modular_message(modular_response)

        # Update legacy properties
        self.last_activity = datetime.utcnow()
        self.status = AgentStatus.from_modular_status(modular_agent.status)

        return legacy_response

    async def execute_capability(self, capability: str, params: Dict[str, Any]) -> Any:
        """
        Execute a specific capability with given parameters.

        Args:
            capability: Name of the capability to execute
            params: Parameters for the capability execution

        Returns:
            Any: Result of the capability execution
        """
        # Delegate to modular agent
        modular_agent = self._ensure_modular_agent()
        result = await modular_agent.execute_capability(capability, params)

        # Update legacy properties
        self.last_activity = datetime.utcnow()
        self.status = AgentStatus.from_modular_status(modular_agent.status)
        self.execution_history.append({
            "capability": capability,
            "params": params,
            "timestamp": datetime.utcnow().isoformat(),
            "result_type": type(result).__name__
        })

        return result

    async def initialize(self) -> None:
        """
        Initialize the agent. Called after construction.
        Override in subclasses for custom initialization.
        """
        if self._modular_agent:
            await self._modular_agent.initialize()
            self.status = AgentStatus.from_modular_status(self._modular_agent.status)
        else:
            self.status = AgentStatus.ACTIVE

        self.last_activity = datetime.utcnow()

    async def shutdown(self) -> None:
        """
        Shutdown the agent gracefully.
        Override in subclasses for custom cleanup.
        """
        if self._modular_agent:
            await self._modular_agent.shutdown()
            self.status = AgentStatus.from_modular_status(self._modular_agent.status)
        else:
            self.status = AgentStatus.INACTIVE

    async def register_mcp_tool(self, tool_name: str, server_id: str) -> None:
        """
        Register an MCP tool with the agent.

        Args:
            tool_name: Name of the tool
            server_id: ID of the MCP server providing the tool
        """
        self.mcp_tools[tool_name] = server_id

        # Update modular config if available
        if self._modular_agent:
            self._modular_agent.config.add_mcp_tool(tool_name)

    async def unregister_mcp_tool(self, tool_name: str) -> None:
        """
        Unregister an MCP tool from the agent.

        Args:
            tool_name: Name of the tool to unregister
        """
        self.mcp_tools.pop(tool_name, None)

        # Update modular config if available
        if self._modular_agent:
            self._modular_agent.config.remove_mcp_tool(tool_name)

    def get_capabilities(self) -> List[AgentCapability]:
        """
        Get the list of agent capabilities.

        Returns:
            List[AgentCapability]: List of capabilities (legacy format)
        """
        if self._modular_agent:
            # Convert modular capabilities to legacy format
            modular_capabilities = self._modular_agent.get_capabilities()
            return [AgentCapability.from_modular_capability(cap) for cap in modular_capabilities]
        else:
            # Default implementation for backward compatibility
            return [
                AgentCapability(
                    name=cap,
                    description=f"Capability: {cap}",
                    parameters={},
                    required_tools=[],
                    required_resources=[]
                ) for cap in self.capabilities
            ]

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current agent status.

        Returns:
            Dict[str, Any]: Status information
        """
        if self._modular_agent:
            # Get status from modular agent and convert
            modular_status = self._modular_agent.get_status()
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "type": self.type,
                "status": AgentStatus.from_modular_status(self._modular_agent.status).value,
                "capabilities": self.capabilities,
                "mcp_tools": list(self.mcp_tools.keys()),
                "created_at": self.created_at.isoformat(),
                "last_activity": self.last_activity.isoformat(),
                "error_count": self.error_count,
                "execution_count": len(self.execution_history),
                "modular_status": modular_status
            }
        else:
            # Legacy status format
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "type": self.type,
                "status": self.status.value,
                "capabilities": self.capabilities,
                "mcp_tools": list(self.mcp_tools.keys()),
                "created_at": self.created_at.isoformat(),
                "last_activity": self.last_activity.isoformat(),
                "error_count": self.error_count,
                "execution_count": len(self.execution_history)
            }


# Re-export modular components for direct access
# This allows users to gradually migrate to the new modular system
ModularAgentMessage = ModularAgentMessage
ModularAgentCapability = ModularAgentCapability
ModularAgentConfig = ModularAgentConfig
ModularAgentStatus = ModularAgentStatus
ModularBaseAgent = ModularBaseAgent

# Backward compatibility alias
Agent = BaseAgent

# Export all for backward compatibility
__all__ = [
    # Legacy classes
    "BaseAgent",
    "Agent",  # Backward compatibility alias
    "AgentMessage",
    "AgentCapability",
    "AgentConfig",
    "AgentStatus",
    "MessageType",

    # Modular classes for direct access
    "ModularBaseAgent",
    "ModularAgentMessage",
    "ModularAgentCapability",
    "ModularAgentConfig",
    "ModularAgentStatus",
    "ModularMessageType",
    "MessagePriority",
    "AgentError",

    # Factory functions
    "create_request_message",
    "create_notification_message",
    "create_tool_call_message",
    "create_capability_request_message"
]
