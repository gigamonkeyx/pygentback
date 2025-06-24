"""
Core Agent Module

This module provides the core agent functionality for PyGent Factory,
including base agent classes, message handling, capabilities, and configuration.
"""

# Import all core agent components for backward compatibility
from .base import BaseAgent, AgentError
from .message import AgentMessage, MessageType, MessagePriority
from .capability import AgentCapability
from .config import AgentConfig
from .status import AgentStatus

# Backward compatibility alias
Agent = BaseAgent

# Re-export for backward compatibility with existing imports
__all__ = [
    "BaseAgent",
    "Agent",  # Backward compatibility alias
    "AgentError",
    "AgentMessage",
    "MessageType",
    "MessagePriority",
    "AgentCapability",
    "AgentConfig",
    "AgentStatus"
]
