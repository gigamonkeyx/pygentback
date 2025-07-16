"""
A2A Protocol Package

Agent-to-Agent Protocol implementation for PyGent Factory.
This package provides the core types and utilities for A2A communication.
"""

from .types import *

__version__ = "0.2.1"
__all__ = [
    # Re-export all types
    "MessageRole",
    "ContentType", 
    "TaskState",
    "MessagePart",
    "A2AMessage",
    "TaskMessage",
    "AgentCapability",
    "AgentCard",
    "Task",
    "TextPart",
    "JSONPart",
    "Message",
    "create_text_message",
    "create_task_message", 
    "create_agent_card",
]
