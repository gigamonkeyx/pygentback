"""
A2A Protocol Core

Core protocol implementation for Agent-to-Agent communication.
"""

from .messages import *

__all__ = [
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
    "create_text_message",
    "create_task_message", 
    "create_agent_card",
]
