"""
A2A Protocol Types

Re-exports all the core types from the protocol messages module.
This provides a clean import interface for A2A protocol types.
"""

# Import all the core types from the protocol messages
from .protocol.messages import (
    # Enums
    MessageRole,
    ContentType,
    TaskState,
    
    # Core message types
    MessagePart,
    A2AMessage,
    TaskMessage,
    
    # Agent types
    AgentCapability,
    AgentSkill,
    AgentCapabilities,
    AgentProvider,
    AgentCard,
    
    # Task types
    Task,
    TaskStatus,
    
    # Specialized message parts
    TextPart,
    JSONPart,
    
    # Convenience functions
    create_text_message,
    create_task_message,
    create_agent_card,
)

# Re-export everything for easy importing
__all__ = [
    # Enums
    "MessageRole",
    "ContentType", 
    "TaskState",
    
    # Core message types
    "MessagePart",
    "A2AMessage",
    "TaskMessage",
    
    # Agent types
    "AgentCapability",
    "AgentSkill",
    "AgentCapabilities",
    "AgentProvider",
    "AgentCard",
    
    # Task types
    "Task",
    "TaskStatus",
    
    # Specialized message parts
    "TextPart",
    "JSONPart",
    
    # Convenience functions
    "create_text_message",
    "create_task_message", 
    "create_agent_card",
]

# Additional type aliases for convenience
Message = A2AMessage  # Common alias
Part = MessagePart    # Common alias
