from typing import Optional

from .handler import A2AProtocolHandler

# Global handler instance
_a2a_handler: Optional[A2AProtocolHandler] = None

def get_a2a_handler() -> A2AProtocolHandler:
    """Get A2A handler with dependencies"""
    global _a2a_handler
    if _a2a_handler is None:
        # Create default instances for now
        # TODO: Implement proper dependency injection
        _a2a_handler = A2AProtocolHandler(
            agent_factory=None,  # Will be handled by handler
            memory_manager=None,  # Will be handled by handler
            db_manager=None  # Will be handled by handler
        )
    return _a2a_handler

def set_a2a_handler(handler: A2AProtocolHandler):
    """Set global A2A handler instance"""
    global _a2a_handler
    _a2a_handler = handler
