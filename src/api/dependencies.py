"""
API Dependencies

Provides dependency injection for FastAPI routes
"""

from fastapi import HTTPException
from typing import Optional

# Global state storage (in production, use proper dependency injection)
_agent_factory = None
_db_manager = None
_memory_manager = None
_mcp_manager = None
_retrieval_system = None
_message_bus = None
_ollama_manager = None


def set_agent_factory(agent_factory):
    """Set the global agent factory"""
    global _agent_factory
    _agent_factory = agent_factory


def set_db_manager(db_manager):
    """Set the global database manager"""
    global _db_manager
    _db_manager = db_manager


def set_memory_manager(memory_manager):
    """Set the global memory manager"""
    global _memory_manager
    _memory_manager = memory_manager


def set_mcp_manager(mcp_manager):
    """Set the global MCP manager"""
    global _mcp_manager
    _mcp_manager = mcp_manager


def set_retrieval_system(retrieval_system):
    """Set the global retrieval system"""
    global _retrieval_system
    _retrieval_system = retrieval_system


def set_message_bus(message_bus):
    """Set the global message bus"""
    global _message_bus
    _message_bus = message_bus


def set_ollama_manager(ollama_manager):
    """Set the global Ollama manager"""
    global _ollama_manager
    _ollama_manager = ollama_manager


async def get_agent_factory():
    """Get agent factory dependency"""
    # Try to get from app state first (available after startup)
    try:
        from .main import app_state
        if app_state.get("agent_factory"):
            return app_state["agent_factory"]
    except ImportError:
        pass

    # Fallback to module-level factory
    if not _agent_factory:
        raise HTTPException(status_code=503, detail="Agent factory not available")
    return _agent_factory


async def get_db_manager():
    """Get database manager dependency"""
    if not _db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    return _db_manager


async def get_memory_manager():
    """Get memory manager dependency"""
    if not _memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not available")
    return _memory_manager


async def get_mcp_manager():
    """Get MCP manager dependency"""
    if not _mcp_manager:
        raise HTTPException(status_code=503, detail="MCP manager not available")
    return _mcp_manager


async def get_retrieval_system():
    """Get retrieval system dependency"""
    if not _retrieval_system:
        raise HTTPException(status_code=503, detail="Retrieval system not available")
    return _retrieval_system


async def get_message_bus():
    """Get message bus dependency"""
    if not _message_bus:
        raise HTTPException(status_code=503, detail="Message bus not available")
    return _message_bus


async def get_ollama_manager():
    """Get Ollama manager dependency"""
    if not _ollama_manager:
        raise HTTPException(status_code=503, detail="Ollama manager not available")
    return _ollama_manager
