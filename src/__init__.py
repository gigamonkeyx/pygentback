"""
PyGent Factory - MCP-Compliant Agent Factory System

This package provides a comprehensive agent factory system built on the
Model Context Protocol (MCP) specification. It enables creation, management,
and orchestration of AI agents with advanced capabilities including:

- MCP-compliant agent architecture
- Vector-based memory management
- RAG (Retrieval-Augmented Generation) system
- Multi-protocol communication
- Comprehensive evaluation framework
- Production-ready API with FastAPI

Main Components:
- Core: Agent management, messaging, and capability systems
- Database: PostgreSQL with pgvector for vector operations
- Storage: Vector storage with multiple backend support
- Memory: Advanced agent memory with consolidation
- MCP: Model Context Protocol server integration
- RAG: Document processing and retrieval system
- Communication: Multi-protocol messaging framework
- API: FastAPI-based REST API
- Security: Authentication and authorization
- Evaluation: Comprehensive agent testing framework

Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "PyGent Factory Team"
__license__ = "MIT"

# Core imports for easy access - minimal version for testing
try:
    from .config.settings import Settings, get_settings
except ImportError:
    Settings = None
    get_settings = None

try:
    from .memory.vector_store import VectorStore, EmbeddingModel
except ImportError:
    VectorStore = None
    EmbeddingModel = None

try:
    from .mcp.tools.executor import MCPToolExecutor
    from .mcp.tools.discovery import MCPToolDiscovery
except ImportError:
    MCPToolExecutor = None
    MCPToolDiscovery = None

try:
    from .core.factory.factory import AgentFactory
except ImportError:
    AgentFactory = None


# Package-level constants
DEFAULT_CONFIG = {
    "agent": {
        "timeout": 300,
        "max_retries": 3,
        "memory_limit": 1000
    },
    "database": {
        "pool_size": 10,
        "max_overflow": 20
    },
    "vector": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "similarity_threshold": 0.7
    },
    "mcp": {
        "timeout": 30,
        "log_level": "INFO"
    }
}

# Supported file types for document processing
SUPPORTED_DOCUMENT_TYPES = [
    ".txt", ".md", ".markdown", ".pdf", ".docx", ".doc",
    ".html", ".htm", ".json", ".yaml", ".yml",
    ".py", ".js", ".ts", ".java", ".cpp", ".c"
]

# Available agent capabilities
CORE_CAPABILITIES = [
    "text_processing",
    "document_analysis",
    "memory_management",
    "tool_execution",
    "communication",
    "learning",
    "reasoning",
    "planning",
    "vector_search",
    "gpu_acceleration",
    "semantic_similarity",
    "s3_rag",
    "minimal_training",
    "gbr_rewards"
]

# MCP server types
MCP_SERVER_TYPES = [
    "filesystem",
    "postgres",
    "github",
    "brave-search",
    "custom"
]


def get_version() -> str:
    """Get package version"""
    return __version__


def get_package_info() -> dict:
    """Get comprehensive package information"""
    return {
        "name": "pygent-factory",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": "MCP-Compliant Agent Factory System",
        "supported_document_types": SUPPORTED_DOCUMENT_TYPES,
        "core_capabilities": CORE_CAPABILITIES,
        "mcp_server_types": MCP_SERVER_TYPES,
        "default_config": DEFAULT_CONFIG
    }


# Convenience functions for quick setup - simplified
async def create_agent_factory(settings=None):
    """Create and initialize an agent factory (simplified version)"""
    if AgentFactory is None:
        raise ImportError("AgentFactory not available")

    factory = AgentFactory()
    await factory.initialize()
    return factory


# Error classes for the package
class PyGentFactoryError(Exception):
    """Base exception for PyGent Factory"""
    pass


class AgentError(PyGentFactoryError):
    """Agent-related errors"""
    pass


class MCPError(PyGentFactoryError):
    """MCP-related errors"""
    pass


class MemoryError(PyGentFactoryError):
    """Memory-related errors"""
    pass


class RAGError(PyGentFactoryError):
    """RAG-related errors"""
    pass


class DatabaseError(PyGentFactoryError):
    """Database-related errors"""
    pass


class AuthenticationError(PyGentFactoryError):
    """Authentication-related errors"""
    pass


class EvaluationError(PyGentFactoryError):
    """Evaluation-related errors"""
    pass


# Export main classes and functions - simplified
__all__ = [
    # Version and info
    "__version__",
    "get_version",
    "get_package_info",

    # Core classes (if available)
    "Settings",
    "get_settings",
    "VectorStore",
    "EmbeddingModel",
    "MCPToolExecutor",
    "MCPToolDiscovery",
    "AgentFactory",

    # Convenience functions
    "create_agent_factory",

    # Constants
    "DEFAULT_CONFIG",
    "SUPPORTED_DOCUMENT_TYPES",
    "CORE_CAPABILITIES",
    "MCP_SERVER_TYPES",

    # Exceptions
    "PyGentFactoryError",
    "AgentError",
    "MCPError",
    "MemoryError",
    "RAGError",
    "DatabaseError",
    "AuthenticationError",
    "EvaluationError"
]
