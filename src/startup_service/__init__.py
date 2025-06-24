"""
PyGent Factory System Startup Service
Centralized control panel for agent supervisors to initialize and manage the entire multi-agent system.

Real implementation with zero mock code - production ready service orchestration.
"""

__version__ = "1.0.0"
__author__ = "PyGent Factory Team"
__description__ = "System Startup Service for PyGent Factory Multi-Agent Platform"

# Core components
from .core.database import DatabaseManager
from .core.orchestrator import ServiceOrchestrator
from .core.config_manager import ConfigurationManager
from .core.websocket_manager import WebSocketManager

# Models and schemas
from .models.schemas import (
    SystemStatus,
    ServiceStatus,
    StartupRequest,
    ConfigurationProfile,
    ServiceConfiguration
)

# API routers
from .api.routes import startup_router, config_router, monitoring_router

# Utilities
from .utils.logging_config import setup_logging
from .utils.security import verify_jwt_token, create_access_token

__all__ = [
    # Core managers
    "DatabaseManager",
    "ServiceOrchestrator", 
    "ConfigurationManager",
    "WebSocketManager",
    
    # Models
    "SystemStatus",
    "ServiceStatus",
    "StartupRequest",
    "ConfigurationProfile",
    "ServiceConfiguration",
    
    # API routers
    "startup_router",
    "config_router", 
    "monitoring_router",
    
    # Utilities
    "setup_logging",
    "verify_jwt_token",
    "create_access_token",
]
