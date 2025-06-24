"""
PyGent Factory Orchestration Module

This module provides orchestration capabilities for multi-agent
multi-MCP systems, enabling dynamic coordination and management
of agents across heterogeneous MCP servers.

Components:
- MCPOrchestrator: Central coordinator for MCP server management
- AgentRegistry: Dynamic agent discovery and capability mapping
- TaskDispatcher: Intelligent task assignment and load balancing
- MetricsCollector: Performance tracking and analysis
- DocumentationOrchestrator: Documentation management
- ResearchOrchestrator: Advanced research and analysis system
"""

from .mcp_orchestrator import MCPOrchestrator
from .agent_registry import AgentRegistry
from .task_dispatcher import TaskDispatcher
from .metrics_collector import MetricsCollector
from .documentation_orchestrator import DocumentationOrchestrator
from .research_orchestrator import ResearchOrchestrator
from .research_integration import (
    ResearchOrchestrationManager,
    ResearchTaskType,
    ResearchAgentType,
    initialize_research_system
)
from .coordination_models import (
    AgentCapability,
    MCPServerInfo,
    TaskRequest,
    CoordinationStrategy,
    PerformanceMetrics,
    OrchestrationConfig
)
from .documentation_models import (
    DocumentationWorkflow,
    DocumentationTask,
    DocumentationConfig,
    DocumentationWorkflowType,
    DocumentationTaskType,
    DocumentationTaskStatus
)

__all__ = [
    'MCPOrchestrator',
    'AgentRegistry',
    'TaskDispatcher',
    'MetricsCollector',
    'DocumentationOrchestrator',
    'ResearchOrchestrator',
    'ResearchOrchestrationManager',
    'ResearchTaskType',
    'ResearchAgentType',
    'initialize_research_system',
    'AgentCapability',
    'MCPServerInfo',
    'TaskRequest',
    'CoordinationStrategy',
    'PerformanceMetrics',
    'OrchestrationConfig',
    'DocumentationWorkflow',
    'DocumentationTask',
    'DocumentationConfig',
    'DocumentationWorkflowType',
    'DocumentationTaskType',
    'DocumentationTaskStatus'
]

__version__ = "1.0.0"