"""
Multi-Agent Coordination System

Advanced multi-agent system for coordinating AI agents, managing workflows,
and enabling collaborative problem-solving across different AI capabilities.
"""

# Core coordination components
from .core import AgentCoordinator, WorkflowManager, CommunicationHub
from .models import Agent, Task, Workflow, Message, CoordinationResult

# Specialized agents
from .agents import (
    RecipeAgent, TestingAgent, AnalysisAgent, OptimizationAgent,
    MonitoringAgent, DocumentationAgent, IntegrationAgent
)

# Coordination strategies (TODO: Implement these modules)
# from .strategies import (
#     CoordinationStrategy, HierarchicalStrategy, PeerToPeerStrategy,
#     PipelineStrategy, ConsensusStrategy, CompetitiveStrategy
# )

# Communication and messaging (TODO: Implement these modules)
# from .communication import MessageBroker, EventBus, NotificationSystem

# Workflow orchestration (TODO: Implement these modules)
# from .workflows import WorkflowEngine, TaskScheduler, DependencyResolver

# Utilities (TODO: Implement these modules)
# from .utils import AgentRegistry, PerformanceTracker, ConflictResolver

__all__ = [
    # Core
    'AgentCoordinator',
    'WorkflowManager',
    'CommunicationHub',

    # Models
    'Agent',
    'Task',
    'Workflow',
    'Message',
    'CoordinationResult',

    # Agents
    'RecipeAgent',
    'TestingAgent',
    'AnalysisAgent',
    'OptimizationAgent',
    'MonitoringAgent',
    'DocumentationAgent',
    'IntegrationAgent'

    # Additional coordination strategies and utilities available for future expansion
]
