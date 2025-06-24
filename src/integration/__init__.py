"""
Advanced Integration and Orchestration System

Comprehensive system for integrating all AI components, orchestrating complex workflows,
and providing unified interfaces for the complete PyGent Factory ecosystem.
"""

# Core integration components
from .core import IntegrationEngine, OrchestrationEngine, WorkflowOrchestrator
from .models import IntegrationConfig, WorkflowDefinition, ExecutionContext, IntegrationResult

# Workflow management
from .workflows import WorkflowManager, WorkflowExecutor, WorkflowTemplate, WorkflowState

# Monitoring and events
from .monitoring import IntegrationMonitor, SystemHealthMonitor, PerformanceMonitor
from .events import EventBus, EventHandler, Event, EventType

# Configuration management
from .config import IntegrationConfigManager, ComponentConfig

# Utilities
from .utils import OperationResult, RetryConfig, ValidationError

# System coordinators
from .coordinators import (
    AISystemCoordinator, ResourceCoordinator, EventCoordinator
)

# Integration adapters
from .adapters import (
    GeneticAlgorithmAdapter, NeuralSearchAdapter, ReinforcementLearningAdapter,
    MCPAdapter, NLPAdapter, MultiAgentAdapter, PredictiveAdapter
)

__all__ = [
    # Core
    'IntegrationEngine',
    'OrchestrationEngine',
    'WorkflowOrchestrator',

    # Models
    'IntegrationConfig',
    'WorkflowDefinition',
    'ExecutionContext',
    'IntegrationResult',

    # Workflow management
    'WorkflowManager',
    'WorkflowExecutor',
    'WorkflowTemplate',
    'WorkflowState',

    # Monitoring and events
    'IntegrationMonitor',
    'SystemHealthMonitor',
    'PerformanceMonitor',
    'EventBus',
    'EventHandler',
    'Event',
    'EventType',

    # Configuration
    'IntegrationConfigManager',
    'ComponentConfig',

    # Utilities
    'OperationResult',
    'RetryConfig',
    'ValidationError',

    # Coordinators
    'AISystemCoordinator',
    'ResourceCoordinator',
    'EventCoordinator',

    # Adapters
    'GeneticAlgorithmAdapter',
    'NeuralSearchAdapter',
    'ReinforcementLearningAdapter',
    'MCPAdapter',
    'NLPAdapter',
    'MultiAgentAdapter',
    'PredictiveAdapter'
]
