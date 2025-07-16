"""
PyGent Factory Core Module

This module provides the core functionality for PyGent Factory including:
- Agent system (both legacy and modular)
- Agent factory and orchestration
- Message system
- Capability system
- GPU optimization and Ollama integration
"""

# Import agent system from agent directory (modular system)
from .agent import BaseAgent, Agent, AgentMessage, AgentCapability, AgentConfig, AgentStatus
from .agent import MessageType, MessagePriority, AgentError

# Aliases for backward compatibility
ModularBaseAgent = BaseAgent
ModularAgentMessage = AgentMessage
ModularAgentCapability = AgentCapability
ModularAgentConfig = AgentConfig
ModularAgentStatus = AgentStatus
ModularMessageType = MessageType

# Import agent factory and orchestration
from .agent_factory import AgentFactory
from .agent_orchestrator import AgentOrchestrator
from .agent_builder import AgentBuilder
from .agent_validator import AgentValidator

# Import capability system
from .capability_system import CapabilitySystem

# Import message system
from .message_system import MessageSystem

# Import GPU and AI integrations
from .gpu_config import GPUConfig
from .gpu_optimization import GPUOptimizer
from .ollama_integration import OllamaIntegration
from .ollama_manager import OllamaManager
from .openrouter_integration import OpenRouterIntegration

# Import research workflow
from .research_workflow import ResearchWorkflow

# Import model manager
from .model_manager import ModelManager

# Phase 4 Integration: Import simulation and behavior detection modules
try:
    from .sim_env import SimulationEnvironment, create_simulation_environment
    from .emergent_behavior_detector import Docker443EmergentBehaviorDetector
    from .graceful_shutdown import GracefulShutdownManager
    PHASE4_MODULES_AVAILABLE = True
except ImportError as e:
    # Phase 4 modules not available - continue without them
    PHASE4_MODULES_AVAILABLE = False

__all__ = [
    # Agent system
    "BaseAgent", "Agent", "AgentMessage", "AgentCapability", "AgentConfig", "AgentStatus", "MessageType",
    "ModularBaseAgent", "ModularAgentMessage", "ModularAgentCapability", "ModularAgentConfig", 
    "ModularAgentStatus", "ModularMessageType", "MessagePriority", "AgentError",
    "create_request_message", "create_notification_message", "create_tool_call_message", 
    "create_capability_request_message",
    
    # Factory and orchestration
    "AgentFactory", "AgentOrchestrator", "AgentBuilder", "AgentValidator",
    
    # Systems
    "CapabilitySystem", "MessageSystem",
    
    # GPU and AI
    "GPUConfig", "GPUOptimizer", "OllamaIntegration", "OllamaManager", "OpenRouterIntegration",
    
    # Workflow and models
    "ResearchWorkflow", "ModelManager",

    # Phase 4 modules (if available)
    "PHASE4_MODULES_AVAILABLE"
]

# Conditionally add Phase 4 modules to __all__ if available
if PHASE4_MODULES_AVAILABLE:
    __all__.extend([
        "SimulationEnvironment", "create_simulation_environment",
        "Docker443EmergentBehaviorDetector", "GracefulShutdownManager"
    ])
