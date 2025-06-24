"""
PyGent Factory Agents

Comprehensive agent system with orchestration, coordination, and specialized agents:

Legacy Agents:
- ReasoningAgent: Uses Tree of Thought (ToT) reasoning
- SearchAgent: Uses s3 RAG pipeline for enhanced search
- GeneralAgent: Direct Ollama integration for general tasks
- EvolutionAgent: Recipe optimization using ToT + evolution
- CodingAgent: Code generation and analysis

New Orchestration System:
- BaseAgent: Core agent architecture with lifecycle management
- OrchestrationManager: Agent lifecycle and task distribution
- CoordinationSystem: Multi-agent workflow coordination
- CommunicationSystem: Inter-agent messaging and protocols
- SpecializedAgents: Research, Analysis, and Generation agents
"""

# Legacy agents (if they exist)
try:
    from .reasoning_agent import ReasoningAgent
except ImportError:
    ReasoningAgent = None

try:
    from .search_agent import SearchAgent
except ImportError:
    SearchAgent = None

try:
    from .general_agent import GeneralAgent
except ImportError:
    GeneralAgent = None

try:
    from .evolution_agent import EvolutionAgent
except ImportError:
    EvolutionAgent = None

try:
    from .coding_agent import CodingAgent
except ImportError:
    CodingAgent = None

# New orchestration system
from .base_agent import (
    BaseAgent,
    AgentType,
    AgentStatus,
    AgentCapability,
    AgentMessage,
    MessageType,
    AgentMetrics
)

from .orchestration_manager import (
    AgentOrchestrationManager,
    TaskDefinition,
    OrchestrationConfig,
    OrchestrationMetrics,
    orchestration_manager
)

from .coordination_system import (
    AgentCoordinationSystem,
    Workflow,
    WorkflowTask,
    CoordinationPattern,
    WorkflowStatus,
    TaskStatus,
    coordination_system
)

from .communication_system import (
    MultiAgentCommunicationSystem,
    CommunicationProtocol,
    MessageRoute,
    CommunicationChannel,
    MessagePriority,
    communication_system
)

from .specialized_agents import (
    ResearchAgent,
    AnalysisAgent,
    GenerationAgent
)

# Build __all__ dynamically
__all__ = [
    # New orchestration system
    'BaseAgent',
    'AgentType',
    'AgentStatus',
    'AgentCapability',
    'AgentMessage',
    'MessageType',
    'AgentMetrics',
    'AgentOrchestrationManager',
    'TaskDefinition',
    'OrchestrationConfig',
    'OrchestrationMetrics',
    'orchestration_manager',
    'AgentCoordinationSystem',
    'Workflow',
    'WorkflowTask',
    'CoordinationPattern',
    'WorkflowStatus',
    'TaskStatus',
    'coordination_system',
    'MultiAgentCommunicationSystem',
    'CommunicationProtocol',
    'MessageRoute',
    'CommunicationChannel',
    'MessagePriority',
    'communication_system',
    'ResearchAgent',
    'AnalysisAgent',
    'GenerationAgent'
]

# Add legacy agents if they exist
if ReasoningAgent:
    __all__.append("ReasoningAgent")
if SearchAgent:
    __all__.append("SearchAgent")
if GeneralAgent:
    __all__.append("GeneralAgent")
if EvolutionAgent:
    __all__.append("EvolutionAgent")
if CodingAgent:
    __all__.append("CodingAgent")
