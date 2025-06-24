"""
Multi-Agent Agents Module

Agent implementations and management for the multi-agent system.
"""

from .base import (
    SpecializedAgent, ConfigurableAgent, 
    TaskProcessingMixin, ErrorHandlingMixin, CommunicationMixin
)
from .specialized import RecipeAgent, TestingAgent
from .registry import AgentRegistry, AgentFactory
from .manager import AgentManager, AgentPool

# New specialized agents for historical research
from .analysis_agent import AnalysisAgent, AnalysisResult
from .coordination_agent import CoordinationAgent, Task, Agent, TaskStatus
from .research_agent import ResearchAgent, ResearchSource, ResearchFinding, ResearchPhase
from .optimization_agent import OptimizationAgent, OptimizationTarget, OptimizationResult, OptimizationType
from .monitoring_agent import MonitoringAgent, MetricData, Alert, AlertLevel
from .documentation_agent import DocumentationAgent, DocumentationItem, DocumentationType
from .integration_agent import IntegrationAgent, IntegrationEndpoint, IntegrationStatus
from .academic_research_agent import AcademicResearchAgent, AcademicPaper, AcademicDatabase

__all__ = [
    # Base classes
    'SpecializedAgent',
    'ConfigurableAgent',
    'TaskProcessingMixin',
    'ErrorHandlingMixin', 
    'CommunicationMixin',
    
    # Specialized agents
    'RecipeAgent',
    'TestingAgent',
    
    # Management
    'AgentRegistry',
    'AgentFactory',
    'AgentManager',
    'AgentPool',

    # Historical Research Agents
    'AnalysisAgent',
    'AnalysisResult',
    'CoordinationAgent',
    'Task',
    'Agent',
    'TaskStatus',
    'ResearchAgent',
    'ResearchSource',
    'ResearchFinding',
    'ResearchPhase',
    'OptimizationAgent',
    'OptimizationTarget',
    'OptimizationResult',
    'OptimizationType',
    'MonitoringAgent',
    'MetricData',
    'Alert',
    'AlertLevel',
    'DocumentationAgent',
    'DocumentationItem',
    'DocumentationType',
    'IntegrationAgent',
    'IntegrationEndpoint',
    'IntegrationStatus',
    'AcademicResearchAgent',
    'AcademicPaper',
    'AcademicDatabase'
]
