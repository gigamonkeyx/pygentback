"""
Comprehensive Agent + MCP Recipe Testing Framework

This module provides a complete testing framework for Agent + MCP combinations
with ML/RL-powered validation, real-world MCP server integration, and build
validation capabilities.
"""

# Core testing framework
from .core.framework import RecipeTestingFramework
from .core.runner import RecipeTestRunner
from .core.scoring import RecipeScorer, ScoringMetrics
from .core.ranking import RecipeRanker, RankingAlgorithms

# Recipe management
from .recipes.schema import RecipeSchema, RecipeDefinition
from .recipes.registry import RecipeRegistry
from .recipes.generator import RecipeGenerator
from .recipes.validator import RecipeValidator

# MCP server pool management
from .mcp.discovery import MCPServerDiscovery
from .mcp.pool_manager import MCPServerPoolManager
from .mcp.installer import MCPServerInstaller
from .mcp.health_monitor import MCPHealthMonitor

# Test execution engine
from .engine.executor import TestExecutor
from .engine.scheduler import TestScheduler
from .engine.profiler import PerformanceProfiler
from .engine.reporter import TestReporter

# ML intelligence
from .ml.predictor import RecipeSuccessPredictor
from .ml.optimizer import RecipeOptimizer
from .ml.failure_analyzer import FailurePatternAnalyzer

# RL intelligence
from .rl.execution_agent import RecipeExecutionAgent
from .rl.evolution import RecipeEvolutionSystem

# Analytics
from .analytics.analyzer import RecipeAnalyzer
from .analytics.dashboard import PerformanceDashboard
from .analytics.trends import TrendAnalyzer

# Monitoring
from .monitoring.health_monitor import RecipeHealthMonitor

# Build integration
from .ci.smart_selector import SmartTestSelector
from .ci.pipeline import ValidationPipeline
from .ci.quality_gates import QualityGates

# Scenario generation
from .scenarios.generator import TestScenarioGenerator
from .scenarios.templates import ScenarioTemplates
from .scenarios.real_data import RealDataProvider

__all__ = [
    # Core framework
    "RecipeTestingFramework",
    "RecipeTestRunner",
    "RecipeScorer",
    "ScoringMetrics",
    "RecipeRanker",
    "RankingAlgorithms",

    # Recipe management
    "RecipeSchema",
    "RecipeDefinition",
    "RecipeRegistry",
    "RecipeGenerator",
    "RecipeValidator",

    # MCP server pool
    "MCPServerDiscovery",
    "MCPServerPoolManager",
    "MCPServerInstaller",
    "MCPHealthMonitor",

    # Test execution
    "TestExecutor",
    "TestScheduler",
    "PerformanceProfiler",
    "TestReporter",

    # ML intelligence
    "RecipeSuccessPredictor",
    "RecipeOptimizer",
    "FailurePatternAnalyzer",

    # RL intelligence
    "RecipeExecutionAgent",
    "RecipeEvolutionSystem",

    # Analytics
    "RecipeAnalyzer",
    "PerformanceDashboard",
    "TrendAnalyzer",

    # Monitoring
    "RecipeHealthMonitor",

    # Build integration
    "SmartTestSelector",
    "ValidationPipeline",
    "QualityGates",

    # Scenario generation
    "TestScenarioGenerator",
    "ScenarioTemplates",
    "RealDataProvider"
]

# Version information
__version__ = "1.0.0"
__author__ = "PyGent Factory Team"
__description__ = "Comprehensive Agent + MCP Recipe Testing Framework"
