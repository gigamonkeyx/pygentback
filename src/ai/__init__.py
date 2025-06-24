"""
PyGent Factory AI Module

Advanced AI capabilities for recipe evolution, optimization, and intelligence.
Includes genetic algorithms, neural architecture search, reinforcement learning,
advanced reasoning systems, and intelligent analysis systems.
"""

from .nas import *
from .rl import *

# Import reasoning with fallback for import issues
try:
    from .reasoning import *
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import reasoning module: {e}")
    # Provide fallback or stub implementations if needed

__all__ = [
    # Neural Architecture Search
    'RecipeNAS',
    'ArchitectureEncoder',
    'SearchSpace',
    'PerformancePredictor',
    'SearchStrategy',

    # Reinforcement Learning
    'RecipeRLAgent',
    'RecipeEnvironment',
    'ActionSpace',
    'RewardFunction',
    'PolicyNetwork',
    'ExperienceReplay',

    # Advanced Reasoning
    'ToTEngine',
    'ThoughtGenerator',
    'StateEvaluator',
    'SearchStrategy',
    'BFSStrategy',
    'DFSStrategy',
    'ThoughtState',
    'ThoughtTree',
    'SearchResult',
    'RecipeOptimizationTask',
    'ResearchAnalysisTask',
    'ProblemSolvingTask'
]
