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
    # Use absolute import fallback
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from src.utils.utf8_logger import get_pygent_logger
        logger = get_pygent_logger("ai_module")
    except ImportError:
        import logging
        logger = logging.getLogger("ai_module")
    logger.warning(f"Failed to import reasoning module: {e}")
    # Provide fallback or stub implementations if needed

# Observer-approved evolution system with absolute import fallback
try:
    from .evolution.evo_loop_fixed import ObserverEvolutionLoop
    OBSERVER_EVOLUTION_AVAILABLE = True
except ImportError as e:
    # Use absolute import fallback
    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from src.utils.utf8_logger import get_pygent_logger
        logger = get_pygent_logger("ai_module")
    except ImportError:
        import logging
        logger = logging.getLogger("ai_module")
    logger.warning(f"Observer evolution system not available: {e}")
    OBSERVER_EVOLUTION_AVAILABLE = False

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

# Add Observer evolution system if available
if OBSERVER_EVOLUTION_AVAILABLE:
    __all__.append('ObserverEvolutionLoop')
    # Export for direct access
    try:
        from .evolution.evo_loop_fixed import ObserverEvolutionLoop
        globals()['ObserverEvolutionLoop'] = ObserverEvolutionLoop
    except ImportError:
        pass

# Export evolution module contents for easier access
try:
    from .evolution import *
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Evolution module export failed: {e}")
