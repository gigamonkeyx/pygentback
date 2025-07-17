"""
DGM (Darwin Gödel Machine) Implementation for PyGent Factory.

This module provides the core components for Sakana AI's DGM self-improving
agent architecture, including self-improvement engines, goal networks, and
performance tracking with Observer-approved formal verification.
"""

__version__ = "1.0.0"
__author__ = "PyGent Factory Team"

# Core DGM components
try:
    from .core.engine import DGMEngine
    from .core.safety_monitor import SafetyMonitor
    from .core.validator import DGMValidator
    from .models import DGMModel, GoalNetwork
    DGM_CORE_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"DGM core components not available: {e}")
    DGM_CORE_AVAILABLE = False

# Observer-approved autonomy system with formal verification
try:
    from .autonomy_fixed import (
        FormalProofSystem,
        ObserverAutonomyController
    )
    OBSERVER_AUTONOMY_AVAILABLE = True
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Observer autonomy system not available: {e}")
    OBSERVER_AUTONOMY_AVAILABLE = False

# Build __all__ dynamically
__all__ = ['__version__', '__author__']

# Add core DGM components if available
if DGM_CORE_AVAILABLE:
    __all__.extend([
        'DGMEngine',
        'SafetyMonitor',
        'DGMValidator',
        'DGMModel',
        'GoalNetwork'
    ])

# Add Observer autonomy system if available
if OBSERVER_AUTONOMY_AVAILABLE:
    __all__.extend([
        'FormalProofSystem',
        'ObserverAutonomyController'
    ])
