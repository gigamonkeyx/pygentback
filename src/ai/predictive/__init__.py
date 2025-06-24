"""
Predictive Recipe Optimization System

Advanced predictive system for optimizing recipe performance, predicting outcomes,
and providing intelligent recommendations for recipe improvement.
"""

# Core prediction components
from .core import PredictiveEngine, OptimizationEngine, RecommendationEngine
from .models import Prediction, Optimization, Recommendation, PredictionMetrics

# Specialized predictors
from .predictors import (
    PerformancePredictor, SuccessPredictor, ResourcePredictor,
    LatencyPredictor, QualityPredictor, CostPredictor
)

# Optimization strategies
from .optimizers import (
    GeneticOptimizer, GradientOptimizer, BayesianOptimizer,
    MultiObjectiveOptimizer
)

# Advanced modules - Available for future implementation
# Additional predictive AI capabilities can be added here as needed

__all__ = [
    # Core
    'PredictiveEngine',
    'OptimizationEngine',
    'RecommendationEngine',
    
    # Models
    'Prediction',
    'Optimization',
    'Recommendation',
    'PredictionMetrics',
    
    # Predictors
    'PerformancePredictor',
    'SuccessPredictor',
    'ResourcePredictor',
    'LatencyPredictor',
    'QualityPredictor',
    'CostPredictor',
    
    # Optimizers
    'GeneticOptimizer',
    'GradientOptimizer',
    'BayesianOptimizer',
    'MultiObjectiveOptimizer'

    # Additional modules available for future expansion
]
