"""
Task-Specific Tree of Thought Implementations

This module contains specialized ToT implementations for different types of problems:
- Recipe optimization and evolution
- Academic research analysis
- General problem solving
"""

from .recipe_optimization import RecipeOptimizationTask
from .research_analysis import ResearchAnalysisTask
from .problem_solving import ProblemSolvingTask

__all__ = [
    'RecipeOptimizationTask',
    'ResearchAnalysisTask', 
    'ProblemSolvingTask'
]
