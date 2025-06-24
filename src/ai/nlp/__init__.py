"""
Natural Language Processing Module

Modular NLP system for recipe understanding, test result interpretation,
natural language queries, and intelligent documentation generation.
"""

# Core components
from .core import NLPProcessor, TextProcessor, PatternMatcher
from .models import ParsedRecipe, RecipeIntent, TestResult, QueryResponse

# Specialized processors
from .recipe import RecipeParser, RecipeAnalyzer
from .testing import TestInterpreter, ResultAnalyzer
from .queries import QueryProcessor, IntentClassifier
from .documentation import DocumentationGenerator, TemplateEngine

# Utilities
from .utils import SemanticAnalyzer, EmbeddingEngine, SimilarityCalculator

__all__ = [
    # Core
    'NLPProcessor',
    'TextProcessor',
    'PatternMatcher',

    # Models
    'ParsedRecipe',
    'RecipeIntent',
    'TestResult',
    'QueryResponse',

    # Processors
    'RecipeParser',
    'RecipeAnalyzer',
    'TestInterpreter',
    'ResultAnalyzer',
    'QueryProcessor',
    'IntentClassifier',
    'DocumentationGenerator',
    'TemplateEngine',

    # Utils
    'SemanticAnalyzer',
    'EmbeddingEngine',
    'SimilarityCalculator'
]
