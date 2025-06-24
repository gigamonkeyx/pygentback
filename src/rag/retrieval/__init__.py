"""
RAG Retrieval Module

This module provides modular retrieval functionality for the RAG system,
including semantic search, ranking, and context-aware retrieval.
"""

# Import all retrieval components
from .base import RetrievalStrategy, RetrievalQuery, RetrievalResult, RetrievalEngine
from .semantic import SemanticRetriever
from .hybrid import HybridRetriever
from .contextual import ContextualRetriever
from .scorer import RetrievalScorer
from .manager import RetrievalManager

# Re-export for easy importing
__all__ = [
    # Core interfaces
    "RetrievalStrategy",
    "RetrievalQuery", 
    "RetrievalResult",
    "RetrievalEngine",
    
    # Retrieval implementations
    "SemanticRetriever",
    "HybridRetriever",
    "ContextualRetriever",
    
    # Scoring and management
    "RetrievalScorer",
    "RetrievalManager"
]
