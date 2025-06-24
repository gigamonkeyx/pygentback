"""
RAG (Retrieval-Augmented Generation) System - Modular Architecture

This module provides the modular RAG system for PyGent Factory, including document processing,
retrieval, and generation capabilities. The system has been modularized for better organization
and maintainability while maintaining full backward compatibility.

Modular Structure:
- retrieval/: Semantic search, ranking, and context-aware retrieval
- indexing/: Document processing, chunking, and index management
- generation/: Response generation and context integration
- pipeline/: End-to-end RAG pipeline management
- s3/: Advanced s3 RAG framework with minimal training data

Legacy compatibility is maintained through wrapper classes.
"""

# Import modular components
from .retrieval import (
    RetrievalManager as ModularRetrievalManager,
    SemanticRetriever,
    RetrievalScorer as ModularRetrievalScorer
)

from .indexing import (
    DocumentProcessorManager,
    IndexingPipelineManager
)

# Import s3 RAG framework
try:
    from .s3 import (
        S3Pipeline,
        S3Result,
        S3SearchAgent,
        S3Config,
        SearchState,
        SearchAction,
        GBRReward,
        GBRRewardCalculator,
        S3RLTrainer,
        BaselineRAG,
        NaiveRAG,
        SearchStrategy
    )
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# Import legacy compatibility layer
from .retrieval_system import (
    RetrievalSystem,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
    RetrievalScorer
)
from .document_processor import (
    DocumentProcessor,
    ProcessedDocument,
    DocumentChunk
)

# Export both modular and legacy interfaces
__all__ = [
    # Legacy compatibility (default imports)
    "RetrievalSystem",
    "RetrievalQuery", 
    "RetrievalResult",
    "RetrievalStrategy",
    "RetrievalScorer",
    "DocumentProcessor",
    "ProcessedDocument",
    "DocumentChunk",
    
    # Modular components (for direct access)
    "ModularRetrievalManager",
    "SemanticRetriever",
    "ModularRetrievalScorer",
    
    # Indexing components
    "DocumentProcessorManager",
    "IndexingPipelineManager"
]

# Add s3 components if available
if S3_AVAILABLE:
    __all__.extend([
        # s3 RAG Framework
        "S3Pipeline",
        "S3Result",
        "S3SearchAgent",
        "S3Config",
        "SearchState",
        "SearchAction",
        "GBRReward",
        "GBRRewardCalculator",
        "S3RLTrainer",
        "BaselineRAG",
        "NaiveRAG",
        "SearchStrategy"
    ])
