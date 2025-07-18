"""
Agent Augmentation Module

This module provides augmentation capabilities for PyGent Factory agents,
including RAG (Retrieval-Augmented Generation), LoRA fine-tuning,
RIPER-Ω protocol integration, and cooperative multi-agent workflows.

Phase 2: RAG Augmentation Integration
"""

# Import simple versions that work without heavy dependencies
from .simple_rag_augmenter import SimpleRAGAugmenter, SimpleCodeRetriever, SimpleEmbeddingService

# Try to import full versions, fall back to simple versions
try:
    from .rag_augmenter import RAGAugmenter
    from .code_retriever import CodeRetriever
    _FULL_RAG_AVAILABLE = True
except ImportError:
    # Use simple versions as fallback
    RAGAugmenter = SimpleRAGAugmenter
    CodeRetriever = SimpleCodeRetriever
    _FULL_RAG_AVAILABLE = False

__all__ = [
    "RAGAugmenter",
    "CodeRetriever",
    "SimpleRAGAugmenter",
    "SimpleCodeRetriever",
    "SimpleEmbeddingService"
]
