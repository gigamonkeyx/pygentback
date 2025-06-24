"""
RAG Indexing Module

This module provides modular document indexing functionality for the RAG system,
including document processing, chunking, embedding generation, and index management.
"""

# Import all indexing components
from .base import DocumentProcessor, ChunkingStrategy, IndexingPipeline
from .chunker import TextChunker, SemanticChunker, HierarchicalChunker
from .processor import DocumentProcessorManager
from .pipeline import IndexingPipelineManager

# Re-export for easy importing
__all__ = [
    # Core interfaces
    "DocumentProcessor",
    "ChunkingStrategy", 
    "IndexingPipeline",
    
    # Chunking implementations
    "TextChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    
    # Processing and pipeline management
    "DocumentProcessorManager",
    "IndexingPipelineManager"
]
