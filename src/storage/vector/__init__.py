"""
Vector Storage Module

This module provides modular vector storage implementations for PyGent Factory,
including abstract interfaces and concrete implementations for different vector databases.
"""

# Import all vector storage components
from .base import VectorStore, VectorDocument, VectorQuery, VectorSearchResult
from .postgresql import PostgreSQLVectorStore
from .chromadb import ChromaDBVectorStore
from .faiss import FAISSVectorStore
from .manager import VectorStoreManager

# Re-export for backward compatibility
__all__ = [
    # Core interfaces
    "VectorStore",
    "VectorDocument", 
    "VectorQuery",
    "VectorSearchResult",
    
    # Implementations
    "PostgreSQLVectorStore",
    "ChromaDBVectorStore", 
    "FAISSVectorStore",
    
    # Manager
    "VectorStoreManager"
]
