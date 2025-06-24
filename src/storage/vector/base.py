"""
Vector Storage Base Classes

This module defines the abstract interfaces for vector storage implementations.
All vector storage backends must implement these interfaces.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import numpy as np


logger = logging.getLogger(__name__)


class VectorStoreType(Enum):
    """Vector store implementation types"""
    POSTGRESQL = "postgresql"
    CHROMADB = "chromadb"
    FAISS = "faiss"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"


class DistanceMetric(Enum):
    """Distance metrics for vector similarity"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class VectorDocument:
    """
    Represents a document with vector embedding.
    
    This is the core data structure for vector storage operations.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    collection: str = "default"
    
    # Document metadata
    title: Optional[str] = None
    source: Optional[str] = None
    document_type: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_count: Optional[int] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.embedding is not None and not isinstance(self.embedding, list):
            # Convert numpy arrays or other formats to list
            self.embedding = list(self.embedding)
    
    @property
    def embedding_dimension(self) -> Optional[int]:
        """Get the dimension of the embedding vector"""
        return len(self.embedding) if self.embedding else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "collection": self.collection,
            "title": self.title,
            "source": self.source,
            "document_type": self.document_type,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorDocument':
        """Create from dictionary"""
        doc = cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            collection=data.get("collection", "default"),
            title=data.get("title"),
            source=data.get("source"),
            document_type=data.get("document_type"),
            chunk_index=data.get("chunk_index"),
            chunk_count=data.get("chunk_count")
        )
        
        # Parse timestamps
        if "created_at" in data:
            doc.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            doc.updated_at = datetime.fromisoformat(data["updated_at"])
        
        return doc


@dataclass
class VectorQuery:
    """
    Represents a vector similarity search query.
    """
    query_vector: Optional[List[float]] = None
    query_text: Optional[str] = None
    collection: str = "default"
    limit: int = 10
    similarity_threshold: float = 0.0
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    
    # Filtering options
    metadata_filter: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_content: bool = True
    
    # Advanced options
    offset: int = 0
    hybrid_search: bool = False
    rerank: bool = False
    
    def __post_init__(self):
        """Validate query parameters"""
        if self.query_vector is None and self.query_text is None:
            raise ValueError("Either query_vector or query_text must be provided")
        
        if self.limit <= 0:
            raise ValueError("Limit must be positive")
        
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")


@dataclass
class VectorSearchResult:
    """
    Represents a single search result from vector similarity search.
    """
    document: VectorDocument
    similarity_score: float
    distance: float
    rank: int = 0
    
    # Additional result metadata
    search_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "document": self.document.to_dict(),
            "similarity_score": self.similarity_score,
            "distance": self.distance,
            "rank": self.rank,
            "search_metadata": self.search_metadata
        }


class VectorStore(ABC):
    """
    Abstract base class for vector storage implementations.
    
    This defines the interface that all vector storage backends must implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the vector store.
        
        Args:
            config: Configuration dictionary for the vector store
        """
        self.config = config
        self.store_type = VectorStoreType(config.get("type", "postgresql"))
        self.default_collection = config.get("default_collection", "default")
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the vector store connection"""
        pass
    
    @abstractmethod
    async def create_collection(self, collection_name: str, 
                               dimension: int, 
                               distance_metric: DistanceMetric = DistanceMetric.COSINE,
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection
            dimension: Dimension of the vectors
            distance_metric: Distance metric to use
            metadata: Optional collection metadata
            
        Returns:
            bool: True if collection was created successfully
        """
        pass
    
    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            bool: True if collection was deleted successfully
        """
        pass
    
    @abstractmethod
    async def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Returns:
            List[str]: List of collection names
        """
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List[str]: List of document IDs that were added
        """
        pass
    
    @abstractmethod
    async def update_documents(self, documents: List[VectorDocument]) -> List[str]:
        """
        Update existing documents in the vector store.
        
        Args:
            documents: List of documents to update
            
        Returns:
            List[str]: List of document IDs that were updated
        """
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str], 
                              collection: str = None) -> int:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            collection: Collection name (optional)
            
        Returns:
            int: Number of documents deleted
        """
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str, 
                          collection: str = None) -> Optional[VectorDocument]:
        """
        Get a document by ID.
        
        Args:
            document_id: ID of the document
            collection: Collection name (optional)
            
        Returns:
            VectorDocument or None if not found
        """
        pass
    
    @abstractmethod
    async def search_similar(self, query: VectorQuery) -> List[VectorSearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Vector query parameters
            
        Returns:
            List[VectorSearchResult]: List of search results
        """
        pass
    
    @abstractmethod
    async def count_documents(self, collection: str = None) -> int:
        """
        Count documents in a collection.
        
        Args:
            collection: Collection name (optional)
            
        Returns:
            int: Number of documents
        """
        pass
    
    # Utility methods that can be overridden by implementations
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the vector store.
        
        Returns:
            Dict with health information
        """
        try:
            collections = await self.list_collections()
            return {
                "status": "healthy",
                "collections": len(collections),
                "type": self.store_type.value,
                "initialized": self._initialized
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "type": self.store_type.value,
                "initialized": self._initialized
            }
    
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict with collection information or None if not found
        """
        try:
            count = await self.count_documents(collection_name)
            return {
                "name": collection_name,
                "document_count": count,
                "type": self.store_type.value
            }
        except Exception:
            return None
    
    def is_initialized(self) -> bool:
        """Check if the vector store is initialized"""
        return self._initialized
    
    def get_config(self) -> Dict[str, Any]:
        """Get the vector store configuration"""
        return self.config.copy()
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(type={self.store_type.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"{self.__class__.__name__}(type={self.store_type.value}, "
                f"initialized={self._initialized})")
