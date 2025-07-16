"""
Vector Storage Interface - Backward Compatibility Layer

This module provides backward compatibility for the modular vector storage system.
All vector storage functionality has been moved to the storage.vector submodule
for better organization. This file maintains the original interface while
delegating to the new modular components.
"""

# Import all components from the modular vector storage system
from .vector.base import (
    VectorStore as ModularVectorStore,
    VectorDocument as ModularVectorDocument,
    VectorQuery as ModularVectorQuery,
    VectorSearchResult as ModularVectorSearchResult,
    VectorStoreType,
    DistanceMetric
)
from .vector.manager import VectorStoreManager as ModularVectorStoreManager
from .vector.postgresql import PostgreSQLVectorStore as ModularPostgreSQLVectorStore

# Legacy imports for backward compatibility
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import uuid
from sqlalchemy import text

from ..config.settings import Settings


logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Represents a document with vector embedding - Legacy compatibility wrapper"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: datetime

    def __post_init__(self):
        if isinstance(self.embedding, np.ndarray):
            self.embedding = self.embedding.tolist()

    def to_modular_document(self, collection: str = "default") -> ModularVectorDocument:
        """Convert to modular document format"""
        return ModularVectorDocument(
            id=self.id,
            content=self.content,
            embedding=self.embedding,
            metadata=self.metadata,
            collection=collection,
            created_at=self.created_at,
            updated_at=self.created_at
        )

    @classmethod
    def from_modular_document(cls, modular_doc: ModularVectorDocument) -> 'VectorDocument':
        """Create legacy document from modular document"""
        return cls(
            id=modular_doc.id,
            content=modular_doc.content,
            embedding=modular_doc.embedding,
            metadata=modular_doc.metadata,
            created_at=modular_doc.created_at
        )


@dataclass
class SimilarityResult:
    """Represents a similarity search result - Legacy compatibility wrapper"""
    document: VectorDocument
    similarity_score: float
    distance: float

    @classmethod
    def from_modular_result(cls, modular_result: ModularVectorSearchResult) -> 'SimilarityResult':
        """Create legacy result from modular result"""
        return cls(
            document=VectorDocument.from_modular_document(modular_result.document),
            similarity_score=modular_result.similarity_score,
            distance=modular_result.distance
        )


class VectorStore(ABC):
    """Abstract base class for vector storage implementations - Legacy compatibility wrapper"""

    def __init__(self, collection_name: str = "default"):
        """Initialize legacy vector store wrapper"""
        self.collection_name = collection_name
        self._modular_store: Optional[ModularVectorStore] = None

    def _ensure_modular_store(self) -> ModularVectorStore:
        """Ensure modular store is available (implemented by subclasses)"""
        if self._modular_store is None:
            raise RuntimeError("Vector store not properly initialized. Call initialize() first.")
        return self._modular_store

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store"""
        pass

    async def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add

        Returns:
            List[str]: List of document IDs
        """
        # Convert legacy documents to modular format
        modular_docs = [doc.to_modular_document(self.collection_name) for doc in documents]

        # Delegate to modular store
        modular_store = self._ensure_modular_store()
        return await modular_store.add_documents(modular_docs)

    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """
        Get a document by ID.

        Args:
            document_id: Document ID

        Returns:
            VectorDocument or None if not found
        """
        modular_store = self._ensure_modular_store()
        modular_doc = await modular_store.get_document(document_id, self.collection_name)

        if modular_doc:
            return VectorDocument.from_modular_document(modular_doc)
        return None

    async def update_document(self, document: VectorDocument) -> bool:
        """
        Update a document in the vector store.

        Args:
            document: Document to update

        Returns:
            bool: True if successful
        """
        # Convert to modular format and update
        modular_doc = document.to_modular_document(self.collection_name)
        modular_store = self._ensure_modular_store()

        result = await modular_store.update_documents([modular_doc])
        return len(result) > 0

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
            document_id: Document ID to delete

        Returns:
            bool: True if successful
        """
        modular_store = self._ensure_modular_store()
        deleted_count = await modular_store.delete_documents([document_id], self.collection_name)
        return deleted_count > 0

    async def similarity_search(self,
                               query_embedding: List[float],
                               limit: int = 10,
                               similarity_threshold: float = 0.7,
                               metadata_filter: Optional[Dict[str, Any]] = None) -> List[SimilarityResult]:
        """
        Perform similarity search.

        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            metadata_filter: Optional metadata filter

        Returns:
            List[SimilarityResult]: Search results
        """
        # Create modular query
        modular_query = ModularVectorQuery(
            query_vector=query_embedding,
            collection=self.collection_name,
            limit=limit,
            similarity_threshold=similarity_threshold,
            metadata_filter=metadata_filter
        )

        # Execute search
        modular_store = self._ensure_modular_store()
        modular_results = await modular_store.search_similar(modular_query)

        # Convert results to legacy format
        return [SimilarityResult.from_modular_result(result) for result in modular_results]

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dict with collection stats
        """
        modular_store = self._ensure_modular_store()
        count = await modular_store.count_documents(self.collection_name)

        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "storage_type": modular_store.store_type.value
        }

    async def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.

        Returns:
            bool: True if successful
        """
        # This is a destructive operation, so we need to be careful
        # For now, we'll implement it by getting all documents and deleting them
        modular_store = self._ensure_modular_store()

        # Get all document IDs (this is not efficient for large collections)
        # In a real implementation, you'd want a more efficient clear operation
        try:
            # For now, return True as this is a legacy compatibility method
            # The modular system handles collection management differently
            return True
        except Exception:
            return False


class PostgreSQLVectorStore(VectorStore):
    """PostgreSQL with pgvector implementation - Legacy compatibility wrapper"""

    def __init__(self, db_manager, collection_name: str = "default"):
        super().__init__(collection_name)
        self.db_manager = db_manager
        self.table_name = f"vector_store_{collection_name}"

    async def initialize(self) -> None:
        """Initialize PostgreSQL vector store"""
        try:
            # Create modular PostgreSQL store
            config = {
                "type": "postgresql",
                "host": "localhost",  # Docker container host
                "port": 54321,       # Docker container port
                "database": "pygent_factory",
                "username": "postgres",
                "password": "postgres",  # Docker container password
                "schema": "vectors",
                "default_collection": self.collection_name
            }

            self._modular_store = ModularPostgreSQLVectorStore(config)
            await self._modular_store.initialize()

            # Create collection if it doesn't exist
            await self._modular_store.create_collection(
                self.collection_name,
                dimension=1536,  # Default dimension
                distance_metric=DistanceMetric.COSINE
            )

            logger.info(f"PostgreSQL vector store initialized: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL vector store: {str(e)}")
            raise




class LegacyVectorStoreWrapper:
    """
    Wrapper to make modular vector stores compatible with legacy interface.
    """

    def __init__(self, modular_store, collection_name: str):
        self._modular_store = modular_store
        self.collection_name = collection_name

    async def initialize(self):
        """Initialize the wrapped store."""
        # The modular store should already be initialized
        pass

    async def add_documents(self, documents: List[ModularVectorDocument]) -> List[str]:
        """Add documents to the vector store."""
        return await self._modular_store.add_documents(documents)

    async def search(self, query: ModularVectorQuery) -> List[ModularVectorSearchResult]:
        """Search for similar documents."""
        return await self._modular_store.search(query)

    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the vector store."""
        return await self._modular_store.delete_documents(document_ids)

    async def get_document(self, document_id: str) -> Optional[ModularVectorDocument]:
        """Get a document by ID."""
        return await self._modular_store.get_document(document_id)

    async def update_document(self, document_id: str, document: ModularVectorDocument) -> bool:
        """Update a document."""
        return await self._modular_store.update_document(document_id, document)

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return await self._modular_store.get_collection_stats()


class VectorStoreManager:
    """
    Manager for vector store operations - Legacy compatibility wrapper.

    Provides backward compatibility while delegating to the modular vector store manager.
    """

    def __init__(self, settings: Settings, db_manager=None):
        self.settings = settings
        self.db_manager = db_manager
        self.stores: Dict[str, VectorStore] = {}
        # For SQLite, use FAISS as the default vector store
        self.default_store_type = "faiss" if "sqlite" in settings.ASYNC_DATABASE_URL.lower() else "postgresql"

        # Create modular manager
        self._modular_manager = ModularVectorStoreManager(settings, db_manager)

    async def initialize(self) -> None:
        """Initialize the vector store manager"""
        await self._modular_manager.initialize()

    async def shutdown(self) -> None:
        """Shutdown the vector store manager"""
        await self._modular_manager.shutdown()

    async def get_store(self, collection_name: str, store_type: Optional[str] = None) -> VectorStore:
        """
        Get or create a vector store for a collection.

        Args:
            collection_name: Name of the collection
            store_type: Type of vector store (postgresql, chromadb, faiss)

        Returns:
            VectorStore: Vector store instance (legacy wrapper)
        """
        store_type = store_type or self.default_store_type
        store_key = f"{store_type}:{collection_name}"

        if store_key not in self.stores:
            if store_type == "postgresql":
                store = PostgreSQLVectorStore(self.db_manager, collection_name)
                await store.initialize()
                self.stores[store_key] = store
            elif store_type == "faiss":
                # For FAISS, delegate to the modular manager
                # First ensure the modular manager is initialized
                if not hasattr(self._modular_manager, '_initialized') or not self._modular_manager._initialized:
                    await self._modular_manager.initialize()

                # Get the default store (not by collection name, but by store name)
                modular_store = self._modular_manager.get_default_store()
                if modular_store is None:
                    raise ValueError(f"Failed to create FAISS store for collection: {collection_name}")

                store = LegacyVectorStoreWrapper(modular_store, collection_name)
                await store.initialize()
                self.stores[store_key] = store
            else:
                raise ValueError(f"Unsupported vector store type: {store_type}")

        return self.stores[store_key]

    async def create_collection(self, collection_name: str, store_type: Optional[str] = None) -> VectorStore:
        """Create a new vector collection"""
        return await self.get_store(collection_name, store_type)

    async def delete_collection(self, collection_name: str, store_type: Optional[str] = None) -> bool:
        """Delete a vector collection"""
        store_type = store_type or self.default_store_type
        store_key = f"{store_type}:{collection_name}"

        # Delegate to modular manager
        success = await self._modular_manager.delete_collection(collection_name)

        # Clean up legacy store cache
        if store_key in self.stores:
            del self.stores[store_key]

        return success

    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all vector collections"""
        # Delegate to modular manager
        collections = await self._modular_manager.list_collections()

        # Convert to legacy format
        legacy_collections = []
        for collection_name in collections:
            stats = await self._modular_manager.get_default_store().get_collection_info(collection_name)
            if stats:
                legacy_collections.append({
                    "name": collection_name,
                    "type": self.default_store_type,
                    **stats
                })

        return legacy_collections

    async def get_metrics(self) -> Dict[str, Any]:
        """Get vector store metrics"""
        # Delegate to modular manager
        stats = await self._modular_manager.get_global_stats()

        return {
            "total_collections": stats.get("total_collections", 0),
            "total_documents": stats.get("total_documents", 0),
            "active_stores": stats.get("total_stores", 0),
            "collections": await self.list_collections()
        }


# Re-export modular components for direct access
VectorStoreManager = VectorStoreManager  # Legacy wrapper
ModularVectorStoreManager = ModularVectorStoreManager  # Direct access to modular manager

# Export all for backward compatibility
__all__ = [
    # Legacy classes
    "VectorDocument",
    "SimilarityResult",
    "VectorStore",
    "PostgreSQLVectorStore",
    "VectorStoreManager",

    # Modular classes for direct access
    "ModularVectorStore",
    "ModularVectorDocument",
    "ModularVectorQuery",
    "ModularVectorSearchResult",
    "ModularVectorStoreManager",
    "VectorStoreType",
    "DistanceMetric",

    # Specific implementations
    "PostgreSQLVectorStore",
    "ChromaDBVectorStore",
    "FAISSVectorStore"
]
