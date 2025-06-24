"""
Vector Store Manager

This module provides a unified interface for managing multiple vector store implementations.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Type, Union
from datetime import datetime

from .base import VectorStore, VectorDocument, VectorQuery, VectorSearchResult, VectorStoreType, DistanceMetric
from .postgresql import PostgreSQLVectorStore
from .chromadb import ChromaDBVectorStore
from .faiss import FAISSVectorStore
try:
    from config.settings import VectorStoreSettings
except ImportError:
    # Fallback for when config module is not available
    class VectorStoreSettings:
        def __init__(self):
            self.CHROMADB_COLLECTION_NAME = "documents"


logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Vector Store Manager
    
    Provides a unified interface for managing multiple vector store implementations
    and routing operations to the appropriate backend.
    """
    
    # Registry of available vector store implementations
    STORE_IMPLEMENTATIONS: Dict[VectorStoreType, Type[VectorStore]] = {
        VectorStoreType.POSTGRESQL: PostgreSQLVectorStore,
        VectorStoreType.CHROMADB: ChromaDBVectorStore,
        VectorStoreType.FAISS: FAISSVectorStore,
    }
    
    def __init__(self, settings, db_manager=None):
        """
        Initialize the vector store manager.
        
        Args:
            settings: Application settings
            db_manager: Database manager (for PostgreSQL backend)
        """
        self.settings = settings
        self.db_manager = db_manager
        
        # Active vector stores
        self.stores: Dict[str, VectorStore] = {}
        self.default_store_name = "default"
        
        # Configuration
        self.vector_config = getattr(settings, 'vector', VectorStoreSettings())
        # For SQLite, use FAISS as the default vector store
        default_type = "faiss" if "sqlite" in settings.ASYNC_DATABASE_URL.lower() else "postgresql"
        self.default_store_type = VectorStoreType(default_type)
        
        # Initialize flag
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the vector store manager"""
        try:
            # Initialize default vector store
            await self._initialize_default_store()
            
            # Initialize additional configured stores
            # For now, skip additional stores since we're using simple config
            # TODO: Add support for additional stores configuration
            
            self._initialized = True
            logger.info("Vector store manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store manager: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the vector store manager"""
        try:
            # Close all vector stores
            for store in self.stores.values():
                await store.close()
            
            self.stores.clear()
            self._initialized = False
            
            logger.info("Vector store manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during vector store manager shutdown: {str(e)}")
    
    async def _initialize_default_store(self) -> None:
        """Initialize the default vector store"""
        # Get default store configuration
        default_config = self._get_default_store_config()
        
        # Create and initialize default store
        store = await self._create_store(self.default_store_type, default_config)
        await store.initialize()
        
        self.stores[self.default_store_name] = store
        logger.info(f"Initialized default vector store: {self.default_store_type.value}")
    
    def _get_default_store_config(self) -> Dict[str, Any]:
        """Get configuration for the default vector store"""
        base_config = {
            "type": self.default_store_type.value,
            "default_collection": getattr(self.vector_config, 'CHROMADB_COLLECTION_NAME', "documents")
        }

        if self.default_store_type == VectorStoreType.POSTGRESQL:
            # PostgreSQL configuration
            db_config = getattr(self.settings, 'database', None)
            if db_config:
                base_config.update({
                    "host": getattr(db_config, 'HOST', "localhost"),
                    "port": getattr(db_config, 'PORT', 5432),
                    "database": getattr(db_config, 'NAME', "pygent_factory"),
                    "username": getattr(db_config, 'USER', "postgres"),
                    "password": getattr(db_config, 'PASSWORD', ""),
                    "schema": "vectors",
                    "max_connections": 10
                })

        elif self.default_store_type == VectorStoreType.CHROMADB:
            # ChromaDB configuration
            base_config.update({
                "persist_directory": "./chroma_db",
                "host": "localhost",
                "port": 8000,
                "persistent": True
            })

        elif self.default_store_type == VectorStoreType.FAISS:
            # FAISS configuration
            base_config.update({
                "persist_directory": "./faiss_db",
                "index_type": "IVFFlat",
                "nlist": 100,
                "nprobe": 10
            })

        return base_config
    
    async def _create_store(self, store_type: VectorStoreType, config: Dict[str, Any]) -> VectorStore:
        """Create a vector store instance"""
        if store_type not in self.STORE_IMPLEMENTATIONS:
            raise ValueError(f"Unsupported vector store type: {store_type.value}")
        
        store_class = self.STORE_IMPLEMENTATIONS[store_type]
        return store_class(config)
    
    async def add_store(self, store_name: str, config: Dict[str, Any]) -> bool:
        """
        Add a new vector store.
        
        Args:
            store_name: Name for the store
            config: Store configuration
            
        Returns:
            bool: True if store was added successfully
        """
        try:
            store_type = VectorStoreType(config.get("type", "postgresql"))
            store = await self._create_store(store_type, config)
            await store.initialize()
            
            self.stores[store_name] = store
            logger.info(f"Added vector store: {store_name} ({store_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vector store {store_name}: {str(e)}")
            return False
    
    async def remove_store(self, store_name: str) -> bool:
        """
        Remove a vector store.
        
        Args:
            store_name: Name of the store to remove
            
        Returns:
            bool: True if store was removed successfully
        """
        try:
            if store_name == self.default_store_name:
                raise ValueError("Cannot remove default vector store")
            
            if store_name in self.stores:
                store = self.stores[store_name]
                await store.close()
                del self.stores[store_name]
                
                logger.info(f"Removed vector store: {store_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove vector store {store_name}: {str(e)}")
            return False
    
    def get_store(self, store_name: str = None) -> Optional[VectorStore]:
        """
        Get a vector store by name.
        
        Args:
            store_name: Name of the store (default: default store)
            
        Returns:
            VectorStore or None if not found
        """
        if store_name is None:
            store_name = self.default_store_name
        
        return self.stores.get(store_name)
    
    def list_stores(self) -> List[str]:
        """Get list of available store names"""
        return list(self.stores.keys())
    
    def get_default_store(self) -> Optional[VectorStore]:
        """Get the default vector store"""
        return self.stores.get(self.default_store_name)
    
    # Convenience methods that delegate to the default store
    async def create_collection(self, collection_name: str, 
                               dimension: int, 
                               distance_metric: DistanceMetric = DistanceMetric.COSINE,
                               metadata: Optional[Dict[str, Any]] = None,
                               store_name: str = None) -> bool:
        """Create a collection in the specified store"""
        store = self.get_store(store_name)
        if not store:
            return False
        
        return await store.create_collection(collection_name, dimension, distance_metric, metadata)
    
    async def delete_collection(self, collection_name: str, store_name: str = None) -> bool:
        """Delete a collection from the specified store"""
        store = self.get_store(store_name)
        if not store:
            return False
        
        return await store.delete_collection(collection_name)
    
    async def list_collections(self, store_name: str = None) -> List[str]:
        """List collections in the specified store"""
        store = self.get_store(store_name)
        if not store:
            return []
        
        return await store.list_collections()
    
    async def add_documents(self, documents: List[VectorDocument], 
                           store_name: str = None) -> List[str]:
        """Add documents to the specified store"""
        store = self.get_store(store_name)
        if not store:
            return []
        
        return await store.add_documents(documents)
    
    async def update_documents(self, documents: List[VectorDocument], 
                              store_name: str = None) -> List[str]:
        """Update documents in the specified store"""
        store = self.get_store(store_name)
        if not store:
            return []
        
        return await store.update_documents(documents)
    
    async def delete_documents(self, document_ids: List[str], 
                              collection: str = None, 
                              store_name: str = None) -> int:
        """Delete documents from the specified store"""
        store = self.get_store(store_name)
        if not store:
            return 0
        
        return await store.delete_documents(document_ids, collection)
    
    async def get_document(self, document_id: str, 
                          collection: str = None, 
                          store_name: str = None) -> Optional[VectorDocument]:
        """Get a document from the specified store"""
        store = self.get_store(store_name)
        if not store:
            return None
        
        return await store.get_document(document_id, collection)
    
    async def search_similar(self, query: VectorQuery,
                            store_name: str = None) -> List[VectorSearchResult]:
        """Search for similar documents in the specified store"""
        store = self.get_store(store_name)
        if not store:
            return []

        return await store.search_similar(query)

    async def search(self, query_embedding: List[float],
                    collection_name: str = "default",
                    limit: int = 10,
                    threshold: float = 0.7,
                    store_name: str = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using embedding vector.

        Args:
            query_embedding: Query vector embedding
            collection_name: Collection to search in
            limit: Maximum number of results
            threshold: Similarity threshold
            store_name: Store to search in (default: default store)

        Returns:
            List of search results with metadata
        """
        store = self.get_store(store_name)
        if not store:
            return []

        # Create VectorQuery object
        vector_query = VectorQuery(
            vector=query_embedding,
            collection=collection_name,
            limit=limit,
            threshold=threshold
        )

        # Execute search
        results = await store.search_similar(vector_query)

        # Convert to dict format expected by validation
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.metadata.get("title", "Untitled"),
                "content": result.metadata.get("content", ""),
                "score": result.score,
                "metadata": result.metadata,
                "source": result.metadata.get("source", "unknown")
            })

        return formatted_results
    
    async def count_documents(self, collection: str = None, 
                             store_name: str = None) -> int:
        """Count documents in the specified store"""
        store = self.get_store(store_name)
        if not store:
            return 0
        
        return await store.count_documents(collection)
    
    # Multi-store operations
    async def search_all_stores(self, query: VectorQuery) -> Dict[str, List[VectorSearchResult]]:
        """Search across all vector stores"""
        results = {}
        
        for store_name, store in self.stores.items():
            try:
                store_results = await store.search_similar(query)
                results[store_name] = store_results
            except Exception as e:
                logger.error(f"Search failed in store {store_name}: {str(e)}")
                results[store_name] = []
        
        return results
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics across all vector stores"""
        stats = {
            "total_stores": len(self.stores),
            "stores": {},
            "total_collections": 0,
            "total_documents": 0
        }
        
        for store_name, store in self.stores.items():
            try:
                collections = await store.list_collections()
                total_docs = await store.count_documents()
                health = await store.health_check()
                
                store_stats = {
                    "type": store.store_type.value,
                    "collections": len(collections),
                    "documents": total_docs,
                    "health": health,
                    "initialized": store.is_initialized()
                }
                
                stats["stores"][store_name] = store_stats
                stats["total_collections"] += len(collections)
                stats["total_documents"] += total_docs
                
            except Exception as e:
                logger.error(f"Failed to get stats for store {store_name}: {str(e)}")
                stats["stores"][store_name] = {
                    "error": str(e),
                    "initialized": False
                }
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all vector stores"""
        health_info = {
            "status": "healthy",
            "stores": {},
            "total_stores": len(self.stores),
            "healthy_stores": 0,
            "unhealthy_stores": 0
        }
        
        for store_name, store in self.stores.items():
            try:
                store_health = await store.health_check()
                health_info["stores"][store_name] = store_health
                
                if store_health.get("status") == "healthy":
                    health_info["healthy_stores"] += 1
                else:
                    health_info["unhealthy_stores"] += 1
                    
            except Exception as e:
                health_info["stores"][store_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_info["unhealthy_stores"] += 1
        
        # Determine overall status
        if health_info["unhealthy_stores"] > 0:
            if health_info["healthy_stores"] > 0:
                health_info["status"] = "degraded"
            else:
                health_info["status"] = "unhealthy"
        
        return health_info
    
    def is_initialized(self) -> bool:
        """Check if the manager is initialized"""
        return self._initialized
    
    def __str__(self) -> str:
        """String representation"""
        return f"VectorStoreManager(stores={len(self.stores)}, default={self.default_store_type.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"VectorStoreManager(stores={list(self.stores.keys())}, "
                f"default_type={self.default_store_type.value}, "
                f"initialized={self._initialized})")
