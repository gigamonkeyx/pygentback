"""
ChromaDB Vector Store Implementation

This module provides a ChromaDB-based vector storage implementation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import uuid

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

from .base import VectorStore, VectorDocument, VectorQuery, VectorSearchResult, DistanceMetric


logger = logging.getLogger(__name__)


class ChromaDBVectorStore(VectorStore):
    """
    ChromaDB vector store implementation.
    
    This implementation provides vector storage using ChromaDB,
    which is optimized for embeddings and similarity search.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ChromaDB vector store.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
        
        # ChromaDB configuration
        self.persist_directory = config.get("persist_directory", "./chroma_db")
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 8000)
        self.use_persistent = config.get("persistent", True)
        
        # ChromaDB client
        self.client = None
        self.collections_cache = {}
    
    async def initialize(self) -> None:
        """Initialize the ChromaDB vector store"""
        try:
            if self.use_persistent:
                # Use persistent client
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            else:
                # Use HTTP client
                self.client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=Settings(
                        anonymized_telemetry=False
                    )
                )
            
            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.client.heartbeat
            )
            
            self._initialized = True
            logger.info("ChromaDB vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB vector store: {str(e)}")
            raise
    
    async def close(self) -> None:
        """Close the ChromaDB vector store connection"""
        if self.client:
            # ChromaDB doesn't require explicit closing
            self.client = None
            self.collections_cache.clear()
        
        self._initialized = False
        logger.info("ChromaDB vector store closed")
    
    async def create_collection(self, collection_name: str, 
                               dimension: int, 
                               distance_metric: DistanceMetric = DistanceMetric.COSINE,
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection"""
        try:
            # Map distance metric to ChromaDB format
            if distance_metric == DistanceMetric.COSINE:
                chroma_metric = "cosine"
            elif distance_metric == DistanceMetric.EUCLIDEAN:
                chroma_metric = "l2"
            elif distance_metric == DistanceMetric.DOT_PRODUCT:
                chroma_metric = "ip"
            else:
                chroma_metric = "cosine"  # Default
            
            # Create collection
            collection = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.create_collection(
                    name=collection_name,
                    metadata={
                        "dimension": dimension,
                        "distance_metric": distance_metric.value,
                        **(metadata or {})
                    },
                    embedding_function=None  # We'll provide embeddings directly
                )
            )
            
            # Cache the collection
            self.collections_cache[collection_name] = collection
            
            logger.info(f"Created ChromaDB collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create ChromaDB collection {collection_name}: {str(e)}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.delete_collection(name=collection_name)
            )
            
            # Remove from cache
            self.collections_cache.pop(collection_name, None)
            
            logger.info(f"Deleted ChromaDB collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete ChromaDB collection {collection_name}: {str(e)}")
            return False
    
    async def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = await asyncio.get_event_loop().run_in_executor(
                None, self.client.list_collections
            )
            return [coll.name for coll in collections]
            
        except Exception as e:
            logger.error(f"Failed to list ChromaDB collections: {str(e)}")
            return []
    
    async def _get_collection(self, collection_name: str):
        """Get or cache a collection"""
        if collection_name not in self.collections_cache:
            try:
                collection = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.get_collection(name=collection_name)
                )
                self.collections_cache[collection_name] = collection
            except Exception as e:
                logger.error(f"Failed to get ChromaDB collection {collection_name}: {str(e)}")
                return None
        
        return self.collections_cache[collection_name]
    
    async def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Add documents to the vector store"""
        if not documents:
            return []
        
        added_ids = []
        
        try:
            # Group documents by collection
            collections = {}
            for doc in documents:
                if doc.collection not in collections:
                    collections[doc.collection] = []
                collections[doc.collection].append(doc)
            
            for collection_name, docs in collections.items():
                collection = await self._get_collection(collection_name)
                if not collection:
                    continue
                
                # Prepare data for ChromaDB
                ids = [doc.id for doc in docs]
                embeddings = [doc.embedding for doc in docs if doc.embedding]
                documents_text = [doc.content for doc in docs]
                metadatas = []
                
                for doc in docs:
                    metadata = {
                        "title": doc.title,
                        "source": doc.source,
                        "document_type": doc.document_type,
                        "chunk_index": doc.chunk_index,
                        "chunk_count": doc.chunk_count,
                        "created_at": doc.created_at.isoformat(),
                        "updated_at": doc.updated_at.isoformat(),
                        **doc.metadata
                    }
                    # Remove None values
                    metadata = {k: v for k, v in metadata.items() if v is not None}
                    metadatas.append(metadata)
                
                # Add to ChromaDB
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.add(
                        ids=ids,
                        embeddings=embeddings if embeddings else None,
                        documents=documents_text,
                        metadatas=metadatas
                    )
                )
                
                added_ids.extend(ids)
            
            logger.info(f"Added {len(added_ids)} documents to ChromaDB")
            return added_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}")
            return []
    
    async def update_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Update existing documents"""
        if not documents:
            return []
        
        updated_ids = []
        
        try:
            # Group documents by collection
            collections = {}
            for doc in documents:
                if doc.collection not in collections:
                    collections[doc.collection] = []
                collections[doc.collection].append(doc)
            
            for collection_name, docs in collections.items():
                collection = await self._get_collection(collection_name)
                if not collection:
                    continue
                
                # Prepare data for ChromaDB
                ids = [doc.id for doc in docs]
                embeddings = [doc.embedding for doc in docs if doc.embedding]
                documents_text = [doc.content for doc in docs]
                metadatas = []
                
                for doc in docs:
                    metadata = {
                        "title": doc.title,
                        "source": doc.source,
                        "document_type": doc.document_type,
                        "chunk_index": doc.chunk_index,
                        "chunk_count": doc.chunk_count,
                        "created_at": doc.created_at.isoformat(),
                        "updated_at": doc.updated_at.isoformat(),
                        **doc.metadata
                    }
                    # Remove None values
                    metadata = {k: v for k, v in metadata.items() if v is not None}
                    metadatas.append(metadata)
                
                # Update in ChromaDB
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.update(
                        ids=ids,
                        embeddings=embeddings if embeddings else None,
                        documents=documents_text,
                        metadatas=metadatas
                    )
                )
                
                updated_ids.extend(ids)
            
            logger.info(f"Updated {len(updated_ids)} documents in ChromaDB")
            return updated_ids
            
        except Exception as e:
            logger.error(f"Failed to update documents in ChromaDB: {str(e)}")
            return []
    
    async def delete_documents(self, document_ids: List[str], 
                              collection: str = None) -> int:
        """Delete documents from the vector store"""
        if not document_ids:
            return 0
        
        try:
            deleted_count = 0
            
            if collection:
                # Delete from specific collection
                coll = await self._get_collection(collection)
                if coll:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: coll.delete(ids=document_ids)
                    )
                    deleted_count = len(document_ids)
            else:
                # Delete from all collections
                collections = await self.list_collections()
                for coll_name in collections:
                    coll = await self._get_collection(coll_name)
                    if coll:
                        try:
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: coll.delete(ids=document_ids)
                            )
                            deleted_count += len(document_ids)
                        except Exception:
                            # Document might not exist in this collection
                            pass
            
            logger.info(f"Deleted {deleted_count} documents from ChromaDB")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete documents from ChromaDB: {str(e)}")
            return 0
    
    async def get_document(self, document_id: str, 
                          collection: str = None) -> Optional[VectorDocument]:
        """Get a document by ID"""
        try:
            collections_to_search = [collection] if collection else await self.list_collections()
            
            for coll_name in collections_to_search:
                coll = await self._get_collection(coll_name)
                if not coll:
                    continue
                
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: coll.get(ids=[document_id], include=["embeddings", "documents", "metadatas"])
                    )
                    
                    if result["ids"]:
                        metadata = result["metadatas"][0] if result["metadatas"] else {}
                        
                        return VectorDocument(
                            id=result["ids"][0],
                            content=result["documents"][0] if result["documents"] else "",
                            embedding=result["embeddings"][0] if result["embeddings"] else None,
                            metadata={k: v for k, v in metadata.items() 
                                    if k not in ["title", "source", "document_type", "chunk_index", 
                                               "chunk_count", "created_at", "updated_at"]},
                            collection=coll_name,
                            title=metadata.get("title"),
                            source=metadata.get("source"),
                            document_type=metadata.get("document_type"),
                            chunk_index=metadata.get("chunk_index"),
                            chunk_count=metadata.get("chunk_count")
                        )
                except Exception:
                    # Document not in this collection
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id} from ChromaDB: {str(e)}")
            return None
    
    async def search_similar(self, query: VectorQuery) -> List[VectorSearchResult]:
        """Search for similar documents"""
        try:
            if not query.query_vector:
                return []
            
            collection = await self._get_collection(query.collection)
            if not collection:
                return []
            
            # Prepare where clause for metadata filtering
            where_clause = None
            if query.metadata_filter:
                where_clause = query.metadata_filter
            
            # Perform similarity search
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: collection.query(
                    query_embeddings=[query.query_vector],
                    n_results=query.limit,
                    where=where_clause,
                    include=["embeddings", "documents", "metadatas", "distances"]
                )
            )
            
            search_results = []
            
            if results["ids"] and results["ids"][0]:
                for i, (doc_id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
                    # Convert distance to similarity score
                    if query.distance_metric == DistanceMetric.COSINE:
                        similarity = 1.0 - distance
                    elif query.distance_metric == DistanceMetric.DOT_PRODUCT:
                        similarity = distance  # Higher is better
                    else:
                        similarity = 1.0 / (1.0 + distance)
                    
                    # Skip results below threshold
                    if similarity < query.similarity_threshold:
                        continue
                    
                    metadata = results["metadatas"][0][i] if results["metadatas"] and results["metadatas"][0] else {}
                    
                    document = VectorDocument(
                        id=doc_id,
                        content=results["documents"][0][i] if query.include_content and results["documents"] else "",
                        embedding=results["embeddings"][0][i] if results["embeddings"] else None,
                        metadata={k: v for k, v in metadata.items() 
                                if k not in ["title", "source", "document_type", "chunk_index", 
                                           "chunk_count", "created_at", "updated_at"]} if query.include_metadata else {},
                        collection=query.collection,
                        title=metadata.get("title"),
                        source=metadata.get("source"),
                        document_type=metadata.get("document_type"),
                        chunk_index=metadata.get("chunk_index"),
                        chunk_count=metadata.get("chunk_count")
                    )
                    
                    result = VectorSearchResult(
                        document=document,
                        similarity_score=similarity,
                        distance=distance,
                        rank=i + 1
                    )
                    
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search similar documents in ChromaDB: {str(e)}")
            return []
    
    async def count_documents(self, collection: str = None) -> int:
        """Count documents in a collection"""
        try:
            if collection:
                coll = await self._get_collection(collection)
                if coll:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: coll.count()
                    )
                    return result
                return 0
            else:
                # Count across all collections
                total = 0
                collections = await self.list_collections()
                for coll_name in collections:
                    coll = await self._get_collection(coll_name)
                    if coll:
                        count = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: coll.count()
                        )
                        total += count
                return total
                
        except Exception as e:
            logger.error(f"Failed to count documents in ChromaDB: {str(e)}")
            return 0
