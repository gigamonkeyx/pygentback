"""
FAISS Vector Store Implementation

This module provides a FAISS-based vector storage implementation for high-performance
similarity search with local storage.
"""

import asyncio
import logging
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    np = None

from .base import VectorStore, VectorDocument, VectorQuery, VectorSearchResult, DistanceMetric


logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """
    FAISS vector store implementation.
    
    This implementation provides high-performance vector storage using FAISS
    for similarity search with local file-based persistence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FAISS vector store.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu or faiss-gpu")
        
        # FAISS configuration
        self.persist_directory = config.get("persist_directory", "./faiss_db")
        self.index_type = config.get("index_type", "IVFFlat")  # IVFFlat, HNSW, Flat
        self.nlist = config.get("nlist", 100)  # Number of clusters for IVF
        self.nprobe = config.get("nprobe", 10)  # Number of clusters to search
        
        # Collections storage
        self.collections = {}  # collection_name -> collection_data
        self.indexes = {}      # collection_name -> faiss_index
        self.documents = {}    # collection_name -> {doc_id -> VectorDocument}
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
    
    async def initialize(self) -> None:
        """Initialize the FAISS vector store"""
        try:
            # Load existing collections from disk
            await self._load_collections()
            
            self._initialized = True
            logger.info("FAISS vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS vector store: {str(e)}")
            raise
    
    async def close(self) -> None:
        """Close the FAISS vector store"""
        try:
            # Save all collections to disk
            await self._save_collections()
            
            # Clear in-memory data
            self.collections.clear()
            self.indexes.clear()
            self.documents.clear()
            
            self._initialized = False
            logger.info("FAISS vector store closed")
            
        except Exception as e:
            logger.error(f"Error closing FAISS vector store: {str(e)}")
    
    async def _load_collections(self) -> None:
        """Load collections from disk"""
        try:
            collections_file = os.path.join(self.persist_directory, "collections.json")
            if os.path.exists(collections_file):
                with open(collections_file, 'r') as f:
                    self.collections = json.load(f)
            
            # Load each collection's index and documents
            for collection_name in self.collections:
                await self._load_collection(collection_name)
                
        except Exception as e:
            logger.error(f"Failed to load collections: {str(e)}")
    
    async def _save_collections(self) -> None:
        """Save collections to disk"""
        try:
            # Save collections metadata
            collections_file = os.path.join(self.persist_directory, "collections.json")
            with open(collections_file, 'w') as f:
                json.dump(self.collections, f, indent=2)
            
            # Save each collection's index and documents
            for collection_name in self.collections:
                await self._save_collection(collection_name)
                
        except Exception as e:
            logger.error(f"Failed to save collections: {str(e)}")
    
    async def _load_collection(self, collection_name: str) -> None:
        """Load a specific collection from disk"""
        try:
            collection_dir = os.path.join(self.persist_directory, collection_name)
            if not os.path.exists(collection_dir):
                return
            
            # Load FAISS index
            index_file = os.path.join(collection_dir, "index.faiss")
            if os.path.exists(index_file):
                self.indexes[collection_name] = faiss.read_index(index_file)
            
            # Load documents
            docs_file = os.path.join(collection_dir, "documents.pkl")
            if os.path.exists(docs_file):
                with open(docs_file, 'rb') as f:
                    self.documents[collection_name] = pickle.load(f)
            else:
                self.documents[collection_name] = {}
                
        except Exception as e:
            logger.error(f"Failed to load collection {collection_name}: {str(e)}")
    
    async def _save_collection(self, collection_name: str) -> None:
        """Save a specific collection to disk"""
        try:
            collection_dir = os.path.join(self.persist_directory, collection_name)
            os.makedirs(collection_dir, exist_ok=True)
            
            # Save FAISS index
            if collection_name in self.indexes:
                index_file = os.path.join(collection_dir, "index.faiss")
                faiss.write_index(self.indexes[collection_name], index_file)
            
            # Save documents
            if collection_name in self.documents:
                docs_file = os.path.join(collection_dir, "documents.pkl")
                with open(docs_file, 'wb') as f:
                    pickle.dump(self.documents[collection_name], f)
                    
        except Exception as e:
            logger.error(f"Failed to save collection {collection_name}: {str(e)}")
    
    def _create_faiss_index(self, dimension: int, distance_metric: DistanceMetric) -> faiss.Index:
        """Create a FAISS index based on configuration"""
        if self.index_type == "Flat":
            if distance_metric == DistanceMetric.COSINE:
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
            elif distance_metric == DistanceMetric.EUCLIDEAN:
                index = faiss.IndexFlatL2(dimension)
            else:
                index = faiss.IndexFlatIP(dimension)  # Default
                
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(dimension)
            if distance_metric == DistanceMetric.COSINE:
                index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
            elif distance_metric == DistanceMetric.EUCLIDEAN:
                index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_L2)
            else:
                index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
                
        elif self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is M parameter
            if distance_metric == DistanceMetric.COSINE:
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            elif distance_metric == DistanceMetric.EUCLIDEAN:
                index.metric_type = faiss.METRIC_L2
            else:
                index.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            # Default to Flat
            index = faiss.IndexFlatIP(dimension)
        
        return index
    
    async def create_collection(self, collection_name: str, 
                               dimension: int, 
                               distance_metric: DistanceMetric = DistanceMetric.COSINE,
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection"""
        try:
            # Create collection metadata
            self.collections[collection_name] = {
                "dimension": dimension,
                "distance_metric": distance_metric.value,
                "metadata": metadata or {},
                "created_at": datetime.utcnow().isoformat(),
                "document_count": 0
            }
            
            # Create FAISS index
            index = self._create_faiss_index(dimension, distance_metric)
            self.indexes[collection_name] = index
            
            # Initialize documents storage
            self.documents[collection_name] = {}
            
            # Save to disk
            await self._save_collection(collection_name)
            
            logger.info(f"Created FAISS collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create FAISS collection {collection_name}: {str(e)}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            # Remove from memory
            self.collections.pop(collection_name, None)
            self.indexes.pop(collection_name, None)
            self.documents.pop(collection_name, None)
            
            # Remove from disk
            collection_dir = os.path.join(self.persist_directory, collection_name)
            if os.path.exists(collection_dir):
                import shutil
                shutil.rmtree(collection_dir)
            
            # Update collections file
            await self._save_collections()
            
            logger.info(f"Deleted FAISS collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete FAISS collection {collection_name}: {str(e)}")
            return False
    
    async def list_collections(self) -> List[str]:
        """List all collections"""
        return list(self.collections.keys())
    
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
                if collection_name not in self.indexes:
                    logger.warning(f"Collection {collection_name} does not exist")
                    continue
                
                index = self.indexes[collection_name]
                doc_storage = self.documents[collection_name]
                
                # Prepare embeddings and document mapping
                embeddings = []
                doc_ids = []
                
                for doc in docs:
                    if doc.embedding:
                        embeddings.append(doc.embedding)
                        doc_ids.append(doc.id)
                        doc_storage[doc.id] = doc
                
                if embeddings:
                    # Convert to numpy array
                    embeddings_array = np.array(embeddings, dtype=np.float32)
                    
                    # Normalize for cosine similarity if needed
                    collection_info = self.collections[collection_name]
                    if collection_info["distance_metric"] == "cosine":
                        faiss.normalize_L2(embeddings_array)
                    
                    # Add to index
                    if hasattr(index, 'is_trained') and not index.is_trained:
                        # Train index if needed (for IVF indexes)
                        if embeddings_array.shape[0] >= self.nlist:
                            index.train(embeddings_array)
                        else:
                            logger.warning(f"Not enough vectors to train index for {collection_name}")
                    
                    index.add(embeddings_array)
                    
                    # Update document count
                    self.collections[collection_name]["document_count"] = len(doc_storage)
                    
                    added_ids.extend(doc_ids)
            
            # Save to disk
            await self._save_collections()
            
            logger.info(f"Added {len(added_ids)} documents to FAISS")
            return added_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to FAISS: {str(e)}")
            return []
    
    async def update_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Update existing documents"""
        # For FAISS, we need to rebuild the index for updates
        # This is a simplified implementation - in production, you might want
        # to use a more sophisticated approach
        
        updated_ids = []
        
        try:
            # Group documents by collection
            collections = {}
            for doc in documents:
                if doc.collection not in collections:
                    collections[doc.collection] = []
                collections[doc.collection].append(doc)
            
            for collection_name, docs in collections.items():
                if collection_name not in self.documents:
                    continue
                
                doc_storage = self.documents[collection_name]
                
                # Update documents in storage
                for doc in docs:
                    if doc.id in doc_storage:
                        doc_storage[doc.id] = doc
                        updated_ids.append(doc.id)
                
                # Rebuild index with all documents
                await self._rebuild_collection_index(collection_name)
            
            # Save to disk
            await self._save_collections()
            
            logger.info(f"Updated {len(updated_ids)} documents in FAISS")
            return updated_ids
            
        except Exception as e:
            logger.error(f"Failed to update documents in FAISS: {str(e)}")
            return []
    
    async def _rebuild_collection_index(self, collection_name: str) -> None:
        """Rebuild the FAISS index for a collection"""
        if collection_name not in self.collections:
            return
        
        collection_info = self.collections[collection_name]
        doc_storage = self.documents[collection_name]
        
        # Create new index
        distance_metric = DistanceMetric(collection_info["distance_metric"])
        new_index = self._create_faiss_index(collection_info["dimension"], distance_metric)
        
        # Add all documents to new index
        embeddings = []
        for doc in doc_storage.values():
            if doc.embedding:
                embeddings.append(doc.embedding)
        
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            if collection_info["distance_metric"] == "cosine":
                faiss.normalize_L2(embeddings_array)
            
            if hasattr(new_index, 'is_trained') and not new_index.is_trained:
                if embeddings_array.shape[0] >= self.nlist:
                    new_index.train(embeddings_array)
            
            new_index.add(embeddings_array)
        
        # Replace old index
        self.indexes[collection_name] = new_index
    
    async def delete_documents(self, document_ids: List[str], 
                              collection: str = None) -> int:
        """Delete documents from the vector store"""
        if not document_ids:
            return 0
        
        try:
            deleted_count = 0
            
            collections_to_process = [collection] if collection else list(self.documents.keys())
            
            for coll_name in collections_to_process:
                if coll_name not in self.documents:
                    continue
                
                doc_storage = self.documents[coll_name]
                
                # Remove documents from storage
                for doc_id in document_ids:
                    if doc_id in doc_storage:
                        del doc_storage[doc_id]
                        deleted_count += 1
                
                # Rebuild index if any documents were deleted
                if deleted_count > 0:
                    await self._rebuild_collection_index(coll_name)
                    self.collections[coll_name]["document_count"] = len(doc_storage)
            
            # Save to disk
            await self._save_collections()
            
            logger.info(f"Deleted {deleted_count} documents from FAISS")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete documents from FAISS: {str(e)}")
            return 0
    
    async def get_document(self, document_id: str, 
                          collection: str = None) -> Optional[VectorDocument]:
        """Get a document by ID"""
        try:
            collections_to_search = [collection] if collection else list(self.documents.keys())
            
            for coll_name in collections_to_search:
                if coll_name in self.documents:
                    doc_storage = self.documents[coll_name]
                    if document_id in doc_storage:
                        return doc_storage[document_id]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id} from FAISS: {str(e)}")
            return None
    
    async def search_similar(self, query: VectorQuery) -> List[VectorSearchResult]:
        """Search for similar documents"""
        try:
            if not query.query_vector or query.collection not in self.indexes:
                return []
            
            index = self.indexes[query.collection]
            doc_storage = self.documents[query.collection]
            collection_info = self.collections[query.collection]
            
            # Prepare query vector
            query_vector = np.array([query.query_vector], dtype=np.float32)
            
            # Normalize for cosine similarity if needed
            if collection_info["distance_metric"] == "cosine":
                faiss.normalize_L2(query_vector)
            
            # Set search parameters for IVF indexes
            if hasattr(index, 'nprobe'):
                index.nprobe = self.nprobe
            
            # Perform search
            distances, indices = index.search(query_vector, query.limit + query.offset)
            
            results = []
            doc_list = list(doc_storage.values())
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                
                if i < query.offset:  # Skip offset results
                    continue
                
                if idx >= len(doc_list):  # Index out of bounds
                    continue
                
                document = doc_list[idx]
                
                # Convert distance to similarity score
                if collection_info["distance_metric"] == "cosine":
                    similarity = distance  # FAISS returns similarity for inner product
                elif collection_info["distance_metric"] == "euclidean":
                    similarity = 1.0 / (1.0 + distance)
                else:
                    similarity = distance
                
                # Skip results below threshold
                if similarity < query.similarity_threshold:
                    continue
                
                # Apply metadata filtering
                if query.metadata_filter:
                    match = True
                    for key, value in query.metadata_filter.items():
                        if document.metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                
                # Create result
                result_doc = document
                if not query.include_content:
                    result_doc = VectorDocument(
                        id=document.id,
                        content="",
                        embedding=document.embedding,
                        metadata=document.metadata if query.include_metadata else {},
                        collection=document.collection,
                        title=document.title,
                        source=document.source,
                        document_type=document.document_type,
                        chunk_index=document.chunk_index,
                        chunk_count=document.chunk_count,
                        created_at=document.created_at,
                        updated_at=document.updated_at
                    )
                
                result = VectorSearchResult(
                    document=result_doc,
                    similarity_score=similarity,
                    distance=distance,
                    rank=len(results) + 1
                )
                
                results.append(result)
                
                if len(results) >= query.limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar documents in FAISS: {str(e)}")
            return []
    
    async def count_documents(self, collection: str = None) -> int:
        """Count documents in a collection"""
        try:
            if collection:
                if collection in self.documents:
                    return len(self.documents[collection])
                return 0
            else:
                # Count across all collections
                total = 0
                for doc_storage in self.documents.values():
                    total += len(doc_storage)
                return total
                
        except Exception as e:
            logger.error(f"Failed to count documents in FAISS: {str(e)}")
            return 0
