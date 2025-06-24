"""
Historical Research Vector Store Configuration
Specialized FAISS configuration optimized for historical document research.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from ..storage.vector.manager import VectorStoreManager
from ..storage.vector.base import VectorStoreType, DistanceMetric, VectorDocument
from ..utils.embedding import EmbeddingService

logger = logging.getLogger(__name__)

class HistoricalResearchVectorConfig:
    """Configuration for historical research vector database."""
    
    def __init__(self, 
                 collection_name: str = "historical_documents",
                 storage_path: str = "data/vector_store/historical",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 faiss_index_type: str = "IVFFlat",
                 nlist: int = 200,
                 nprobe: int = 20):
        
        self.collection_name = collection_name
        self.storage_path = Path(storage_path)
        self.embedding_model = embedding_model
        self.faiss_index_type = faiss_index_type
        self.nlist = nlist  # Number of clusters for IVF
        self.nprobe = nprobe  # Number of clusters to search
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize config
        self._validate_config()
    
    def _validate_config(self):
        """Validate vector store configuration."""
        if self.nlist <= 0:
            raise ValueError("nlist must be positive")
        if self.nprobe <= 0 or self.nprobe > self.nlist:
            raise ValueError("nprobe must be positive and <= nlist")
    
    def get_faiss_config(self) -> Dict[str, Any]:
        """Get FAISS-specific configuration."""
        return {
            'index_type': self.faiss_index_type,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'storage_path': str(self.storage_path),
            'collection_name': self.collection_name
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding service configuration."""
        return {
            'model_name': self.embedding_model,
            'batch_size': 32,
            'normalize_embeddings': True
        }

class HistoricalResearchVectorStore:
    """Vector store specialized for historical research documents."""
    
    def __init__(self, config: HistoricalResearchVectorConfig):
        self.config = config
        self.embedding_service = None
        self.vector_manager = None
        self.store = None
        self._initialized = False
    
    async def initialize(self, settings) -> bool:
        """Initialize the historical research vector store."""
        try:
            # Initialize embedding service
            embedding_config = self.config.get_embedding_config()
            self.embedding_service = EmbeddingService(
                model_name=embedding_config['model_name']
            )
            await self.embedding_service.initialize()
            
            # Initialize vector store manager  
            self.vector_manager = VectorStoreManager(settings)
            await self.vector_manager.initialize()
            
            # Get or create historical documents collection
            self.store = await self.vector_manager.get_store(
                store_name=self.config.collection_name,
                store_type=VectorStoreType.FAISS,
                config=self.config.get_faiss_config()
            )
            
            # Ensure collection exists
            await self._ensure_collection_exists()
            
            self._initialized = True
            logger.info(f"Historical research vector store initialized: {self.config.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize historical research vector store: {e}")
            return False
    
    async def _ensure_collection_exists(self):
        """Ensure the historical documents collection exists."""
        try:
            # Check if collection exists
            collections = await self.store.list_collections()
            if self.config.collection_name not in collections:
                # Create collection
                await self.store.create_collection(
                    collection_name=self.config.collection_name,
                    dimension=384,  # Default for all-MiniLM-L6-v2
                    distance_metric=DistanceMetric.COSINE
                )
                logger.info(f"Created collection: {self.config.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.config.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise
    
    async def add_document(self, 
                          document_id: str,
                          text_content: str, 
                          metadata: Dict[str, Any]) -> bool:
        """Add a document to the vector store."""
        if not self._initialized:
            logger.error("Vector store not initialized")
            return False
        
        try:
            # Generate embedding
            embedding = await self.embedding_service.get_embedding(text_content)
            if embedding is None:
                logger.error(f"Failed to generate embedding for document {document_id}")
                return False
            
            # Create vector document
            vector_doc = VectorDocument(
                id=document_id,
                content=text_content,
                embedding=embedding,
                metadata=metadata
            )
            
            # Add to vector store
            await self.store.add_documents(
                collection_name=self.config.collection_name,
                documents=[vector_doc]
            )
            
            logger.info(f"Added document to vector store: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {document_id} to vector store: {e}")
            return False
    
    async def add_document_chunks(self,
                                 document_id: str,
                                 chunks: List[Dict[str, Any]]) -> int:
        """Add multiple chunks from a document to the vector store."""
        if not self._initialized:
            logger.error("Vector store not initialized")
            return 0
        
        added_count = 0
        
        try:
            vector_docs = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                text_content = chunk.get('text', '')
                
                if not text_content.strip():
                    continue
                
                # Generate embedding
                embedding = await self.embedding_service.get_embedding(text_content)
                if embedding is None:
                    logger.warning(f"Failed to generate embedding for chunk {chunk_id}")
                    continue
                
                # Prepare metadata
                chunk_metadata = {
                    'document_id': document_id,
                    'chunk_id': chunk_id,
                    'chunk_index': i,
                    'page_number': chunk.get('page_number'),
                    'section': chunk.get('section'),
                    'source_type': chunk.get('source_type', 'unknown'),
                    **chunk.get('metadata', {})
                }
                
                # Create vector document
                vector_doc = VectorDocument(
                    id=chunk_id,
                    content=text_content,
                    embedding=embedding,
                    metadata=chunk_metadata
                )
                
                vector_docs.append(vector_doc)
            
            # Add all chunks in batch
            if vector_docs:
                await self.store.add_documents(
                    collection_name=self.config.collection_name,
                    documents=vector_docs
                )
                added_count = len(vector_docs)
                
                logger.info(f"Added {added_count} chunks from document {document_id} to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add document chunks for {document_id}: {e}")
        
        return added_count
    
    async def search_documents(self,
                              query: str,
                              limit: int = 10,
                              metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents using semantic similarity."""
        if not self._initialized:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.get_embedding(query)
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search vector store
            results = await self.store.search(
                collection_name=self.config.collection_name,
                query_embedding=query_embedding,
                limit=limit,
                metadata_filter=metadata_filter
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'document_id': result.metadata.get('document_id', result.id),
                    'chunk_id': result.id,
                    'content': result.content,
                    'score': result.score,
                    'metadata': result.metadata
                })
            
            logger.info(f"Found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    async def search_by_timeframe(self,
                                 query: str,
                                 start_year: Optional[int] = None,
                                 end_year: Optional[int] = None,
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents within a specific timeframe."""
        metadata_filter = {}
        
        if start_year is not None:
            metadata_filter['start_year_gte'] = start_year
        if end_year is not None:
            metadata_filter['end_year_lte'] = end_year
            
        return await self.search_documents(
            query=query,
            limit=limit,
            metadata_filter=metadata_filter if metadata_filter else None
        )
    
    async def search_by_source_type(self,
                                   query: str,
                                   source_types: List[str],
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents by source type (e.g., 'primary', 'secondary', 'government')."""
        metadata_filter = {
            'source_type': {'$in': source_types}
        }
        
        return await self.search_documents(
            query=query,
            limit=limit,
            metadata_filter=metadata_filter
        )
    
    async def get_document_count(self) -> int:
        """Get total number of documents in the vector store."""
        if not self._initialized:
            return 0
        
        try:
            return await self.store.get_document_count(self.config.collection_name)
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks from the vector store."""
        if not self._initialized:
            logger.error("Vector store not initialized")
            return False
        
        try:
            # Delete all chunks for this document
            await self.store.delete_documents(
                collection_name=self.config.collection_name,
                document_ids=[document_id],
                metadata_filter={'document_id': document_id}
            )
            
            logger.info(f"Deleted document {document_id} from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the historical documents collection."""
        if not self._initialized:
            return {}
        
        try:
            stats = await self.store.get_collection_stats(self.config.collection_name)
            
            # Add additional historical research specific stats
            stats['embedding_model'] = self.config.embedding_model
            stats['faiss_config'] = self.config.get_faiss_config()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    async def close(self):
        """Close the vector store and cleanup resources."""
        try:
            if self.vector_manager:
                await self.vector_manager.shutdown()
            
            if self.embedding_service:
                await self.embedding_service.cleanup()
            
            self._initialized = False
            logger.info("Historical research vector store closed")
            
        except Exception as e:
            logger.error(f"Error closing vector store: {e}")
