"""
Semantic Retrieval Implementation

This module provides semantic similarity-based retrieval using vector embeddings.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from .base import BaseRetriever, RetrievalQuery, RetrievalResult, RetrievalStats
from ...storage.vector import VectorStoreManager, VectorQuery, VectorSearchResult


logger = logging.getLogger(__name__)


class SemanticRetriever(BaseRetriever):
    """
    Semantic retrieval implementation using vector similarity search.
    
    This retriever uses embedding-based similarity to find semantically
    related documents, providing the foundation for RAG systems.
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager, 
                 embedding_service, collections: Optional[List[str]] = None):
        """
        Initialize semantic retriever.
        
        Args:
            vector_store_manager: Vector store manager instance
            embedding_service: Embedding generation service
            collections: Default collections to search
        """
        super().__init__("semantic")
        self.vector_store_manager = vector_store_manager
        self.embedding_service = embedding_service
        self.default_collections = collections or ["documents", "knowledge_base"]
        
        # Performance tracking
        self.embedding_cache = {}  # Simple cache for embeddings
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Perform semantic retrieval using vector similarity.
        
        Args:
            query: Retrieval query with text and parameters
            
        Returns:
            List[RetrievalResult]: Semantically similar results
        """
        start_time = time.time()
        
        try:
            # Generate query embedding if not provided
            embedding_start = time.time()
            query_embedding = await self._get_query_embedding(query)
            embedding_time = (time.time() - embedding_start) * 1000
            
            # Determine collections to search
            collections = query.collections or self.default_collections
            
            # Perform vector search across collections
            search_start = time.time()
            all_results = []
            
            for collection in collections:
                try:
                    # Create vector query
                    vector_query = VectorQuery(
                        query_vector=query_embedding,
                        collection=collection,
                        limit=query.max_results * 2,  # Get more for better ranking
                        similarity_threshold=query.similarity_threshold,
                        metadata_filter=query.filters,
                        distance_metric="cosine"
                    )
                    
                    # Execute search
                    collection_results = await self.vector_store_manager.search_similar(vector_query)
                    
                    # Convert to retrieval results
                    for result in collection_results:
                        retrieval_result = self._convert_vector_result(result, collection, query)
                        all_results.append(retrieval_result)
                        
                except Exception as e:
                    logger.warning(f"Failed to search collection {collection}: {str(e)}")
                    continue
            
            search_time = (time.time() - search_start) * 1000
            
            # Sort by similarity score and limit results
            all_results.sort(key=lambda x: x.similarity_score, reverse=True)
            final_results = all_results[:query.max_results]
            
            # Update statistics
            execution_time = (time.time() - start_time) * 1000
            self._update_stats(query, final_results, execution_time)
            
            # Log performance metrics
            logger.debug(f"Semantic retrieval completed: {len(final_results)} results in {execution_time:.2f}ms "
                        f"(embedding: {embedding_time:.2f}ms, search: {search_time:.2f}ms)")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {str(e)}")
            return []
    
    async def _get_query_embedding(self, query: RetrievalQuery) -> List[float]:
        """Get or generate embedding for query text"""
        if query.embedding:
            return query.embedding
        
        # Check cache first
        cache_key = hash(query.text)
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]
        
        # Generate new embedding
        self.cache_misses += 1
        try:
            embedding_result = await self.embedding_service.generate_embedding(query.text)
            embedding = embedding_result.embedding
            
            # Cache the result (with simple size limit)
            if len(self.embedding_cache) < 1000:
                self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {str(e)}")
            raise
    
    def _convert_vector_result(self, vector_result: VectorSearchResult, 
                              collection: str, query: RetrievalQuery) -> RetrievalResult:
        """Convert vector search result to retrieval result"""
        document = vector_result.document
        metadata = document.metadata
        
        return RetrievalResult(
            document_id=metadata.get('document_id', document.id),
            chunk_id=document.id,
            content=document.content,
            title=metadata.get('document_title', metadata.get('title', 'Untitled')),
            similarity_score=vector_result.similarity_score,
            relevance_score=vector_result.similarity_score,  # Will be updated by scorer
            metadata=metadata,
            source_path=metadata.get('source_path'),
            source_url=metadata.get('source_url'),
            collection=collection,
            chunk_index=metadata.get('chunk_index', 0),
            total_chunks=metadata.get('total_chunks', 1),
            semantic_score=vector_result.similarity_score,
            retrieval_strategy="semantic"
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed semantic retrieval statistics"""
        base_stats = await super().get_stats()
        
        semantic_stats = {
            "cache_stats": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                "cache_size": len(self.embedding_cache)
            },
            "collections": {
                "default_collections": self.default_collections,
                "total_collections": len(self.default_collections)
            }
        }
        
        base_stats["semantic_stats"] = semantic_stats
        return base_stats
    
    async def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Semantic retriever cache cleared")
    
    async def warm_cache(self, common_queries: List[str]) -> None:
        """Pre-populate cache with common queries"""
        logger.info(f"Warming semantic retriever cache with {len(common_queries)} queries")
        
        for query_text in common_queries:
            try:
                cache_key = hash(query_text)
                if cache_key not in self.embedding_cache:
                    embedding_result = await self.embedding_service.generate_embedding(query_text)
                    self.embedding_cache[cache_key] = embedding_result.embedding
            except Exception as e:
                logger.warning(f"Failed to warm cache for query '{query_text}': {str(e)}")
        
        logger.info(f"Cache warmed. Current size: {len(self.embedding_cache)}")
    
    async def get_similar_documents(self, document_id: str, collection: str,
                                   limit: int = 5) -> List[RetrievalResult]:
        """
        Find documents similar to a given document.
        
        Args:
            document_id: ID of the reference document
            collection: Collection containing the document
            limit: Maximum number of similar documents to return
            
        Returns:
            List[RetrievalResult]: Similar documents
        """
        try:
            # Get the reference document
            reference_doc = await self.vector_store_manager.get_default_store().get_document(
                document_id, collection
            )
            
            if not reference_doc or not reference_doc.embedding:
                logger.warning(f"Document {document_id} not found or has no embedding")
                return []
            
            # Create similarity query
            query = RetrievalQuery(
                text="",  # Empty text since we're using embedding directly
                embedding=reference_doc.embedding,
                max_results=limit + 1,  # +1 to exclude the reference document
                similarity_threshold=0.5,
                collections=[collection]
            )
            
            # Perform retrieval
            results = await self.retrieve(query)
            
            # Filter out the reference document itself
            similar_docs = [r for r in results if r.document_id != document_id]
            
            return similar_docs[:limit]
            
        except Exception as e:
            logger.error(f"Failed to find similar documents for {document_id}: {str(e)}")
            return []
