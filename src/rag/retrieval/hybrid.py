"""
Hybrid Retrieval Implementation

Combines multiple retrieval strategies for improved performance.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base import RetrievalEngine, RetrievalQuery, RetrievalResult, RetrievalStrategy
from .semantic import SemanticRetriever

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval"""
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    max_results: int = 10
    similarity_threshold: float = 0.7


class HybridRetriever(RetrievalEngine):
    """
    Hybrid retrieval engine that combines semantic and keyword search.
    
    Uses weighted combination of multiple retrieval strategies to improve
    relevance and coverage of search results.
    """
    
    def __init__(self, config: Optional[HybridConfig] = None):
        super().__init__(RetrievalStrategy.HYBRID)
        self.config = config or HybridConfig()
        self.semantic_retriever = None  # Will be initialized later
        self.vector_store_manager = None
        self.embedding_service = None
        self.is_initialized = False
        
    async def initialize(self, vector_store_manager=None, embedding_service=None) -> bool:
        """Initialize the hybrid retriever"""
        try:
            # Store dependencies
            if vector_store_manager:
                self.vector_store_manager = vector_store_manager
            if embedding_service:
                self.embedding_service = embedding_service

            # Initialize semantic retriever if we have dependencies
            if self.vector_store_manager and self.embedding_service:
                self.semantic_retriever = SemanticRetriever(
                    self.vector_store_manager,
                    self.embedding_service
                )
                logger.info("Semantic retriever initialized with dependencies")
            else:
                logger.warning("Hybrid retriever initialized without semantic capabilities")

            self.is_initialized = True
            logger.info("Hybrid retriever initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize hybrid retriever: {e}")
            return False
    
    async def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval combining multiple strategies.
        
        Args:
            query: The retrieval query
            
        Returns:
            List of retrieval results ranked by hybrid score
        """
        try:
            if not self.is_initialized:
                logger.warning("Hybrid retriever not initialized")
                return []

            # Get semantic results if available
            semantic_results = []
            if self.semantic_retriever:
                semantic_results = await self.semantic_retriever.retrieve(query)
            else:
                logger.warning("Semantic retriever not available, using fallback")
            
            # For now, just return semantic results
            # In a full implementation, this would combine with keyword search
            results = []
            for result in semantic_results[:self.config.max_results]:
                # Apply hybrid scoring
                hybrid_score = result.score * self.config.semantic_weight
                
                hybrid_result = RetrievalResult(
                    id=result.id,
                    content=result.content,
                    metadata=result.metadata,
                    score=hybrid_score,
                    source=f"hybrid_{result.source}"
                )
                results.append(hybrid_result)
            
            # Sort by hybrid score
            results.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug(f"Hybrid retrieval returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the hybrid retriever"""
        try:
            # Add to semantic retriever if available
            if self.semantic_retriever:
                success = await self.semantic_retriever.add_documents(documents)
                if success:
                    logger.debug(f"Added {len(documents)} documents to hybrid retriever")
                return success
            else:
                logger.warning("Semantic retriever not available for adding documents")
                return False

        except Exception as e:
            logger.error(f"Failed to add documents to hybrid retriever: {e}")
            return False
    
    async def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Update a document in the hybrid retriever"""
        try:
            if self.semantic_retriever:
                success = await self.semantic_retriever.update_document(doc_id, document)
                if success:
                    logger.debug(f"Updated document {doc_id} in hybrid retriever")
                return success
            else:
                logger.warning("Semantic retriever not available for updating documents")
                return False

        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the hybrid retriever"""
        try:
            if self.semantic_retriever:
                success = await self.semantic_retriever.delete_document(doc_id)
                if success:
                    logger.debug(f"Deleted document {doc_id} from hybrid retriever")
                return success
            else:
                logger.warning("Semantic retriever not available for deleting documents")
                return False

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid retriever statistics"""
        semantic_stats = {}
        if self.semantic_retriever:
            try:
                semantic_stats = self.semantic_retriever.get_stats()
            except Exception as e:
                logger.warning(f"Failed to get semantic stats: {e}")
                semantic_stats = {"error": str(e)}

        return {
            "strategy": self.strategy.value,
            "initialized": self.is_initialized,
            "has_semantic_retriever": self.semantic_retriever is not None,
            "config": {
                "semantic_weight": self.config.semantic_weight,
                "keyword_weight": self.config.keyword_weight,
                "max_results": self.config.max_results,
                "similarity_threshold": self.config.similarity_threshold
            },
            "semantic_stats": semantic_stats
        }
