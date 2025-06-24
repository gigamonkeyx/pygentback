"""
Contextual Retrieval Implementation

Provides context-aware retrieval with conversation history and user context.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base import RetrievalEngine, RetrievalQuery, RetrievalResult, RetrievalStrategy
from .semantic import SemanticRetriever

logger = logging.getLogger(__name__)


@dataclass
class ContextualConfig:
    """Configuration for contextual retrieval"""
    context_window: int = 5
    context_weight: float = 0.3
    max_results: int = 10
    similarity_threshold: float = 0.7


class ContextualRetriever(RetrievalEngine):
    """
    Contextual retrieval engine that considers conversation history and user context.
    
    Enhances retrieval by incorporating previous queries, user preferences,
    and conversation context to improve relevance.
    """
    
    def __init__(self, vector_store_manager=None, embedding_service=None, config: Optional[ContextualConfig] = None):
        super().__init__(RetrievalStrategy.CONTEXTUAL)
        self.config = config or ContextualConfig()
        self.vector_store_manager = vector_store_manager
        self.embedding_service = embedding_service
        self.semantic_retriever = SemanticRetriever(vector_store_manager, embedding_service)
        self.conversation_history: List[str] = []
        self.user_context: Dict[str, Any] = {}
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the contextual retriever"""
        try:
            # Initialize semantic retriever
            if not await self.semantic_retriever.initialize():
                logger.error("Failed to initialize semantic retriever")
                return False
            
            self.is_initialized = True
            logger.info("Contextual retriever initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize contextual retriever: {e}")
            return False
    
    async def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Perform contextual retrieval considering conversation history.
        
        Args:
            query: The retrieval query
            
        Returns:
            List of retrieval results enhanced with contextual relevance
        """
        try:
            if not self.is_initialized:
                logger.warning("Contextual retriever not initialized")
                return []
            
            # Build contextual query
            contextual_query = self._build_contextual_query(query)
            
            # Get semantic results
            semantic_results = await self.semantic_retriever.retrieve(contextual_query)
            
            # Apply contextual scoring
            results = []
            for result in semantic_results[:self.config.max_results]:
                # Calculate contextual relevance
                context_score = self._calculate_context_score(result, query)
                
                # Combine semantic and contextual scores
                final_score = (
                    result.score * (1 - self.config.context_weight) +
                    context_score * self.config.context_weight
                )
                
                contextual_result = RetrievalResult(
                    id=result.id,
                    content=result.content,
                    metadata={
                        **result.metadata,
                        "context_score": context_score,
                        "original_score": result.score
                    },
                    score=final_score,
                    source=f"contextual_{result.source}"
                )
                results.append(contextual_result)
            
            # Sort by final score
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Update conversation history
            self._update_conversation_history(query.text)
            
            logger.debug(f"Contextual retrieval returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Contextual retrieval failed: {e}")
            return []
    
    def _build_contextual_query(self, query: RetrievalQuery) -> RetrievalQuery:
        """Build a contextual query incorporating conversation history"""
        try:
            # Get recent conversation context
            recent_context = self.conversation_history[-self.config.context_window:]
            
            # Enhance query text with context
            if recent_context:
                context_text = " ".join(recent_context)
                enhanced_text = f"{query.text} Context: {context_text}"
            else:
                enhanced_text = query.text
            
            # Create enhanced query
            contextual_query = RetrievalQuery(
                text=enhanced_text,
                embedding=query.embedding,
                filters=query.filters,
                max_results=query.max_results,
                similarity_threshold=query.similarity_threshold,
                agent_id=query.agent_id
            )
            
            return contextual_query
            
        except Exception as e:
            logger.error(f"Failed to build contextual query: {e}")
            return query
    
    def _calculate_context_score(self, result: RetrievalResult, query: RetrievalQuery) -> float:
        """Calculate contextual relevance score"""
        try:
            # Simple contextual scoring based on conversation history
            context_score = 0.5  # Base score
            
            # Check if result content relates to recent conversation
            if self.conversation_history:
                recent_context = " ".join(self.conversation_history[-3:]).lower()
                result_content = result.content.lower()
                
                # Simple keyword overlap scoring
                context_words = set(recent_context.split())
                result_words = set(result_content.split())
                overlap = len(context_words.intersection(result_words))
                
                if overlap > 0:
                    context_score += min(0.3, overlap * 0.1)
            
            return min(1.0, context_score)
            
        except Exception as e:
            logger.error(f"Failed to calculate context score: {e}")
            return 0.5
    
    def _update_conversation_history(self, query_text: str) -> None:
        """Update conversation history with new query"""
        try:
            self.conversation_history.append(query_text)
            
            # Keep only recent history
            max_history = self.config.context_window * 2
            if len(self.conversation_history) > max_history:
                self.conversation_history = self.conversation_history[-max_history:]
                
        except Exception as e:
            logger.error(f"Failed to update conversation history: {e}")
    
    def set_user_context(self, context: Dict[str, Any]) -> None:
        """Set user context for personalized retrieval"""
        self.user_context = context
        logger.debug("Updated user context")
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history.clear()
        logger.debug("Cleared conversation history")
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the contextual retriever"""
        return await self.semantic_retriever.add_documents(documents)
    
    async def update_document(self, doc_id: str, document: Dict[str, Any]) -> bool:
        """Update a document in the contextual retriever"""
        return await self.semantic_retriever.update_document(doc_id, document)
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the contextual retriever"""
        return await self.semantic_retriever.delete_document(doc_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get contextual retriever statistics"""
        semantic_stats = self.semantic_retriever.get_stats()
        
        return {
            "strategy": self.strategy.value,
            "initialized": self.is_initialized,
            "conversation_history_length": len(self.conversation_history),
            "user_context_keys": list(self.user_context.keys()),
            "config": {
                "context_window": self.config.context_window,
                "context_weight": self.config.context_weight,
                "max_results": self.config.max_results,
                "similarity_threshold": self.config.similarity_threshold
            },
            "semantic_stats": semantic_stats
        }
