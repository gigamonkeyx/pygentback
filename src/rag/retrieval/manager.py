"""
Retrieval Manager

This module provides the main interface for managing different retrieval strategies
and coordinating retrieval operations across the RAG system.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union

from .base import (
    RetrievalStrategy, RetrievalQuery, RetrievalResult, RetrievalStats,
    BaseRetriever, RetrievalError
)
from .semantic import SemanticRetriever
from .scorer import RetrievalScorer, ScoringWeights
from ...storage.vector import VectorStoreManager


logger = logging.getLogger(__name__)


class RetrievalManager:
    """
    Main manager for retrieval operations.
    
    Coordinates between different retrieval strategies, handles query routing,
    and provides a unified interface for all retrieval operations.
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager, 
                 embedding_service, settings=None):
        """
        Initialize the retrieval manager.
        
        Args:
            vector_store_manager: Vector store manager instance
            embedding_service: Embedding generation service
            settings: Application settings
        """
        self.vector_store_manager = vector_store_manager
        self.embedding_service = embedding_service
        self.settings = settings
        
        # Initialize retrievers (will be populated by async init)
        self.retrievers: Dict[str, BaseRetriever] = {}
        self._initialized = False
        
        # Initialize scorer
        scoring_weights = ScoringWeights()
        if settings and hasattr(settings, 'rag'):
            rag_settings = settings.rag
            scoring_weights = ScoringWeights(
                similarity=getattr(rag_settings, 'similarity_weight', 0.5),
                temporal=getattr(rag_settings, 'temporal_weight', 0.15),
                authority=getattr(rag_settings, 'authority_weight', 0.15),
                quality=getattr(rag_settings, 'quality_weight', 0.1),
                keyword=getattr(rag_settings, 'keyword_weight', 0.1)
            )
        
        self.scorer = RetrievalScorer(scoring_weights)
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "queries_by_strategy": {},
            "avg_results_per_query": 0.0,
            "avg_execution_time": 0.0
        }

    async def initialize(self) -> None:
        """Initialize retrievers asynchronously"""
        if not self._initialized:
            await self._initialize_retrievers()
            self._initialized = True

    async def _initialize_retrievers(self) -> None:
        """Initialize available retrieval strategies"""
        # Semantic retriever (always available)
        self.retrievers["semantic"] = SemanticRetriever(
            self.vector_store_manager,
            self.embedding_service
        )
        
        # Add hybrid retriever
        try:
            from .hybrid import HybridRetriever
            hybrid_retriever = HybridRetriever()
            # Initialize with dependencies
            await hybrid_retriever.initialize(self.vector_store_manager, self.embedding_service)
            self.retrievers["hybrid"] = hybrid_retriever
            logger.info("Hybrid retriever initialized successfully")
        except ImportError as e:
            logger.warning(f"Hybrid retriever not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize hybrid retriever: {e}")

        # Add contextual retriever
        try:
            from .contextual import ContextualRetriever
            self.retrievers["contextual"] = ContextualRetriever(
                self.vector_store_manager,
                self.embedding_service
            )
            logger.info("Contextual retriever initialized successfully")
        except ImportError as e:
            logger.warning(f"Contextual retriever not available: {e}")
        
        logger.info(f"Initialized {len(self.retrievers)} retrieval strategies")
    
    async def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Perform retrieval using the specified strategy.
        
        Args:
            query: Retrieval query with strategy and parameters
            
        Returns:
            List[RetrievalResult]: Retrieved and ranked results
        """
        start_time = time.time()
        
        try:
            # Ensure initialization
            if not self._initialized:
                await self.initialize()

            # Validate query
            self._validate_query(query)
            
            # Route to appropriate retriever
            retriever = self._get_retriever(query.strategy)
            
            # Perform retrieval
            results = await retriever.retrieve(query)
            
            # Apply scoring and ranking if requested
            if query.rerank and results:
                results = await self.scorer.score_results(results, query)
            
            # Update statistics
            execution_time = (time.time() - start_time) * 1000
            self._update_stats(query, results, execution_time)
            
            logger.debug(f"Retrieved {len(results)} results using {query.strategy.value} strategy "
                        f"in {execution_time:.2f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query.text}': {str(e)}")
            raise RetrievalError(f"Retrieval failed: {str(e)}") from e
    
    async def multi_strategy_retrieve(self, query: RetrievalQuery, 
                                    strategies: List[RetrievalStrategy]) -> Dict[str, List[RetrievalResult]]:
        """
        Perform retrieval using multiple strategies and return all results.
        
        Args:
            query: Base retrieval query
            strategies: List of strategies to use
            
        Returns:
            Dict[str, List[RetrievalResult]]: Results by strategy
        """
        results = {}
        
        # Execute retrievals in parallel
        tasks = []
        for strategy in strategies:
            strategy_query = RetrievalQuery(
                text=query.text,
                embedding=query.embedding,
                filters=query.filters,
                strategy=strategy,
                mode=query.mode,
                max_results=query.max_results,
                similarity_threshold=query.similarity_threshold,
                context=query.context,
                agent_id=query.agent_id,
                collections=query.collections,
                rerank=query.rerank,
                diversify=query.diversify
            )
            tasks.append(self._retrieve_with_strategy(strategy_query))
        
        # Wait for all retrievals to complete
        strategy_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for i, strategy in enumerate(strategies):
            strategy_result = strategy_results[i]
            if isinstance(strategy_result, Exception):
                logger.warning(f"Strategy {strategy.value} failed: {str(strategy_result)}")
                results[strategy.value] = []
            else:
                results[strategy.value] = strategy_result
        
        return results
    
    async def _retrieve_with_strategy(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Helper method for parallel retrieval"""
        try:
            return await self.retrieve(query)
        except Exception as e:
            logger.error(f"Strategy {query.strategy.value} failed: {str(e)}")
            return []
    
    async def ensemble_retrieve(self, query: RetrievalQuery,
                              strategies: Optional[List[RetrievalStrategy]] = None) -> List[RetrievalResult]:
        """
        Perform ensemble retrieval combining multiple strategies.
        
        Args:
            query: Base retrieval query
            strategies: Strategies to combine (default: all available)
            
        Returns:
            List[RetrievalResult]: Combined and ranked results
        """
        if not strategies:
            strategies = [RetrievalStrategy.SEMANTIC]  # Default for now
        
        # Get results from multiple strategies
        strategy_results = await self.multi_strategy_retrieve(query, strategies)
        
        # Combine and deduplicate results
        combined_results = []
        seen_chunks = set()
        
        for strategy_name, results in strategy_results.items():
            for result in results:
                chunk_key = (result.document_id, result.chunk_id)
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    # Add strategy information
                    result.retrieval_strategy = f"ensemble_{strategy_name}"
                    combined_results.append(result)
        
        # Re-rank combined results
        if combined_results:
            combined_results = await self.scorer.score_results(combined_results, query)
        
        # Limit to requested number of results
        return combined_results[:query.max_results]
    
    def _get_retriever(self, strategy: RetrievalStrategy) -> BaseRetriever:
        """Get retriever for the specified strategy"""
        strategy_name = strategy.value
        
        if strategy_name not in self.retrievers:
            # Fall back to semantic retrieval
            logger.warning(f"Strategy {strategy_name} not available, falling back to semantic")
            strategy_name = "semantic"
        
        return self.retrievers[strategy_name]
    
    def _validate_query(self, query: RetrievalQuery) -> None:
        """Validate retrieval query"""
        if not query.text.strip():
            raise RetrievalError("Query text cannot be empty")
        
        if query.max_results <= 0:
            raise RetrievalError("max_results must be positive")
        
        if not 0 <= query.similarity_threshold <= 1:
            raise RetrievalError("similarity_threshold must be between 0 and 1")
    
    def _update_stats(self, query: RetrievalQuery, results: List[RetrievalResult], 
                     execution_time: float) -> None:
        """Update retrieval statistics"""
        self.stats["total_queries"] += 1
        
        # Update strategy stats
        strategy = query.strategy.value
        if strategy not in self.stats["queries_by_strategy"]:
            self.stats["queries_by_strategy"][strategy] = 0
        self.stats["queries_by_strategy"][strategy] += 1
        
        # Update averages
        total_queries = self.stats["total_queries"]
        
        # Average results per query
        current_avg_results = self.stats["avg_results_per_query"]
        self.stats["avg_results_per_query"] = (
            (current_avg_results * (total_queries - 1) + len(results)) / total_queries
        )
        
        # Average execution time
        current_avg_time = self.stats["avg_execution_time"]
        self.stats["avg_execution_time"] = (
            (current_avg_time * (total_queries - 1) + execution_time) / total_queries
        )
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive retrieval statistics"""
        # Get stats from individual retrievers
        retriever_stats = {}
        for name, retriever in self.retrievers.items():
            retriever_stats[name] = await retriever.get_stats()
        
        return {
            "manager_stats": self.stats.copy(),
            "retriever_stats": retriever_stats,
            "available_strategies": list(self.retrievers.keys()),
            "scorer_weights": {
                "similarity": self.scorer.weights.similarity,
                "temporal": self.scorer.weights.temporal,
                "authority": self.scorer.weights.authority,
                "quality": self.scorer.weights.quality,
                "keyword": self.scorer.weights.keyword
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on retrieval system"""
        health = {
            "status": "healthy",
            "retrievers": {},
            "vector_store": "unknown",
            "embedding_service": "unknown"
        }
        
        # Check retrievers
        for name, retriever in self.retrievers.items():
            try:
                stats = await retriever.get_stats()
                health["retrievers"][name] = "healthy"
            except Exception as e:
                health["retrievers"][name] = f"error: {str(e)}"
                health["status"] = "degraded"
        
        # Check vector store
        try:
            vector_stats = await self.vector_store_manager.get_global_stats()
            health["vector_store"] = "healthy"
        except Exception as e:
            health["vector_store"] = f"error: {str(e)}"
            health["status"] = "degraded"
        
        # Check embedding service
        try:
            # Simple test embedding
            test_result = await self.embedding_service.generate_embedding("test")
            health["embedding_service"] = "healthy"
        except Exception as e:
            health["embedding_service"] = f"error: {str(e)}"
            health["status"] = "degraded"
        
        return health
    
    async def clear_caches(self) -> None:
        """Clear all retrieval caches"""
        for retriever in self.retrievers.values():
            if hasattr(retriever, 'clear_cache'):
                await retriever.clear_cache()
        
        logger.info("Cleared all retrieval caches")
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available retrieval strategies"""
        return list(self.retrievers.keys())
