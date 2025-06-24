"""
RAG Retrieval Base Classes

This module defines the core interfaces and data structures for the RAG retrieval system.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Retrieval strategy options"""
    SEMANTIC = "semantic"           # Pure semantic similarity
    HYBRID = "hybrid"              # Semantic + keyword matching
    CONTEXTUAL = "contextual"      # Context-aware retrieval
    ADAPTIVE = "adaptive"          # Adaptive based on query type


class RetrievalMode(Enum):
    """Retrieval mode options"""
    STANDARD = "standard"          # Standard retrieval
    EXHAUSTIVE = "exhaustive"      # Comprehensive search
    FAST = "fast"                  # Quick retrieval
    PRECISE = "precise"            # High precision retrieval


@dataclass
class RetrievalQuery:
    """Represents a retrieval query with all parameters"""
    text: str
    embedding: Optional[List[float]] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC
    mode: RetrievalMode = RetrievalMode.STANDARD
    max_results: int = 10
    similarity_threshold: float = 0.7
    context: Optional[str] = None
    agent_id: Optional[str] = None
    collections: Optional[List[str]] = None
    
    # Advanced parameters
    rerank: bool = True
    diversify: bool = False
    temporal_weight: float = 0.1
    authority_weight: float = 0.1
    
    def __post_init__(self):
        """Validate query parameters"""
        if not self.text.strip():
            raise ValueError("Query text cannot be empty")
        
        if self.max_results <= 0:
            raise ValueError("max_results must be positive")
        
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")


@dataclass
class RetrievalResult:
    """Represents a retrieval result with comprehensive metadata"""
    document_id: str
    chunk_id: str
    content: str
    title: str
    similarity_score: float
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Source information
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    collection: Optional[str] = None
    
    # Content structure
    chunk_index: int = 0
    total_chunks: int = 1
    
    # Scoring details
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    temporal_score: float = 0.0
    authority_score: float = 0.0
    quality_score: float = 0.0
    
    # Retrieval metadata
    retrieved_at: datetime = field(default_factory=datetime.utcnow)
    retrieval_strategy: Optional[str] = None
    
    def get_context_snippet(self, max_length: int = 200) -> str:
        """Get a context snippet of the content"""
        if len(self.content) <= max_length:
            return self.content
        
        # Try to end at sentence boundary
        truncated = self.content[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.7:  # If period is reasonably close to end
            return truncated[:last_period + 1]
        
        return truncated + "..."
    
    def get_highlighted_content(self, query_terms: List[str], 
                              max_length: int = 500) -> str:
        """Get content with query terms highlighted"""
        content = self.content[:max_length] if len(self.content) > max_length else self.content
        
        # Simple highlighting (in a real implementation, use proper highlighting)
        for term in query_terms:
            if term.lower() in content.lower():
                # Case-insensitive replacement with highlighting
                import re
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                content = pattern.sub(f"**{term}**", content)
        
        return content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "title": self.title,
            "similarity_score": self.similarity_score,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
            "source_path": self.source_path,
            "source_url": self.source_url,
            "collection": self.collection,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "scores": {
                "semantic": self.semantic_score,
                "keyword": self.keyword_score,
                "temporal": self.temporal_score,
                "authority": self.authority_score,
                "quality": self.quality_score
            },
            "retrieved_at": self.retrieved_at.isoformat(),
            "retrieval_strategy": self.retrieval_strategy
        }


@dataclass
class RetrievalStats:
    """Statistics about a retrieval operation"""
    query: str
    strategy: str
    total_results: int
    filtered_results: int
    execution_time_ms: float
    collections_searched: List[str]
    
    # Performance metrics
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0
    ranking_time_ms: float = 0.0
    
    # Quality metrics
    avg_similarity_score: float = 0.0
    avg_relevance_score: float = 0.0
    score_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "query": self.query,
            "strategy": self.strategy,
            "total_results": self.total_results,
            "filtered_results": self.filtered_results,
            "execution_time_ms": self.execution_time_ms,
            "collections_searched": self.collections_searched,
            "performance": {
                "embedding_time_ms": self.embedding_time_ms,
                "search_time_ms": self.search_time_ms,
                "ranking_time_ms": self.ranking_time_ms
            },
            "quality": {
                "avg_similarity_score": self.avg_similarity_score,
                "avg_relevance_score": self.avg_relevance_score,
                "score_distribution": self.score_distribution
            }
        }


@runtime_checkable
class RetrievalEngine(Protocol):
    """Protocol defining the interface for retrieval engines"""
    
    async def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve documents for a query"""
        ...
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        ...


class BaseRetriever(ABC):
    """Abstract base class for retrieval implementations"""
    
    def __init__(self, name: str):
        self.name = name
        self.stats = {
            "total_queries": 0,
            "total_results": 0,
            "avg_execution_time": 0.0,
            "last_query_time": None
        }
    
    @abstractmethod
    async def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Retrieval query
            
        Returns:
            List[RetrievalResult]: Retrieved results
        """
        pass
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            "name": self.name,
            "stats": self.stats.copy()
        }
    
    def _update_stats(self, query: RetrievalQuery, results: List[RetrievalResult], 
                     execution_time: float) -> None:
        """Update internal statistics"""
        self.stats["total_queries"] += 1
        self.stats["total_results"] += len(results)
        
        # Update average execution time
        current_avg = self.stats["avg_execution_time"]
        total_queries = self.stats["total_queries"]
        self.stats["avg_execution_time"] = (
            (current_avg * (total_queries - 1) + execution_time) / total_queries
        )
        
        self.stats["last_query_time"] = datetime.utcnow().isoformat()


class RetrievalError(Exception):
    """Base exception for retrieval errors"""
    pass


class QueryValidationError(RetrievalError):
    """Exception raised for invalid queries"""
    pass


class RetrievalTimeoutError(RetrievalError):
    """Exception raised when retrieval times out"""
    pass


class EmbeddingError(RetrievalError):
    """Exception raised for embedding generation errors"""
    pass
