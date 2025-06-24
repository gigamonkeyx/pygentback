"""
Vector Store Implementation

Provides vector storage and retrieval capabilities for PyGent Factory,
supporting embeddings, similarity search, and knowledge management.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from abc import ABC, abstractmethod

try:
    from ..config.settings import Settings
except ImportError:
    # Fallback for direct import
    Settings = None

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Document with vector embedding"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SimilarityResult:
    """Result from similarity search"""
    document: VectorDocument
    similarity_score: float
    distance: float


class EmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass


class SimpleEmbeddingModel(EmbeddingModel):
    """Simple embedding model for testing"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Simple hash-based embedding for testing"""
        embeddings = []
        for text in texts:
            # Simple deterministic embedding based on text hash
            hash_val = hash(text)
            embedding = []
            for i in range(self.dimension):
                embedding.append(float((hash_val + i) % 1000) / 1000.0)
            embeddings.append(embedding)
        return embeddings
    
    def get_dimension(self) -> int:
        return self.dimension


class VectorStoreManager:
    """
    Vector store manager for embeddings and similarity search.
    
    Provides high-level interface for vector operations including
    document storage, embedding generation, and similarity search.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        if Settings is not None:
            self.settings = settings or Settings()
        else:
            self.settings = None
        self.embedding_model: Optional[EmbeddingModel] = None
        self.documents: Dict[str, VectorDocument] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.is_initialized = False
    
    async def initialize(self, embedding_model: Optional[EmbeddingModel] = None):
        """Initialize vector store"""
        try:
            if embedding_model:
                self.embedding_model = embedding_model
            else:
                # Use simple embedding model as default
                self.embedding_model = SimpleEmbeddingModel()
            
            self.is_initialized = True
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def add_document(self, document: VectorDocument) -> bool:
        """Add document to vector store"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Generate embedding if not provided
            if document.embedding is None:
                embeddings = await self.embedding_model.encode([document.content])
                document.embedding = embeddings[0]
            
            # Store document and embedding
            self.documents[document.id] = document
            self.embeddings[document.id] = document.embedding
            
            logger.debug(f"Added document {document.id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {document.id}: {e}")
            return False
    
    async def add_documents(self, documents: List[VectorDocument]) -> int:
        """Add multiple documents to vector store"""
        added_count = 0
        for document in documents:
            if await self.add_document(document):
                added_count += 1
        return added_count
    
    async def search_similar(self, 
                           query: str, 
                           limit: int = 10,
                           threshold: float = 0.0) -> List[SimilarityResult]:
        """Search for similar documents"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if not self.documents:
                return []
            
            # Generate query embedding
            query_embeddings = await self.embedding_model.encode([query])
            query_embedding = query_embeddings[0]
            
            # Calculate similarities
            results = []
            for doc_id, doc_embedding in self.embeddings.items():
                similarity = self._calculate_cosine_similarity(query_embedding, doc_embedding)
                
                if similarity >= threshold:
                    distance = 1.0 - similarity
                    result = SimilarityResult(
                        document=self.documents[doc_id],
                        similarity_score=similarity,
                        distance=distance
                    )
                    results.append(result)
            
            # Sort by similarity (highest first) and limit
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return []
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Convert to numpy arrays for calculation
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get document by ID"""
        return self.documents.get(document_id)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from vector store"""
        try:
            if document_id in self.documents:
                del self.documents[document_id]
                del self.embeddings[document_id]
                logger.debug(f"Deleted document {document_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def clear(self):
        """Clear all documents from vector store"""
        self.documents.clear()
        self.embeddings.clear()
        logger.info("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_documents": len(self.documents),
            "is_initialized": self.is_initialized,
            "embedding_dimension": self.embedding_model.get_dimension() if self.embedding_model else 0
        }


# Backward compatibility aliases
VectorStore = VectorStoreManager
