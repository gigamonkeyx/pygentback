"""
Vector Store Compatibility Layer

Fixes interface compatibility between legacy and modular vector storage.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class CompatDocument:
    """Compatibility document wrapper"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class CompatResult:
    """Compatibility search result wrapper"""
    document: CompatDocument
    similarity_score: float
    distance: float = 0.0
    
    def __post_init__(self):
        if self.distance == 0.0:
            self.distance = 1.0 - self.similarity_score

class CompatVectorStore:
    """Compatibility vector store wrapper"""
    
    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.documents: Dict[str, CompatDocument] = {}
        self.embeddings: List[List[float]] = []
        self.doc_ids: List[str] = []
        
    async def add_documents(self, documents: List[CompatDocument]) -> None:
        """Add documents to store"""
        for doc in documents:
            self.documents[doc.id] = doc
            if doc.embedding:
                self.embeddings.append(doc.embedding)
                self.doc_ids.append(doc.id)
        
        logger.info(f"Added {len(documents)} documents to {self.collection_name}")
    
    async def similarity_search(self, 
                               query_embedding: List[float],
                               limit: int = 10,
                               similarity_threshold: float = 0.0,
                               metadata_filter: Optional[Dict[str, Any]] = None) -> List[CompatResult]:
        """Perform similarity search"""
        if not self.embeddings:
            return []
        
        # Simple cosine similarity (fallback implementation)
        import numpy as np
        
        query_np = np.array(query_embedding)
        query_norm = np.linalg.norm(query_np)
        
        similarities = []
        for i, emb in enumerate(self.embeddings):
            emb_np = np.array(emb)
            emb_norm = np.linalg.norm(emb_np)
            
            if query_norm > 0 and emb_norm > 0:
                similarity = np.dot(query_np, emb_np) / (query_norm * emb_norm)
            else:
                similarity = 0.0
            
            if similarity >= similarity_threshold:
                doc_id = self.doc_ids[i]
                doc = self.documents[doc_id]
                
                # Apply metadata filter if specified
                if metadata_filter:
                    match = all(
                        doc.metadata.get(k) == v 
                        for k, v in metadata_filter.items()
                    )
                    if not match:
                        continue
                
                similarities.append((similarity, doc))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for similarity, doc in similarities[:limit]:
            results.append(CompatResult(
                document=doc,
                similarity_score=similarity,
                distance=1.0 - similarity
            ))
        
        return results
    
    async def update_document(self, document: CompatDocument) -> None:
        """Update document in store"""
        if document.id in self.documents:
            self.documents[document.id] = document
            # Update embedding if changed
            if document.id in self.doc_ids:
                idx = self.doc_ids.index(document.id)
                if document.embedding:
                    self.embeddings[idx] = document.embedding

class CompatVectorStoreManager:
    """Compatibility vector store manager"""
    
    def __init__(self, settings=None):
        self.settings = settings
        self.stores: Dict[str, CompatVectorStore] = {}
        
    async def initialize(self):
        """Initialize manager"""
        logger.info("Compatibility vector store manager initialized")
    
    async def get_store(self, collection_name: str) -> CompatVectorStore:
        """Get or create vector store"""
        if collection_name not in self.stores:
            self.stores[collection_name] = CompatVectorStore(collection_name)
        return self.stores[collection_name]
