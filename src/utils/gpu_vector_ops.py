"""
GPU Vector Operations

Provides GPU-accelerated vector operations using CuPy as fallback
for FAISS-GPU when not available.
"""

import logging
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available, falling back to CPU operations")

class GPUVectorOps:
    """GPU-accelerated vector operations"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        if self.use_gpu:
            logger.info("GPU vector operations enabled")
        else:
            logger.info("Using CPU vector operations")
    
    def cosine_similarity_batch(self, 
                               query: np.ndarray, 
                               vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and batch of vectors"""
        
        if self.use_gpu and CUPY_AVAILABLE:
            return self._gpu_cosine_similarity(query, vectors)
        else:
            return self._cpu_cosine_similarity(query, vectors)
    
    def _gpu_cosine_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """GPU cosine similarity computation"""
        try:
            # Transfer to GPU
            query_gpu = cp.asarray(query)
            vectors_gpu = cp.asarray(vectors)
            
            # Normalize
            query_norm = cp.linalg.norm(query_gpu)
            vectors_norm = cp.linalg.norm(vectors_gpu, axis=1)
            
            # Compute dot products
            dots = cp.dot(vectors_gpu, query_gpu)
            
            # Compute similarities
            similarities = dots / (query_norm * vectors_norm + 1e-8)
            
            # Transfer back to CPU
            return cp.asnumpy(similarities)
            
        except Exception as e:
            logger.error(f"GPU similarity computation failed: {e}")
            return self._cpu_cosine_similarity(query, vectors)
    
    def _cpu_cosine_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """CPU cosine similarity computation"""
        query_norm = np.linalg.norm(query)
        vectors_norm = np.linalg.norm(vectors, axis=1)
        
        dots = np.dot(vectors, query)
        similarities = dots / (query_norm * vectors_norm + 1e-8)
        
        return similarities
    
    def top_k_indices(self, similarities: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get top k indices and scores"""
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                similarities_gpu = cp.asarray(similarities)
                top_indices = cp.argpartition(similarities_gpu, -k)[-k:]
                top_indices = top_indices[cp.argsort(similarities_gpu[top_indices])][::-1]
                top_scores = similarities_gpu[top_indices]
                
                return cp.asnumpy(top_indices), cp.asnumpy(top_scores)
            except:
                pass
        
        # CPU fallback
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores

# Global instance
gpu_ops = GPUVectorOps()
