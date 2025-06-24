"""
GPU-Accelerated Memory System

A hybrid approach that uses FAISS-CPU for indexing but CuPy for 
GPU-accelerated vector operations on the NVIDIA RTX 3080.
"""

import asyncio
import logging
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# GPU acceleration
try:
    import cupy as cp
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    CUPY_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info(f"GPU acceleration available: CUDA={GPU_AVAILABLE}, CuPy={CUPY_AVAILABLE}")
except ImportError:
    GPU_AVAILABLE = False
    CUPY_AVAILABLE = False
    cp = None
    torch = None
    logger = logging.getLogger(__name__)
    logger.warning("GPU acceleration not available")

# Vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


@dataclass
class GPUVectorDocument:
    """Vector document optimized for GPU operations"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    gpu_embedding: Optional[Any] = None  # CuPy array
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_gpu(self) -> bool:
        """Move embedding to GPU memory"""
        if not CUPY_AVAILABLE or self.embedding is None:
            return False
        
        try:
            self.gpu_embedding = cp.asarray(self.embedding, dtype=cp.float32)
            return True
        except Exception as e:
            logger.error(f"Failed to move embedding to GPU: {e}")
            return False
    
    def to_cpu(self) -> bool:
        """Move embedding back to CPU memory"""
        if self.gpu_embedding is None:
            return False
        
        try:
            self.embedding = cp.asnumpy(self.gpu_embedding)
            return True
        except Exception as e:
            logger.error(f"Failed to move embedding to CPU: {e}")
            return False


class GPUAcceleratedVectorStore:
    """
    GPU-accelerated vector store using CuPy for computations
    and FAISS for indexing
    """
    
    def __init__(self, dimension: int = 384, use_gpu: bool = True):
        self.dimension = dimension
        self.use_gpu = use_gpu and GPU_AVAILABLE and CUPY_AVAILABLE
        self.documents: Dict[str, GPUVectorDocument] = {}
        self.index = None
        self.gpu_embeddings = None
        self.document_ids = []
        
        # Initialize FAISS index
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dimension)
        
        logger.info(f"GPU Vector Store initialized: GPU={self.use_gpu}, FAISS={FAISS_AVAILABLE}")
    
    async def add_document(self, doc: GPUVectorDocument) -> bool:
        """Add document to the store"""
        try:
            if doc.embedding is None:
                logger.warning(f"Document {doc.id} has no embedding")
                return False
            
            # Move to GPU if enabled
            if self.use_gpu:
                doc.to_gpu()
            
            # Add to FAISS index
            if self.index is not None:
                embedding = doc.embedding.reshape(1, -1).astype(np.float32)
                self.index.add(embedding)
                self.document_ids.append(doc.id)
            
            # Store document
            self.documents[doc.id] = doc
            
            # Update GPU embeddings matrix
            if self.use_gpu:
                await self._update_gpu_embeddings()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {doc.id}: {e}")
            return False
    
    async def _update_gpu_embeddings(self):
        """Update the GPU embeddings matrix"""
        if not self.use_gpu or not self.documents:
            return
        
        try:
            # Collect all embeddings
            embeddings = []
            for doc in self.documents.values():
                if doc.embedding is not None:
                    embeddings.append(doc.embedding)
            
            if embeddings:
                # Create GPU matrix
                embeddings_matrix = np.vstack(embeddings).astype(np.float32)
                self.gpu_embeddings = cp.asarray(embeddings_matrix)
                
        except Exception as e:
            logger.error(f"Failed to update GPU embeddings: {e}")
    
    async def similarity_search_gpu(self, 
                                   query_embedding: np.ndarray, 
                                   k: int = 5,
                                   threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Perform similarity search using GPU acceleration
        """
        if not self.use_gpu or self.gpu_embeddings is None:
            return await self.similarity_search_cpu(query_embedding, k, threshold)
        
        try:
            start_time = time.time()
            
            # Move query to GPU
            query_gpu = cp.asarray(query_embedding.reshape(1, -1), dtype=cp.float32)
            
            # Compute cosine similarity on GPU
            # Normalize vectors
            query_norm = query_gpu / cp.linalg.norm(query_gpu, axis=1, keepdims=True)
            embeddings_norm = self.gpu_embeddings / cp.linalg.norm(
                self.gpu_embeddings, axis=1, keepdims=True
            )
            
            # Compute similarities
            similarities = cp.dot(query_norm, embeddings_norm.T).flatten()
            
            # Move back to CPU for processing
            similarities_cpu = cp.asnumpy(similarities)
            
            # Get top k results
            if len(similarities_cpu) < k:
                k = len(similarities_cpu)
            
            top_indices = np.argpartition(similarities_cpu, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities_cpu[top_indices])][::-1]
            
            results = []
            doc_ids = list(self.documents.keys())
            
            for idx in top_indices:
                if idx < len(doc_ids) and similarities_cpu[idx] >= threshold:
                    results.append((doc_ids[idx], float(similarities_cpu[idx])))
            
            search_time = time.time() - start_time
            logger.debug(f"GPU similarity search completed in {search_time:.4f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"GPU similarity search failed: {e}")
            return await self.similarity_search_cpu(query_embedding, k, threshold)
    
    async def similarity_search_cpu(self, 
                                   query_embedding: np.ndarray, 
                                   k: int = 5,
                                   threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Fallback CPU similarity search"""
        try:
            if self.index is None or self.index.ntotal == 0:
                return []
            
            start_time = time.time()
            
            # Search using FAISS
            query = query_embedding.reshape(1, -1).astype(np.float32)
            distances, indices = self.index.search(query, k)
            
            # Convert distances to similarities (L2 to cosine approximation)
            similarities = 1.0 / (1.0 + distances[0])
            
            results = []
            for i, (idx, sim) in enumerate(zip(indices[0], similarities)):
                if idx != -1 and idx < len(self.document_ids) and sim >= threshold:
                    doc_id = self.document_ids[idx]
                    results.append((doc_id, float(sim)))
            
            search_time = time.time() - start_time
            logger.debug(f"CPU similarity search completed in {search_time:.4f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"CPU similarity search failed: {e}")
            return []
    
    async def benchmark_performance(self, query_embedding: np.ndarray, iterations: int = 100):
        """Benchmark GPU vs CPU performance"""
        results = {
            'gpu_enabled': self.use_gpu,
            'document_count': len(self.documents),
            'gpu_times': [],
            'cpu_times': []
        }
        
        # GPU benchmark
        if self.use_gpu:
            for _ in range(iterations):
                start_time = time.time()
                await self.similarity_search_gpu(query_embedding, k=10)
                results['gpu_times'].append(time.time() - start_time)
        
        # CPU benchmark
        for _ in range(iterations):
            start_time = time.time()
            await self.similarity_search_cpu(query_embedding, k=10)
            results['cpu_times'].append(time.time() - start_time)
        
        # Calculate statistics
        if results['gpu_times']:
            results['gpu_avg_time'] = np.mean(results['gpu_times'])
            results['gpu_std_time'] = np.std(results['gpu_times'])
            results['cpu_avg_time'] = np.mean(results['cpu_times'])
            results['cpu_std_time'] = np.std(results['cpu_times'])
            results['speedup'] = results['cpu_avg_time'] / results['gpu_avg_time']
        
        return results
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        stats = {
            'cpu_documents': len(self.documents),
            'faiss_index_size': self.index.ntotal if self.index else 0,
            'gpu_enabled': self.use_gpu
        }
        
        if self.use_gpu and CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                stats['gpu_memory_used_mb'] = mempool.used_bytes() / 1024 / 1024
                stats['gpu_memory_total_mb'] = mempool.total_bytes() / 1024 / 1024
                
                if self.gpu_embeddings is not None:
                    stats['gpu_embeddings_shape'] = self.gpu_embeddings.shape
                    stats['gpu_embeddings_size_mb'] = (
                        self.gpu_embeddings.nbytes / 1024 / 1024
                    )
            except Exception as e:
                logger.warning(f"Failed to get GPU memory stats: {e}")
        
        if GPU_AVAILABLE and torch is not None:
            try:
                stats['torch_gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                stats['torch_gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
            except Exception as e:
                logger.warning(f"Failed to get PyTorch GPU memory stats: {e}")
        
        return stats


async def test_gpu_memory_system():
    """Test the GPU-accelerated memory system"""
    print("🧠 Testing GPU-Accelerated Memory System")
    print("=" * 50)
    
    # Create vector store
    store = GPUAcceleratedVectorStore(dimension=384, use_gpu=True)
    
    # Generate test documents
    print("Generating test documents...")
    documents = []
    for i in range(1000):
        embedding = np.random.random(384).astype(np.float32)
        doc = GPUVectorDocument(
            id=f"doc_{i}",
            content=f"This is test document {i}",
            embedding=embedding
        )
        documents.append(doc)
    
    # Add documents
    print("Adding documents to store...")
    start_time = time.time()
    for doc in documents:
        await store.add_document(doc)
    add_time = time.time() - start_time
    
    print(f"✅ Added {len(documents)} documents in {add_time:.2f}s")
    
    # Test search
    query_embedding = np.random.random(384).astype(np.float32)
    
    print("\nTesting search performance...")
    results = await store.benchmark_performance(query_embedding, iterations=50)
    
    print(f"GPU Search: {results.get('gpu_avg_time', 'N/A'):.4f}s avg")
    print(f"CPU Search: {results.get('cpu_avg_time', 'N/A'):.4f}s avg")
    if 'speedup' in results:
        print(f"Speedup: {results['speedup']:.2f}x")
    
    # Memory usage
    memory_stats = store.get_memory_usage()
    print(f"\nMemory Usage:")
    for key, value in memory_stats.items():
        print(f"  {key}: {value}")
    
    return store, results, memory_stats


if __name__ == "__main__":
    asyncio.run(test_gpu_memory_system())
