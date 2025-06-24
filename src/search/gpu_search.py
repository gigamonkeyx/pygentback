"""
GPU-Accelerated Vector Search using FAISS

High-performance vector similarity search with GPU acceleration,
supporting various index types and optimization strategies.
"""

import logging
import numpy as np
import os
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)

# Optional FAISS import with fallback
try:
    import faiss
    FAISS_AVAILABLE = True

    # Check for GPU support - can be disabled via environment variable
    FAISS_GPU_AVAILABLE = False
    
    # Skip GPU check if explicitly disabled
    if os.getenv('FAISS_FORCE_CPU', '').lower() in ('1', 'true', 'yes'):
        logger.info("FAISS GPU check skipped (FAISS_FORCE_CPU=true), using CPU only")
    else:
        try:
            # Try to create GPU resources
            gpu_res = faiss.StandardGpuResources()
            # Try to access GPU-specific classes
            if hasattr(faiss, 'GpuIndexIVFFlat'):
                FAISS_GPU_AVAILABLE = True
                logger.info("FAISS GPU support detected and available")
            else:
                logger.warning("FAISS GPU classes not available, using CPU only")
        except Exception as e:
            logger.warning(f"FAISS GPU support not available ({e}), falling back to CPU")

except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False
    logger.warning("FAISS not available, vector search will use fallback implementation")


class IndexType(Enum):
    """Supported FAISS index types"""
    FLAT = "flat"                    # Brute force exact search
    IVF_FLAT = "ivf_flat"           # Inverted file with flat vectors
    IVF_PQ = "ivf_pq"               # Inverted file with product quantization
    HNSW = "hnsw"                   # Hierarchical navigable small world


@dataclass
class VectorSearchConfig:
    """Configuration for vector search operations"""
    # Index configuration
    index_type: IndexType = IndexType.IVF_FLAT
    dimension: int = 768
    nlist: int = 100                # Number of clusters for IVF
    nprobe: int = 10               # Number of clusters to search

    # GPU configuration
    use_gpu: bool = False          # Default to CPU since GPU often not available
    gpu_id: int = 0
    use_float16: bool = False
    use_cuvs: bool = True          # Use NVIDIA cuVS if available

    def __post_init__(self):
        """Auto-detect and enable GPU if available"""
        if FAISS_GPU_AVAILABLE and not self.use_gpu:
            logger.info("GPU support detected, enabling GPU acceleration")
            self.use_gpu = True
    
    # Memory configuration
    gpu_memory_fraction: float = 0.8
    temp_memory_mb: int = 1024
    
    # Search configuration
    metric_type: str = "L2"        # L2 or INNER_PRODUCT
    batch_size: int = 1000
    max_vectors: int = 1000000
    
    # Optimization
    enable_precomputed_tables: bool = True
    indices_options: str = "INDICES_64_BIT"  # INDICES_CPU, INDICES_32_BIT, INDICES_64_BIT


@dataclass
class SearchResult:
    """Result of a vector search operation"""
    distances: np.ndarray
    indices: np.ndarray
    query_time: float
    total_results: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchSearchResult:
    """Result of a batch search operation"""
    results: List[SearchResult]
    total_time: float
    average_query_time: float
    throughput_qps: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class FaissGpuManager:
    """Manages FAISS GPU resources and configuration"""
    
    def __init__(self, config: VectorSearchConfig):
        self.config = config
        self.resources = None
        self.initialized = False
        
        if FAISS_GPU_AVAILABLE and config.use_gpu:
            self._initialize_gpu_resources()
        elif not FAISS_AVAILABLE:
            logger.error("FAISS not available - cannot initialize GPU manager")
    
    def _initialize_gpu_resources(self):
        """Initialize GPU resources for FAISS"""
        try:
            self.resources = faiss.StandardGpuResources()
            
            # Configure temporary memory
            temp_memory_bytes = self.config.temp_memory_mb * 1024 * 1024
            self.resources.setTempMemory(temp_memory_bytes)
            
            # Set default stream if needed
            if hasattr(self.resources, 'setDefaultNullStreamAllDevices'):
                self.resources.setDefaultNullStreamAllDevices()
            
            self.initialized = True
            logger.info(f"FAISS GPU resources initialized with {self.config.temp_memory_mb}MB temp memory")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS GPU resources: {e}")
            self.initialized = False
    
    def get_resources(self):
        """Get GPU resources object"""
        return self.resources if self.initialized else None
    
    def cleanup(self):
        """Clean up GPU resources"""
        if self.resources:
            # FAISS handles cleanup automatically
            self.resources = None
            self.initialized = False


class GpuVectorIndex:
    """GPU-accelerated vector index using FAISS"""
    
    def __init__(self, config: VectorSearchConfig):
        self.config = config
        self.gpu_manager = FaissGpuManager(config)
        self.index = None
        self.cpu_index = None
        self.trained = False
        self.vector_count = 0
        
        # Statistics
        self.search_count = 0
        self.total_search_time = 0.0
        
        if FAISS_AVAILABLE:
            self._create_index()
        else:
            logger.error("FAISS not available - using fallback implementation")
    
    def _create_index(self):
        """Create the appropriate FAISS index"""
        try:
            # Create CPU index first
            if self.config.index_type == IndexType.FLAT:
                if self.config.metric_type == "L2":
                    self.cpu_index = faiss.IndexFlatL2(self.config.dimension)
                else:
                    self.cpu_index = faiss.IndexFlatIP(self.config.dimension)
                    
            elif self.config.index_type == IndexType.IVF_FLAT:
                # Create quantizer
                if self.config.metric_type == "L2":
                    quantizer = faiss.IndexFlatL2(self.config.dimension)
                    self.cpu_index = faiss.IndexIVFFlat(
                        quantizer, self.config.dimension, self.config.nlist
                    )
                else:
                    quantizer = faiss.IndexFlatIP(self.config.dimension)
                    self.cpu_index = faiss.IndexIVFFlat(
                        quantizer, self.config.dimension, self.config.nlist, 
                        faiss.METRIC_INNER_PRODUCT
                    )
                    
            elif self.config.index_type == IndexType.IVF_PQ:
                # Product quantization parameters
                m = 8  # Number of subquantizers
                bits = 8  # Bits per subquantizer
                
                if self.config.metric_type == "L2":
                    quantizer = faiss.IndexFlatL2(self.config.dimension)
                    self.cpu_index = faiss.IndexIVFPQ(
                        quantizer, self.config.dimension, self.config.nlist, m, bits
                    )
                else:
                    quantizer = faiss.IndexFlatIP(self.config.dimension)
                    self.cpu_index = faiss.IndexIVFPQ(
                        quantizer, self.config.dimension, self.config.nlist, m, bits,
                        faiss.METRIC_INNER_PRODUCT
                    )
            
            # Move to GPU if available
            if FAISS_GPU_AVAILABLE and self.config.use_gpu:
                self._move_to_gpu()
            else:
                self.index = self.cpu_index
                
            logger.info(f"Created {self.config.index_type.value} index with dimension {self.config.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            self.index = None
    
    def _move_to_gpu(self):
        """Move index to GPU"""
        try:
            resources = self.gpu_manager.get_resources()
            if not resources:
                logger.warning("GPU resources not available, using CPU index")
                self.index = self.cpu_index
                return
            
            # Configure GPU cloner options
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.config.use_float16
            
            # Set cuVS if available
            if hasattr(co, 'use_cuvs'):
                co.use_cuvs = self.config.use_cuvs
            
            # Set indices options
            if self.config.indices_options == "INDICES_CPU":
                co.indicesOptions = faiss.INDICES_CPU
            elif self.config.indices_options == "INDICES_32_BIT":
                co.indicesOptions = faiss.INDICES_32_BIT
            else:
                co.indicesOptions = faiss.INDICES_64_BIT
            
            # Clone to GPU
            self.index = faiss.index_cpu_to_gpu(
                resources, self.config.gpu_id, self.cpu_index, co
            )
            
            logger.info(f"Moved index to GPU {self.config.gpu_id}")
            
        except Exception as e:
            logger.error(f"Failed to move index to GPU: {e}")
            self.index = self.cpu_index
    
    def train(self, vectors: np.ndarray) -> bool:
        """Train the index with vectors"""
        if not self.index:
            logger.error("Index not initialized")
            return False
        
        try:
            # Ensure vectors are float32
            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)
            
            # Check dimensions
            if vectors.shape[1] != self.config.dimension:
                raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match config {self.config.dimension}")
            
            # Train the index
            start_time = time.time()
            self.index.train(vectors)
            train_time = time.time() - start_time
            
            self.trained = True
            logger.info(f"Trained index with {len(vectors)} vectors in {train_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train index: {e}")
            return False
    
    def add(self, vectors: np.ndarray) -> bool:
        """Add vectors to the index"""
        if not self.index:
            logger.error("Index not initialized")
            return False
        
        if not self.trained and self.config.index_type != IndexType.FLAT:
            logger.error("Index must be trained before adding vectors")
            return False
        
        try:
            # Ensure vectors are float32
            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)
            
            # Add in batches if necessary
            batch_size = self.config.batch_size
            total_vectors = len(vectors)
            
            start_time = time.time()
            for i in range(0, total_vectors, batch_size):
                batch = vectors[i:i + batch_size]
                self.index.add(batch)
            
            add_time = time.time() - start_time
            self.vector_count += total_vectors
            
            logger.info(f"Added {total_vectors} vectors in {add_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False
    
    def search(self, query_vectors: np.ndarray, k: int = 10) -> SearchResult:
        """Search for nearest neighbors"""
        if not self.index:
            logger.error("Index not initialized")
            return SearchResult(np.array([]), np.array([]), 0.0, 0)
        
        try:
            # Ensure query vectors are float32
            if query_vectors.dtype != np.float32:
                query_vectors = query_vectors.astype(np.float32)
            
            # Set search parameters for IVF indices
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.config.nprobe
            
            # Perform search
            start_time = time.time()
            distances, indices = self.index.search(query_vectors, k)
            search_time = time.time() - start_time
            
            # Update statistics
            self.search_count += len(query_vectors)
            self.total_search_time += search_time
            
            return SearchResult(
                distances=distances,
                indices=indices,
                query_time=search_time,
                total_results=len(query_vectors) * k,
                metadata={
                    'k': k,
                    'nprobe': getattr(self.index, 'nprobe', None),
                    'index_type': self.config.index_type.value
                }
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResult(np.array([]), np.array([]), 0.0, 0)
    
    def batch_search(self, query_batches: List[np.ndarray], k: int = 10) -> BatchSearchResult:
        """Perform batch search operations"""
        results = []
        start_time = time.time()
        
        for batch in query_batches:
            result = self.search(batch, k)
            results.append(result)
        
        total_time = time.time() - start_time
        total_queries = sum(len(batch) for batch in query_batches)
        avg_query_time = total_time / max(1, total_queries)
        throughput = total_queries / max(0.001, total_time)
        
        return BatchSearchResult(
            results=results,
            total_time=total_time,
            average_query_time=avg_query_time,
            throughput_qps=throughput,
            metadata={
                'total_queries': total_queries,
                'batch_count': len(query_batches),
                'k': k
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        stats = {
            'vector_count': self.vector_count,
            'search_count': self.search_count,
            'total_search_time': self.total_search_time,
            'avg_search_time': self.total_search_time / max(1, self.search_count),
            'trained': self.trained,
            'index_type': self.config.index_type.value,
            'dimension': self.config.dimension,
            'gpu_enabled': FAISS_GPU_AVAILABLE and self.config.use_gpu
        }
        
        if hasattr(self.index, 'ntotal'):
            stats['index_size'] = self.index.ntotal
            
        return stats
    
    def save(self, filepath: str) -> bool:
        """Save index to disk"""
        try:
            # Move to CPU for saving
            if FAISS_GPU_AVAILABLE and self.config.use_gpu and self.index:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, filepath)
            elif self.index:
                faiss.write_index(self.index, filepath)
            else:
                logger.error("No index to save")
                return False
                
            logger.info(f"Saved index to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def load(self, filepath: str) -> bool:
        """Load index from disk"""
        try:
            # Load CPU index
            self.cpu_index = faiss.read_index(filepath)
            self.trained = True
            self.vector_count = self.cpu_index.ntotal
            
            # Move to GPU if configured
            if FAISS_GPU_AVAILABLE and self.config.use_gpu:
                self._move_to_gpu()
            else:
                self.index = self.cpu_index
                
            logger.info(f"Loaded index from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        self.index = None
        self.cpu_index = None
        self.gpu_manager.cleanup()


# Fallback implementation when FAISS is not available
class FallbackVectorIndex:
    """Simple fallback vector index using numpy"""
    
    def __init__(self, config: VectorSearchConfig):
        self.config = config
        self.vectors = None
        self.vector_count = 0
        
    def train(self, vectors: np.ndarray) -> bool:
        return True
    
    def add(self, vectors: np.ndarray) -> bool:
        if self.vectors is None:
            self.vectors = vectors.copy()
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.vector_count += len(vectors)
        return True
    
    def search(self, query_vectors: np.ndarray, k: int = 10) -> SearchResult:
        if self.vectors is None:
            return SearchResult(np.array([]), np.array([]), 0.0, 0)
        
        start_time = time.time()
        
        # Simple brute force search
        distances = []
        indices = []
        
        for query in query_vectors:
            if self.config.metric_type == "L2":
                dists = np.linalg.norm(self.vectors - query, axis=1)
            else:
                dists = -np.dot(self.vectors, query)  # Negative for top-k
            
            top_k_indices = np.argpartition(dists, min(k, len(dists)-1))[:k]
            top_k_indices = top_k_indices[np.argsort(dists[top_k_indices])]
            
            distances.append(dists[top_k_indices])
            indices.append(top_k_indices)
        
        search_time = time.time() - start_time
        
        return SearchResult(
            distances=np.array(distances),
            indices=np.array(indices),
            query_time=search_time,
            total_results=len(query_vectors) * k
        )
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'vector_count': self.vector_count,
            'index_type': 'fallback',
            'dimension': self.config.dimension
        }


# Factory function
def create_vector_index(config: VectorSearchConfig) -> Union[GpuVectorIndex, FallbackVectorIndex]:
    """Create appropriate vector index based on availability"""
    if FAISS_AVAILABLE:
        return GpuVectorIndex(config)
    else:
        logger.warning("Using fallback vector index - install faiss-gpu for better performance")
        return FallbackVectorIndex(config)
