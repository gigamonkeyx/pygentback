"""
GPU Vector Search Example

Demonstrates GPU-accelerated vector search capabilities using FAISS
with PyGent Factory's search infrastructure.
"""

import asyncio
import numpy as np
import logging
from typing import List, Dict, Any

from src.search.gpu_search import (
    create_vector_index, VectorSearchConfig, IndexType, 
    FAISS_AVAILABLE, FAISS_GPU_AVAILABLE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_vectors(n_vectors: int, dimension: int) -> np.ndarray:
    """Generate sample vectors for testing"""
    # Create some structured data with clusters
    np.random.seed(42)
    
    # Create 3 clusters
    cluster_centers = np.random.randn(3, dimension) * 2
    vectors = []
    
    for i in range(n_vectors):
        cluster_id = i % 3
        # Add noise around cluster center
        vector = cluster_centers[cluster_id] + np.random.randn(dimension) * 0.5
        vectors.append(vector)
    
    return np.array(vectors, dtype=np.float32)


def test_basic_vector_search():
    """Test basic vector search functionality"""
    print("🔍 Testing Basic Vector Search")
    print("=" * 40)
    
    # Configuration
    config = VectorSearchConfig(
        index_type=IndexType.FLAT,  # Start with simple flat index
        dimension=128,
        use_gpu=FAISS_GPU_AVAILABLE,
        batch_size=1000
    )
    
    print(f"FAISS Available: {FAISS_AVAILABLE}")
    print(f"FAISS GPU Available: {FAISS_GPU_AVAILABLE}")
    print(f"Using GPU: {config.use_gpu}")
    print()
    
    # Create index
    index = create_vector_index(config)
    
    # Generate test data
    n_vectors = 10000
    vectors = generate_sample_vectors(n_vectors, config.dimension)
    query_vectors = generate_sample_vectors(5, config.dimension)
    
    print(f"Generated {n_vectors} vectors with {config.dimension} dimensions")
    
    # Train and add vectors
    print("Training index...")
    train_success = index.train(vectors)
    print(f"Training successful: {train_success}")
    
    print("Adding vectors...")
    add_success = index.add(vectors)
    print(f"Adding successful: {add_success}")
    
    # Search
    print("Performing search...")
    k = 10
    result = index.search(query_vectors, k=k)
    
    print(f"Search completed in {result.query_time:.4f}s")
    print(f"Found {result.total_results} results")
    print(f"Result shape: {result.distances.shape}")
    print()
    
    # Show some results
    print("Sample results:")
    for i in range(min(3, len(query_vectors))):
        print(f"Query {i}: Top 3 distances = {result.distances[i][:3]}")
    
    # Get statistics
    stats = index.get_stats()
    print("\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return index


def test_ivf_index():
    """Test IVF (Inverted File) index"""
    print("\n🏗️ Testing IVF Index")
    print("=" * 30)
    
    # Configuration for IVF
    config = VectorSearchConfig(
        index_type=IndexType.IVF_FLAT,
        dimension=256,
        nlist=50,  # Number of clusters
        nprobe=5,  # Number of clusters to search
        use_gpu=FAISS_GPU_AVAILABLE,
        use_float16=True,  # Memory optimization
        batch_size=2000
    )
    
    print(f"Index type: {config.index_type.value}")
    print(f"Clusters (nlist): {config.nlist}")
    print(f"Search clusters (nprobe): {config.nprobe}")
    print(f"Using float16: {config.use_float16}")
    print()
    
    # Create index
    index = create_vector_index(config)
    
    # Generate larger dataset
    n_vectors = 50000
    vectors = generate_sample_vectors(n_vectors, config.dimension)
    query_vectors = generate_sample_vectors(10, config.dimension)
    
    print(f"Generated {n_vectors} vectors for IVF test")
    
    # Train with subset (IVF needs training)
    training_vectors = vectors[:10000]  # Use subset for training
    print("Training IVF index...")
    train_success = index.train(training_vectors)
    print(f"Training successful: {train_success}")
    
    # Add all vectors
    print("Adding vectors to index...")
    add_success = index.add(vectors)
    print(f"Adding successful: {add_success}")
    
    # Benchmark search
    print("Benchmarking search performance...")
    k_values = [1, 5, 10, 50]
    
    for k in k_values:
        result = index.search(query_vectors, k=k)
        qps = len(query_vectors) / result.query_time
        print(f"k={k:2d}: {result.query_time:.4f}s, {qps:.1f} QPS")
    
    return index


def test_batch_search():
    """Test batch search capabilities"""
    print("\n📦 Testing Batch Search")
    print("=" * 25)
    
    config = VectorSearchConfig(
        index_type=IndexType.IVF_FLAT,
        dimension=128,
        nlist=20,
        nprobe=3,
        use_gpu=FAISS_GPU_AVAILABLE,
        batch_size=1000
    )
    
    # Create and populate index
    index = create_vector_index(config)
    vectors = generate_sample_vectors(20000, config.dimension)
    
    index.train(vectors[:5000])
    index.add(vectors)
    
    # Create multiple query batches
    batch_sizes = [10, 50, 100, 500]
    query_batches = []
    
    for batch_size in batch_sizes:
        batch = generate_sample_vectors(batch_size, config.dimension)
        query_batches.append(batch)
    
    print(f"Created {len(query_batches)} query batches")
    print(f"Batch sizes: {[len(batch) for batch in query_batches]}")
    
    # Perform batch search
    batch_result = index.batch_search(query_batches, k=10)
    
    print(f"\nBatch Search Results:")
    print(f"Total time: {batch_result.total_time:.4f}s")
    print(f"Average query time: {batch_result.average_query_time:.6f}s")
    print(f"Throughput: {batch_result.throughput_qps:.1f} QPS")
    print(f"Total queries: {batch_result.metadata['total_queries']}")
    
    return index


def test_memory_optimization():
    """Test memory optimization features"""
    print("\n💾 Testing Memory Optimization")
    print("=" * 35)
    
    # Test different memory configurations
    configs = [
        ("Standard", VectorSearchConfig(
            index_type=IndexType.IVF_FLAT,
            dimension=512,
            use_float16=False,
            temp_memory_mb=2048
        )),
        ("Float16", VectorSearchConfig(
            index_type=IndexType.IVF_FLAT,
            dimension=512,
            use_float16=True,
            temp_memory_mb=1024
        )),
        ("CPU Indices", VectorSearchConfig(
            index_type=IndexType.IVF_FLAT,
            dimension=512,
            use_float16=True,
            indices_options="INDICES_CPU",
            temp_memory_mb=512
        ))
    ]
    
    vectors = generate_sample_vectors(10000, 512)
    query_vectors = generate_sample_vectors(100, 512)
    
    for name, config in configs:
        print(f"\nTesting {name} configuration:")
        print(f"  Float16: {config.use_float16}")
        print(f"  Indices: {config.indices_options}")
        print(f"  Temp memory: {config.temp_memory_mb}MB")
        
        try:
            index = create_vector_index(config)
            index.train(vectors[:2000])
            index.add(vectors)
            
            result = index.search(query_vectors, k=5)
            stats = index.get_stats()
            
            print(f"  Search time: {result.query_time:.4f}s")
            print(f"  Index size: {stats.get('index_size', 'N/A')}")
            print(f"  GPU enabled: {stats.get('gpu_enabled', False)}")
            
        except Exception as e:
            print(f"  Error: {e}")


def test_save_load():
    """Test index persistence"""
    print("\n💾 Testing Index Save/Load")
    print("=" * 30)
    
    config = VectorSearchConfig(
        index_type=IndexType.IVF_FLAT,
        dimension=128,
        nlist=10,
        use_gpu=FAISS_GPU_AVAILABLE
    )
    
    # Create and populate index
    index1 = create_vector_index(config)
    vectors = generate_sample_vectors(5000, config.dimension)
    query_vectors = generate_sample_vectors(3, config.dimension)
    
    index1.train(vectors[:1000])
    index1.add(vectors)
    
    # Search with original index
    result1 = index1.search(query_vectors, k=5)
    print(f"Original index search time: {result1.query_time:.4f}s")
    
    # Save index
    filepath = "test_index.faiss"
    save_success = index1.save(filepath)
    print(f"Save successful: {save_success}")
    
    if save_success:
        # Create new index and load
        index2 = create_vector_index(config)
        load_success = index2.load(filepath)
        print(f"Load successful: {load_success}")
        
        if load_success:
            # Search with loaded index
            result2 = index2.search(query_vectors, k=5)
            print(f"Loaded index search time: {result2.query_time:.4f}s")
            
            # Compare results
            distance_diff = np.abs(result1.distances - result2.distances).max()
            print(f"Max distance difference: {distance_diff:.6f}")
            
            if distance_diff < 1e-5:
                print("✅ Save/load successful - results match!")
            else:
                print("⚠️ Results differ after save/load")
        
        # Cleanup
        try:
            import os
            os.remove(filepath)
            print("Cleaned up test file")
        except:
            pass


async def main():
    """Run all GPU vector search examples"""
    print("🚀 PyGent Factory GPU Vector Search Examples")
    print("=" * 50)
    
    if not FAISS_AVAILABLE:
        print("❌ FAISS not available. Install with:")
        print("   pip install faiss-cpu  # or faiss-gpu for GPU support")
        return
    
    print(f"✅ FAISS available")
    print(f"🎮 GPU support: {FAISS_GPU_AVAILABLE}")
    print()
    
    try:
        # Run tests
        test_basic_vector_search()
        test_ivf_index()
        test_batch_search()
        test_memory_optimization()
        test_save_load()
        
        print("\n🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("1. Integrate with your embedding models")
        print("2. Use with ToT reasoning for semantic search")
        print("3. Scale up to larger datasets")
        print("4. Experiment with different index types")
        
    except Exception as e:
        logger.exception("Example failed")
        print(f"\n❌ Example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
