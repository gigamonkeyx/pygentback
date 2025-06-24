#!/usr/bin/env python3
"""
Real GPU Usage Test

Test actual GPU utilization in the PyGent Factory memory system
and identify what components are and aren't using the NVIDIA 3080.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_pytorch_gpu():
    """Test PyTorch GPU usage"""
    print("🔥 PyTorch GPU Test")
    print("=" * 30)
    
    try:
        import torch
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            
            # Test actual GPU computation
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            start_time = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            print(f"✅ GPU computation successful: {gpu_time:.4f}s")
            print(f"✅ GPU memory used: {memory_allocated:.1f}MB")
            
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch GPU test failed: {e}")
        return False


def test_cupy_gpu():
    """Test CuPy GPU usage"""
    print("\n🚀 CuPy GPU Test")
    print("=" * 25)
    
    try:
        import cupy as cp
        
        print(f"CuPy version: {cp.__version__}")
        print(f"CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}")
        
        # Test actual GPU computation
        x_gpu = cp.random.random((1000, 1000), dtype=cp.float32)
        y_gpu = cp.random.random((1000, 1000), dtype=cp.float32)
        
        start_time = time.time()
        z_gpu = cp.dot(x_gpu, y_gpu)
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - start_time
        
        # Check memory usage
        mempool = cp.get_default_memory_pool()
        memory_used = mempool.used_bytes() / 1024 / 1024  # MB
        
        print(f"✅ CuPy GPU computation successful: {gpu_time:.4f}s")
        print(f"✅ GPU memory used: {memory_used:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ CuPy GPU test failed: {e}")
        return False


def test_faiss_gpu():
    """Test FAISS GPU capabilities"""
    print("\n🔍 FAISS GPU Test")
    print("=" * 25)
    
    try:
        import faiss
        
        print(f"FAISS version: {faiss.__version__}")
        
        # Check for GPU support
        has_gpu_support = hasattr(faiss, 'GpuIndexFlatL2')
        print(f"GPU support compiled: {has_gpu_support}")
        
        if has_gpu_support:
            try:
                # Try to create GPU resources
                res = faiss.StandardGpuResources()
                
                # Create a simple GPU index
                dimension = 128
                index_cpu = faiss.IndexFlatL2(dimension)
                index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
                
                # Test with some data
                vectors = np.random.random((1000, dimension)).astype('float32')
                index_gpu.add(vectors)
                
                # Search test
                query = np.random.random((10, dimension)).astype('float32')
                start_time = time.time()
                distances, indices = index_gpu.search(query, 5)
                search_time = time.time() - start_time
                
                print(f"✅ FAISS GPU search successful: {search_time:.4f}s")
                return True
                
            except Exception as e:
                print(f"❌ FAISS GPU resources failed: {e}")
                return False
        else:
            print("❌ FAISS compiled without GPU support")
            return False
            
    except Exception as e:
        print(f"❌ FAISS test failed: {e}")
        return False


def test_memory_system_gpu():
    """Test PyGent Factory memory system GPU usage"""
    print("\n🧠 Memory System GPU Test")
    print("=" * 35)
    
    try:
        # Try to import memory components
        from src.memory.memory_manager import MemorySpace, MemoryType
        from src.storage.vector_store import VectorStoreManager
        
        print("✅ Memory components imported")
        
        # Check if vector store can use GPU
        # This will likely fail because FAISS-GPU isn't installed
        config = {
            "max_entries": 1000,
            "retention_threshold": 0.1,
            "use_gpu": True,
            "gpu_id": 0
        }
        
        vector_manager = VectorStoreManager()
        print("✅ Vector store manager created")
        
        # This is where it will likely fail
        try:
            memory_space = MemorySpace("test_agent", vector_manager, config)
            print("✅ Memory space created")
            return True
        except Exception as e:
            print(f"❌ Memory space creation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        return False


def test_embedding_gpu():
    """Test embedding generation with GPU"""
    print("\n📊 Embedding GPU Test")
    print("=" * 30)
    
    try:
        from src.utils.embedding import EmbeddingService
        from src.config.settings import Settings
        
        settings = Settings()
        embedding_service = EmbeddingService(settings)
        
        print("✅ Embedding service created")
        
        # Test if sentence transformers can use GPU
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            
            print(f"✅ SentenceTransformer device: {device}")
            
            # Test encoding
            texts = ["This is a test sentence", "Another test sentence"]
            start_time = time.time()
            embeddings = model.encode(texts)
            encode_time = time.time() - start_time
            
            print(f"✅ Embedding generation: {encode_time:.4f}s")
            print(f"✅ Embedding shape: {embeddings.shape}")
            
            if device == 'cuda':
                print("✅ Embeddings using GPU!")
                return True
            else:
                print("⚠️ Embeddings using CPU only")
                return False
                
        except Exception as e:
            print(f"❌ Embedding GPU test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Embedding service test failed: {e}")
        return False


def main():
    """Run comprehensive GPU usage tests"""
    print("🔬 PyGent Factory GPU Usage Analysis")
    print("=" * 50)
    print(f"Target GPU: NVIDIA GeForce RTX 3080")
    print()
    
    results = {
        'pytorch_gpu': test_pytorch_gpu(),
        'cupy_gpu': test_cupy_gpu(),
        'faiss_gpu': test_faiss_gpu(),
        'memory_system_gpu': test_memory_system_gpu(),
        'embedding_gpu': test_embedding_gpu()
    }
    
    print("\n" + "=" * 50)
    print("🏁 GPU Usage Summary")
    print("=" * 50)
    
    working_count = sum(results.values())
    total_count = len(results)
    
    for component, working in results.items():
        status = "✅ WORKING" if working else "❌ NOT WORKING"
        print(f"{component.replace('_', ' ').title()}: {status}")
    
    print(f"\nGPU Utilization: {working_count}/{total_count} components")
    
    if working_count == 0:
        print("🚨 CRITICAL: NO GPU UTILIZATION DETECTED!")
    elif working_count < total_count:
        print("⚠️ WARNING: Partial GPU utilization")
    else:
        print("🎉 EXCELLENT: Full GPU utilization")
    
    print("\n🔧 Recommendations:")
    
    if not results['faiss_gpu']:
        print("- Install FAISS-GPU: conda install -c conda-forge faiss-gpu")
    
    if not results['embedding_gpu']:
        print("- Ensure SentenceTransformers uses GPU device")
    
    if not results['memory_system_gpu']:
        print("- Fix memory system GPU integration")
    
    # Save results
    import json
    with open('gpu_usage_analysis.json', 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'results': results,
            'gpu_utilization_ratio': working_count / total_count,
            'recommendations': [
                "Install FAISS-GPU for vector search acceleration",
                "Configure memory system for GPU usage",
                "Optimize embedding generation for GPU"
            ]
        }, f, indent=2)
    
    print(f"📄 Results saved to: gpu_usage_analysis.json")


if __name__ == "__main__":
    main()
