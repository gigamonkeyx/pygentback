#!/usr/bin/env python3
"""
Comprehensive Memory System Test

Tests all aspects of the PyGent Factory memory system including:
- GPU acceleration (PyTorch, CuPy, FAISS)
- Vector storage and retrieval
- Embedding generation
- Memory management
- Database persistence
"""

import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test results
test_results = {
    "timestamp": datetime.now().isoformat(),
    "gpu_tests": {},
    "memory_tests": {},
    "vector_tests": {},
    "embedding_tests": {},
    "issues": [],
    "recommendations": []
}

def log_issue(category: str, description: str, severity: str = "error"):
    """Log an issue found during testing"""
    test_results["issues"].append({
        "category": category,
        "description": description,
        "severity": severity,
        "timestamp": datetime.now().isoformat()
    })
    print(f"⚠️ {severity.upper()}: {description}")

def log_recommendation(description: str):
    """Log a recommendation"""
    test_results["recommendations"].append(description)
    print(f"💡 RECOMMENDATION: {description}")

def test_gpu_acceleration():
    """Test GPU acceleration capabilities"""
    print("🚀 Testing GPU Acceleration")
    print("=" * 40)
    
    # Test PyTorch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        test_results["gpu_tests"]["pytorch_cuda"] = {
            "available": cuda_available,
            "version": torch.version.cuda if cuda_available else None,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None
        }
        
        if cuda_available:
            print(f"✅ PyTorch CUDA: {torch.version.cuda}")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
              # Test basic GPU operation
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000, device=device)
            _ = torch.matmul(x, x)  # Test GPU computation
            torch.cuda.synchronize()
            print("   ✅ Basic GPU operations working")
            
        else:
            log_issue("gpu", "PyTorch CUDA not available", "error")
            
    except Exception as e:
        log_issue("gpu", f"PyTorch CUDA test failed: {e}", "error")
    
    # Test CuPy
    try:
        import cupy as cp
        test_results["gpu_tests"]["cupy"] = {
            "available": True,
            "version": cp.__version__,
            "cuda_version": cp.cuda.runtime.runtimeGetVersion()
        }
        
        print(f"✅ CuPy: {cp.__version__}")
        
        # Test basic CuPy operation
        x_gpu = cp.random.random((1000, 1000))
        y_gpu = cp.dot(x_gpu, x_gpu)
        cp.cuda.Stream.null.synchronize()
        print("   ✅ Basic CuPy operations working")
        
    except Exception as e:
        log_issue("gpu", f"CuPy test failed: {e}", "error")
        test_results["gpu_tests"]["cupy"] = {"available": False, "error": str(e)}
    
    # Test FAISS
    try:
        import faiss
        test_results["gpu_tests"]["faiss"] = {
            "available": True,
            "version": faiss.__version__ if hasattr(faiss, '__version__') else "unknown"
        }
        
        # Check for GPU support
        try:
            gpu_res = faiss.StandardGpuResources()
            test_results["gpu_tests"]["faiss"]["gpu_support"] = True
            print(f"✅ FAISS with GPU support")
            
            # Test GPU index creation
            index = faiss.IndexFlatL2(128)
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
            print("   ✅ FAISS GPU index creation working")
            
        except Exception as e:
            test_results["gpu_tests"]["faiss"]["gpu_support"] = False
            log_issue("gpu", f"FAISS GPU support not available: {e}", "warning")
            print(f"⚠️ FAISS CPU-only: {e}")
            
    except ImportError:
        log_issue("gpu", "FAISS not installed", "error")
        test_results["gpu_tests"]["faiss"] = {"available": False}

async def test_memory_system():
    """Test the memory management system"""
    print("\n🧠 Testing Memory System")
    print("=" * 40)
    
    try:
        from memory.memory_manager import MemoryManager, MemoryType, MemoryImportance
        from storage.vector_store import VectorStoreManager
        from config.settings import Settings
        
        # Initialize components
        settings = Settings()
        vector_store_manager = VectorStoreManager(settings)
        await vector_store_manager.initialize()
        
        memory_manager = MemoryManager(
            vector_store_manager=vector_store_manager,
            config={"max_entries": 1000}
        )
        await memory_manager.initialize()
        
        print("✅ Memory system initialized")
        test_results["memory_tests"]["initialization"] = True
        
        # Test memory storage
        agent_id = "test_agent"
        memory_space = await memory_manager.get_or_create_memory_space(agent_id)
        
        memory_id = await memory_space.store_memory(
            content="Test memory content",
            memory_type=MemoryType.SHORT_TERM,
            importance=MemoryImportance.HIGH
        )
        
        print("✅ Memory storage working")
        test_results["memory_tests"]["storage"] = True
        
        # Test memory retrieval
        memories = await memory_space.retrieve_memories(
            query="test content",
            limit=5
        )
        
        if memories:
            print("✅ Memory retrieval working")
            test_results["memory_tests"]["retrieval"] = True
        else:
            log_issue("memory", "Memory retrieval returned no results", "warning")
            test_results["memory_tests"]["retrieval"] = False
        
    except Exception as e:
        log_issue("memory", f"Memory system test failed: {e}", "error")
        test_results["memory_tests"]["error"] = str(e)

async def test_vector_storage():
    """Test vector storage capabilities"""
    print("\n📊 Testing Vector Storage")
    print("=" * 40)
    
    try:
        from storage.vector_store import VectorStoreManager, VectorDocument
        from config.settings import Settings
        
        settings = Settings()
        vector_manager = VectorStoreManager(settings)
        await vector_manager.initialize()
        
        print("✅ Vector store manager initialized")
        
        # Test document storage
        doc = VectorDocument(
            id="test_doc_1",
            content="This is a test document for vector storage",
            embedding=[0.1] * 384,  # Mock embedding
            metadata={"test": True}
        )
        
        store = await vector_manager.get_store("test_collection")
        await store.add_documents([doc])
        
        print("✅ Vector document storage working")
        test_results["vector_tests"]["storage"] = True
        
        # Test similarity search
        results = await store.similarity_search(
            query_embedding=[0.1] * 384,
            limit=5
        )
        
        if results:
            print("✅ Vector similarity search working")
            test_results["vector_tests"]["search"] = True
        else:
            log_issue("vector", "Vector search returned no results", "warning")
            test_results["vector_tests"]["search"] = False
        
    except Exception as e:
        log_issue("vector", f"Vector storage test failed: {e}", "error")
        test_results["vector_tests"]["error"] = str(e)

async def test_embedding_generation():
    """Test embedding generation"""
    print("\n🔤 Testing Embedding Generation")
    print("=" * 40)
    
    try:
        from utils.embedding import EmbeddingService
        from config.settings import Settings
        
        settings = Settings()
        embedding_service = EmbeddingService(settings)
        
        # Test single embedding
        result = await embedding_service.generate_embedding("Test text for embedding")
        
        if result and result.embedding:
            print(f"✅ Embedding generation working (dim: {len(result.embedding)})")
            test_results["embedding_tests"]["generation"] = True
            test_results["embedding_tests"]["dimension"] = len(result.embedding)
        else:
            log_issue("embedding", "Embedding generation failed", "error")
            test_results["embedding_tests"]["generation"] = False
        
        # Test batch embedding
        texts = ["Text 1", "Text 2", "Text 3"]
        batch_results = await embedding_service.generate_embeddings(texts)
        
        if len(batch_results) == len(texts):
            print("✅ Batch embedding generation working")
            test_results["embedding_tests"]["batch"] = True
        else:
            log_issue("embedding", "Batch embedding generation failed", "error")
            test_results["embedding_tests"]["batch"] = False
        
    except Exception as e:
        log_issue("embedding", f"Embedding generation test failed: {e}", "error")
        test_results["embedding_tests"]["error"] = str(e)

def test_database_connectivity():
    """Test database connectivity"""
    print("\n🗄️ Testing Database Connectivity")
    print("=" * 40)
    
    try:
        from config.database import get_database_url, create_engine
        from sqlalchemy import text
        
        # Test database connection
        db_url = get_database_url()
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            if result.fetchone():
                print("✅ Database connectivity working")
                test_results["memory_tests"]["database"] = True
            else:
                log_issue("database", "Database query failed", "error")
                test_results["memory_tests"]["database"] = False
        
    except Exception as e:
        log_issue("database", f"Database connectivity test failed: {e}", "error")
        test_results["memory_tests"]["database"] = False

def analyze_issues_and_recommendations():
    """Analyze found issues and generate recommendations"""
    print("\n📋 Analysis and Recommendations")
    print("=" * 40)
    
    # Check for GPU acceleration issues
    if not test_results["gpu_tests"].get("pytorch_cuda", {}).get("available"):
        log_recommendation("Install PyTorch with CUDA support for GPU acceleration")
    
    if not test_results["gpu_tests"].get("cupy", {}).get("available"):
        log_recommendation("Install CuPy for GPU-accelerated array operations")
    
    if not test_results["gpu_tests"].get("faiss", {}).get("gpu_support"):
        log_recommendation("Install FAISS-GPU for GPU-accelerated vector search")
    
    # Check for memory system issues
    if not test_results["memory_tests"].get("initialization"):
        log_recommendation("Fix memory system initialization issues")
    
    if not test_results["vector_tests"].get("storage"):
        log_recommendation("Fix vector storage system")
    
    if not test_results["embedding_tests"].get("generation"):
        log_recommendation("Fix embedding generation system")
    
    # Performance recommendations
    if test_results["gpu_tests"].get("pytorch_cuda", {}).get("available"):
        log_recommendation("Enable GPU acceleration in memory and vector systems")
    
    log_recommendation("Implement vector store interface compatibility fixes")
    log_recommendation("Add GPU memory monitoring and optimization")
    log_recommendation("Implement embedding caching for better performance")

async def main():
    """Run comprehensive memory system test"""
    print("🔍 PyGent Factory Memory System Comprehensive Test")
    print("=" * 50)
    
    # Run all tests
    test_gpu_acceleration()
    await test_memory_system()
    await test_vector_storage()
    await test_embedding_generation()
    test_database_connectivity()
    
    # Analyze and provide recommendations
    analyze_issues_and_recommendations()
    
    # Save results
    with open("memory_system_analysis.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n📊 Test completed. Found {len(test_results['issues'])} issues.")
    print(f"📝 Generated {len(test_results['recommendations'])} recommendations.")
    print("📁 Full results saved to memory_system_analysis.json")

if __name__ == "__main__":
    asyncio.run(main())
