#!/usr/bin/env python3
"""
Final Memory System Verification Test

Verifies that the critical fixes have resolved the memory system issues.
"""

import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_fixed_memory_system():
    """Test the fixed memory system components"""
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "overall_status": "unknown"
    }
    
    print("🔍 Final Memory System Verification")
    print("=" * 40)
    
    # Test 1: FAISS Import and Basic Operations
    try:
        import faiss
        
        # Test basic FAISS operations
        index = faiss.IndexFlatL2(128)
        print("✅ FAISS: Import and basic index creation working")
        results["tests"]["faiss"] = {"status": "pass", "version": getattr(faiss, '__version__', 'unknown')}
        
    except Exception as e:
        print(f"❌ FAISS: Failed - {e}")
        results["tests"]["faiss"] = {"status": "fail", "error": str(e)}
    
    # Test 2: GPU Operations
    try:
        from utils.gpu_vector_ops import gpu_ops
        import numpy as np
        
        # Test GPU vector operations
        query = np.random.random(384).astype(np.float32)
        vectors = np.random.random((1000, 384)).astype(np.float32)
        
        similarities = gpu_ops.cosine_similarity_batch(query, vectors)
        top_indices, top_scores = gpu_ops.top_k_indices(similarities, 5)
        
        print("✅ GPU Operations: Vector similarity computation working")
        results["tests"]["gpu_ops"] = {
            "status": "pass", 
            "gpu_enabled": gpu_ops.use_gpu,
            "top_score": float(top_scores[0]) if len(top_scores) > 0 else 0.0
        }
        
    except Exception as e:
        print(f"❌ GPU Operations: Failed - {e}")
        results["tests"]["gpu_ops"] = {"status": "fail", "error": str(e)}
    
    # Test 3: Vector Store Compatibility
    try:
        from storage.vector_compat import CompatVectorStore, CompatDocument
        
        # Test compatibility vector store
        store = CompatVectorStore("test")
        
        doc = CompatDocument(
            id="test_1",
            content="Test document",
            embedding=[0.1] * 384,
            metadata={"test": True}
        )
        
        await store.add_documents([doc])
        
        results_search = await store.similarity_search(
            query_embedding=[0.1] * 384,
            limit=5
        )
        
        print("✅ Vector Store Compatibility: Working")
        results["tests"]["vector_compat"] = {
            "status": "pass",
            "documents_added": 1,
            "search_results": len(results_search)
        }
        
    except Exception as e:
        print(f"❌ Vector Store Compatibility: Failed - {e}")
        results["tests"]["vector_compat"] = {"status": "fail", "error": str(e)}
    
    # Test 4: Database Configuration
    try:
        from config.database import get_database_url, create_engine
        
        db_url = get_database_url()
        engine = create_engine(db_url)
        
        print("✅ Database Configuration: Module import and engine creation working")
        results["tests"]["database_config"] = {
            "status": "pass",
            "url_configured": bool(db_url)
        }
        
    except Exception as e:
        print(f"❌ Database Configuration: Failed - {e}")
        results["tests"]["database_config"] = {"status": "fail", "error": str(e)}
    
    # Test 5: Embedding System (should still work)
    try:
        from utils.embedding import EmbeddingService
        from config.settings import Settings
        
        settings = Settings()
        embedding_service = EmbeddingService(settings)
        
        # Test embedding generation
        result = await embedding_service.generate_embedding("Test embedding")
        
        print(f"✅ Embedding System: Working (dim: {len(result.embedding)})")
        results["tests"]["embedding"] = {
            "status": "pass",
            "dimension": len(result.embedding),
            "provider": result.model
        }
        
    except Exception as e:
        print(f"❌ Embedding System: Failed - {e}")
        results["tests"]["embedding"] = {"status": "fail", "error": str(e)}
    
    # Test 6: Integration Test - Full Memory Pipeline
    try:
        from storage.vector_compat import CompatVectorStoreManager, CompatDocument
        from utils.embedding import EmbeddingService
        from config.settings import Settings
        
        # Initialize components
        settings = Settings()
        embedding_service = EmbeddingService(settings)
        vector_manager = CompatVectorStoreManager(settings)
        await vector_manager.initialize()
        
        # Test full pipeline
        store = await vector_manager.get_store("integration_test")
        
        # Generate embedding and store document
        text = "This is a test document for integration testing"
        embedding_result = await embedding_service.generate_embedding(text)
        
        doc = CompatDocument(
            id="integration_1",
            content=text,
            embedding=embedding_result.embedding,
            metadata={"integration_test": True}
        )
        
        await store.add_documents([doc])
        
        # Test retrieval
        search_results = await store.similarity_search(
            query_embedding=embedding_result.embedding,
            limit=5
        )
        
        print("✅ Integration Test: Full memory pipeline working")
        results["tests"]["integration"] = {
            "status": "pass",
            "pipeline_complete": True,
            "search_results": len(search_results),
            "similarity_score": search_results[0].similarity_score if search_results else 0.0
        }
        
    except Exception as e:
        print(f"❌ Integration Test: Failed - {e}")
        results["tests"]["integration"] = {"status": "fail", "error": str(e)}
    
    # Determine overall status
    passed_tests = sum(1 for test in results["tests"].values() if test["status"] == "pass")
    total_tests = len(results["tests"])
    
    if passed_tests == total_tests:
        results["overall_status"] = "all_pass"
        print(f"\n🎉 ALL TESTS PASSED! ({passed_tests}/{total_tests})")
        print("🚀 Memory system is fully operational!")
    elif passed_tests >= total_tests * 0.8:
        results["overall_status"] = "mostly_pass"
        print(f"\n✅ MOSTLY OPERATIONAL ({passed_tests}/{total_tests})")
        print("⚠️ Some components may need additional work")
    else:
        results["overall_status"] = "needs_work"
        print(f"\n⚠️ NEEDS MORE WORK ({passed_tests}/{total_tests})")
        print("❌ Several critical components are not working")
    
    # Save results
    with open("memory_system_verification.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

async def main():
    await test_fixed_memory_system()

if __name__ == "__main__":
    asyncio.run(main())
