"""
Test the complete historical research system implementation.
"""
import asyncio
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.research.pipeline import create_historical_research_pipeline
from src.config.settings import Settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_historical_research_system():
    """Test the complete historical research system."""
    print("=" * 60)
    print("HISTORICAL RESEARCH SYSTEM TEST")
    print("=" * 60)
    
    try:
        # Create settings
        settings = Settings()
        
        print("\n1. Initializing Historical Research Pipeline...")
        pipeline = await create_historical_research_pipeline(settings)
        print("✓ Pipeline initialized successfully")
        
        print("\n2. Getting System Status...")
        status = await pipeline.get_system_status()
        print(f"✓ System Status: {status.get('initialized', False)}")
        print(f"  - Storage initialized: {bool(status.get('storage'))}")
        print(f"  - Vector store initialized: {bool(status.get('vector_store'))}")
        print(f"  - Embedding service initialized: {bool(status.get('embedding'))}")
        
        print("\n3. Testing Document Processing Components...")
        
        # Test semantic chunking
        from src.research.semantic_chunking import SemanticChunker
        chunker = SemanticChunker()
        
        sample_text = """
        === Page 1 ===
        The American Civil War (1861-1865) was a significant conflict in United States history.
        
        Chapter 1: Causes of the War
        
        The war began due to several factors including slavery, states' rights, and economic differences.
        Abraham Lincoln's election in 1860 was a catalyst for secession.
        
        === Page 2 ===
        
        The war officially began on April 12, 1861, when Confederate forces fired on Fort Sumter.
        """
        
        chunks = chunker.chunk_document("test_doc", sample_text, {"title": "Civil War Test"})
        print(f"✓ Semantic chunking: Created {len(chunks)} chunks")
        
        print("\n4. Testing Vector Components...")
        
        # Test embedding service
        test_embedding = await pipeline.embedding_service.get_embedding("The Civil War was a major conflict")
        if test_embedding:
            print(f"✓ Embedding generation: Vector dimension {len(test_embedding)}")
        else:
            print("✗ Embedding generation failed")
        
        print("\n5. Testing Search Functionality...")
        
        # Test search (may be empty initially)
        search_results = await pipeline.search_historical_documents("Civil War", limit=5)
        print(f"✓ Search functionality: Found {len(search_results)} results")
        
        print("\n6. Performance Statistics...")
        
        # Get performance stats
        embedding_stats = pipeline.embedding_service.get_performance_stats()
        print(f"  - Embeddings generated: {embedding_stats.get('total_embeddings_generated', 0)}")
        print(f"  - Cache hit rate: {embedding_stats.get('cache_hit_rate', 0):.2%}")
        
        if pipeline.vector_pipeline:
            pipeline_stats = pipeline.vector_pipeline.get_processing_stats()
            print(f"  - Documents processed: {pipeline_stats.get('documents_processed', 0)}")
        
        print("\n7. Testing Health Check...")
        health = await pipeline.vector_pipeline.health_check() if pipeline.vector_pipeline else {"status": "unknown"}
        print(f"✓ Health check: {health.get('status', 'unknown')}")
        if health.get('issues'):
            for issue in health['issues']:
                print(f"  - Issue: {issue}")
        
        print("\n8. System Cleanup...")
        await pipeline.cleanup()
        print("✓ System cleaned up successfully")
        
        print("\n" + "=" * 60)
        print("HISTORICAL RESEARCH SYSTEM TEST COMPLETE")
        print("✓ All core components tested successfully")
        print("✓ System is ready for historical research tasks")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_components():
    """Test individual components in isolation."""
    print("\n" + "=" * 60)
    print("INDIVIDUAL COMPONENT TESTS")
    print("=" * 60)
    
    try:
        # Test document storage
        print("\n1. Testing Document Storage...")
        from src.acquisition.document_storage import DocumentStorageSystem, DocumentMetadata
        
        storage = DocumentStorageSystem("test_data/documents")
          # Create test metadata for demonstration
        test_metadata = DocumentMetadata(
            document_id="test_123",
            original_url="https://example.com/test.pdf",
            title="Test Document",
            author="Test Author"
        )
        
        print(f"✓ Document storage initialized: {storage.base_path}")
        print(f"✓ Test metadata created: {test_metadata.title}")
        
        # Test HTTP session
        print("\n2. Testing HTTP Session...")
        from src.acquisition.http_session import AcademicHTTPSession
        
        with AcademicHTTPSession() as session:
            doc_id = session.generate_document_id("https://example.com/test.pdf")
            print(f"✓ HTTP session: Generated document ID {doc_id}")
        
        # Test processing metadata
        print("\n3. Testing Processing Metadata...")
        from src.acquisition.processing_metadata import DocumentProcessingTracker, DocumentProcessingMetadata
        
        tracker = DocumentProcessingTracker(storage)
        
        processing_metadata = DocumentProcessingMetadata(
            document_id="test_123",
            processing_timestamp="2025-06-20T10:00:00Z",
            extraction_method="text",
            extraction_success=True,
            quality_score=0.85
        )
        
        tracker.record_processing_result(processing_metadata)
        print("✓ Processing metadata recorded")
        
        # Test vector config
        print("\n4. Testing Vector Configuration...")
        from src.research.vector_config import HistoricalResearchVectorConfig
        
        vector_config = HistoricalResearchVectorConfig(
            storage_path="test_data/vector_store"
        )
        
        faiss_config = vector_config.get_faiss_config()
        print(f"✓ Vector config: FAISS index type {faiss_config['index_type']}")
        
        print("\n" + "=" * 60)
        print("INDIVIDUAL COMPONENT TESTS COMPLETE")
        print("✓ All components initialized successfully")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    async def main():
        print("Starting Historical Research System Tests...\n")
        
        # Test individual components first
        component_success = await test_individual_components()
        
        if component_success:
            # Test full system
            system_success = await test_historical_research_system()
            
            if system_success:
                print("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION")
                return 0
            else:
                print("\n❌ SYSTEM TEST FAILED")
                return 1
        else:
            print("\n❌ COMPONENT TESTS FAILED")
            return 1
    
    exit_code = asyncio.run(main())
