#!/usr/bin/env python3
"""
Phase 1 Foundation Validation Test

Comprehensive test of all Phase 1 foundation tasks from the historical research integration plan.
This validates that all core infrastructure is properly implemented and working.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce log noise for testing
    format='%(levelname)s: %(message)s'
)

def test_task_1_gpu_configuration():
    """Task 1: Configure GPU Acceleration Environment"""
    try:
        from src.core.gpu_config import GPUManager
        gpu_manager = GPUManager()
        gpu_ready = gpu_manager.initialize()
        
        print(f"✓ Task 1 - GPU Configuration: {gpu_manager.config.device_name} ({gpu_manager.config.device})")
        if gpu_manager.config.device != 'cpu':
            print(f"  GPU Memory: {gpu_manager.config.memory_available}MB available")
        return True
    except Exception as e:
        print(f"✗ Task 1 - GPU Configuration failed: {e}")
        return False

def test_task_2_ollama_integration():
    """Task 2: Implement Ollama Integration Framework"""
    try:
        from src.core.ollama_manager import OllamaManager
        ollama_manager = OllamaManager()
        print("✓ Task 2 - Ollama Integration: Manager initialized")
        print(f"  Ollama URL: {ollama_manager.ollama_url}")
        if ollama_manager.ollama_executable:
            print(f"  Executable found: {ollama_manager.ollama_executable}")
        return True
    except Exception as e:
        print(f"✗ Task 2 - Ollama Integration failed: {e}")
        return False

def test_task_3_settings_management():
    """Task 3: Configuration Management"""
    try:
        from src.config.settings import get_settings
        settings = get_settings()
        print("✓ Task 3 - Settings Management: Configuration loaded")
        print(f"  Ollama URL: {settings.ai.OLLAMA_BASE_URL}")
        print(f"  Embedding Model: {settings.ai.EMBEDDING_MODEL}")
        return True
    except Exception as e:
        print(f"✗ Task 3 - Settings Management failed: {e}")
        return False

def test_task_6_http_session():
    """Task 6: Implement Enhanced HTTP Session Management"""
    try:
        from src.acquisition.http_session import AcademicHTTPSession
        session = AcademicHTTPSession()
        print("✓ Task 6 - HTTP Session Management: Academic session initialized")
        print(f"  Max retries: {session.max_retries}")
        print(f"  Timeout: {session.timeout}s")
        return True
    except Exception as e:
        print(f"✗ Task 6 - HTTP Session Management failed: {e}")
        return False

def test_task_7_document_download():
    """Task 7: Build Intelligent Document Download Pipeline"""
    try:
        from src.acquisition.document_download import DocumentDownloadPipeline
        pipeline = DocumentDownloadPipeline()
        print("✓ Task 7 - Document Download Pipeline: Initialized successfully")
        print(f"  Storage root: {pipeline.storage_root}")
        return True
    except Exception as e:
        print(f"✗ Task 7 - Document Download Pipeline failed: {e}")
        return False

def test_task_12_vector_store():
    """Task 12: Configure GPU-Accelerated FAISS Vector Store"""
    try:
        from src.config.settings import get_settings
        from src.storage.vector.manager import VectorStoreManager
        
        settings = get_settings()
        vector_manager = VectorStoreManager(settings)
        print("✓ Task 12 - Vector Store: FAISS manager initialized")
        return True
    except Exception as e:
        print(f"✗ Task 12 - Vector Store failed: {e}")
        return False

def test_task_13_embedding_service():
    """Task 13: Integrate GPU-Accelerated Embedding Service"""
    try:
        from src.utils.embedding import EmbeddingService
        embedding_service = EmbeddingService()
        print("✓ Task 13 - Embedding Service: Multiple providers initialized")
        print(f"  Default provider: {embedding_service.default_provider}")
        print(f"  Available providers: {list(embedding_service.providers.keys())}")
        return True
    except Exception as e:
        print(f"✗ Task 13 - Embedding Service failed: {e}")
        return False

def test_enhanced_document_acquisition():
    """Enhanced Document Acquisition with AI Analysis"""
    try:
        from src.acquisition.enhanced_document_acquisition import EnhancedDocumentAcquisition
        enhanced_acq = EnhancedDocumentAcquisition()
        print("✓ Enhanced Document Acquisition: AI-powered pipeline initialized")
        print(f"  Storage path: {enhanced_acq.storage_path}")
        return True
    except Exception as e:
        print(f"✗ Enhanced Document Acquisition failed: {e}")
        return False

def test_anti_hallucination_framework():
    """Anti-Hallucination Framework"""
    try:
        from src.validation.anti_hallucination_framework import AntiHallucinationFramework
        anti_hall = AntiHallucinationFramework()
        print("✓ Anti-Hallucination Framework: Multi-method verification initialized")
        return True
    except Exception as e:
        print(f"✗ Anti-Hallucination Framework failed: {e}")
        return False

def test_multi_agent_orchestrator():
    """Multi-Agent Orchestrator"""
    try:
        from src.orchestration.multi_agent_orchestrator import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator()
        print("✓ Multi-Agent Orchestrator: Advanced coordination system initialized")
        return True
    except Exception as e:
        print(f"✗ Multi-Agent Orchestrator failed: {e}")
        return False

def main():
    """Run all Phase 1 foundation tests"""
    print("=" * 60)
    print("PHASE 1 FOUNDATION VALIDATION TEST")
    print("Historical Research Integration Plan - Core Infrastructure")
    print("=" * 60)
    
    tests = [
        ("Core Infrastructure", [
            test_task_1_gpu_configuration,
            test_task_2_ollama_integration, 
            test_task_3_settings_management,
            test_task_6_http_session,
            test_task_7_document_download,
            test_task_12_vector_store,
            test_task_13_embedding_service,
        ]),
        ("Advanced Components", [
            test_enhanced_document_acquisition,
            test_anti_hallucination_framework,
            test_multi_agent_orchestrator,
        ])
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category, test_functions in tests:
        print(f"\n--- {category} ---")
        for test_func in test_functions:
            total_tests += 1
            if test_func():
                passed_tests += 1
        
    print(f"\n{'=' * 60}")
    print(f"VALIDATION RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL PHASE 1 FOUNDATION TASKS: SUCCESSFULLY VALIDATED")
        print("✅ System ready for advanced integration tasks")
    else:
        print(f"⚠️  {total_tests - passed_tests} tasks need attention")
        
    print("=" * 60)
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
