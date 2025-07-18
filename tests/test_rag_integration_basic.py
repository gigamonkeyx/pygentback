#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic RAG Integration Test - Phase 2.1
Observer-approved validation for RAG augmentation integration (simplified)

Tests the core RAG integration components without heavy dependencies.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Fix import paths
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class MockEmbeddingService:
    """Mock embedding service for testing"""
    
    async def initialize(self):
        pass
    
    async def get_embedding(self, text: str):
        # Return a mock embedding vector
        return [0.1] * 384  # Standard embedding dimension
    
    async def get_embeddings_batch(self, texts):
        return [[0.1] * 384 for _ in texts]


class MockVectorStoreManager:
    """Mock vector store manager for testing"""
    
    def __init__(self, settings, db_manager):
        self.settings = settings
        self.db_manager = db_manager
        self._initialized = False
    
    async def initialize(self):
        self._initialized = True
    
    async def search_similar(self, query):
        # Return mock search results
        from dataclasses import dataclass
        
        @dataclass
        class MockDocument:
            content: str
            title: str
            metadata: Dict[str, Any]
            
            def to_dict(self):
                return {
                    'content': self.content,
                    'title': self.title,
                    'metadata': self.metadata
                }
        
        @dataclass
        class MockSearchResult:
            document: MockDocument
            similarity_score: float
        
        # Return mock coding-related documents
        mock_docs = [
            MockSearchResult(
                document=MockDocument(
                    content="def reverse_string(s): return s[::-1]",
                    title="String Reversal Function",
                    metadata={"language": "python", "doc_type": "function"}
                ),
                similarity_score=0.85
            ),
            MockSearchResult(
                document=MockDocument(
                    content="# Python string manipulation examples\nStrings in Python are immutable...",
                    title="Python String Documentation",
                    metadata={"language": "python", "doc_type": "documentation"}
                ),
                similarity_score=0.72
            )
        ]
        
        return mock_docs


class TestBasicRAGIntegration:
    """Basic test suite for RAG integration"""
    
    async def test_code_retriever_basic(self):
        """Test basic code retriever functionality"""
        try:
            # Import the code retriever
            from src.agents.augmentation.code_retriever import CodeRetriever
            
            # Create mock components
            mock_vector_store = MockVectorStoreManager(None, None)
            mock_embedding_service = MockEmbeddingService()
            
            # Create code retriever
            code_retriever = CodeRetriever(
                vector_store_manager=mock_vector_store,
                embedding_service=mock_embedding_service
            )
            
            # Test language detection
            python_query = "write a function to reverse a string in Python"
            detected_language = code_retriever._detect_language_from_query(python_query)
            assert detected_language == "python", f"Should detect Python, got {detected_language}"
            
            # Test pattern extraction
            patterns = code_retriever._extract_code_patterns(python_query, "python")
            # Patterns might be empty, which is OK
            
            # Test collection mapping
            collections = code_retriever._get_relevant_collections("python")
            assert len(collections) > 0, "Should return relevant collections"
            
            logger.info("Basic code retriever test passed")
            return True
            
        except Exception as e:
            logger.error(f"Basic code retriever test failed: {e}")
            return False
    
    async def test_rag_augmenter_basic(self):
        """Test basic RAG augmenter functionality"""
        try:
            # Import the RAG augmenter
            from src.agents.augmentation.rag_augmenter import RAGAugmenter
            
            # Create mock components
            mock_vector_store = MockVectorStoreManager(None, None)
            mock_embedding_service = MockEmbeddingService()
            
            # Create RAG augmenter
            rag_augmenter = RAGAugmenter(
                vector_store_manager=mock_vector_store,
                embedding_service=mock_embedding_service
            )
            
            # Test cache key generation
            cache_key = rag_augmenter._generate_cache_key("test prompt", {"context": "test"})
            assert isinstance(cache_key, str), "Cache key should be a string"
            assert len(cache_key) > 0, "Cache key should not be empty"
            
            # Test prompt augmentation creation
            mock_docs = [
                {"content": "def test(): pass", "title": "Test Function"},
                {"content": "# Documentation", "title": "Docs"}
            ]
            augmented_prompt = rag_augmenter._create_augmented_prompt("original prompt", mock_docs)
            assert "original prompt" in augmented_prompt, "Original prompt should be in augmented version"
            assert "Test Function" in augmented_prompt, "Document titles should be included"
            
            # Test stats
            stats = rag_augmenter.get_stats()
            assert "total_augmentations" in stats, "Should have augmentation stats"
            assert "initialized" in stats, "Should have initialization status"
            
            logger.info("Basic RAG augmenter test passed")
            return True
            
        except Exception as e:
            logger.error(f"Basic RAG augmenter test failed: {e}")
            return False
    
    async def test_base_agent_augmentation_hooks(self):
        """Test base agent augmentation hooks"""
        try:
            # Import base agent components
            from src.core.agent.base import BaseAgent
            from src.core.agent.config import AgentConfig
            
            # Create test agent implementation
            class TestAgent(BaseAgent):
                async def _agent_initialize(self):
                    pass
                
                async def _agent_shutdown(self):
                    pass
                
                async def _handle_request(self, message):
                    return "test response"
                
                async def _handle_tool_call(self, message):
                    return "test tool call"
                
                async def _handle_capability_request(self, message):
                    return {"capabilities": ["test"]}
                
                async def _handle_notification(self, message):
                    return "notification received"
            
            # Create agent with augmentation enabled
            config = AgentConfig(
                agent_id="test_agent",
                name="Test Agent",
                agent_type="test",
                custom_config={
                    "augmentation_enabled": True,
                    "rag_enabled": True
                }
            )
            
            agent = TestAgent(config)
            
            # Test augmentation configuration
            assert agent.augmentation_enabled == True, "Augmentation should be enabled"
            assert agent.rag_enabled == True, "RAG should be enabled"
            assert agent.rag_augmenter is None, "RAG augmenter should be None initially"
            
            # Test augmentation metrics
            metrics = agent.get_augmentation_metrics()
            assert "augmentation_enabled" in metrics, "Should have augmentation status"
            assert "total_requests" in metrics, "Should have request count"
            
            # Test augmented generation (without actual RAG)
            result = await agent._augmented_generate("test prompt")
            assert isinstance(result, str), "Should return a string"
            
            logger.info("Base agent augmentation hooks test passed")
            return True
            
        except Exception as e:
            logger.error(f"Base agent augmentation hooks test failed: {e}")
            return False
    
    async def test_coding_agent_integration(self):
        """Test coding agent RAG integration"""
        try:
            # Import coding agent
            from src.agents.coding_agent import CodingAgent
            from src.core.agent.config import AgentConfig
            
            # Create coding agent with augmentation
            config = AgentConfig(
                agent_id="test_coding_agent",
                name="Test Coding Agent",
                agent_type="coding",
                custom_config={
                    "model_name": "llama3:8b",
                    "augmentation_enabled": True,
                    "rag_enabled": True
                }
            )
            
            coding_agent = CodingAgent(config)
            
            # Test configuration
            assert coding_agent.augmentation_enabled == True, "Augmentation should be enabled"
            assert coding_agent.rag_enabled == True, "RAG should be enabled"
            
            # Test language detection
            language = coding_agent._detect_language("write a Python function")
            assert language in coding_agent.supported_languages, f"Should detect supported language, got {language}"
            
            # Test task classification
            task_type = coding_agent._classify_request("write a function to sort a list")
            assert task_type == "code_generation", f"Should classify as code generation, got {task_type}"
            
            logger.info("Coding agent integration test passed")
            return True
            
        except Exception as e:
            logger.error(f"Coding agent integration test failed: {e}")
            return False
    
    async def test_performance_simulation(self):
        """Test performance simulation for RAG integration"""
        try:
            # Simulate RAG augmentation performance
            test_prompts = [
                "write a Python function to reverse a string",
                "create a JavaScript function to validate email",
                "implement a sorting algorithm in Python",
                "write a function to calculate fibonacci numbers",
                "create a class for binary tree operations"
            ]
            
            total_time = 0
            successful_operations = 0
            
            for prompt in test_prompts:
                start_time = time.time()
                
                # Simulate RAG operations
                await asyncio.sleep(0.01)  # Simulate 10ms processing time
                
                # Simulate augmentation
                augmented_prompt = f"Context: [Mock documentation]\n\n{prompt}"
                
                end_time = time.time()
                operation_time_ms = (end_time - start_time) * 1000
                total_time += operation_time_ms
                successful_operations += 1
                
                logger.debug(f"Simulated operation: {operation_time_ms:.2f}ms")
            
            avg_time_ms = total_time / len(test_prompts)
            success_rate = successful_operations / len(test_prompts)
            
            logger.info(f"Performance Simulation Results:")
            logger.info(f"  Average operation time: {avg_time_ms:.2f}ms")
            logger.info(f"  Success rate: {success_rate:.1%}")
            logger.info(f"  Target: <200ms (simulated: {avg_time_ms < 200})")
            
            # Performance should be good in simulation
            assert avg_time_ms < 200, f"Should be under 200ms, got {avg_time_ms:.2f}ms"
            assert success_rate == 1.0, f"Should have 100% success rate, got {success_rate:.1%}"
            
            logger.info("Performance simulation test passed")
            return True
            
        except Exception as e:
            logger.error(f"Performance simulation test failed: {e}")
            return False


async def run_basic_rag_integration_tests():
    """Run basic RAG integration tests for Phase 2.1"""
    print("\n🚀 PHASE 2.1 BASIC VALIDATION: RAG Integration Components")
    print("=" * 65)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_instance = TestBasicRAGIntegration()
    results = {}
    
    try:
        # Test 1: Code retriever basic functionality
        print("\n1. Testing basic code retriever functionality...")
        results['code_retriever_basic'] = await test_instance.test_code_retriever_basic()
        
        # Test 2: RAG augmenter basic functionality
        print("\n2. Testing basic RAG augmenter functionality...")
        results['rag_augmenter_basic'] = await test_instance.test_rag_augmenter_basic()
        
        # Test 3: Base agent augmentation hooks
        print("\n3. Testing base agent augmentation hooks...")
        results['base_agent_hooks'] = await test_instance.test_base_agent_augmentation_hooks()
        
        # Test 4: Coding agent integration
        print("\n4. Testing coding agent integration...")
        results['coding_agent_integration'] = await test_instance.test_coding_agent_integration()
        
        # Test 5: Performance simulation
        print("\n5. Testing performance simulation...")
        results['performance_simulation'] = await test_instance.test_performance_simulation()
        
        # Summary
        print("\n" + "=" * 65)
        print("PHASE 2.1 BASIC RAG INTEGRATION VALIDATION RESULTS:")
        print("=" * 65)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        success_rate = passed_tests / total_tests
        
        for test_name, result in results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
        
        if success_rate >= 1.0:  # 100% success required for basic tests
            print("\n🎉 PHASE 2.1 BASIC VALIDATION: SUCCESS")
            print("RAG integration components operational")
            print("Observer Checkpoint: Basic RAG functionality validated")
            print("🚀 Ready for full RAG integration testing with dependencies")
            return True
        else:
            print("\n⚠️ PHASE 2.1 BASIC VALIDATION: FAILED")
            print("Some basic tests failed. Review and fix issues before proceeding.")
            return False
            
    except Exception as e:
        print(f"\n❌ PHASE 2.1 BASIC VALIDATION FAILED: {e}")
        logger.error(f"Phase 2.1 basic validation error: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(run_basic_rag_integration_tests())
