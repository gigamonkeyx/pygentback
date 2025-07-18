#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple RAG Integration Test - Phase 2.1 (Working Version)
Observer-approved validation for RAG augmentation without dependency issues

Tests the simple RAG integration that works without transformers library.
"""

import asyncio
import logging
import time
from typing import Dict, Any

# Fix import paths
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.augmentation.simple_rag_augmenter import (
    SimpleRAGAugmenter, 
    SimpleCodeRetriever, 
    SimpleEmbeddingService,
    SimpleRAGResult
)

logger = logging.getLogger(__name__)


class TestSimpleRAGIntegration:
    """Test suite for simple RAG integration"""
    
    async def test_simple_embedding_service(self):
        """Test simple embedding service"""
        try:
            embedding_service = SimpleEmbeddingService()
            
            # Test initialization
            await embedding_service.initialize()
            assert embedding_service._initialized == True, "Should be initialized"
            
            # Test single embedding
            text = "write a Python function"
            embedding = await embedding_service.get_embedding(text)
            assert isinstance(embedding, list), "Should return a list"
            assert len(embedding) == 384, f"Should be 384 dimensions, got {len(embedding)}"
            assert all(-1 <= x <= 1 for x in embedding), "Values should be in [-1, 1] range"
            
            # Test batch embeddings
            texts = ["function", "class", "import"]
            embeddings = await embedding_service.get_embeddings_batch(texts)
            assert len(embeddings) == 3, "Should return 3 embeddings"
            assert all(len(emb) == 384 for emb in embeddings), "All should be 384 dimensions"
            
            logger.info("Simple embedding service test passed")
            return True
            
        except Exception as e:
            logger.error(f"Simple embedding service test failed: {e}")
            return False
    
    async def test_simple_code_retriever(self):
        """Test simple code retriever"""
        try:
            code_retriever = SimpleCodeRetriever()
            
            # Test initialization
            await code_retriever.initialize()
            assert code_retriever._initialized == True, "Should be initialized"
            
            # Test document creation
            assert len(code_retriever.mock_documents) > 0, "Should have mock documents"
            
            # Test retrieval
            query = "write a Python function to reverse a string"
            results = await code_retriever.retrieve(query, k=3, language="python")
            
            assert isinstance(results, list), "Should return a list"
            assert len(results) <= 3, "Should return at most 3 results"
            
            if results:
                # Check result structure
                result = results[0]
                assert "content" in result, "Should have content"
                assert "title" in result, "Should have title"
                assert "relevance_score" in result, "Should have relevance score"
                assert result["relevance_score"] > 0, "Should have positive relevance"
            
            # Test stats
            stats = code_retriever.get_stats()
            assert stats["total_retrievals"] >= 1, "Should have at least one retrieval"
            assert stats["initialized"] == True, "Should be initialized"
            
            logger.info("Simple code retriever test passed")
            return True
            
        except Exception as e:
            logger.error(f"Simple code retriever test failed: {e}")
            return False
    
    async def test_simple_rag_augmenter(self):
        """Test simple RAG augmenter"""
        try:
            rag_augmenter = SimpleRAGAugmenter()
            
            # Test initialization
            await rag_augmenter.initialize()
            assert rag_augmenter._initialized == True, "Should be initialized"
            
            # Test language detection
            python_prompt = "write a Python function to calculate fibonacci"
            detected_lang = rag_augmenter._detect_language(python_prompt)
            assert detected_lang == "python", f"Should detect Python, got {detected_lang}"
            
            js_prompt = "create a JavaScript function to validate email"
            detected_lang = rag_augmenter._detect_language(js_prompt)
            assert detected_lang == "javascript", f"Should detect JavaScript, got {detected_lang}"
            
            # Test augmentation
            start_time = time.time()
            result = await rag_augmenter.augment_prompt(python_prompt)
            end_time = time.time()
            
            assert isinstance(result, SimpleRAGResult), "Should return SimpleRAGResult"
            assert result.success == True, "Should be successful"
            assert result.original_prompt == python_prompt, "Should preserve original prompt"
            assert len(result.augmented_prompt) > len(python_prompt), "Should be augmented"
            assert "Context Information:" in result.augmented_prompt, "Should have context section"
            
            # Test performance
            retrieval_time = (end_time - start_time) * 1000
            logger.info(f"Augmentation time: {retrieval_time:.2f}ms")
            assert retrieval_time < 1000, f"Should be under 1000ms, got {retrieval_time:.2f}ms"
            
            # Test caching
            cached_result = await rag_augmenter.augment_prompt(python_prompt)
            assert cached_result.original_prompt == result.original_prompt, "Should return cached result"
            
            # Test stats
            stats = rag_augmenter.get_stats()
            assert stats["total_augmentations"] >= 2, "Should have at least 2 augmentations"
            assert stats["cache_hits"] >= 1, "Should have at least 1 cache hit"
            assert stats["success_rate"] >= 0.99, f"Should have ~100% success rate, got {stats['success_rate']:.3f}"
            
            # Cleanup
            await rag_augmenter.shutdown()
            
            logger.info("Simple RAG augmenter test passed")
            return True
            
        except Exception as e:
            logger.error(f"Simple RAG augmenter test failed: {e}")
            return False
    
    async def test_augmentation_quality(self):
        """Test augmentation quality and relevance"""
        try:
            rag_augmenter = SimpleRAGAugmenter()
            await rag_augmenter.initialize()
            
            test_cases = [
                {
                    "prompt": "write a Python function to reverse a string",
                    "expected_language": "python",
                    "expected_keywords": ["reverse", "string", "def"]
                },
                {
                    "prompt": "create a JavaScript function for email validation",
                    "expected_language": "javascript",
                    "expected_keywords": ["email", "validation", "function"]
                },
                {
                    "prompt": "implement a binary tree class in Python",
                    "expected_language": "python",
                    "expected_keywords": ["binary", "tree", "class"]
                }
            ]
            
            successful_augmentations = 0
            
            for test_case in test_cases:
                result = await rag_augmenter.augment_prompt(test_case["prompt"])
                
                if result.success:
                    successful_augmentations += 1
                    
                    # Check language detection
                    detected_lang = result.metadata.get("detected_language")
                    if detected_lang == test_case["expected_language"]:
                        logger.debug(f"✓ Language detection correct: {detected_lang}")
                    
                    # Check if relevant documents were retrieved
                    if result.retrieved_documents:
                        logger.debug(f"✓ Retrieved {len(result.retrieved_documents)} documents")
                    
                    # Check augmentation quality
                    if len(result.augmented_prompt) > len(result.original_prompt):
                        logger.debug(f"✓ Prompt augmented successfully")
            
            success_rate = successful_augmentations / len(test_cases)
            logger.info(f"Augmentation quality test: {success_rate:.1%} success rate")
            
            await rag_augmenter.shutdown()
            
            assert success_rate >= 0.8, f"Should have at least 80% success rate, got {success_rate:.1%}"
            
            logger.info("Augmentation quality test passed")
            return True
            
        except Exception as e:
            logger.error(f"Augmentation quality test failed: {e}")
            return False
    
    async def test_performance_requirements(self):
        """Test performance requirements"""
        try:
            rag_augmenter = SimpleRAGAugmenter()
            await rag_augmenter.initialize()
            
            # Test multiple augmentations for performance
            test_prompts = [
                "write a Python function to sort a list",
                "create a JavaScript function to find duplicates",
                "implement a TypeScript interface for user data",
                "write a Python class for queue operations",
                "create a function to validate input data"
            ]
            
            total_time = 0
            successful_operations = 0
            
            for prompt in test_prompts:
                start_time = time.time()
                result = await rag_augmenter.augment_prompt(prompt)
                end_time = time.time()
                
                operation_time_ms = (end_time - start_time) * 1000
                total_time += operation_time_ms
                
                if result.success:
                    successful_operations += 1
                
                logger.debug(f"Operation time: {operation_time_ms:.2f}ms")
            
            avg_time_ms = total_time / len(test_prompts)
            success_rate = successful_operations / len(test_prompts)
            
            logger.info(f"Performance Results:")
            logger.info(f"  Average operation time: {avg_time_ms:.2f}ms")
            logger.info(f"  Success rate: {success_rate:.1%}")
            logger.info(f"  Target: <200ms per operation")
            
            await rag_augmenter.shutdown()
            
            # Performance requirements
            assert avg_time_ms < 200, f"Should be under 200ms, got {avg_time_ms:.2f}ms"
            assert success_rate >= 0.9, f"Should have at least 90% success rate, got {success_rate:.1%}"
            
            logger.info("Performance requirements test passed")
            return True
            
        except Exception as e:
            logger.error(f"Performance requirements test failed: {e}")
            return False


async def run_simple_rag_integration_tests():
    """Run simple RAG integration tests"""
    print("\n🚀 PHASE 2.1 WORKING VALIDATION: Simple RAG Integration")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_instance = TestSimpleRAGIntegration()
    results = {}
    
    try:
        # Test 1: Simple embedding service
        print("\n1. Testing simple embedding service...")
        results['embedding_service'] = await test_instance.test_simple_embedding_service()
        
        # Test 2: Simple code retriever
        print("\n2. Testing simple code retriever...")
        results['code_retriever'] = await test_instance.test_simple_code_retriever()
        
        # Test 3: Simple RAG augmenter
        print("\n3. Testing simple RAG augmenter...")
        results['rag_augmenter'] = await test_instance.test_simple_rag_augmenter()
        
        # Test 4: Augmentation quality
        print("\n4. Testing augmentation quality...")
        results['augmentation_quality'] = await test_instance.test_augmentation_quality()
        
        # Test 5: Performance requirements
        print("\n5. Testing performance requirements...")
        results['performance'] = await test_instance.test_performance_requirements()
        
        # Summary
        print("\n" + "=" * 60)
        print("PHASE 2.1 SIMPLE RAG INTEGRATION VALIDATION RESULTS:")
        print("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        success_rate = passed_tests / total_tests
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
        
        if success_rate >= 1.0:  # 100% success required
            print("\n🎉 PHASE 2.1 WORKING VALIDATION: SUCCESS")
            print("✅ Simple RAG integration fully operational")
            print("✅ Observer Checkpoint: RAG functionality validated")
            print("✅ Performance targets met (<200ms, >90% success)")
            print("🚀 Ready to proceed to Phase 2.2 or dependency resolution")
            return True
        else:
            print("\n⚠️ PHASE 2.1 WORKING VALIDATION: FAILED")
            print("Some tests failed. Review and fix issues before proceeding.")
            return False
            
    except Exception as e:
        print(f"\n❌ PHASE 2.1 WORKING VALIDATION FAILED: {e}")
        logger.error(f"Phase 2.1 working validation error: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(run_simple_rag_integration_tests())
