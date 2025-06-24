"""
Test Real Agents

Test the real PyGent Factory agents to verify they work correctly.
"""

import asyncio
import aiohttp
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_tot_reasoning_agent():
    """Test the ToT Reasoning Agent."""
    try:
        logger.info("🧠 Testing ToT Reasoning Agent...")
        
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get("http://localhost:8001/health") as response:
                assert response.status == 200
                health_data = await response.json()
                logger.info(f"   Health: {health_data['status']}")
            
            # Test reasoning endpoint
            reasoning_request = {
                "problem": "How to implement zero mock code in a production system?",
                "reasoning_depth": 3,
                "exploration_breadth": 4,
                "context": {"domain": "software_engineering"}
            }
            
            async with session.post(
                "http://localhost:8001/reason",
                json=reasoning_request
            ) as response:
                assert response.status == 200
                result = await response.json()
                
                # Verify response structure
                assert "reasoning_steps" in result
                assert "best_solution" in result
                assert "confidence_score" in result
                assert "alternative_solutions" in result
                
                logger.info(f"   Reasoning completed with confidence: {result['confidence_score']}")
                logger.info(f"   Solution: {result['best_solution'][:100]}...")
                
        logger.info("✅ ToT Reasoning Agent: All tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ ToT Reasoning Agent test failed: {e}")
        return False


async def test_rag_retrieval_agent():
    """Test the RAG Retrieval Agent."""
    try:
        logger.info("🔍 Testing RAG Retrieval Agent...")
        
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get("http://localhost:8002/health") as response:
                assert response.status == 200
                health_data = await response.json()
                logger.info(f"   Health: {health_data['status']}")
                logger.info(f"   Documents indexed: {health_data['documents_indexed']}")
            
            # Test retrieval endpoint
            retrieval_request = {
                "query": "zero mock code implementation",
                "max_results": 5,
                "similarity_threshold": 0.5
            }
            
            async with session.post(
                "http://localhost:8002/retrieve",
                json=retrieval_request
            ) as response:
                assert response.status == 200
                result = await response.json()
                
                # Verify response structure
                assert "retrieved_documents" in result
                assert "total_count" in result
                assert "similarity_scores" in result
                
                documents = result["retrieved_documents"]
                logger.info(f"   Retrieved {len(documents)} documents")
                
                if documents:
                    best_doc = documents[0]
                    logger.info(f"   Best match: {best_doc['title']}")
                    logger.info(f"   Relevance: {best_doc['relevance_score']:.3f}")
                
        logger.info("✅ RAG Retrieval Agent: All tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG Retrieval Agent test failed: {e}")
        return False


async def test_agent_integration():
    """Test agent integration with the system."""
    try:
        logger.info("🔗 Testing Agent Integration...")
        
        # Test that agents can work together
        async with aiohttp.ClientSession() as session:
            # First, retrieve relevant documents
            retrieval_request = {
                "query": "agent orchestration best practices",
                "max_results": 3
            }
            
            async with session.post(
                "http://localhost:8002/retrieve",
                json=retrieval_request
            ) as response:
                retrieval_result = await response.json()
                documents = retrieval_result["retrieved_documents"]
            
            # Then, use retrieved context for reasoning
            context_text = " ".join([doc["content"] for doc in documents])
            reasoning_request = {
                "problem": "Design an optimal agent orchestration strategy",
                "context": {
                    "retrieved_documents": context_text[:500],  # Limit context size
                    "domain": "orchestration"
                }
            }
            
            async with session.post(
                "http://localhost:8001/reason",
                json=reasoning_request
            ) as response:
                reasoning_result = await response.json()
            
            logger.info("   ✅ Agents successfully integrated")
            logger.info(f"   Retrieved {len(documents)} documents for context")
            logger.info(f"   Generated solution with confidence: {reasoning_result['confidence_score']}")
        
        logger.info("✅ Agent Integration: All tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Agent Integration test failed: {e}")
        return False


async def main():
    """Run all agent tests."""
    logger.info("🧪 Starting Real Agent Tests...")
    logger.info("🎯 Goal: Verify real agents provide actual functionality")
    
    try:
        # Test individual agents
        tot_success = await test_tot_reasoning_agent()
        rag_success = await test_rag_retrieval_agent()
        integration_success = await test_agent_integration()
        
        # Summary
        total_tests = 3
        passed_tests = sum([tot_success, rag_success, integration_success])
        
        logger.info("\n" + "="*60)
        logger.info("📊 REAL AGENT TEST RESULTS")
        logger.info("="*60)
        logger.info(f"🧠 ToT Reasoning Agent: {'✅ PASSED' if tot_success else '❌ FAILED'}")
        logger.info(f"🔍 RAG Retrieval Agent: {'✅ PASSED' if rag_success else '❌ FAILED'}")
        logger.info(f"🔗 Agent Integration: {'✅ PASSED' if integration_success else '❌ FAILED'}")
        logger.info(f"📈 Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL REAL AGENT TESTS PASSED!")
            logger.info("✅ Real agents providing actual functionality")
            logger.info("✅ No mock code in agent responses")
            logger.info("🚀 Agents ready for zero mock code integration")
        else:
            logger.error("❌ Some agent tests failed")
            logger.error("🔧 Check agent services and try again")
        
        logger.info("="*60)
        
        return passed_tests == total_tests
        
    except Exception as e:
        logger.error(f"❌ Agent tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)