"""
Test script for validating user-associated agent and document services

This script tests the new AgentService and DocumentService to ensure
all agents and documents are properly associated with users.
"""

import asyncio
import logging
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.database.connection import initialize_database
from src.config.settings import get_settings
from src.services.user_service import UserService, get_user_service
from src.services.agent_service import AgentService, set_agent_service  
from src.services.document_service import DocumentService, set_document_service
from src.core.agent_factory import AgentFactory
from src.memory.memory_manager import MemoryManager
from src.mcp.server_registry import MCPServerManager
from src.storage.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_user_associated_services():
    """Test that agents and documents are properly associated with users."""
    
    logger.info("Starting user-associated services test...")
    
    try:
        # Initialize settings and database
        settings = get_settings()
        db_manager = await initialize_database(settings)
        
        # Initialize core services
        vector_store_manager = VectorStoreManager(settings, db_manager)
        memory_manager = MemoryManager(vector_store_manager, settings)
        await memory_manager.start()
        
        mcp_manager = MCPServerManager(settings)
        await mcp_manager.start()
        
        agent_factory = AgentFactory(mcp_manager, memory_manager, settings, None)
          # Initialize services
        user_service = UserService()
        
        agent_service = AgentService(agent_factory)
        set_agent_service(agent_service)
        
        document_service = DocumentService()
        set_document_service(document_service)
        
        logger.info("Services initialized successfully")
        
        # Test 1: Create test users
        logger.info("Test 1: Creating test users...")
        test_user1 = await user_service.create_user("testuser1@example.com", "Test User 1", "password123")
        test_user2 = await user_service.create_user("testuser2@example.com", "Test User 2", "password456")
        
        logger.info(f"Created users: {test_user1.id}, {test_user2.id}")
        
        # Test 2: Create agents for different users
        logger.info("Test 2: Creating agents for different users...")
        
        agent1 = await agent_service.create_agent(
            user_id=test_user1.id,
            agent_type="research_agent",
            name="User1 Research Agent",
            capabilities=["research", "analysis"]
        )
        
        agent2 = await agent_service.create_agent(
            user_id=test_user2.id, 
            agent_type="documentation_agent",
            name="User2 Doc Agent",
            capabilities=["documentation", "writing"]
        )
        
        logger.info(f"Created agents: {agent1.id} for user {test_user1.id}, {agent2.id} for user {test_user2.id}")
        
        # Test 3: Verify user isolation
        logger.info("Test 3: Testing user isolation...")
        
        user1_agents = agent_service.get_user_agents(test_user1.id)
        user2_agents = agent_service.get_user_agents(test_user2.id)
        
        assert len(user1_agents) == 1
        assert len(user2_agents) == 1
        assert user1_agents[0].id == agent1.id
        assert user2_agents[0].id == agent2.id
        
        # Verify cross-user access denial
        user1_cannot_access_agent2 = agent_service.get_agent_by_id(agent2.id, test_user1.id)
        assert user1_cannot_access_agent2 is None
        
        logger.info("User isolation test passed")
        
        # Test 4: Create documents for different users
        logger.info("Test 4: Creating documents for different users...")
        
        doc1 = document_service.create_document(
            user_id=test_user1.id,
            title="User1 Research Document",
            content="This is research content for user 1",
            document_type="research"
        )
        
        doc2 = document_service.create_document(
            user_id=test_user2.id,
            title="User2 Documentation",
            content="This is documentation content for user 2", 
            document_type="documentation"
        )
        
        logger.info(f"Created documents: {doc1.id} for user {test_user1.id}, {doc2.id} for user {test_user2.id}")
        
        # Test 5: Verify document isolation
        logger.info("Test 5: Testing document isolation...")
        
        user1_docs = document_service.get_user_documents(test_user1.id)
        user2_docs = document_service.get_user_documents(test_user2.id)
        
        assert len(user1_docs) == 1
        assert len(user2_docs) == 1
        assert user1_docs[0].id == doc1.id
        assert user2_docs[0].id == doc2.id
        
        # Verify cross-user access denial
        user1_cannot_access_doc2 = document_service.get_document_by_id(doc2.id, test_user1.id)
        assert user1_cannot_access_doc2 is None
        
        logger.info("Document isolation test passed")
        
        # Test 6: Test statistics
        logger.info("Test 6: Testing statistics...")
        
        agent_stats1 = agent_service.get_agent_statistics(test_user1.id)
        doc_stats1 = document_service.get_document_statistics(test_user1.id)
        
        assert agent_stats1['total_agents'] == 1
        assert agent_stats1['agents_by_type']['research_agent'] == 1
        
        assert doc_stats1['total_documents'] == 1
        assert doc_stats1['documents_by_type']['research'] == 1
        
        logger.info("Statistics test passed")
        
        # Test 7: Test search
        logger.info("Test 7: Testing document search...")
        
        search_results = document_service.search_user_documents(
            user_id=test_user1.id,
            query="research"
        )
        
        assert len(search_results) == 1
        assert search_results[0].id == doc1.id
        
        # Search for user2's content should not return user1's documents
        cross_search = document_service.search_user_documents(
            user_id=test_user1.id,
            query="documentation"
        )
        
        assert len(cross_search) == 0  # user1 shouldn't see user2's docs
        
        logger.info("Document search test passed")
        
        logger.info("All tests passed! User-associated services are working correctly.")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            await memory_manager.stop()
            await mcp_manager.stop()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


if __name__ == "__main__":
    success = asyncio.run(test_user_associated_services())
    sys.exit(0 if success else 1)
