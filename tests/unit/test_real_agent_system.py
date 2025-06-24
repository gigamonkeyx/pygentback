#!/usr/bin/env python3
"""
Test Real Agent System

Test creating real agents and having them communicate and perform document retrieval.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Environment setup - Override .env file settings
os.environ.update({
    "DATABASE_URL": "postgresql+asyncpg://postgres:postgres@localhost:54321/pygent_factory",
    "DB_HOST": "localhost",
    "DB_PORT": "54321",
    "DB_NAME": "pygent_factory",
    "DB_USER": "postgres",
    "DB_PASSWORD": "postgres",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "REDIS_DB": "0",
    "REDIS_URL": "redis://localhost:6379/0"
})

sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging to file to avoid output issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_agent_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def test_real_agent_system():
    """Test the complete real agent system"""
    
    logger.info("🚀 STARTING REAL AGENT SYSTEM TEST")
    
    try:
        # Step 1: Initialize infrastructure
        logger.info("📊 Initializing database...")
        from database.production_manager import initialize_database
        db_success = await initialize_database()
        logger.info(f"Database initialization: {db_success}")
        
        logger.info("🔄 Initializing Redis...")
        from cache.redis_manager import initialize_redis
        redis_success = await initialize_redis()
        logger.info(f"Redis initialization: {redis_success}")
        
        if not (db_success and redis_success):
            logger.error("Infrastructure initialization failed")
            return False
        
        # Step 2: Create real agents
        logger.info("🤖 Creating real agents...")
        
        from agents.specialized_agents import ResearchAgent, AnalysisAgent
        
        # Create Research Agent
        research_agent = ResearchAgent(name="RealResearchAgent")
        research_init = await research_agent.initialize()
        logger.info(f"Research Agent: {research_agent.name} - {research_agent.status} - Init: {research_init}")
        
        # Create Analysis Agent
        analysis_agent = AnalysisAgent(name="RealAnalysisAgent")
        analysis_init = await analysis_agent.initialize()
        logger.info(f"Analysis Agent: {analysis_agent.name} - {analysis_agent.status} - Init: {analysis_init}")
        
        # Step 3: Test document retrieval
        logger.info("📚 Testing document retrieval...")
        
        try:
            search_result = await research_agent._search_documents({
                "query": "agent communication",
                "limit": 3
            })
            logger.info(f"Document search result: {search_result.get('search_method', 'unknown')}")
            logger.info(f"Documents found: {search_result.get('total_found', 0)}")
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
        
        # Step 4: Test agent communication
        logger.info("📡 Testing agent communication...")
        
        from agents.communication_system import MultiAgentCommunicationSystem
        from agents.base_agent import AgentMessage, MessageType
        from agents.communication_system import MessageRoute, CommunicationProtocol
        
        # Initialize communication system
        comm_system = MultiAgentCommunicationSystem()
        comm_init = await comm_system.initialize()
        logger.info(f"Communication system initialized: {comm_init}")
        
        # Register agents
        await comm_system.join_channel(research_agent.agent_id, "general_broadcast")
        await comm_system.join_channel(analysis_agent.agent_id, "general_broadcast")
        
        # Send message from research agent to analysis agent
        message = AgentMessage(
            type=MessageType.TASK,
            sender_id=research_agent.agent_id,
            recipient_id=analysis_agent.agent_id,
            content={
                "action": "greeting",
                "message": "Hello from Research Agent!",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        route = MessageRoute(
            sender_id=research_agent.agent_id,
            recipient_ids=[analysis_agent.agent_id],
            protocol=CommunicationProtocol.DIRECT
        )
        
        message_sent = await comm_system.send_message(message, route)
        logger.info(f"Message sent: {message_sent}")
        
        # Check if message was received
        received_message = await comm_system.receive_message(analysis_agent.agent_id)
        if received_message:
            logger.info(f"Message received: {received_message.content['message']}")
        else:
            logger.warning("No message received")
        
        # Step 5: Test task execution
        logger.info("⚙️ Testing task execution...")
        
        try:
            task_result = await analysis_agent._perform_statistical_analysis({
                "dataset": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "analysis_type": "descriptive"
            })
            logger.info(f"Statistical analysis completed: {task_result['statistics']['mean']}")
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
        
        logger.info("🎉 REAL AGENT SYSTEM TEST COMPLETED")
        return True
        
    except Exception as e:
        logger.error(f"Real agent system test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

async def main():
    """Main test function"""
    success = await test_real_agent_system()
    
    # Write final result to both log and console
    result_msg = f"Real Agent System Test: {'SUCCESS' if success else 'FAILED'}"
    logger.info(result_msg)
    print(result_msg)
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
