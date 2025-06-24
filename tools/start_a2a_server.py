#!/usr/bin/env python3
"""
A2A Server Startup Script

Launches the A2A server with full agent registration and monitoring.
"""

import os
import sys
import asyncio
import logging
import signal
from pathlib import Path
from datetime import datetime

# Environment setup
os.environ.update({
    "DATABASE_URL": "postgresql+asyncpg://postgres:postgres@localhost:54321/pygent_factory",
    "DB_HOST": "localhost",
    "DB_PORT": "54321", 
    "DB_NAME": "pygent_factory",
    "DB_USER": "postgres",
    "DB_PASSWORD": "postgres",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "REDIS_DB": "0"
})

sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('a2a_server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class A2AServerManager:
    """A2A Server Manager for production deployment"""
    
    def __init__(self):
        self.running = False
        self.server_task = None
        self.agents = []
        
    async def initialize_infrastructure(self):
        """Initialize database and Redis infrastructure"""
        logger.info("🔧 Initializing infrastructure...")
        
        try:
            from database.production_manager import initialize_database, db_manager
            from cache.redis_manager import initialize_redis, redis_manager
            
            # Initialize database
            db_success = await initialize_database()
            if not db_success:
                raise Exception("Database initialization failed")
            
            # Initialize Redis
            redis_success = await initialize_redis()
            if not redis_success:
                raise Exception("Redis initialization failed")
            
            logger.info("✅ Infrastructure initialized successfully")
            return db_manager, redis_manager
            
        except Exception as e:
            logger.error(f"❌ Infrastructure initialization failed: {e}")
            raise
    
    async def initialize_a2a_system(self, db_manager, redis_manager):
        """Initialize A2A protocol system"""
        logger.info("🤖 Initializing A2A system...")
        
        try:
            from a2a_protocol import a2a_manager
            
            # Initialize A2A manager
            success = await a2a_manager.initialize(
                database_manager=db_manager,
                redis_manager=redis_manager
            )
            
            if not success:
                raise Exception("A2A manager initialization failed")
            
            logger.info("✅ A2A system initialized successfully")
            return a2a_manager
            
        except Exception as e:
            logger.error(f"❌ A2A system initialization failed: {e}")
            raise
    
    async def register_agents(self, a2a_manager):
        """Register agents with A2A protocol"""
        logger.info("👥 Registering agents...")
        
        try:
            from agents.specialized_agents import ResearchAgent, AnalysisAgent
            
            # Create and initialize agents
            research_agent = ResearchAgent(name="ProductionResearchAgent")
            analysis_agent = AnalysisAgent(name="ProductionAnalysisAgent")
            
            # Initialize agents
            research_init = await research_agent.initialize()
            analysis_init = await analysis_agent.initialize()
            
            if not (research_init and analysis_init):
                raise Exception("Agent initialization failed")
            
            # Register with A2A
            research_reg = await a2a_manager.register_agent(research_agent)
            analysis_reg = await a2a_manager.register_agent(analysis_agent)
            
            if not (research_reg and analysis_reg):
                raise Exception("Agent registration failed")
            
            self.agents = [research_agent, analysis_agent]
            
            logger.info("✅ Agents registered successfully:")
            logger.info(f"   - Research Agent: {research_agent.name} ({research_agent.agent_id})")
            logger.info(f"   - Analysis Agent: {analysis_agent.name} ({analysis_agent.agent_id})")
            
            return self.agents
            
        except Exception as e:
            logger.error(f"❌ Agent registration failed: {e}")
            raise
    
    async def start_server(self, host="localhost", port=8080):
        """Start the A2A server"""
        logger.info(f"🚀 Starting A2A server on {host}:{port}...")
        
        try:
            from a2a_protocol import a2a_manager
            
            # Start server
            success = await a2a_manager.start_server(host=host, port=port)
            
            if not success:
                raise Exception("Server startup failed")
            
            self.running = True
            logger.info(f"✅ A2A server started successfully on http://{host}:{port}")
            logger.info(f"📡 Agent discovery endpoint: http://{host}:{port}/.well-known/agent.json")
            logger.info(f"🔗 JSON-RPC endpoint: http://{host}:{port}/")
            logger.info(f"📊 Health check endpoint: http://{host}:{port}/health")
            logger.info(f"👥 Agents list endpoint: http://{host}:{port}/agents")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Server startup failed: {e}")
            raise
    
    async def monitor_system(self):
        """Monitor system health and status"""
        logger.info("📊 Starting system monitoring...")
        
        try:
            from a2a_protocol import a2a_manager
            
            while self.running:
                # Get system status
                status = await a2a_manager.get_agent_status()
                
                logger.info(f"System Status:")
                logger.info(f"   - Total Agents: {status.get('total_agents', 0)}")
                logger.info(f"   - Active Tasks: {status.get('active_tasks', 0)}")
                logger.info(f"   - Timestamp: {datetime.utcnow().isoformat()}")
                
                # Wait before next check
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
        except Exception as e:
            logger.error(f"❌ System monitoring error: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down A2A server...")
        
        try:
            self.running = False
            
            from a2a_protocol import a2a_manager
            await a2a_manager.shutdown()
            
            logger.info("✅ A2A server shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")


async def main():
    """Main server startup function"""
    
    print("PYGENT FACTORY A2A SERVER")
    print("=" * 50)
    print(f"Starting at: {datetime.utcnow().isoformat()}")
    print("=" * 50)
    
    server_manager = A2AServerManager()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler():
        logger.info("📡 Received shutdown signal")
        asyncio.create_task(server_manager.shutdown())
    
    # Register signal handlers
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, lambda s, f: signal_handler())
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, lambda s, f: signal_handler())
    
    try:
        # Initialize infrastructure
        db_manager, redis_manager = await server_manager.initialize_infrastructure()
        
        # Initialize A2A system
        a2a_manager = await server_manager.initialize_a2a_system(db_manager, redis_manager)
        
        # Register agents
        agents = await server_manager.register_agents(a2a_manager)
        
        # Start server
        await server_manager.start_server(host="0.0.0.0", port=8080)
        
        print("\nA2A SERVER READY!")
        print("=" * 50)
        print("Endpoints Available:")
        print("   - Agent Discovery: http://localhost:8080/.well-known/agent.json")
        print("   - JSON-RPC API: http://localhost:8080/")
        print("   - Health Check: http://localhost:8080/health")
        print("   - Agents List: http://localhost:8080/agents")
        print("\nRegistered Agents:")
        for agent in agents:
            print(f"   - {agent.name} ({str(agent.agent_type)})")
        print("\nMonitoring active...")
        print("   Press Ctrl+C to shutdown gracefully")
        print("=" * 50)
        
        # Start monitoring
        monitor_task = asyncio.create_task(server_manager.monitor_system())
        
        # Keep server running
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
    except KeyboardInterrupt:
        logger.info("📡 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await server_manager.shutdown()
        print("\nA2A Server stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
