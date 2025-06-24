"""
Agent Startup Script

Starts all PyGent Factory agent services locally.
"""

import asyncio
import subprocess
import sys
import os
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentManager:
    """Manages PyGent Factory agent services."""
    
    def __init__(self):
        self.agents = {
            "tot_reasoning": {
                "script": "agents/tot_reasoning_agent.py",
                "port": 8001,
                "process": None
            },
            "rag_retrieval": {
                "script": "agents/rag_retrieval_agent.py", 
                "port": 8002,
                "process": None
            }
        }
    
    async def start_agent(self, agent_name):
        """Start a single agent."""
        try:
            agent_info = self.agents[agent_name]
            script_path = agent_info["script"]
            port = agent_info["port"]
            
            logger.info(f"🚀 Starting {agent_name} on port {port}...")
            
            # Start the agent process
            process = subprocess.Popen([
                sys.executable, script_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            agent_info["process"] = process
            
            # Wait a moment for startup
            await asyncio.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                logger.info(f"✅ {agent_name} started successfully on port {port}")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ {agent_name} failed to start:")
                logger.error(f"   stdout: {stdout.decode()}")
                logger.error(f"   stderr: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start {agent_name}: {e}")
            return False
    
    async def start_all_agents(self):
        """Start all agents."""
        logger.info("🚀 Starting PyGent Factory Agent Services...")
        
        success_count = 0
        for agent_name in self.agents.keys():
            if await self.start_agent(agent_name):
                success_count += 1
        
        total_agents = len(self.agents)
        logger.info(f"📊 Agent startup complete: {success_count}/{total_agents} agents running")
        
        if success_count == total_agents:
            logger.info("🎉 All agents started successfully!")
            return True
        else:
            logger.warning(f"⚠️ Only {success_count}/{total_agents} agents started")
            return False
    
    async def check_agent_health(self):
        """Check health of all running agents."""
        try:
            import aiohttp
            
            logger.info("🔍 Checking agent health...")
            
            async with aiohttp.ClientSession() as session:
                for agent_name, agent_info in self.agents.items():
                    port = agent_info["port"]
                    
                    try:
                        async with session.get(f"http://localhost:{port}/health", timeout=5) as response:
                            if response.status == 200:
                                health_data = await response.json()
                                logger.info(f"✅ {agent_name}: {health_data.get('status', 'unknown')}")
                            else:
                                logger.warning(f"⚠️ {agent_name}: HTTP {response.status}")
                                
                    except Exception as e:
                        logger.error(f"❌ {agent_name}: Health check failed - {e}")
            
            return True
            
        except ImportError:
            logger.warning("⚠️ aiohttp not available - skipping health checks")
            return False
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
    
    def stop_all_agents(self):
        """Stop all running agents."""
        logger.info("🛑 Stopping all agents...")
        
        for agent_name, agent_info in self.agents.items():
            process = agent_info.get("process")
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    logger.info(f"✅ {agent_name} stopped")
                except Exception as e:
                    logger.error(f"❌ Failed to stop {agent_name}: {e}")
                    try:
                        process.kill()
                    except:
                        pass


async def main():
    """Main function to start and manage agents."""
    manager = AgentManager()
    
    try:
        # Start all agents
        success = await manager.start_all_agents()
        
        if success:
            # Check health
            await asyncio.sleep(3)
            await manager.check_agent_health()
            
            logger.info("\n" + "="*50)
            logger.info("🎯 PyGent Factory Agents Running")
            logger.info("="*50)
            logger.info("🧠 ToT Reasoning: http://localhost:8001")
            logger.info("🔍 RAG Retrieval: http://localhost:8002")
            logger.info("="*50)
            logger.info("Press Ctrl+C to stop all agents")
            
            # Keep running until interrupted
            try:
                await asyncio.Future()
            except KeyboardInterrupt:
                logger.info("\n🛑 Shutdown requested...")
        
        else:
            logger.error("❌ Failed to start all agents")
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutdown requested...")
    except Exception as e:
        logger.error(f"❌ Agent manager failed: {e}")
    finally:
        manager.stop_all_agents()
        logger.info("👋 Agent manager shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())