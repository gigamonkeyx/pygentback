"""
Real Environment Setup

Sets up REAL infrastructure required for zero mock code operation.
"""

import asyncio
import logging
import subprocess
import sys
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealEnvironmentSetup:
    """Setup real infrastructure for zero mock code operation."""
    
    def __init__(self):
        self.services_status = {
            "postgresql": False,
            "redis": False,
            "github_api": False,
            "pygent_agents": False
        }
    
    async def setup_postgresql(self):
        """Setup real PostgreSQL database."""
        try:
            logger.info("🗄️ Setting up PostgreSQL...")
            
            # Check if PostgreSQL is running
            try:
                import asyncpg
                conn = await asyncpg.connect("postgresql://postgres:postgres@localhost:54321/pygent_factory")
                await conn.fetchval("SELECT 1")
                await conn.close()
                
                logger.info("✅ PostgreSQL: Already running and accessible")
                self.services_status["postgresql"] = True
                return True
                
            except Exception as e:
                logger.error(f"❌ PostgreSQL: Not accessible - {e}")
                logger.info("🔧 PostgreSQL setup instructions:")
                logger.info("   1. Install PostgreSQL: https://www.postgresql.org/download/")
                logger.info("   2. Create database: createdb pygent_factory")
                logger.info("   3. Set connection: postgresql://postgres:postgres@localhost:54321/pygent_factory")
                return False
                
        except Exception as e:
            logger.error(f"PostgreSQL setup failed: {e}")
            return False
    
    async def setup_redis(self):
        """Setup real Redis cache."""
        try:
            logger.info("💾 Setting up Redis...")
            
            # Check if Redis is running
            try:
                import redis
                r = redis.Redis(host='localhost', port=6379, decode_responses=True)
                r.ping()
                
                logger.info("✅ Redis: Already running and accessible")
                self.services_status["redis"] = True
                return True
                
            except Exception as e:
                logger.error(f"❌ Redis: Not accessible - {e}")
                logger.info("🔧 Redis setup instructions:")
                logger.info("   1. Install Redis: https://redis.io/download")
                logger.info("   2. Start Redis server: redis-server")
                logger.info("   3. Default connection: localhost:6379")
                return False
                
        except Exception as e:
            logger.error(f"Redis setup failed: {e}")
            return False
    
    async def setup_github_api(self):
        """Setup real GitHub API access."""
        try:
            logger.info("🐙 Setting up GitHub API...")
            
            # Check for GitHub token
            github_token = os.getenv("GITHUB_TOKEN")
            if not github_token:
                logger.error("❌ GitHub: No GITHUB_TOKEN environment variable")
                logger.info("🔧 GitHub API setup instructions:")
                logger.info("   1. Go to: https://github.com/settings/tokens")
                logger.info("   2. Generate new token with repo permissions")
                logger.info("   3. Set environment variable: export GITHUB_TOKEN=your_token")
                return False
            
            # Test GitHub API access
            try:
                import requests
                headers = {"Authorization": f"token {github_token}"}
                response = requests.get("https://api.github.com/user", headers=headers, timeout=10)
                
                if response.status_code == 200:
                    user_data = response.json()
                    logger.info(f"✅ GitHub: API access working for user {user_data.get('login', 'unknown')}")
                    self.services_status["github_api"] = True
                    return True
                else:
                    logger.error(f"❌ GitHub: API returned status {response.status_code}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ GitHub: API test failed - {e}")
                return False
                
        except Exception as e:
            logger.error(f"GitHub API setup failed: {e}")
            return False
    
    async def setup_pygent_agents(self):
        """Setup real PyGent Factory agent systems."""
        try:
            logger.info("🤖 Setting up PyGent Factory agents...")
            
            # Check for agent endpoints
            agent_endpoints = [
                "http://localhost:8001",  # ToT reasoning
                "http://localhost:8002",  # RAG retrieval  
                "http://localhost:8003",  # RAG generation
                "http://localhost:8004"   # Evaluation
            ]
            
            working_endpoints = 0
            
            for endpoint in agent_endpoints:
                try:
                    import requests
                    response = requests.get(f"{endpoint}/health", timeout=5)
                    if response.status_code == 200:
                        working_endpoints += 1
                        logger.info(f"✅ Agent endpoint working: {endpoint}")
                except:
                    logger.warning(f"⚠️ Agent endpoint not available: {endpoint}")
            
            if working_endpoints > 0:
                logger.info(f"✅ PyGent Agents: {working_endpoints}/{len(agent_endpoints)} endpoints available")
                self.services_status["pygent_agents"] = True
                return True
            else:
                logger.error("❌ PyGent Agents: No agent endpoints available")
                logger.info("🔧 PyGent Factory agents setup instructions:")
                logger.info("   1. Start ToT reasoning agent on port 8001")
                logger.info("   2. Start RAG retrieval agent on port 8002")
                logger.info("   3. Start RAG generation agent on port 8003")
                logger.info("   4. Start evaluation agent on port 8004")
                return False
                
        except Exception as e:
            logger.error(f"PyGent agents setup failed: {e}")
            return False
    
    async def verify_zero_mock_environment(self):
        """Verify that environment supports zero mock code operation."""
        logger.info("🎯 Verifying Zero Mock Code Environment...")
        
        # Setup all services
        postgresql_ok = await self.setup_postgresql()
        redis_ok = await self.setup_redis()
        github_ok = await self.setup_github_api()
        agents_ok = await self.setup_pygent_agents()
        
        # Summary
        total_services = 4
        working_services = sum([postgresql_ok, redis_ok, github_ok, agents_ok])
        
        logger.info("\n" + "="*60)
        logger.info("📊 REAL ENVIRONMENT STATUS:")
        logger.info(f"🗄️ PostgreSQL: {'✅ Ready' if postgresql_ok else '❌ Not Ready'}")
        logger.info(f"💾 Redis: {'✅ Ready' if redis_ok else '❌ Not Ready'}")
        logger.info(f"🐙 GitHub API: {'✅ Ready' if github_ok else '❌ Not Ready'}")
        logger.info(f"🤖 PyGent Agents: {'✅ Ready' if agents_ok else '❌ Not Ready'}")
        logger.info(f"📈 Overall: {working_services}/{total_services} services ready")
        
        if working_services == total_services:
            logger.info("🏆 ZERO MOCK CODE ENVIRONMENT: READY!")
            logger.info("🚀 All real integrations available - no fallbacks needed!")
            return True
        else:
            logger.error("❌ ZERO MOCK CODE ENVIRONMENT: NOT READY")
            logger.error("🔧 Fix the issues above before proceeding")
            logger.error("🚫 System will FAIL without real integrations")
            return False
        
        logger.info("="*60)


async def main():
    """Setup and verify real environment."""
    logger.info("🚀 Setting up Real Environment for Zero Mock Code...")
    
    setup = RealEnvironmentSetup()
    success = await setup.verify_zero_mock_environment()
    
    if success:
        logger.info("\n🎉 ENVIRONMENT READY FOR ZERO MOCK CODE OPERATION!")
        return True
    else:
        logger.error("\n💥 ENVIRONMENT NOT READY - REAL INTEGRATIONS REQUIRED!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)