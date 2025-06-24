"""
Strict Integration Manager

NO FALLBACKS. NO MOCK CODE. REAL INTEGRATIONS ONLY.
If a real integration fails, the system fails. Period.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .real_database_client import create_real_database_client
from .real_memory_client import create_real_memory_client  
from .real_github_client import create_real_github_client
from .real_agent_integration import create_real_agent_client

logger = logging.getLogger(__name__)


class StrictIntegrationManager:
    """
    Strict integration manager with ZERO fallbacks.
    
    Philosophy: Real integrations or failure. No mock code allowed.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Real clients ONLY
        self.database_client = None
        self.memory_client = None
        self.github_client = None
        self.agent_client = None
        
        # Integration status
        self.integration_status = {
            "database": {"connected": False, "error": None},
            "memory": {"connected": False, "error": None},
            "github": {"connected": False, "error": None},
            "agents": {"connected": False, "error": None}
        }
        
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize ALL real integrations. Fail if any fail."""
        try:
            logger.info("🎯 Initializing STRICT real integrations (no fallbacks)...")
            
            # Database - MUST work
            try:
                self.database_client = await create_real_database_client()
                self.integration_status["database"]["connected"] = True
                logger.info("✅ Database: Real PostgreSQL connected")
            except Exception as e:
                self.integration_status["database"]["error"] = str(e)
                logger.error(f"❌ Database: FAILED - {e}")
                raise ConnectionError(f"Database connection required but failed: {e}")
            
            # Memory - MUST work
            try:
                self.memory_client = await create_real_memory_client()
                # Force Redis mode, no local fallback
                if not self.memory_client.use_redis:
                    raise ConnectionError("Redis required but not available")
                self.integration_status["memory"]["connected"] = True
                logger.info("✅ Memory: Real Redis connected")
            except Exception as e:
                self.integration_status["memory"]["error"] = str(e)
                logger.error(f"❌ Memory: FAILED - {e}")
                raise ConnectionError(f"Redis connection required but failed: {e}")
            
            # GitHub - MUST work
            try:
                self.github_client = await create_real_github_client()
                if not self.github_client.is_connected:
                    raise ConnectionError("GitHub API connection failed")
                self.integration_status["github"]["connected"] = True
                logger.info("✅ GitHub: Real API connected")
            except Exception as e:
                self.integration_status["github"]["error"] = str(e)
                logger.error(f"❌ GitHub: FAILED - {e}")
                raise ConnectionError(f"GitHub API connection required but failed: {e}")
            
            # Agents - MUST work
            try:
                self.agent_client = await create_real_agent_client()
                if not self.agent_client.is_connected:
                    raise ConnectionError("Agent systems connection failed")
                self.integration_status["agents"]["connected"] = True
                logger.info("✅ Agents: Real PyGent Factory agents connected")
            except Exception as e:
                self.integration_status["agents"]["error"] = str(e)
                logger.error(f"❌ Agents: FAILED - {e}")
                raise ConnectionError(f"Agent systems connection required but failed: {e}")
            
            self.is_initialized = True
            logger.info("🎉 ALL REAL INTEGRATIONS CONNECTED - ZERO MOCK CODE!")
            return True
            
        except Exception as e:
            logger.error(f"💥 STRICT INTEGRATION FAILED: {e}")
            logger.error("🚫 NO FALLBACKS AVAILABLE - SYSTEM CANNOT START")
            raise
    
    async def shutdown(self):
        """Shutdown all real integrations."""
        try:
            if self.database_client:
                await self.database_client.disconnect()
            if self.memory_client:
                await self.memory_client.disconnect()
            if self.github_client:
                await self.github_client.disconnect()
            if self.agent_client:
                await self.agent_client.disconnect()
            
            self.is_initialized = False
            logger.info("🔌 All real integrations disconnected")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
    
    async def execute_database_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database request - REAL ONLY."""
        if not self.database_client or not self.integration_status["database"]["connected"]:
            raise ConnectionError("Real database connection required but not available")
        
        try:
            from .real_database_client import DatabaseIntegrationAdapter
            adapter = DatabaseIntegrationAdapter(self.database_client)
            result = await adapter.execute_postgres_request(request)
            result["integration_type"] = "real"
            return result
            
        except Exception as e:
            logger.error(f"Real database request failed: {e}")
            raise
    
    async def execute_memory_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory request - REAL ONLY."""
        if not self.memory_client or not self.integration_status["memory"]["connected"]:
            raise ConnectionError("Real memory connection required but not available")
        
        try:
            from .real_memory_client import MemoryIntegrationAdapter
            adapter = MemoryIntegrationAdapter(self.memory_client)
            result = await adapter.execute_memory_request(request)
            result["integration_type"] = "real"
            return result
            
        except Exception as e:
            logger.error(f"Real memory request failed: {e}")
            raise
    
    async def execute_github_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GitHub request - REAL ONLY."""
        if not self.github_client or not self.integration_status["github"]["connected"]:
            raise ConnectionError("Real GitHub connection required but not available")
        
        try:
            from .real_github_client import GitHubIntegrationAdapter
            adapter = GitHubIntegrationAdapter(self.github_client)
            result = await adapter.execute_github_request(request)
            result["integration_type"] = "real"
            return result
            
        except Exception as e:
            logger.error(f"Real GitHub request failed: {e}")
            raise
    
    async def create_real_agent_executor(self, agent_id: str, agent_type: str):
        """Create real agent executor - REAL ONLY."""
        if not self.agent_client or not self.integration_status["agents"]["connected"]:
            raise ConnectionError("Real agent connection required but not available")
        
        try:
            from .real_agent_integration import RealAgentExecutor
            return RealAgentExecutor(agent_id, agent_type, self.agent_client)
            
        except Exception as e:
            logger.error(f"Real agent executor creation failed: {e}")
            raise
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status - all must be real."""
        connected_count = sum(1 for status in self.integration_status.values() if status["connected"])
        
        return {
            "is_initialized": self.is_initialized,
            "integrations": dict(self.integration_status),
            "connected_systems": connected_count,
            "total_systems": len(self.integration_status),
            "all_real": connected_count == len(self.integration_status),
            "zero_mock_code": connected_count == len(self.integration_status)
        }


# Global strict integration manager
_strict_integration_manager: Optional[StrictIntegrationManager] = None


async def get_strict_integration_manager(config: Dict[str, Any] = None) -> StrictIntegrationManager:
    """Get strict integration manager - REAL INTEGRATIONS ONLY."""
    global _strict_integration_manager
    
    if _strict_integration_manager is None:
        if config is None:
            config = {
                "require_real_database": True,
                "require_real_memory": True,
                "require_real_github": True,
                "require_real_agents": True
            }
        
        _strict_integration_manager = StrictIntegrationManager(config)
        await _strict_integration_manager.initialize()  # Will fail if any integration fails
    
    return _strict_integration_manager


async def shutdown_strict_integration_manager():
    """Shutdown strict integration manager."""
    global _strict_integration_manager
    
    if _strict_integration_manager:
        await _strict_integration_manager.shutdown()
        _strict_integration_manager = None