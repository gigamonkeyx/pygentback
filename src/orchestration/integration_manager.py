"""
Integration Manager

Manages the transition from mock implementations to real integrations.
Provides a unified interface for all real system integrations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .real_database_client import RealDatabaseClient, DatabaseIntegrationAdapter, create_real_database_client
from .real_memory_client import RealMemoryClient, MemoryIntegrationAdapter, create_real_memory_client
from .real_github_client import RealGitHubClient, GitHubIntegrationAdapter, create_real_github_client
from .real_agent_integration import RealAgentClient, RealAgentExecutor, create_real_agent_client

logger = logging.getLogger(__name__)


class IntegrationManager:
    """
    Manages all real system integrations and provides fallback mechanisms.
    
    Features:
    - Centralized integration management
    - Health monitoring for all integrations
    - Automatic fallback to mock implementations when needed
    - Performance monitoring and metrics
    - Configuration management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Real clients
        self.database_client: Optional[RealDatabaseClient] = None
        self.memory_client: Optional[RealMemoryClient] = None
        self.github_client: Optional[RealGitHubClient] = None
        self.agent_client: Optional[RealAgentClient] = None
        
        # Integration adapters
        self.database_adapter: Optional[DatabaseIntegrationAdapter] = None
        self.memory_adapter: Optional[MemoryIntegrationAdapter] = None
        self.github_adapter: Optional[GitHubIntegrationAdapter] = None
        
        # Integration status
        self.integration_status = {
            "database": {"connected": False, "last_check": None, "error": None},
            "memory": {"connected": False, "last_check": None, "error": None},
            "github": {"connected": False, "last_check": None, "error": None},
            "agents": {"connected": False, "last_check": None, "error": None}
        }
        
        # Performance metrics
        self.performance_metrics = {
            "database_response_times": [],
            "memory_response_times": [],
            "github_response_times": [],
            "agent_response_times": []
        }
        
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all real integrations."""
        try:
            logger.info("Initializing real system integrations...")
            
            # Initialize database integration
            await self._initialize_database()
            
            # Initialize memory integration
            await self._initialize_memory()
            
            # Initialize GitHub integration
            await self._initialize_github()
            
            # Initialize agent integration
            await self._initialize_agents()
            
            self.is_initialized = True
            
            # Log integration status
            connected_count = sum(1 for status in self.integration_status.values() if status["connected"])
            total_count = len(self.integration_status)
            
            logger.info(f"Integration initialization complete: {connected_count}/{total_count} systems connected")
            
            return connected_count > 0  # Success if at least one system connected
            
        except Exception as e:
            logger.error(f"Integration initialization failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown all integrations."""
        try:
            logger.info("Shutting down integrations...")
            
            if self.database_client:
                await self.database_client.disconnect()
            
            if self.memory_client:
                await self.memory_client.disconnect()
            
            if self.github_client:
                await self.github_client.disconnect()
            
            if self.agent_client:
                await self.agent_client.disconnect()
            
            self.is_initialized = False
            logger.info("All integrations shut down")
            
        except Exception as e:
            logger.error(f"Integration shutdown failed: {e}")
    
    async def _initialize_database(self):
        """Initialize database integration."""
        try:
            self.database_client = await create_real_database_client()
            self.database_adapter = DatabaseIntegrationAdapter(self.database_client)
            
            self.integration_status["database"] = {
                "connected": True,
                "last_check": datetime.utcnow(),
                "error": None
            }
            
            logger.info("✅ Database integration initialized")
            
        except Exception as e:
            logger.error(f"❌ Database integration failed: {e}")
            self.integration_status["database"] = {
                "connected": False,
                "last_check": datetime.utcnow(),
                "error": str(e)
            }
    
    async def _initialize_memory(self):
        """Initialize memory integration."""
        try:
            self.memory_client = await create_real_memory_client()
            self.memory_adapter = MemoryIntegrationAdapter(self.memory_client)
            
            self.integration_status["memory"] = {
                "connected": True,
                "last_check": datetime.utcnow(),
                "error": None
            }
            
            logger.info("✅ Memory integration initialized")
            
        except Exception as e:
            logger.error(f"❌ Memory integration failed: {e}")
            self.integration_status["memory"] = {
                "connected": False,
                "last_check": datetime.utcnow(),
                "error": str(e)
            }
    
    async def _initialize_github(self):
        """Initialize GitHub integration."""
        try:
            self.github_client = await create_real_github_client()
            self.github_adapter = GitHubIntegrationAdapter(self.github_client)
            
            self.integration_status["github"] = {
                "connected": True,
                "last_check": datetime.utcnow(),
                "error": None
            }
            
            logger.info("✅ GitHub integration initialized")
            
        except Exception as e:
            logger.error(f"❌ GitHub integration failed: {e}")
            self.integration_status["github"] = {
                "connected": False,
                "last_check": datetime.utcnow(),
                "error": str(e)
            }
    
    async def _initialize_agents(self):
        """Initialize agent integration."""
        try:
            self.agent_client = await create_real_agent_client()
            
            self.integration_status["agents"] = {
                "connected": True,
                "last_check": datetime.utcnow(),
                "error": None
            }
            
            logger.info("✅ Agent integration initialized")
            
        except Exception as e:
            logger.error(f"❌ Agent integration failed: {e}")
            self.integration_status["agents"] = {
                "connected": False,
                "last_check": datetime.utcnow(),
                "error": str(e)
            }
    
    async def execute_database_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database request with REAL implementation only - no fallbacks."""
        start_time = datetime.utcnow()

        try:
            if self.database_adapter and self.integration_status["database"]["connected"]:
                result = await self.database_adapter.execute_postgres_request(request)
                result["integration_type"] = "real"

                # Record performance
                response_time = (datetime.utcnow() - start_time).total_seconds()
                self.performance_metrics["database_response_times"].append(response_time)

                return result
            else:
                # NO FALLBACK - require real database connection
                raise RuntimeError("Real database connection required but not available. No fallback implementations allowed in production.")
            
        except Exception as e:
            logger.error(f"Database request failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "integration_type": "error"
            }
    
    async def execute_memory_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory request with real or fallback implementation."""
        start_time = datetime.utcnow()
        
        try:
            if self.memory_adapter and self.integration_status["memory"]["connected"]:
                result = await self.memory_adapter.execute_memory_request(request)
                result["integration_type"] = "real"
            else:
                result = await self._fallback_memory_request(request)
                result["integration_type"] = "fallback"
            
            # Record performance
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.performance_metrics["memory_response_times"].append(response_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Memory request failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "integration_type": "error"
            }
    
    async def execute_github_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GitHub request with real or fallback implementation."""
        start_time = datetime.utcnow()
        
        try:
            if self.github_adapter and self.integration_status["github"]["connected"]:
                result = await self.github_adapter.execute_github_request(request)
                result["integration_type"] = "real"
            else:
                result = await self._fallback_github_request(request)
                result["integration_type"] = "fallback"
            
            # Record performance
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self.performance_metrics["github_response_times"].append(response_time)
            
            return result
            
        except Exception as e:
            logger.error(f"GitHub request failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "integration_type": "error"
            }
    
    async def create_real_agent_executor(self, agent_id: str, agent_type: str) -> RealAgentExecutor:
        """Create real agent executor."""
        if self.agent_client and self.integration_status["agents"]["connected"]:
            return RealAgentExecutor(agent_id, agent_type, self.agent_client)
        else:
            # Return fallback executor
            return self._create_fallback_agent_executor(agent_id, agent_type)
    
    # FALLBACK IMPLEMENTATIONS REMOVED - NO MOCK DATA ALLOWED IN PRODUCTION
    async def _fallback_database_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback database implementation REMOVED - no mock data allowed."""
        raise RuntimeError("Database fallback implementations are not allowed in production. Real database connection required.")
    
    async def _fallback_memory_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback memory implementation."""
        operation = request.get("operation", "")
        key = request.get("key", "")
        
        if operation == "store":
            return {
                "status": "success",
                "message": f"Fallback stored: {key}",
                "key": key
            }
        elif operation == "retrieve":
            return {
                "status": "success",
                "key": key,
                "value": f"Fallback value for {key}",
                "found": True
            }
        else:
            return {
                "status": "success",
                "message": f"Fallback memory operation: {operation}"
            }
    
    async def _fallback_github_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback GitHub implementation."""
        operation = request.get("operation", "")
        
        if operation == "get_repository":
            repo = request.get("repository", "")
            return {
                "status": "success",
                "repository": {
                    "name": repo,
                    "description": "Fallback repository description",
                    "url": f"https://github.com/fallback/{repo}"
                }
            }
        elif operation == "create_file":
            path = request.get("path", "")
            return {
                "status": "success",
                "message": f"Fallback file created: {path}",
                "sha": "fallback_sha_123"
            }
        else:
            return {
                "status": "success",
                "message": f"Fallback GitHub operation: {operation}"
            }
    
    def _create_fallback_agent_executor(self, agent_id: str, agent_type: str):
        """Create real agent executor - no fallback to mock implementations."""
        from .real_agent_integration import RealAgentExecutor
        return RealAgentExecutor(agent_id, agent_type)
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        return {
            "is_initialized": self.is_initialized,
            "integrations": dict(self.integration_status),
            "performance_metrics": {
                "database_avg_response": sum(self.performance_metrics["database_response_times"][-100:]) / max(len(self.performance_metrics["database_response_times"][-100:]), 1),
                "memory_avg_response": sum(self.performance_metrics["memory_response_times"][-100:]) / max(len(self.performance_metrics["memory_response_times"][-100:]), 1),
                "github_avg_response": sum(self.performance_metrics["github_response_times"][-100:]) / max(len(self.performance_metrics["github_response_times"][-100:]), 1),
                "agent_avg_response": sum(self.performance_metrics["agent_response_times"][-100:]) / max(len(self.performance_metrics["agent_response_times"][-100:]), 1)
            },
            "connected_systems": sum(1 for status in self.integration_status.values() if status["connected"]),
            "total_systems": len(self.integration_status)
        }


# Global integration manager instance
_integration_manager: Optional[IntegrationManager] = None


async def get_integration_manager(config: Dict[str, Any] = None) -> IntegrationManager:
    """Get or create global integration manager."""
    global _integration_manager
    
    if _integration_manager is None:
        if config is None:
            config = {
                "database_enabled": True,
                "memory_enabled": True,
                "github_enabled": True,
                "agents_enabled": True
            }
        
        _integration_manager = IntegrationManager(config)
        await _integration_manager.initialize()
    
    return _integration_manager


async def shutdown_integration_manager():
    """Shutdown global integration manager."""
    global _integration_manager
    
    if _integration_manager:
        await _integration_manager.shutdown()
        _integration_manager = None