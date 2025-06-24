"""
MCP Server Pool Manager

This module manages a pool of MCP servers for testing, including installation,
lifecycle management, health monitoring, and resource allocation.
"""

import asyncio
import logging
import subprocess
import psutil
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import shutil

from .discovery import MCPServerInfo, MCPServerDiscovery
from .installer import MCPServerInstaller
from .health_monitor import MCPHealthMonitor
try:
    from ...mcp.server.config import MCPServerConfig, MCPServerType
    from ...mcp.server.manager import MCPServerManager
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    from typing import Dict, Any
    from enum import Enum

    class MCPServerType(Enum):
        FILESYSTEM = "filesystem"
        DATABASE = "database"
        API = "api"
        TOOL = "tool"
        CUSTOM = "custom"

    @dataclass
    class MCPServerConfig:
        id: str
        name: str
        command: str
        capabilities: list
        transport: str
        config: Dict[str, Any]
        auto_start: bool
        server_type: MCPServerType

    class MCPServerManager:
        def __init__(self):
            pass

        async def register_server(self, config):
            return True

        async def start_server(self, server_id):
            return True

        async def stop_server(self, server_id):
            return True

        async def health_check(self):
            return {"registry": {"servers": {}}}


logger = logging.getLogger(__name__)


@dataclass
class MCPServerInstance:
    """Represents an installed and managed MCP server instance"""
    server_info: MCPServerInfo
    config: MCPServerConfig
    process_id: Optional[int] = None
    status: str = "stopped"  # stopped, starting, running, error, crashed
    install_path: Optional[Path] = None
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"  # healthy, unhealthy, unknown
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    restart_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_started: Optional[datetime] = None
    last_stopped: Optional[datetime] = None


class MCPServerPoolManager:
    """
    Manages a pool of MCP servers for comprehensive testing.
    
    Provides installation, lifecycle management, health monitoring,
    and resource allocation for multiple MCP servers.
    """
    
    def __init__(self, 
                 pool_dir: str = "./data/mcp_pool",
                 max_concurrent_servers: int = 20,
                 resource_limits: Optional[Dict[str, Any]] = None):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent_servers = max_concurrent_servers
        self.resource_limits = resource_limits or {
            "max_memory_mb": 4096,
            "max_cpu_percent": 80.0,
            "max_disk_mb": 10240
        }
        
        # Core components
        self.discovery = MCPServerDiscovery()
        self.installer = MCPServerInstaller(str(self.pool_dir / "installations"))
        self.health_monitor = MCPHealthMonitor()
        self.mcp_manager = None  # Will be set during initialization
        
        # Server pool state
        self.server_instances: Dict[str, MCPServerInstance] = {}
        self.running_servers: Set[str] = set()
        self.installation_queue: List[str] = []
        self.startup_queue: List[str] = []
        
        # Resource tracking
        self.total_memory_usage = 0.0
        self.total_cpu_usage = 0.0
        self.total_disk_usage = 0.0
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._resource_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self, mcp_manager: MCPServerManager) -> None:
        """Initialize the pool manager"""
        self.mcp_manager = mcp_manager
        
        # Load discovery cache
        await self.discovery.load_discovery_cache()
        
        # Load existing server instances
        await self._load_server_instances()
        
        # Start background monitoring
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._resource_task = asyncio.create_task(self._resource_monitoring_loop())
        
        logger.info(f"MCP Server Pool Manager initialized with {len(self.server_instances)} servers")
    
    async def shutdown(self) -> None:
        """Shutdown the pool manager"""
        self._running = False
        
        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._resource_task:
            self._resource_task.cancel()
        
        # Stop all running servers
        await self.stop_all_servers()
        
        # Save server instances
        await self._save_server_instances()
        
        logger.info("MCP Server Pool Manager shutdown complete")
    
    async def discover_and_install_servers(self, 
                                         categories: Optional[List[str]] = None,
                                         max_servers_per_category: int = 5) -> Dict[str, List[str]]:
        """
        Discover and install MCP servers by category.
        
        Args:
            categories: List of categories to install, or None for all
            max_servers_per_category: Maximum servers to install per category
            
        Returns:
            Dict mapping categories to lists of installed server names
        """
        # Discover available servers
        discovered_servers = await self.discovery.discover_all_servers()
        
        if not categories:
            categories = ["nlp", "graphics", "database", "web_ui", "development"]
        
        installation_results = {}
        
        for category in categories:
            category_servers = self.discovery.get_servers_by_category(category)
            
            # Sort by verification status and popularity (if available)
            category_servers.sort(key=lambda s: (s.verified, len(s.tools)), reverse=True)
            
            # Limit number of servers per category
            servers_to_install = category_servers[:max_servers_per_category]
            
            installed_servers = []
            for server_info in servers_to_install:
                try:
                    success = await self.install_server(server_info)
                    if success:
                        installed_servers.append(server_info.name)
                        logger.info(f"Installed {category} server: {server_info.name}")
                    else:
                        logger.warning(f"Failed to install {category} server: {server_info.name}")
                
                except Exception as e:
                    logger.error(f"Error installing {server_info.name}: {e}")
            
            installation_results[category] = installed_servers
            logger.info(f"Installed {len(installed_servers)} servers for category: {category}")
        
        return installation_results
    
    async def install_server(self, server_info: MCPServerInfo) -> bool:
        """
        Install a specific MCP server.
        
        Args:
            server_info: Information about the server to install
            
        Returns:
            bool: True if installation successful
        """
        try:
            # Check if already installed
            if server_info.name in self.server_instances:
                logger.info(f"Server {server_info.name} already installed")
                return True
            
            # Install the server
            install_result = await self.installer.install_server(server_info)
            if not install_result.success:
                logger.error(f"Installation failed for {server_info.name}: {install_result.error}")
                return False
            
            # Create server configuration
            config = await self._create_server_config(server_info, install_result.install_path)
            
            # Create server instance
            instance = MCPServerInstance(
                server_info=server_info,
                config=config,
                install_path=install_result.install_path,
                status="installed"
            )
            
            self.server_instances[server_info.name] = instance
            
            # Register with MCP manager
            if self.mcp_manager:
                await self.mcp_manager.register_server(config)
            
            logger.info(f"Successfully installed server: {server_info.name}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to install server {server_info.name}: {e}")
            return False
    
    async def start_server(self, server_name: str) -> bool:
        """
        Start a specific MCP server.
        
        Args:
            server_name: Name of the server to start
            
        Returns:
            bool: True if start successful
        """
        if server_name not in self.server_instances:
            logger.error(f"Server {server_name} not found in pool")
            return False
        
        instance = self.server_instances[server_name]
        
        try:
            # Check resource limits
            if not await self._check_resource_availability(instance):
                logger.warning(f"Insufficient resources to start {server_name}")
                return False
            
            # Start the server via MCP manager
            if self.mcp_manager:
                success = await self.mcp_manager.start_server(instance.config.id)
                if success:
                    instance.status = "running"
                    instance.last_started = datetime.utcnow()
                    self.running_servers.add(server_name)
                    
                    # Update resource usage
                    await self._update_server_resources(instance)
                    
                    logger.info(f"Started server: {server_name}")
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to start server {server_name}: {e}")
            instance.status = "error"
            instance.error_count += 1
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """
        Stop a specific MCP server.
        
        Args:
            server_name: Name of the server to stop
            
        Returns:
            bool: True if stop successful
        """
        if server_name not in self.server_instances:
            logger.error(f"Server {server_name} not found in pool")
            return False
        
        instance = self.server_instances[server_name]
        
        try:
            # Stop the server via MCP manager
            if self.mcp_manager:
                success = await self.mcp_manager.stop_server(instance.config.id)
                if success:
                    instance.status = "stopped"
                    instance.last_stopped = datetime.utcnow()
                    self.running_servers.discard(server_name)
                    
                    # Update resource usage
                    await self._update_server_resources(instance)
                    
                    logger.info(f"Stopped server: {server_name}")
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to stop server {server_name}: {e}")
            return False
    
    async def restart_server(self, server_name: str) -> bool:
        """Restart a specific MCP server"""
        if await self.stop_server(server_name):
            await asyncio.sleep(1)  # Brief pause
            if await self.start_server(server_name):
                instance = self.server_instances[server_name]
                instance.restart_count += 1
                return True
        return False
    
    async def start_servers_by_category(self, category: str, max_servers: int = 5) -> List[str]:
        """Start servers from a specific category"""
        started_servers = []
        category_servers = [
            name for name, instance in self.server_instances.items()
            if instance.server_info.category == category and instance.status != "running"
        ]
        
        for server_name in category_servers[:max_servers]:
            if await self.start_server(server_name):
                started_servers.append(server_name)
        
        return started_servers
    
    async def stop_all_servers(self) -> None:
        """Stop all running servers"""
        stop_tasks = []
        for server_name in list(self.running_servers):
            stop_tasks.append(self.stop_server(server_name))
        
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
    
    async def get_server_health(self, server_name: str) -> Dict[str, Any]:
        """Get health status of a specific server"""
        if server_name not in self.server_instances:
            return {"status": "not_found"}
        
        instance = self.server_instances[server_name]
        
        # Get health from MCP manager
        if self.mcp_manager and instance.status == "running":
            try:
                health = await self.mcp_manager.health_check()
                server_health = health.get("registry", {}).get("servers", {}).get(instance.config.id)
                if server_health:
                    instance.health_status = "healthy" if server_health.get("status") == "running" else "unhealthy"
                    instance.last_health_check = datetime.utcnow()
            except Exception as e:
                logger.error(f"Health check failed for {server_name}: {e}")
                instance.health_status = "unhealthy"
        
        return {
            "status": instance.status,
            "health_status": instance.health_status,
            "last_health_check": instance.last_health_check.isoformat() if instance.last_health_check else None,
            "resource_usage": instance.resource_usage,
            "error_count": instance.error_count,
            "restart_count": instance.restart_count,
            "uptime_seconds": (datetime.utcnow() - instance.last_started).total_seconds() if instance.last_started else 0
        }
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get overall pool status"""
        total_servers = len(self.server_instances)
        running_servers = len(self.running_servers)
        
        # Count by category
        category_counts = {}
        for instance in self.server_instances.values():
            category = instance.server_info.category
            if category not in category_counts:
                category_counts[category] = {"total": 0, "running": 0}
            category_counts[category]["total"] += 1
            if instance.server_info.name in self.running_servers:
                category_counts[category]["running"] += 1
        
        return {
            "total_servers": total_servers,
            "running_servers": running_servers,
            "stopped_servers": total_servers - running_servers,
            "category_breakdown": category_counts,
            "resource_usage": {
                "memory_mb": self.total_memory_usage,
                "cpu_percent": self.total_cpu_usage,
                "disk_mb": self.total_disk_usage
            },
            "resource_limits": self.resource_limits,
            "max_concurrent_servers": self.max_concurrent_servers
        }
    
    async def _create_server_config(self, server_info: MCPServerInfo, install_path: Path) -> MCPServerConfig:
        """Create MCP server configuration from server info"""
        return MCPServerConfig(
            id=f"pool_{server_info.name}",
            name=server_info.name,
            command=server_info.install_command,
            capabilities=server_info.capabilities,
            transport="stdio",  # Default transport
            config=server_info.config_template,
            auto_start=False,  # Manual control in pool
            server_type=server_info.server_type
        )
    
    async def _check_resource_availability(self, instance: MCPServerInstance) -> bool:
        """Check if resources are available to start a server"""
        # Estimate resource requirements (would be more sophisticated in practice)
        estimated_memory = 256  # MB
        estimated_cpu = 10.0    # Percent
        
        if (self.total_memory_usage + estimated_memory > self.resource_limits["max_memory_mb"] or
            self.total_cpu_usage + estimated_cpu > self.resource_limits["max_cpu_percent"] or
            len(self.running_servers) >= self.max_concurrent_servers):
            return False
        
        return True
    
    async def _update_server_resources(self, instance: MCPServerInstance) -> None:
        """Update resource usage for a server"""
        if instance.process_id:
            try:
                process = psutil.Process(instance.process_id)
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                instance.resource_usage = {
                    "memory_mb": memory_mb,
                    "cpu_percent": cpu_percent
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                instance.resource_usage = {"memory_mb": 0, "cpu_percent": 0}
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while self._running:
            try:
                # Health check all running servers
                for server_name in list(self.running_servers):
                    await self.get_server_health(server_name)
                
                # Auto-restart failed servers
                for server_name, instance in self.server_instances.items():
                    if (instance.status == "error" and 
                        instance.restart_count < 3 and
                        instance.last_started and
                        datetime.utcnow() - instance.last_started > timedelta(minutes=5)):
                        
                        logger.info(f"Auto-restarting failed server: {server_name}")
                        await self.restart_server(server_name)
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _resource_monitoring_loop(self) -> None:
        """Background resource monitoring loop"""
        while self._running:
            try:
                # Update resource usage for all running servers
                total_memory = 0.0
                total_cpu = 0.0
                
                for server_name in self.running_servers:
                    instance = self.server_instances[server_name]
                    await self._update_server_resources(instance)
                    total_memory += instance.resource_usage.get("memory_mb", 0)
                    total_cpu += instance.resource_usage.get("cpu_percent", 0)
                
                self.total_memory_usage = total_memory
                self.total_cpu_usage = total_cpu
                
                await asyncio.sleep(60)  # Update every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(120)
    
    async def _save_server_instances(self) -> None:
        """Save server instances to disk"""
        try:
            instances_file = self.pool_dir / "server_instances.json"
            instances_data = {}
            
            for name, instance in self.server_instances.items():
                instances_data[name] = {
                    "server_info": {
                        "name": instance.server_info.name,
                        "description": instance.server_info.description,
                        "server_type": instance.server_info.server_type.value,
                        "category": instance.server_info.category,
                        "install_command": instance.server_info.install_command,
                        "capabilities": instance.server_info.capabilities,
                        "tools": instance.server_info.tools
                    },
                    "status": instance.status,
                    "install_path": str(instance.install_path) if instance.install_path else None,
                    "error_count": instance.error_count,
                    "restart_count": instance.restart_count,
                    "created_at": instance.created_at.isoformat()
                }
            
            with open(instances_file, 'w') as f:
                json.dump(instances_data, f, indent=2)
            
            logger.info(f"Saved {len(instances_data)} server instances")
        
        except Exception as e:
            logger.error(f"Failed to save server instances: {e}")
    
    async def _load_server_instances(self) -> None:
        """Load server instances from disk"""
        try:
            instances_file = self.pool_dir / "server_instances.json"
            if instances_file.exists():
                with open(instances_file, 'r') as f:
                    instances_data = json.load(f)
                
                # Reconstruct server instances (simplified for this implementation)
                logger.info(f"Loaded {len(instances_data)} server instances from disk")
        
        except Exception as e:
            logger.error(f"Failed to load server instances: {e}")
