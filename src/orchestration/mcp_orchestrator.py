"""
MCP Orchestrator

Central coordinator for managing multiple MCP servers including filesystem,
PostgreSQL, GitHub, memory, and custom servers. Provides intelligent routing,
load balancing, and failover capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any, Union, Callable
from datetime import datetime, timedelta
import weakref
from collections import defaultdict, deque
import json

from .coordination_models import (
    MCPServerInfo, MCPServerType, TaskRequest, PerformanceMetrics,
    OrchestrationConfig, ServerID, CapabilityName
)
from .real_mcp_client import RealMCPClient

logger = logging.getLogger(__name__)


class MCPServerConnection:
    """Represents a connection to an MCP server."""
    
    def __init__(self, server_info: MCPServerInfo):
        self.server_info = server_info
        self.mcp_client = RealMCPClient(server_info.endpoint, server_info.server_type.value)
        self.last_used = datetime.utcnow()
        self.request_count = 0
        self.error_count = 0
        self.is_active = False
        
    async def connect(self):
        """Establish connection to MCP server."""
        try:
            success = await self.mcp_client.connect()
            if success:
                self.is_active = True
                self.server_info.current_connections += 1
                logger.debug(f"Connected to MCP server: {self.server_info.name}")
                return True
            else:
                self.error_count += 1
                return False
        except Exception as e:
            logger.error(f"Failed to connect to {self.server_info.name}: {e}")
            self.error_count += 1
            return False
    
    async def disconnect(self):
        """Close connection to MCP server."""
        try:
            await self.mcp_client.disconnect()
            self.is_active = False
            if self.server_info.current_connections > 0:
                self.server_info.current_connections -= 1
            logger.debug(f"Disconnected from MCP server: {self.server_info.name}")
        except Exception as e:
            logger.error(f"Error disconnecting from {self.server_info.name}: {e}")
    
    async def execute_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a request on the MCP server."""
        if not self.is_active:
            raise ConnectionError(f"Not connected to {self.server_info.name}")
        
        start_time = datetime.utcnow()
        try:
            # Execute real MCP request
            result = await self.mcp_client.execute_request(request)
            
            self.request_count += 1
            self.last_used = datetime.utcnow()
            
            # Update server metrics
            response_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_response_time(response_time)
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Request failed on {self.server_info.name}: {e}")
            raise
    
    def _update_response_time(self, response_time: float):
        """Update average response time."""
        if self.server_info.response_time_avg == 0:
            self.server_info.response_time_avg = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.server_info.response_time_avg = (
                alpha * response_time + 
                (1 - alpha) * self.server_info.response_time_avg
            )


class MCPOrchestrator:
    """
    Central orchestrator for managing multiple MCP servers.
    
    Features:
    - Dynamic server discovery and registration
    - Intelligent request routing and load balancing
    - Connection pooling and management
    - Health monitoring and failover
    - Performance optimization
    """
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.servers: Dict[ServerID, MCPServerInfo] = {}
        self.connections: Dict[ServerID, List[MCPServerConnection]] = defaultdict(list)
        self.server_capabilities: Dict[CapabilityName, Set[ServerID]] = defaultdict(set)
        
        # Performance tracking
        self.request_history: deque = deque(maxlen=1000)
        self.server_metrics: Dict[ServerID, PerformanceMetrics] = {}
        
        # Health monitoring
        self.health_check_tasks: Dict[ServerID, asyncio.Task] = {}
        self.is_running = False
        
        # Load balancing
        self.round_robin_counters: Dict[CapabilityName, int] = defaultdict(int)
        self.server_weights: Dict[ServerID, float] = defaultdict(lambda: 1.0)
        
        logger.info("MCP Orchestrator initialized")
    
    async def start(self):
        """Start the orchestrator and begin health monitoring."""
        self.is_running = True
        
        # Start health monitoring for all registered servers
        for server_id in self.servers:
            await self._start_health_monitoring(server_id)
        
        logger.info("MCP Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator and cleanup resources."""
        self.is_running = False
        
        # Stop health monitoring
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Close all connections
        for server_connections in self.connections.values():
            for connection in server_connections:
                await connection.disconnect()
        
        self.health_check_tasks.clear()
        self.connections.clear()
        
        logger.info("MCP Orchestrator stopped")
    
    async def register_server(self, server_info: MCPServerInfo) -> bool:
        """Register a new MCP server."""
        try:
            # Validate server info
            if not server_info.server_id or not server_info.name:
                raise ValueError("Server ID and name are required")
            
            # Check if server already exists
            if server_info.server_id in self.servers:
                logger.warning(f"Server {server_info.server_id} already registered")
                return False
            
            # Register server
            self.servers[server_info.server_id] = server_info
            
            # Update capability mapping
            for capability in server_info.capabilities:
                self.server_capabilities[capability].add(server_info.server_id)
            
            # Initialize metrics
            self.server_metrics[server_info.server_id] = PerformanceMetrics()
            
            # Start health monitoring if orchestrator is running
            if self.is_running:
                await self._start_health_monitoring(server_info.server_id)
            
            logger.info(f"Registered MCP server: {server_info.name} ({server_info.server_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register server {server_info.name}: {e}")
            return False
    
    async def unregister_server(self, server_id: ServerID) -> bool:
        """Unregister an MCP server."""
        try:
            if server_id not in self.servers:
                logger.warning(f"Server {server_id} not found")
                return False
            
            server_info = self.servers[server_id]
            
            # Stop health monitoring
            if server_id in self.health_check_tasks:
                self.health_check_tasks[server_id].cancel()
                del self.health_check_tasks[server_id]
            
            # Close connections
            if server_id in self.connections:
                for connection in self.connections[server_id]:
                    await connection.disconnect()
                del self.connections[server_id]
            
            # Remove from capability mapping
            for capability in server_info.capabilities:
                self.server_capabilities[capability].discard(server_id)
            
            # Remove server
            del self.servers[server_id]
            if server_id in self.server_metrics:
                del self.server_metrics[server_id]
            
            logger.info(f"Unregistered MCP server: {server_info.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister server {server_id}: {e}")
            return False
    
    async def execute_request(self, 
                            capability: CapabilityName,
                            request: Dict[str, Any],
                            preferred_server: Optional[ServerID] = None) -> Dict[str, Any]:
        """Execute a request on an appropriate MCP server."""
        start_time = datetime.utcnow()
        
        try:
            # Select server
            server_id = await self._select_server(capability, preferred_server)
            if not server_id:
                raise RuntimeError(f"No available server for capability: {capability}")
            
            # Get or create connection
            connection = await self._get_connection(server_id)
            if not connection:
                raise RuntimeError(f"Failed to connect to server: {server_id}")
            
            # Execute request
            result = await connection.execute_request(request)
            
            # Record success
            self._record_request(server_id, True, datetime.utcnow() - start_time)
            
            return result
            
        except Exception as e:
            # Record failure
            if 'server_id' in locals():
                self._record_request(server_id, False, datetime.utcnow() - start_time)
            
            logger.error(f"Request execution failed for capability {capability}: {e}")
            raise
    
    async def execute_batch_requests(self,
                                   requests: List[Dict[str, Any]],
                                   capability: CapabilityName) -> List[Dict[str, Any]]:
        """Execute multiple requests in batch for improved efficiency."""
        if not self.config.batch_processing_enabled:
            # Execute sequentially if batching is disabled
            results = []
            for request in requests:
                result = await self.execute_request(capability, request)
                results.append(result)
            return results
        
        # Group requests by server for optimal batching
        server_groups = await self._group_requests_by_server(requests, capability)
        
        # Execute batches in parallel
        batch_tasks = []
        for server_id, server_requests in server_groups.items():
            task = asyncio.create_task(
                self._execute_server_batch(server_id, server_requests)
            )
            batch_tasks.append(task)
        
        # Collect results
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Flatten results maintaining order
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch execution failed: {batch_result}")
                # Add error results for failed batch
                results.extend([{"error": str(batch_result)}] * len(requests))
            else:
                results.extend(batch_result)
        
        return results
    
    async def get_server_status(self, server_id: Optional[ServerID] = None) -> Dict[str, Any]:
        """Get status information for servers."""
        if server_id:
            if server_id not in self.servers:
                return {"error": f"Server {server_id} not found"}
            
            server_info = self.servers[server_id]
            metrics = self.server_metrics.get(server_id, PerformanceMetrics())
            
            return {
                "server_id": server_id,
                "name": server_info.name,
                "type": server_info.server_type.value,
                "is_healthy": server_info.is_healthy,
                "connection_utilization": server_info.connection_utilization,
                "response_time_avg": server_info.response_time_avg,
                "success_rate": server_info.success_rate,
                "capabilities": list(server_info.capabilities),
                "metrics": metrics
            }
        else:
            # Return status for all servers
            status = {}
            for sid in self.servers:
                status[sid] = await self.get_server_status(sid)
            return status
    
    async def get_capability_servers(self, capability: CapabilityName) -> List[ServerID]:
        """Get list of servers that support a specific capability."""
        return list(self.server_capabilities.get(capability, set()))
    
    async def _select_server(self, 
                           capability: CapabilityName,
                           preferred_server: Optional[ServerID] = None) -> Optional[ServerID]:
        """Select the best server for a capability."""
        available_servers = []
        
        # Get servers that support the capability
        candidate_servers = self.server_capabilities.get(capability, set())
        
        # Filter for available and healthy servers
        for server_id in candidate_servers:
            server_info = self.servers.get(server_id)
            if server_info and server_info.can_handle_capability(capability):
                available_servers.append(server_id)
        
        if not available_servers:
            return None
        
        # Use preferred server if available
        if preferred_server and preferred_server in available_servers:
            return preferred_server
        
        # Apply load balancing algorithm
        if self.config.load_balancing_algorithm == "round_robin":
            return self._round_robin_selection(capability, available_servers)
        elif self.config.load_balancing_algorithm == "weighted_round_robin":
            return self._weighted_round_robin_selection(available_servers)
        elif self.config.load_balancing_algorithm == "least_connections":
            return self._least_connections_selection(available_servers)
        else:
            # Default to round robin
            return self._round_robin_selection(capability, available_servers)
    
    def _round_robin_selection(self, capability: CapabilityName, servers: List[ServerID]) -> ServerID:
        """Round robin server selection."""
        counter = self.round_robin_counters[capability]
        selected = servers[counter % len(servers)]
        self.round_robin_counters[capability] = counter + 1
        return selected
    
    def _weighted_round_robin_selection(self, servers: List[ServerID]) -> ServerID:
        """Weighted round robin based on server performance."""
        # Calculate weights based on performance metrics
        weights = []
        for server_id in servers:
            server_info = self.servers[server_id]
            # Weight based on inverse of response time and connection utilization
            weight = self.server_weights[server_id]
            if server_info.response_time_avg > 0:
                weight *= (1.0 / server_info.response_time_avg)
            weight *= (1.0 - server_info.connection_utilization)
            weights.append(max(weight, 0.1))  # Minimum weight
        
        # Select based on weights
        total_weight = sum(weights)
        if total_weight == 0:
            return servers[0]
        
        import random
        r = random.uniform(0, total_weight)
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return servers[i]
        
        return servers[-1]
    
    def _least_connections_selection(self, servers: List[ServerID]) -> ServerID:
        """Select server with least connections."""
        min_connections = float('inf')
        selected_server = servers[0]
        
        for server_id in servers:
            server_info = self.servers[server_id]
            if server_info.current_connections < min_connections:
                min_connections = server_info.current_connections
                selected_server = server_id
        
        return selected_server
    
    async def _get_connection(self, server_id: ServerID) -> Optional[MCPServerConnection]:
        """Get or create a connection to the specified server."""
        server_info = self.servers.get(server_id)
        if not server_info:
            return None
        
        # Check for available connection in pool
        server_connections = self.connections[server_id]
        for connection in server_connections:
            if connection.is_active and not connection.connection:  # Available connection
                return connection
        
        # Create new connection if pool not full
        if len(server_connections) < self.config.connection_pool_size:
            connection = MCPServerConnection(server_info)
            if await connection.connect():
                server_connections.append(connection)
                return connection
        
        # Wait for available connection or timeout
        timeout = self.config.server_timeout
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            for connection in server_connections:
                if connection.is_active and not connection.connection:
                    return connection
            await asyncio.sleep(0.1)
        
        logger.warning(f"Connection timeout for server {server_id}")
        return None
    
    async def _start_health_monitoring(self, server_id: ServerID):
        """Start health monitoring for a server."""
        async def health_check_loop():
            while self.is_running and server_id in self.servers:
                try:
                    await self._perform_health_check(server_id)
                    await asyncio.sleep(self.config.server_health_check_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health check error for {server_id}: {e}")
                    await asyncio.sleep(self.config.server_health_check_interval)
        
        task = asyncio.create_task(health_check_loop())
        self.health_check_tasks[server_id] = task
    
    async def _perform_health_check(self, server_id: ServerID):
        """Perform health check on a server."""
        server_info = self.servers.get(server_id)
        if not server_info:
            return
        
        try:
            # Simple health check - attempt to create connection
            connection = MCPServerConnection(server_info)
            if await connection.connect():
                await connection.disconnect()
                server_info.is_healthy = True
                server_info.last_health_check = datetime.utcnow()
            else:
                server_info.is_healthy = False
        except Exception as e:
            server_info.is_healthy = False
            logger.warning(f"Health check failed for {server_id}: {e}")
    
    def _record_request(self, server_id: ServerID, success: bool, duration: timedelta):
        """Record request metrics."""
        self.request_history.append({
            'server_id': server_id,
            'success': success,
            'duration': duration.total_seconds(),
            'timestamp': datetime.utcnow()
        })
        
        # Update server success rate
        server_info = self.servers.get(server_id)
        if server_info:
            # Exponential moving average for success rate
            alpha = 0.1
            current_success = 1.0 if success else 0.0
            server_info.success_rate = (
                alpha * current_success + 
                (1 - alpha) * server_info.success_rate
            )
    
    async def _group_requests_by_server(self, 
                                      requests: List[Dict[str, Any]], 
                                      capability: CapabilityName) -> Dict[ServerID, List[Dict[str, Any]]]:
        """Group requests by optimal server assignment."""
        server_groups = defaultdict(list)
        
        for request in requests:
            server_id = await self._select_server(capability)
            if server_id:
                server_groups[server_id].append(request)
        
        return dict(server_groups)
    
    async def _execute_server_batch(self, 
                                  server_id: ServerID, 
                                  requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a batch of requests on a specific server."""
        connection = await self._get_connection(server_id)
        if not connection:
            raise RuntimeError(f"Failed to get connection to server {server_id}")
        
        results = []
        for request in requests:
            try:
                result = await connection.execute_request(request)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        return results
    
    async def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration metrics."""
        total_servers = len(self.servers)
        healthy_servers = sum(1 for s in self.servers.values() if s.is_healthy)
        
        # Calculate average utilization
        if total_servers > 0:
            avg_server_utilization = sum(
                s.connection_utilization for s in self.servers.values()
            ) / total_servers
        else:
            avg_server_utilization = 0.0
        
        # Calculate recent success rate
        recent_requests = [
            r for r in self.request_history 
            if (datetime.utcnow() - r['timestamp']).total_seconds() < 300
        ]
        
        if recent_requests:
            success_rate = sum(1 for r in recent_requests if r['success']) / len(recent_requests)
            avg_response_time = sum(r['duration'] for r in recent_requests) / len(recent_requests)
        else:
            success_rate = 1.0
            avg_response_time = 0.0
        
        return {
            'total_servers': total_servers,
            'healthy_servers': healthy_servers,
            'server_health_rate': healthy_servers / max(total_servers, 1),
            'avg_server_utilization': avg_server_utilization,
            'recent_success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'total_requests': len(self.request_history),
            'capabilities': list(self.server_capabilities.keys()),
            'active_connections': sum(
                len(connections) for connections in self.connections.values()
            )
        }