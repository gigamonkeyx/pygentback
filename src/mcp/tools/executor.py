"""
MCP Tool Executor

Provides secure execution environment for MCP tools with proper
isolation, error handling, and result management.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Tool execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ToolExecutionRequest:
    """Request for tool execution"""
    tool_name: str
    server_name: str
    arguments: Dict[str, Any]
    execution_id: str
    timeout: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    """Result of tool execution"""
    execution_id: str
    tool_name: str
    server_name: str
    status: ExecutionStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution was successful"""
        return self.status == ExecutionStatus.COMPLETED


class MCPToolExecutor:
    """
    MCP Tool Executor for secure tool execution.
    
    Provides isolated execution environment for MCP tools with
    proper error handling, timeout management, and result tracking.
    """
    
    def __init__(self, max_concurrent_executions: int = 10, default_timeout: int = 30):
        self.max_concurrent_executions = max_concurrent_executions
        self.default_timeout = default_timeout
        self.active_executions: Dict[str, ToolExecutionResult] = {}
        self.execution_history: List[ToolExecutionResult] = []
        self.tool_registry: Dict[str, Callable] = {}
        self.is_initialized = False
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
    
    async def initialize(self):
        """Initialize tool executor"""
        try:
            self.is_initialized = True
            logger.info("MCP Tool Executor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MCP Tool Executor: {e}")
            raise
    
    def register_tool(self, tool_name: str, tool_function: Callable):
        """Register a tool function"""
        self.tool_registry[tool_name] = tool_function
        logger.debug(f"Registered tool: {tool_name}")
    
    def unregister_tool(self, tool_name: str):
        """Unregister a tool function"""
        if tool_name in self.tool_registry:
            del self.tool_registry[tool_name]
            logger.debug(f"Unregistered tool: {tool_name}")
    
    async def execute_tool(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        """Execute a tool with the given request"""
        if not self.is_initialized:
            await self.initialize()
        
        # Create execution result
        result = ToolExecutionResult(
            execution_id=request.execution_id,
            tool_name=request.tool_name,
            server_name=request.server_name,
            status=ExecutionStatus.PENDING,
            metadata=request.metadata.copy()
        )
        
        # Add to active executions
        self.active_executions[request.execution_id] = result
        
        try:
            # Acquire execution semaphore
            async with self._execution_semaphore:
                result.status = ExecutionStatus.RUNNING
                start_time = datetime.utcnow()
                result.started_at = start_time
                
                # Execute tool with timeout
                execution_result = await asyncio.wait_for(
                    self._execute_tool_internal(request),
                    timeout=request.timeout
                )
                
                # Update result
                result.result = execution_result
                result.status = ExecutionStatus.COMPLETED
                result.completed_at = datetime.utcnow()
                result.execution_time_ms = (result.completed_at - start_time).total_seconds() * 1000
                
        except asyncio.TimeoutError:
            result.status = ExecutionStatus.TIMEOUT
            result.error = f"Tool execution timed out after {request.timeout} seconds"
            result.completed_at = datetime.utcnow()
            logger.warning(f"Tool execution timeout: {request.tool_name}")
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.utcnow()
            logger.error(f"Tool execution failed: {request.tool_name} - {e}")
            
        finally:
            # Remove from active executions and add to history
            if request.execution_id in self.active_executions:
                del self.active_executions[request.execution_id]
            
            self.execution_history.append(result)
            
            # Keep history limited
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
        
        return result
    
    async def _execute_tool_internal(self, request: ToolExecutionRequest) -> Any:
        """Internal tool execution logic"""
        tool_name = request.tool_name
        
        # Check if tool is registered
        if tool_name not in self.tool_registry:
            # REAL tool execution via MCP server discovery
            return await self._execute_discovered_tool(request)
        
        # Execute registered tool
        tool_function = self.tool_registry[tool_name]
        
        try:
            # Call tool function
            if asyncio.iscoroutinefunction(tool_function):
                result = await tool_function(request.arguments)
            else:
                result = tool_function(request.arguments)
            
            return result
            
        except Exception as e:
            logger.error(f"Tool function execution failed: {tool_name} - {e}")
            raise
    
    async def _execute_discovered_tool(self, request: ToolExecutionRequest) -> Dict[str, Any]:
        """Execute tool via discovered MCP servers"""
        tool_name = request.tool_name
        arguments = request.arguments
        
        try:
            # REAL tool execution via MCP server discovery
            from mcp.tools.discovery import MCPToolDiscovery

            # Use discovery service to find appropriate server
            discovery = MCPToolDiscovery()

            # Find servers that have this tool
            matching_servers = []
            for server_name, server_info in discovery.discovered_servers.items():
                if server_info.status == "active":
                    for capability in server_info.capabilities:
                        if capability.name == tool_name:
                            matching_servers.append((server_name, server_info))
                            break

            if not matching_servers:
                # No server found with this tool
                return {
                    "success": False,
                    "error": f"No MCP server found with tool '{tool_name}'",
                    "timestamp": datetime.utcnow().isoformat()
                }

            # Try executing on the first available server
            server_name, server_info = matching_servers[0]

            # Import real MCP client
            from mcp.client import MCPClient

            # Connect and execute
            mcp_client = MCPClient(server_info.server_url)
            connection_result = await mcp_client.connect()

            if connection_result.get('success'):
                # Execute real tool
                execution_result = await mcp_client.call_tool(tool_name, arguments)
                await mcp_client.disconnect()

                return {
                    "success": execution_result.get('success', True),
                    "result": execution_result.get('result', execution_result),
                    "server": server_name,
                    "tool": tool_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to connect to MCP server {server_name}",
                    "timestamp": datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Real tool execution failed for {tool_name}: {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
                "tool": tool_name,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution"""
        if execution_id in self.active_executions:
            result = self.active_executions[execution_id]
            result.status = ExecutionStatus.CANCELLED
            result.error = "Execution cancelled by user"
            result.completed_at = datetime.utcnow()
            
            # Move to history
            del self.active_executions[execution_id]
            self.execution_history.append(result)
            
            logger.info(f"Cancelled execution: {execution_id}")
            return True
        
        return False
    
    def get_execution_status(self, execution_id: str) -> Optional[ToolExecutionResult]:
        """Get status of an execution"""
        # Check active executions first
        if execution_id in self.active_executions:
            return self.active_executions[execution_id]
        
        # Check history
        for result in reversed(self.execution_history):
            if result.execution_id == execution_id:
                return result
        
        return None
    
    def get_active_executions(self) -> List[ToolExecutionResult]:
        """Get all active executions"""
        return list(self.active_executions.values())
    
    def get_execution_history(self, limit: int = 100) -> List[ToolExecutionResult]:
        """Get execution history"""
        return self.execution_history[-limit:] if self.execution_history else []
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics"""
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for r in self.execution_history if r.status == ExecutionStatus.COMPLETED)
        failed_executions = sum(1 for r in self.execution_history if r.status == ExecutionStatus.FAILED)
        timeout_executions = sum(1 for r in self.execution_history if r.status == ExecutionStatus.TIMEOUT)
        
        avg_execution_time = 0.0
        if self.execution_history:
            total_time = sum(r.execution_time_ms for r in self.execution_history if r.execution_time_ms > 0)
            avg_execution_time = total_time / len(self.execution_history)
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "timeout_executions": timeout_executions,
            "active_executions": len(self.active_executions),
            "registered_tools": len(self.tool_registry),
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0.0,
            "avg_execution_time_ms": avg_execution_time,
            "is_initialized": self.is_initialized
        }
    
    async def shutdown(self):
        """Shutdown executor and cancel all active executions"""
        # Cancel all active executions
        active_ids = list(self.active_executions.keys())
        for execution_id in active_ids:
            await self.cancel_execution(execution_id)
        
        logger.info("MCP Tool Executor shutdown completed")
