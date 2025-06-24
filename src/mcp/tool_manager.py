"""
MCP Tool Manager - Handles MCP tool execution with hyper-availability

Separate module for MCP tool management with circuit breakers and fallbacks.
Keeps provider registry focused on LLM providers only.
"""

import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class MCPToolManager:
    """
    Manages MCP tool execution with hyper-availability features.
    
    Features:
    - Circuit breakers for failing tools
    - Native Python fallbacks
    - Smart fallback hierarchy (MCP first, then alternatives, then native)
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize MCP tool manager."""
        self.mcp_tools: Dict[str, Dict[str, Any]] = {}
        self.native_fallbacks: Dict[str, Callable] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.mcp_client = None  # Will be injected
    
    def set_mcp_client(self, mcp_client):
        """Set the MCP client for tool execution."""
        self.mcp_client = mcp_client
    
    async def register_tool(self, 
                           tool_name: str, 
                           mcp_config: Dict[str, Any],
                           native_fallback: Optional[Callable] = None) -> bool:
        """
        Register an MCP tool with optional native fallback.
        
        Args:
            tool_name: Name of the tool
            mcp_config: MCP server configuration
            native_fallback: Optional Python function as fallback
            
        Returns:
            Success status
        """
        self.mcp_tools[tool_name] = {
            'config': mcp_config,
            'success_count': 0,
            'failure_count': 0,
            'last_used': datetime.utcnow()
        }
        
        # Initialize circuit breaker
        self.circuit_breakers[tool_name] = {
            'state': 'closed',  # closed, open, half-open
            'failure_threshold': 5,
            'timeout': 60,
            'last_failure': None
        }
        
        # Register native fallback if provided
        if native_fallback:
            self.native_fallbacks[tool_name] = native_fallback
            logger.info(f"Registered MCP tool '{tool_name}' with native fallback")
        else:
            logger.info(f"Registered MCP tool '{tool_name}' (no fallback)")
        
        return True
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool with smart fallback hierarchy.
        
        Hierarchy:
        1. Primary MCP server
        2. Alternative MCP servers (if configured)
        3. Native Python fallback (if available)
        4. Helpful error message
        
        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            
        Returns:
            Execution result with metadata
        """
        # Check if circuit breaker is open
        if self._is_circuit_open(tool_name):
            logger.warning(f"Circuit breaker open for {tool_name}, using fallback")
            return await self._execute_fallback(tool_name, parameters)
        
        # Try primary MCP execution
        try:
            result = await self._execute_mcp_tool(tool_name, parameters)
            self._record_success(tool_name)
            return {
                'success': True,
                'result': result,
                'source': 'mcp_primary',
                'tool_name': tool_name
            }
        except Exception as e:
            logger.warning(f"Primary MCP failed for {tool_name}: {e}")
            self._record_failure(tool_name, str(e))
            return await self._execute_fallback(tool_name, parameters)
    
    async def _execute_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute tool via MCP client."""
        if not self.mcp_client:
            raise Exception("No MCP client configured")
        
        if tool_name not in self.mcp_tools:
            raise Exception(f"Tool {tool_name} not registered")
        
        # This would be the actual MCP call
        return await self.mcp_client.call_tool(tool_name, parameters)
    
    async def _execute_fallback(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fallback strategies in order."""
        
        # Try native fallback if available
        if tool_name in self.native_fallbacks:
            try:
                native_func = self.native_fallbacks[tool_name]
                result = await native_func(parameters)
                return {
                    'success': True,
                    'result': result,
                    'source': 'native_fallback',
                    'warning': 'MCP unavailable - using local fallback',
                    'mcp_preferred': True
                }
            except Exception as e:
                logger.error(f"Native fallback failed for {tool_name}: {e}")
        
        # No fallback available
        available_tools = list(self.native_fallbacks.keys())
        return {
            'success': False,
            'error': f'No MCP server or native fallback for {tool_name}',
            'suggestions': [
                f"Install MCP server that supports '{tool_name}'",
                f"Implement native fallback for '{tool_name}'",
                f"Use alternative tool from: {available_tools[:3]}"
            ],
            'recommendation': 'Install MCP server or implement native fallback'
        }
    
    def _is_circuit_open(self, tool_name: str) -> bool:
        """Check if circuit breaker is open."""
        if tool_name not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[tool_name]
        if breaker['state'] == 'open':
            # Check if timeout has passed
            if breaker['last_failure']:
                elapsed = (datetime.utcnow() - breaker['last_failure']).seconds
                if elapsed > breaker['timeout']:
                    breaker['state'] = 'half-open'
                    return False
            return True
        
        return False
    
    def _record_success(self, tool_name: str):
        """Record successful execution."""
        if tool_name in self.mcp_tools:
            self.mcp_tools[tool_name]['success_count'] += 1
            self.mcp_tools[tool_name]['last_used'] = datetime.utcnow()
        
        # Reset circuit breaker
        if tool_name in self.circuit_breakers:
            self.circuit_breakers[tool_name]['state'] = 'closed'
            self.circuit_breakers[tool_name]['last_failure'] = None
    
    def _record_failure(self, tool_name: str, error: str):
        """Record failed execution and update circuit breaker."""
        if tool_name in self.mcp_tools:
            self.mcp_tools[tool_name]['failure_count'] += 1
        
        # Update circuit breaker
        if tool_name in self.circuit_breakers:
            breaker = self.circuit_breakers[tool_name]
            breaker['last_failure'] = datetime.utcnow()
            
            # Check if we should open the circuit
            total_failures = self.mcp_tools.get(tool_name, {}).get('failure_count', 0)
            if total_failures >= breaker['failure_threshold']:
                breaker['state'] = 'open'
                logger.warning(f"Circuit breaker opened for tool: {tool_name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all tools."""
        return {
            'registered_tools': len(self.mcp_tools),
            'native_fallbacks': len(self.native_fallbacks),
            'circuit_breakers': {
                name: breaker['state'] 
                for name, breaker in self.circuit_breakers.items()
            },
            'tool_stats': self.mcp_tools,
            'mcp_client_ready': self.mcp_client is not None,
            'timestamp': datetime.utcnow().isoformat()
        }


# Global instance
_mcp_tool_manager = None

def get_mcp_tool_manager() -> MCPToolManager:
    """Get global MCP tool manager instance."""
    global _mcp_tool_manager
    if _mcp_tool_manager is None:
        _mcp_tool_manager = MCPToolManager()
    return _mcp_tool_manager
