"""
PyGent A2A Client SDK

Official Python client for PyGent Factory A2A Multi-Agent System.
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

__version__ = "1.0.0"
__author__ = "PyGent Factory Team"
__email__ = "support@timpayne.net"

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """A2A task states"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


@dataclass
class A2AConfig:
    """A2A client configuration"""
    base_url: str = "http://localhost:8080"
    api_key: Optional[str] = None
    jwt_token: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    user_agent: str = f"PyGent-A2A-Client/{__version__}"
    
    # Connection pooling
    max_connections: int = 100
    max_connections_per_host: int = 20
    keepalive_timeout: int = 30
    
    # Request configuration
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    enable_compression: bool = True
    
    # Logging
    log_requests: bool = False
    log_responses: bool = False


@dataclass
class TaskResult:
    """A2A task result"""
    task_id: str
    session_id: Optional[str]
    state: TaskState
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    @classmethod
    def from_response(cls, response_data: Dict[str, Any]) -> 'TaskResult':
        """Create TaskResult from API response"""
        result = response_data.get("result", {})
        
        # Parse timestamp
        timestamp = None
        if "status" in result and "timestamp" in result["status"]:
            try:
                timestamp = datetime.fromisoformat(
                    result["status"]["timestamp"].replace('Z', '+00:00')
                )
            except (ValueError, AttributeError):
                pass
        
        return cls(
            task_id=result.get("id", ""),
            session_id=result.get("sessionId"),
            state=TaskState(result.get("status", {}).get("state", "submitted")),
            artifacts=result.get("artifacts", []),
            history=result.get("history", []),
            metadata=result.get("metadata", {}),
            timestamp=timestamp
        )


class A2AClientError(Exception):
    """Base A2A client error"""
    pass


class A2AConnectionError(A2AClientError):
    """A2A connection error"""
    pass


class A2ATimeoutError(A2AClientError):
    """A2A timeout error"""
    pass


class A2AAuthenticationError(A2AClientError):
    """A2A authentication error"""
    pass


class A2ATaskError(A2AClientError):
    """A2A task error"""
    pass


class A2AClient:
    """
    PyGent A2A Client
    
    Official Python client for interacting with PyGent Factory A2A Multi-Agent System.
    
    Example:
        ```python
        import asyncio
        from pygent_a2a_client import A2AClient, A2AConfig
        
        async def main():
            config = A2AConfig(
                base_url="https://api.timpayne.net/a2a",
                api_key="your-api-key"
            )
            
            async with A2AClient(config) as client:
                # Send a document search task
                task = await client.send_task({
                    "role": "user",
                    "parts": [{"type": "text", "text": "Search for AI research papers"}]
                })
                
                # Wait for completion
                result = await client.wait_for_completion(task.task_id)
                print(f"Task completed: {result.artifacts}")
        
        asyncio.run(main())
        ```
    """
    
    def __init__(self, config: A2AConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_id_counter = 0
        
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def connect(self):
        """Establish connection to A2A server"""
        if self.session:
            return
        
        # Configure connector
        connector = aiohttp.TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=self.config.max_connections_per_host,
            keepalive_timeout=self.config.keepalive_timeout,
            enable_cleanup_closed=True
        )
        
        # Configure timeout
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        # Configure headers
        headers = {
            "User-Agent": self.config.user_agent,
            "Content-Type": "application/json"
        }
        
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        
        if self.config.jwt_token:
            headers["Authorization"] = f"Bearer {self.config.jwt_token}"
        
        # Create session
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        
        # Test connection
        try:
            await self.health_check()
            logger.info(f"Connected to A2A server: {self.config.base_url}")
        except Exception as e:
            await self.close()
            raise A2AConnectionError(f"Failed to connect to A2A server: {e}")
    
    async def close(self):
        """Close connection to A2A server"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check A2A server health"""
        if not self.session:
            raise A2AConnectionError("Not connected to A2A server")
        
        try:
            async with self.session.get(f"{self.config.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise A2AConnectionError(f"Health check failed: HTTP {response.status}")
        except aiohttp.ClientError as e:
            raise A2AConnectionError(f"Health check error: {e}")
    
    async def discover_agents(self) -> Dict[str, Any]:
        """Discover available agents and capabilities"""
        if not self.session:
            raise A2AConnectionError("Not connected to A2A server")
        
        try:
            async with self.session.get(f"{self.config.base_url}/.well-known/agent.json") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise A2AConnectionError(f"Agent discovery failed: HTTP {response.status}")
        except aiohttp.ClientError as e:
            raise A2AConnectionError(f"Agent discovery error: {e}")
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        if not self.session:
            raise A2AConnectionError("Not connected to A2A server")
        
        try:
            async with self.session.get(f"{self.config.base_url}/agents") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise A2AConnectionError(f"List agents failed: HTTP {response.status}")
        except aiohttp.ClientError as e:
            raise A2AConnectionError(f"List agents error: {e}")
    
    async def send_task(self, message: Dict[str, Any], session_id: Optional[str] = None) -> TaskResult:
        """
        Send a task to the A2A system
        
        Args:
            message: Task message with role and parts
            session_id: Optional session ID for task grouping
            
        Returns:
            TaskResult with task ID and initial status
            
        Example:
            ```python
            task = await client.send_task({
                "role": "user",
                "parts": [{"type": "text", "text": "Search for documents about quantum computing"}]
            })
            ```
        """
        params = {"message": message}
        if session_id:
            params["sessionId"] = session_id
        
        response = await self._send_jsonrpc_request("tasks/send", params)
        return TaskResult.from_response(response)
    
    async def get_task(self, task_id: str) -> TaskResult:
        """
        Get task status and results
        
        Args:
            task_id: Task ID to retrieve
            
        Returns:
            TaskResult with current status and artifacts
        """
        response = await self._send_jsonrpc_request("tasks/get", {"id": task_id})
        return TaskResult.from_response(response)
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if cancellation was successful
        """
        try:
            await self._send_jsonrpc_request("tasks/cancel", {"id": task_id})
            return True
        except A2ATaskError:
            return False
    
    async def wait_for_completion(
        self, 
        task_id: str, 
        timeout: Optional[int] = None,
        poll_interval: float = 1.0,
        callback: Optional[Callable[[TaskResult], None]] = None
    ) -> TaskResult:
        """
        Wait for task completion with optional progress callback
        
        Args:
            task_id: Task ID to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds
            callback: Optional callback for progress updates
            
        Returns:
            TaskResult when task is completed
            
        Raises:
            A2ATimeoutError: If timeout is reached
            A2ATaskError: If task fails
        """
        start_time = time.time()
        timeout = timeout or self.config.timeout
        
        while True:
            result = await self.get_task(task_id)
            
            # Call progress callback if provided
            if callback:
                try:
                    callback(result)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
            
            # Check completion states
            if result.state == TaskState.COMPLETED:
                return result
            elif result.state == TaskState.FAILED:
                raise A2ATaskError(f"Task {task_id} failed")
            elif result.state == TaskState.CANCELED:
                raise A2ATaskError(f"Task {task_id} was canceled")
            
            # Check timeout
            if time.time() - start_time > timeout:
                raise A2ATimeoutError(f"Task {task_id} did not complete within {timeout}s")
            
            # Wait before next poll
            await asyncio.sleep(poll_interval)
    
    async def search_documents(self, query: str, **kwargs) -> TaskResult:
        """
        Convenience method for document search
        
        Args:
            query: Search query
            **kwargs: Additional parameters for wait_for_completion
            
        Returns:
            TaskResult with search results
        """
        task = await self.send_task({
            "role": "user",
            "parts": [{"type": "text", "text": f"Search for documents about {query}"}]
        })
        
        return await self.wait_for_completion(task.task_id, **kwargs)
    
    async def analyze_data(self, data_description: str, **kwargs) -> TaskResult:
        """
        Convenience method for data analysis
        
        Args:
            data_description: Description of data to analyze
            **kwargs: Additional parameters for wait_for_completion
            
        Returns:
            TaskResult with analysis results
        """
        task = await self.send_task({
            "role": "user",
            "parts": [{"type": "text", "text": f"Analyze: {data_description}"}]
        })
        
        return await self.wait_for_completion(task.task_id, **kwargs)
    
    async def _send_jsonrpc_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC request to A2A server"""
        if not self.session:
            raise A2AConnectionError("Not connected to A2A server")
        
        # Generate request ID
        self._request_id_counter += 1
        request_id = self._request_id_counter
        
        # Build request
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id
        }
        
        # Log request if enabled
        if self.config.log_requests:
            logger.debug(f"A2A Request: {json.dumps(request, indent=2)}")
        
        # Send request with retries
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(self.config.base_url, json=request) as response:
                    response_data = await response.json()
                    
                    # Log response if enabled
                    if self.config.log_responses:
                        logger.debug(f"A2A Response: {json.dumps(response_data, indent=2)}")
                    
                    # Check for JSON-RPC error
                    if "error" in response_data:
                        error = response_data["error"]
                        error_msg = f"A2A Error {error.get('code', 'unknown')}: {error.get('message', 'Unknown error')}"
                        
                        # Determine error type
                        if error.get("code") == -32602:  # Invalid params
                            raise A2ATaskError(error_msg)
                        elif response.status == 401:
                            raise A2AAuthenticationError(error_msg)
                        else:
                            raise A2AClientError(error_msg)
                    
                    # Check HTTP status
                    if response.status != 200:
                        raise A2AConnectionError(f"HTTP {response.status}: {response.reason}")
                    
                    return response_data
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    break
        
        # All retries failed
        raise A2AConnectionError(f"Request failed after {self.config.max_retries + 1} attempts: {last_error}")


# Convenience functions
async def quick_search(query: str, base_url: str = "http://localhost:8080", **kwargs) -> TaskResult:
    """
    Quick document search without managing client lifecycle
    
    Args:
        query: Search query
        base_url: A2A server URL
        **kwargs: Additional configuration parameters
        
    Returns:
        TaskResult with search results
    """
    config = A2AConfig(base_url=base_url, **kwargs)
    async with A2AClient(config) as client:
        return await client.search_documents(query)


async def quick_analysis(data_description: str, base_url: str = "http://localhost:8080", **kwargs) -> TaskResult:
    """
    Quick data analysis without managing client lifecycle
    
    Args:
        data_description: Description of data to analyze
        base_url: A2A server URL
        **kwargs: Additional configuration parameters
        
    Returns:
        TaskResult with analysis results
    """
    config = A2AConfig(base_url=base_url, **kwargs)
    async with A2AClient(config) as client:
        return await client.analyze_data(data_description)


# Export public API
__all__ = [
    "A2AClient",
    "A2AConfig", 
    "TaskResult",
    "TaskState",
    "A2AClientError",
    "A2AConnectionError",
    "A2ATimeoutError",
    "A2AAuthenticationError",
    "A2ATaskError",
    "quick_search",
    "quick_analysis"
]
