"""
MCP Server Lifecycle Management

This module provides lifecycle management for MCP servers including
starting, stopping, monitoring, and restarting servers.
"""

import asyncio
import logging
import subprocess
import signal
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from .config import MCPServerConfig, MCPServerStatus, MCPTransportType
from .registry import MCPServerRegistration

# Import MCP client for remote connections
try:
    # Only import what we need dynamically to avoid unused import warnings
    import importlib.util
    spec = importlib.util.find_spec("mcp.client.sse")
    MCP_CLIENT_AVAILABLE = spec is not None
except ImportError:
    MCP_CLIENT_AVAILABLE = False


logger = logging.getLogger(__name__)


class MCPServerProcess:
    """Represents a running MCP server process or connection"""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.client_session: Optional[Any] = None
        self.started_at: Optional[datetime] = None
        self.stdout_task: Optional[asyncio.Task] = None
        self.stderr_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._remote_connection = None
    
    async def start(self) -> bool:
        """Start the MCP server process or connection"""
        try:
            if self.process and not self.process.returncode:
                logger.warning(f"Server {self.config.name} is already running")
                return True
            
            if self.client_session:
                logger.warning(f"Server {self.config.name} already has an active connection")
                return True
            
            # Handle different transport types
            if self.config.transport == MCPTransportType.SSE:
                return await self._start_sse_connection()
            elif self.config.transport == MCPTransportType.HTTP:
                return await self._start_http_connection()  
            elif self.config.transport == MCPTransportType.WEBSOCKET:
                return await self._start_websocket_connection()
            elif self.config.transport == MCPTransportType.STDIO:
                return await self._start_stdio_process()
            else:
                logger.error(f"Unsupported transport type: {self.config.transport}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start MCP server {self.config.name}: {str(e)}")
            return False
    
    async def _start_sse_connection(self) -> bool:
        """Start SSE connection to remote server"""
        if not MCP_CLIENT_AVAILABLE:
            logger.error("MCP client not available for SSE connections")
            return False
        
        try:
            # Build URL from command (which contains the full URL for remote servers)
            if len(self.config.command) >= 3 and self.config.command[0] == "npx":
                # Extract URL from npx mcp-remote <url> command
                url = self.config.command[2] if len(self.config.command) > 2 else None
            else:
                # Build URL from config
                protocol = "https" if self.config.port == 443 else "http"
                url = f"{protocol}://{self.config.host}"
                if self.config.port and self.config.port not in [80, 443]:
                    url += f":{self.config.port}"
                if self.config.path:
                    url += self.config.path
            
            if not url:
                logger.error(f"No URL specified for SSE connection: {self.config.name}")
                return False
            
            logger.info(f"Connecting to SSE server: {url}")
            
            # Check if authentication is required
            auth_headers = {}
            auth_config = self.config.config.get("authentication", {})
            
            if auth_config.get("required", False) and auth_config.get("type") == "api_token":
                # Try to get authentication token (OAuth or API token)
                auth_token = await self._get_auth_token("cloudflare")
                if auth_token:
                    auth_headers["Authorization"] = f"Bearer {auth_token}"
                    logger.info(f"Using authentication token for {self.config.name}")
                else:
                    logger.warning(f"Authentication required for {self.config.name} but no token found")
                    # Don't fail here - let the server decide if it requires auth
            
            # Test connection with a simple HTTP request first
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=10)
            
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, headers=auth_headers) as response:
                        if response.status in [200, 401]:  # 401 is expected for auth-required servers without token
                            logger.info(f"SSE endpoint accessible: {url} (status: {response.status})")
                            
                            # Store connection info (we'll create actual SSE connection when needed)
                            self._remote_connection = {
                                "url": url,
                                "type": "sse",
                                "status": "accessible",
                                "auth_headers": auth_headers,
                                "last_check": datetime.utcnow()
                            }
                            
                            self.started_at = datetime.utcnow()
                            logger.info(f"SSE connection prepared for: {self.config.name}")
                            return True
                        else:
                            logger.warning(f"SSE endpoint returned status {response.status}: {url}")
                            return False
                            
            except asyncio.TimeoutError:
                logger.warning(f"SSE endpoint timeout (expected for SSE streams): {url}")
                # Timeout is expected for SSE streams, consider it a success
                self._remote_connection = {
                    "url": url,
                    "type": "sse",
                    "status": "timeout_expected",
                    "auth_headers": auth_headers,
                    "last_check": datetime.utcnow()
                }
                self.started_at = datetime.utcnow()
                return True
                
        except Exception as e:
            logger.error(f"Failed to establish SSE connection for {self.config.name}: {str(e)}")
            return False
    
    async def _start_http_connection(self) -> bool:
        """Start HTTP connection to remote server"""
        logger.info(f"HTTP transport not yet implemented for {self.config.name}")
        return False
    
    async def _start_websocket_connection(self) -> bool:
        """Start WebSocket connection to remote server"""
        logger.info(f"WebSocket transport not yet implemented for {self.config.name}")
        return False
    
    async def _start_stdio_process(self) -> bool:
        """Start local STDIO process"""
        # Prepare environment
        env = os.environ.copy()
        env.update(self.config.environment_variables)
        
        # Start STDIO process
        self.process = await asyncio.create_subprocess_exec(
            *self.config.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self.config.working_directory
        )
        
        self.started_at = datetime.utcnow()
        
        # Start output monitoring if enabled
        if self.config.enable_logging and self.process.stdout:
            self.stdout_task = asyncio.create_task(self._monitor_stdout())
        if self.config.enable_logging and self.process.stderr:
            self.stderr_task = asyncio.create_task(self._monitor_stderr())
        
        logger.info(f"Started MCP server: {self.config.name} (PID: {self.process.pid})")
        return True
    
    async def stop(self, timeout: int = 10) -> bool:
        """Stop the MCP server process or connection"""
        try:
            # Set stop event
            self._stop_event.set()
            
            # Cancel monitoring tasks
            if self.stdout_task:
                self.stdout_task.cancel()
            if self.stderr_task:
                self.stderr_task.cancel()
            
            # Handle remote connections
            if self.client_session:
                try:
                    # Close client session if exists
                    await self.client_session.close()
                    self.client_session = None
                except Exception as e:
                    logger.warning(f"Error closing client session for {self.config.name}: {e}")
            
            if self._remote_connection:
                try:
                    # Close remote connection if exists
                    self._remote_connection = None
                except Exception as e:
                    logger.warning(f"Error closing remote connection for {self.config.name}: {e}")
            
            # Handle local process
            if self.process:
                # Try graceful shutdown first
                if self.process.returncode is None:
                    self.process.terminate()
                    
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        # Force kill if graceful shutdown failed
                        logger.warning(f"Force killing MCP server {self.config.name}")
                        self.process.kill()
                        await self.process.wait()
            
            logger.info(f"Stopped MCP server: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop MCP server {self.config.name}: {str(e)}")
            return False
    
    async def restart(self) -> bool:
        """Restart the MCP server process"""
        logger.info(f"Restarting MCP server: {self.config.name}")
        
        # Stop the process
        await self.stop()
        
        # Wait for restart delay
        if self.config.restart_delay > 0:
            await asyncio.sleep(self.config.restart_delay)
        
        # Start the process
        return await self.start()
    
    def is_running(self) -> bool:
        """Check if the server process or connection is running"""
        # For remote connections, check if we have an active connection
        if self.config.transport in [MCPTransportType.SSE, MCPTransportType.HTTP, MCPTransportType.WEBSOCKET]:
            return (self._remote_connection is not None and 
                   self._remote_connection.get("status") in ["accessible", "timeout_expected"])
        
        # For local processes, check if process is running
        return self.process is not None and self.process.returncode is None
    
    def get_pid(self) -> Optional[int]:
        """Get process ID (only applicable for local processes)"""
        if self.process:
            return self.process.pid
        return None
    
    def get_uptime(self) -> Optional[timedelta]:
        """Get process uptime"""
        if self.started_at:
            return datetime.utcnow() - self.started_at
        return None
    
    async def send_input(self, data: bytes) -> bool:
        """Send input to the process (for STDIO transport)"""
        try:
            if self.process and self.process.stdin:
                self.process.stdin.write(data)
                await self.process.stdin.drain()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to send input to {self.config.name}: {str(e)}")
            return False
    
    async def read_output(self, timeout: int = 1) -> Optional[bytes]:
        """Read output from the process (for STDIO transport)"""
        try:
            if self.process and self.process.stdout:
                return await asyncio.wait_for(
                    self.process.stdout.read(1024), 
                    timeout=timeout
                )
            return None
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Failed to read output from {self.config.name}: {str(e)}")
            return None
    
    async def _monitor_stdout(self) -> None:
        """Monitor stdout output"""
        try:
            while not self._stop_event.is_set() and self.process and self.process.stdout:
                line = await self.process.stdout.readline()
                if not line:
                    break
                
                log_message = line.decode('utf-8', errors='ignore').strip()
                if log_message:
                    logger.info(f"[{self.config.name}] {log_message}")
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error monitoring stdout for {self.config.name}: {str(e)}")
    
    async def _monitor_stderr(self) -> None:
        """Monitor stderr output"""
        try:
            while not self._stop_event.is_set() and self.process and self.process.stderr:
                line = await self.process.stderr.readline()
                if not line:
                    break
                
                log_message = line.decode('utf-8', errors='ignore').strip()
                if log_message:
                    logger.error(f"[{self.config.name}] {log_message}")
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error monitoring stderr for {self.config.name}: {str(e)}")
    
    def _get_cloudflare_api_token(self) -> Optional[str]:
        """Get Cloudflare API token from environment variables"""
        # Try multiple environment variable names
        for env_var in ["CLOUDFLARE_API_TOKEN", "CF_API_TOKEN", "CLOUDFLARE_TOKEN"]:
            token = os.environ.get(env_var)
            if token:
                return token
        
        # Try to load from cloudflare_auth.env file
        auth_file_paths = [
            "cloudflare_auth.env",
            ".cloudflare_auth.env",
            os.path.join(os.getcwd(), "cloudflare_auth.env"),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "cloudflare_auth.env")
        ]
        
        for auth_file in auth_file_paths:
            if os.path.exists(auth_file):
                try:
                    with open(auth_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("CLOUDFLARE_API_TOKEN="):
                                token = line.split("=", 1)[1].strip()
                                if token and not token.startswith("#"):
                                    return token
                except Exception as e:
                    logger.warning(f"Error reading auth file {auth_file}: {e}")
        
        return None
    
    async def _get_oauth_token(self, provider: str, user_id: str = "system") -> Optional[str]:
        """Get OAuth token for a provider"""
        try:
            from src.auth import OAuthManager, FileTokenStorage
            
            # Initialize OAuth manager if not already done
            oauth_manager = OAuthManager()
            oauth_manager.set_token_storage(FileTokenStorage())
            
            token = await oauth_manager.get_token(user_id, provider)
            if token and not token.is_expired:
                return token.access_token
            elif token and token.expires_soon and token.refresh_token:
                # Try to refresh the token
                provider_obj = oauth_manager.get_provider(provider)
                if provider_obj:
                    try:
                        new_token = await provider_obj.refresh_token(token.refresh_token)
                        new_token.user_id = user_id
                        await oauth_manager.token_storage.store_token(user_id, provider, new_token)
                        return new_token.access_token
                    except Exception as e:
                        logger.warning(f"Failed to refresh {provider} token: {e}")
            
            return None
        except Exception as e:
            logger.warning(f"Error getting OAuth token for {provider}: {e}")
            return None

    async def _get_auth_token(self, provider: str, user_id: str = "system") -> Optional[str]:
        """Get authentication token for a provider (tries OAuth first, then API token)"""
        # Try OAuth token first
        oauth_token = await self._get_oauth_token(provider, user_id)
        if oauth_token:
            logger.info(f"Using OAuth token for {provider}")
            return oauth_token
        
        # Fall back to API token for Cloudflare
        if provider == "cloudflare":
            api_token = self._get_cloudflare_api_token()
            if api_token:
                logger.info(f"Using API token for {provider}")
                return api_token
        
        return None


class MCPServerLifecycle:
    """
    Manages the lifecycle of MCP servers including starting, stopping,
    monitoring, and automatic restart functionality.
    """
    
    def __init__(self):
        self.processes: Dict[str, MCPServerProcess] = {}
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start_monitoring(self) -> None:
        """Start the lifecycle monitoring"""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_processes())
        logger.info("MCP server lifecycle monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop the lifecycle monitoring"""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop all processes
        for process in self.processes.values():
            await process.stop()
        
        self.processes.clear()
        logger.info("MCP server lifecycle monitoring stopped")
    
    async def start_server(self, registration: MCPServerRegistration) -> bool:
        """
        Start an MCP server.
        
        Args:
            registration: Server registration
            
        Returns:
            bool: True if server started successfully
        """
        try:
            server_id = registration.config.id
            
            # Check if already running
            if server_id in self.processes:
                process = self.processes[server_id]
                if process.is_running():
                    logger.warning(f"Server {registration.config.name} is already running")
                    return True
                else:
                    # Clean up dead process
                    del self.processes[server_id]
            
            # Create and start new process
            process = MCPServerProcess(registration.config)
            success = await process.start()
            
            if success:
                self.processes[server_id] = process
                registration.status = MCPServerStatus.RUNNING
                registration.start_count += 1
                registration.clear_error()
                logger.info(f"Successfully started MCP server: {registration.config.name}")
            else:
                registration.set_error("Failed to start server process")
            
            return success
            
        except Exception as e:
            error_msg = f"Failed to start server: {str(e)}"
            logger.error(error_msg)
            registration.set_error(error_msg)
            return False
    
    async def stop_server(self, registration: MCPServerRegistration) -> bool:
        """
        Stop an MCP server.
        
        Args:
            registration: Server registration
            
        Returns:
            bool: True if server stopped successfully
        """
        try:
            server_id = registration.config.id
            
            if server_id not in self.processes:
                logger.warning(f"Server {registration.config.name} is not running")
                registration.status = MCPServerStatus.STOPPED
                return True
            
            process = self.processes[server_id]
            success = await process.stop(timeout=registration.config.timeout)
            
            if success:
                del self.processes[server_id]
                registration.status = MCPServerStatus.STOPPED
                registration.clear_error()
                logger.info(f"Successfully stopped MCP server: {registration.config.name}")
            else:
                registration.set_error("Failed to stop server process")
            
            return success
            
        except Exception as e:
            error_msg = f"Failed to stop server: {str(e)}"
            logger.error(error_msg)
            registration.set_error(error_msg)
            return False
    
    async def restart_server(self, registration: MCPServerRegistration) -> bool:
        """
        Restart an MCP server.
        
        Args:
            registration: Server registration
            
        Returns:
            bool: True if server restarted successfully
        """
        try:
            server_id = registration.config.id
            
            if server_id in self.processes:
                process = self.processes[server_id]
                success = await process.restart()
                
                if success:
                    registration.status = MCPServerStatus.RUNNING
                    registration.increment_restart_count()
                    registration.clear_error()
                    logger.info(f"Successfully restarted MCP server: {registration.config.name}")
                else:
                    registration.set_error("Failed to restart server process")
                
                return success
            else:
                # Server not running, just start it
                return await self.start_server(registration)
                
        except Exception as e:
            error_msg = f"Failed to restart server: {str(e)}"
            logger.error(error_msg)
            registration.set_error(error_msg)
            return False
    
    def get_server_process(self, server_id: str) -> Optional[MCPServerProcess]:
        """Get server process by ID"""
        return self.processes.get(server_id)
    
    def is_server_running(self, server_id: str) -> bool:
        """Check if server is running"""
        process = self.processes.get(server_id)
        return process is not None and process.is_running()
    
    def get_running_servers(self) -> List[str]:
        """Get list of running server IDs"""
        return [
            server_id for server_id, process in self.processes.items()
            if process.is_running()
        ]
    
    async def send_to_server(self, server_id: str, data: bytes) -> bool:
        """Send data to a server (for STDIO transport)"""
        process = self.processes.get(server_id)
        if process:
            return await process.send_input(data)
        return False
    
    async def read_from_server(self, server_id: str, timeout: int = 1) -> Optional[bytes]:
        """Read data from a server (for STDIO transport)"""
        process = self.processes.get(server_id)
        if process:
            return await process.read_output(timeout)
        return None
    
    async def _monitor_processes(self) -> None:
        """Monitor running processes and handle failures"""
        while self._running:
            try:
                # Check each process
                dead_processes = []
                
                for server_id, process in self.processes.items():
                    if not process.is_running():
                        dead_processes.append(server_id)
                
                # Handle dead processes
                for server_id in dead_processes:
                    process = self.processes[server_id]
                    logger.warning(f"Detected dead MCP server process: {process.config.name}")
                    
                    # Remove from active processes
                    del self.processes[server_id]
                    
                    # Note: Restart logic should be handled by the server manager
                    # based on the registration configuration
                
                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in MCP server process monitor: {str(e)}")
                await asyncio.sleep(30)  # Wait 30 seconds before retry
    
    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get lifecycle statistics"""
        running_count = len([p for p in self.processes.values() if p.is_running()])
        
        return {
            "total_processes": len(self.processes),
            "running_processes": running_count,
            "monitoring": self._running,
            "process_details": {
                server_id: {
                    "running": process.is_running(),
                    "pid": process.get_pid(),
                    "uptime": str(process.get_uptime()) if process.get_uptime() else None
                }
                for server_id, process in self.processes.items()
            }
        }
