#!/usr/bin/env python3
"""
MCP Server Manager

Comprehensive management system for all MCP servers with:
- Ordered startup sequences based on dependencies
- Health monitoring and stability checks
- Automatic restart and failover mechanisms
- Resource monitoring and port conflict detection
- Comprehensive logging and error tracking
"""

import asyncio
import json
import logging
import psutil
import time
import subprocess
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/mcp_server_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    """Server status enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    RESTARTING = "restarting"
    UNHEALTHY = "unhealthy"


@dataclass
class ServerHealth:
    """Server health information"""
    status: ServerStatus
    last_check: datetime
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    uptime: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None


@dataclass
class ServerProcess:
    """Server process information"""
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    restart_count: int = 0
    health: ServerHealth = field(default_factory=lambda: ServerHealth(
        status=ServerStatus.STOPPED,
        last_check=datetime.utcnow()
    ))


class MCPServerManager:
    """Comprehensive MCP server management system"""
    
    def __init__(self, config_file: str = "mcp_server_configs.json"):
        self.config_file = Path(config_file)
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.processes: Dict[str, ServerProcess] = {}
        self.startup_order: List[str] = []
        self.is_running = False
        self.health_check_interval = 30  # seconds
        self.max_startup_time = 120  # seconds
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Load server configurations
        self._load_server_configs()
    
    def _load_server_configs(self):
        """Load server configurations from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            self.servers = {server['id']: server for server in config_data['servers']}
            
            # Sort servers by startup order
            servers_with_order = [
                (server_id, server.get('config', {}).get('startup_order', 999))
                for server_id, server in self.servers.items()
                if server.get('auto_start', True)
            ]
            
            self.startup_order = [
                server_id for server_id, _ in sorted(servers_with_order, key=lambda x: x[1])
            ]
            
            logger.info(f"Loaded {len(self.servers)} server configurations")
            logger.info(f"Startup order: {self.startup_order}")
            
        except Exception as e:
            logger.error(f"Failed to load server configurations: {e}")
            raise
    
    def _check_port_availability(self, port: int) -> bool:
        """Check if a port is available"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result != 0  # Port is available if connection fails
        except Exception:
            return False
    
    def _get_process_info(self, pid: int) -> Tuple[Optional[float], Optional[float]]:
        """Get process memory and CPU usage"""
        try:
            process = psutil.Process(pid)
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            return memory_mb, cpu_percent
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None, None
    
    async def _start_server(self, server_id: str) -> bool:
        """Start a single server"""
        try:
            server_config = self.servers[server_id]
            logger.info(f"Starting server: {server_id}")
            
            # Check if server is already running
            if server_id in self.processes and self.processes[server_id].process:
                if self.processes[server_id].process.poll() is None:
                    logger.warning(f"Server {server_id} is already running")
                    return True
            
            # Check port availability for HTTP servers
            if server_config.get('transport') == 'http':
                port = server_config.get('port')
                if port and not self._check_port_availability(port):
                    logger.error(f"Port {port} is already in use for server {server_id}")
                    return False
            
            # Prepare command
            command = server_config['command']
            env = None
            if 'environment' in server_config:
                import os
                env = os.environ.copy()
                env.update(server_config['environment'])
            
            # Start process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=Path.cwd()
            )
            
            # Store process information
            server_process = ServerProcess(
                process=process,
                pid=process.pid,
                start_time=datetime.utcnow(),
                restart_count=self.processes.get(server_id, ServerProcess()).restart_count
            )
            server_process.health.status = ServerStatus.STARTING
            self.processes[server_id] = server_process
            
            # Wait for server to start
            timeout = server_config.get('timeout', 30)
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Check if process is still running
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    logger.error(f"Server {server_id} failed to start: {stderr.decode()}")
                    server_process.health.status = ServerStatus.FAILED
                    server_process.health.error_message = stderr.decode()
                    return False
                
                # Check health for HTTP servers
                if server_config.get('transport') == 'http':
                    if await self._check_server_health(server_id):
                        server_process.health.status = ServerStatus.RUNNING
                        logger.info(f"Server {server_id} started successfully")
                        return True
                
                await asyncio.sleep(1)
            
            # For stdio servers, assume they're running if process is alive
            if server_config.get('transport') == 'stdio' and process.poll() is None:
                server_process.health.status = ServerStatus.RUNNING
                logger.info(f"Server {server_id} started successfully")
                return True
            
            logger.error(f"Server {server_id} failed to start within timeout")
            server_process.health.status = ServerStatus.FAILED
            return False
            
        except Exception as e:
            logger.error(f"Failed to start server {server_id}: {e}")
            if server_id in self.processes:
                self.processes[server_id].health.status = ServerStatus.FAILED
                self.processes[server_id].health.error_message = str(e)
            return False
    
    async def _check_server_health(self, server_id: str) -> bool:
        """Check health of a specific server"""
        try:
            server_config = self.servers[server_id]
            server_process = self.processes.get(server_id)
            
            if not server_process or not server_process.process:
                return False
            
            # Check if process is still running
            if server_process.process.poll() is not None:
                server_process.health.status = ServerStatus.FAILED
                return False
            
            # Update process info
            if server_process.pid:
                memory, cpu = self._get_process_info(server_process.pid)
                server_process.health.memory_usage = memory
                server_process.health.cpu_usage = cpu
                
                if server_process.start_time:
                    server_process.health.uptime = (
                        datetime.utcnow() - server_process.start_time
                    ).total_seconds()
            
            # For HTTP servers, check health endpoint
            if server_config.get('transport') == 'http':
                host = server_config.get('host', 'localhost')
                port = server_config.get('port')
                
                if port:
                    health_url = f"http://{host}:{port}/health"
                    
                    start_time = time.time()
                    try:
                        response = requests.get(health_url, timeout=5)
                        response_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            server_process.health.status = ServerStatus.RUNNING
                            server_process.health.response_time = response_time
                            server_process.health.consecutive_failures = 0
                            server_process.health.last_check = datetime.utcnow()
                            return True
                        else:
                            server_process.health.consecutive_failures += 1
                            server_process.health.error_message = f"HTTP {response.status_code}"
                            
                    except requests.RequestException as e:
                        server_process.health.consecutive_failures += 1
                        server_process.health.error_message = str(e)
            
            # For stdio servers, assume healthy if process is running
            elif server_config.get('transport') == 'stdio':
                server_process.health.status = ServerStatus.RUNNING
                server_process.health.consecutive_failures = 0
                server_process.health.last_check = datetime.utcnow()
                return True
            
            # Mark as unhealthy if too many consecutive failures
            if server_process.health.consecutive_failures >= 3:
                server_process.health.status = ServerStatus.UNHEALTHY
                return False
            
            server_process.health.last_check = datetime.utcnow()
            return server_process.health.status == ServerStatus.RUNNING
            
        except Exception as e:
            logger.error(f"Health check failed for server {server_id}: {e}")
            if server_id in self.processes:
                self.processes[server_id].health.consecutive_failures += 1
                self.processes[server_id].health.error_message = str(e)
            return False
    
    async def _restart_server(self, server_id: str) -> bool:
        """Restart a failed server"""
        try:
            server_config = self.servers[server_id]
            server_process = self.processes.get(server_id)
            
            if not server_process:
                return await self._start_server(server_id)
            
            # Check restart limits
            max_restarts = server_config.get('max_restarts', 3)
            if server_process.restart_count >= max_restarts:
                logger.error(f"Server {server_id} exceeded max restart attempts ({max_restarts})")
                server_process.health.status = ServerStatus.FAILED
                return False
            
            logger.info(f"Restarting server {server_id} (attempt {server_process.restart_count + 1})")
            
            # Stop existing process
            await self._stop_server(server_id)
            
            # Wait a bit before restarting
            await asyncio.sleep(2)
            
            # Increment restart count
            server_process.restart_count += 1
            server_process.health.status = ServerStatus.RESTARTING
            
            # Start server
            return await self._start_server(server_id)
            
        except Exception as e:
            logger.error(f"Failed to restart server {server_id}: {e}")
            return False
    
    async def _stop_server(self, server_id: str) -> bool:
        """Stop a server"""
        try:
            server_process = self.processes.get(server_id)
            
            if not server_process or not server_process.process:
                return True
            
            logger.info(f"Stopping server: {server_id}")
            
            # Terminate process gracefully
            server_process.process.terminate()
            
            # Wait for graceful shutdown
            try:
                server_process.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                server_process.process.kill()
                server_process.process.wait()
            
            server_process.health.status = ServerStatus.STOPPED
            server_process.process = None
            server_process.pid = None
            
            logger.info(f"Server {server_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop server {server_id}: {e}")
            return False
    
    async def start_all_servers(self) -> Dict[str, bool]:
        """Start all servers in dependency order"""
        logger.info("Starting all MCP servers...")
        results = {}
        
        for server_id in self.startup_order:
            logger.info(f"Starting server {server_id}...")
            success = await self._start_server(server_id)
            results[server_id] = success
            
            if success:
                logger.info(f"✅ Server {server_id} started successfully")
                # Wait a bit between server starts to avoid resource conflicts
                await asyncio.sleep(2)
            else:
                logger.error(f"❌ Server {server_id} failed to start")
                
                # Check if this is a critical server
                server_config = self.servers[server_id]
                if server_config.get('config', {}).get('priority', 3) <= 2:
                    logger.error(f"Critical server {server_id} failed - continuing with other servers")
        
        # Start health monitoring
        self.is_running = True
        asyncio.create_task(self._health_monitoring_loop())
        
        successful_starts = sum(1 for success in results.values() if success)
        total_servers = len(results)
        
        logger.info(f"Server startup completed: {successful_starts}/{total_servers} servers started")
        return results
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring loop"""
        logger.info("Starting health monitoring loop...")
        
        while self.is_running:
            try:
                for server_id in self.startup_order:
                    if server_id not in self.processes:
                        continue
                    
                    server_config = self.servers[server_id]
                    server_process = self.processes[server_id]
                    
                    # Skip if server is not supposed to be running
                    if not server_config.get('auto_start', True):
                        continue
                    
                    # Check health
                    is_healthy = await self._check_server_health(server_id)
                    
                    # Restart if unhealthy and restart is enabled
                    if not is_healthy and server_config.get('restart_on_failure', True):
                        if server_process.health.status in [ServerStatus.FAILED, ServerStatus.UNHEALTHY]:
                            logger.warning(f"Server {server_id} is unhealthy, attempting restart...")
                            await self._restart_server(server_id)
                
                # Wait before next health check cycle
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def stop_all_servers(self):
        """Stop all servers"""
        logger.info("Stopping all MCP servers...")
        self.is_running = False
        
        # Stop servers in reverse order
        for server_id in reversed(self.startup_order):
            await self._stop_server(server_id)
        
        logger.info("All servers stopped")
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers"""
        status = {}
        
        for server_id, server_config in self.servers.items():
            server_process = self.processes.get(server_id)
            
            if server_process:
                status[server_id] = {
                    'name': server_config.get('name', server_id),
                    'status': server_process.health.status.value,
                    'uptime': server_process.health.uptime,
                    'memory_usage_mb': server_process.health.memory_usage,
                    'cpu_usage_percent': server_process.health.cpu_usage,
                    'response_time_ms': server_process.health.response_time * 1000 if server_process.health.response_time else None,
                    'consecutive_failures': server_process.health.consecutive_failures,
                    'restart_count': server_process.restart_count,
                    'last_check': server_process.health.last_check.isoformat(),
                    'error_message': server_process.health.error_message,
                    'transport': server_config.get('transport'),
                    'port': server_config.get('port'),
                    'auto_start': server_config.get('auto_start', True)
                }
            else:
                status[server_id] = {
                    'name': server_config.get('name', server_id),
                    'status': ServerStatus.STOPPED.value,
                    'transport': server_config.get('transport'),
                    'port': server_config.get('port'),
                    'auto_start': server_config.get('auto_start', True)
                }
        
        return status
    
    def print_status_report(self):
        """Print a comprehensive status report"""
        status = self.get_server_status()
        
        print("\n" + "="*80)
        print("🚀 MCP SERVER STATUS REPORT")
        print("="*80)
        
        running_count = 0
        failed_count = 0
        
        for server_id, info in status.items():
            status_emoji = {
                'running': '✅',
                'starting': '🔄',
                'failed': '❌',
                'stopped': '⏹️',
                'unhealthy': '⚠️',
                'restarting': '🔄'
            }.get(info['status'], '❓')
            
            print(f"{status_emoji} {info['name']}")
            print(f"   Status: {info['status'].upper()}")
            
            if info['status'] == 'running':
                running_count += 1
                if info.get('uptime'):
                    print(f"   Uptime: {info['uptime']:.1f}s")
                if info.get('memory_usage_mb'):
                    print(f"   Memory: {info['memory_usage_mb']:.1f}MB")
                if info.get('response_time_ms'):
                    print(f"   Response Time: {info['response_time_ms']:.1f}ms")
            elif info['status'] in ['failed', 'unhealthy']:
                failed_count += 1
                if info.get('error_message'):
                    print(f"   Error: {info['error_message']}")
                if info.get('restart_count', 0) > 0:
                    print(f"   Restarts: {info['restart_count']}")
            
            if info.get('transport') == 'http' and info.get('port'):
                print(f"   Endpoint: http://localhost:{info['port']}")
            
            print()
        
        total_servers = len(status)
        print(f"📊 SUMMARY: {running_count}/{total_servers} servers running, {failed_count} failed")
        print("="*80)


async def main():
    """Main function for testing the MCP server manager"""
    manager = MCPServerManager()
    
    try:
        # Start all servers
        results = await manager.start_all_servers()
        
        # Print initial status
        manager.print_status_report()
        
        # Keep running and monitoring
        print("\n🔍 Monitoring servers... Press Ctrl+C to stop")
        while True:
            await asyncio.sleep(60)  # Print status every minute
            manager.print_status_report()
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        await manager.stop_all_servers()
    except Exception as e:
        logger.error(f"Manager error: {e}")
        await manager.stop_all_servers()


if __name__ == "__main__":
    asyncio.run(main())
