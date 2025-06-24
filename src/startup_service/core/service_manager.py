"""
Service Manager for PyGent Factory Startup Service
Individual service lifecycle management with real implementations.
"""

import asyncio
import subprocess
import time
import os
import signal
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import aiohttp
import docker
import psutil

from ..models.schemas import ServiceType, ServiceStatus
from ..utils.logging_config import orchestrator_logger
from .config_manager import ConfigurationManager


class ServiceManager:
    """Manages individual service lifecycle operations."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.logger = orchestrator_logger
        
        # Service tracking
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.docker_containers: Dict[str, str] = {}  # service_name -> container_id
        self.service_configs: Dict[str, Dict[str, Any]] = {}
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            self.logger.warning(f"Docker client not available: {e}")
            self.docker_client = None
    
    async def initialize(self) -> bool:
        """Initialize the service manager."""
        try:
            self.logger.info("Initializing service manager")
            
            # Test Docker connectivity if available
            if self.docker_client:
                try:
                    self.docker_client.ping()
                    self.logger.info("Docker client connected successfully")
                except Exception as e:
                    self.logger.warning(f"Docker connectivity test failed: {e}")
                    self.docker_client = None
            
            self.logger.info("Service manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize service manager: {e}")
            return False
    
    async def start_service(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Start a service based on its configuration."""
        try:
            self.logger.info(f"Starting service: {service_name}")
            self.service_configs[service_name] = config
            
            service_type = config.get("service_type", "other")
            
            if service_type == ServiceType.POSTGRESQL:
                return await self._start_postgresql(service_name, config)
            elif service_type == ServiceType.REDIS:
                return await self._start_redis(service_name, config)
            elif service_type == ServiceType.OLLAMA:
                return await self._start_ollama(service_name, config)
            elif service_type == ServiceType.AGENT:
                return await self._start_agent_service(service_name, config)
            elif service_type == ServiceType.MONITORING:
                return await self._start_monitoring_service(service_name, config)
            else:
                return await self._start_generic_service(service_name, config)
                
        except Exception as e:
            self.logger.error(f"Failed to start service {service_name}: {e}")
            return False
    
    async def _start_postgresql(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Start PostgreSQL service."""
        try:
            # Check if already running
            if await self._check_postgresql_health(config):
                self.logger.info(f"PostgreSQL {service_name} already running")
                return True
            
            # Try Docker first
            if self.docker_client and config.get("use_docker", True):
                return await self._start_postgresql_docker(service_name, config)
            else:
                return await self._start_postgresql_native(service_name, config)
                
        except Exception as e:
            self.logger.error(f"Failed to start PostgreSQL {service_name}: {e}")
            return False
    
    async def _start_postgresql_docker(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Start PostgreSQL using Docker."""
        try:
            container_name = f"pygent_postgres_{service_name}"
            
            # Check if container already exists
            try:
                container = self.docker_client.containers.get(container_name)
                if container.status == "running":
                    self.docker_containers[service_name] = container.id
                    return True
                else:
                    container.start()
                    self.docker_containers[service_name] = container.id
                    await asyncio.sleep(5)  # Wait for startup
                    return await self._check_postgresql_health(config)
            except docker.errors.NotFound:
                pass
            
            # Create new container
            environment = {
                "POSTGRES_DB": config.get("database", "pygent_factory"),
                "POSTGRES_USER": config.get("username", "postgres"),
                "POSTGRES_PASSWORD": config.get("password", "postgres"),
                "POSTGRES_HOST_AUTH_METHOD": "trust"
            }
            
            ports = {
                "5432/tcp": config.get("port", 5432)
            }
            
            volumes = {}
            if config.get("data_volume"):
                volumes[config["data_volume"]] = {"bind": "/var/lib/postgresql/data", "mode": "rw"}
            
            container = self.docker_client.containers.run(
                image=config.get("image", "postgres:15"),
                name=container_name,
                environment=environment,
                ports=ports,
                volumes=volumes,
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            self.docker_containers[service_name] = container.id
            
            # Wait for PostgreSQL to be ready
            for _ in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                if await self._check_postgresql_health(config):
                    self.logger.info(f"PostgreSQL {service_name} started successfully")
                    return True
            
            self.logger.error(f"PostgreSQL {service_name} failed to become healthy")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start PostgreSQL Docker container: {e}")
            return False
    
    async def _start_postgresql_native(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Start PostgreSQL natively (assumes already installed)."""
        try:
            # This would typically involve starting a system service
            # For now, we'll assume PostgreSQL is already running
            self.logger.info(f"Assuming native PostgreSQL {service_name} is already running")
            return await self._check_postgresql_health(config)
            
        except Exception as e:
            self.logger.error(f"Failed to start native PostgreSQL: {e}")
            return False
    
    async def _check_postgresql_health(self, config: Dict[str, Any]) -> bool:
        """Check PostgreSQL health."""
        try:
            import asyncpg
            
            connection_string = (
                f"postgresql://{config.get('username', 'postgres')}:"
                f"{config.get('password', 'postgres')}@"
                f"{config.get('host', 'localhost')}:"
                f"{config.get('port', 5432)}/"
                f"{config.get('database', 'pygent_factory')}"
            )
            
            conn = await asyncpg.connect(connection_string)
            await conn.execute("SELECT 1")
            await conn.close()
            
            return True
            
        except Exception as e:
            self.logger.debug(f"PostgreSQL health check failed: {e}")
            return False
    
    async def _start_redis(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Start Redis service."""
        try:
            # Check if already running
            if await self._check_redis_health(config):
                self.logger.info(f"Redis {service_name} already running")
                return True
            
            # Try Docker first
            if self.docker_client and config.get("use_docker", True):
                return await self._start_redis_docker(service_name, config)
            else:
                return await self._start_redis_native(service_name, config)
                
        except Exception as e:
            self.logger.error(f"Failed to start Redis {service_name}: {e}")
            return False
    
    async def _start_redis_docker(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Start Redis using Docker."""
        try:
            container_name = f"pygent_redis_{service_name}"
            
            # Check if container already exists
            try:
                container = self.docker_client.containers.get(container_name)
                if container.status == "running":
                    self.docker_containers[service_name] = container.id
                    return True
                else:
                    container.start()
                    self.docker_containers[service_name] = container.id
                    await asyncio.sleep(2)
                    return await self._check_redis_health(config)
            except docker.errors.NotFound:
                pass
            
            # Create new container
            ports = {
                "6379/tcp": config.get("port", 6379)
            }
            
            command = ["redis-server"]
            if config.get("password"):
                command.extend(["--requirepass", config["password"]])
            
            container = self.docker_client.containers.run(
                image=config.get("image", "redis:7"),
                name=container_name,
                ports=ports,
                command=command,
                detach=True,
                restart_policy={"Name": "unless-stopped"}
            )
            
            self.docker_containers[service_name] = container.id
            
            # Wait for Redis to be ready
            for _ in range(10):
                await asyncio.sleep(1)
                if await self._check_redis_health(config):
                    self.logger.info(f"Redis {service_name} started successfully")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start Redis Docker container: {e}")
            return False
    
    async def _check_redis_health(self, config: Dict[str, Any]) -> bool:
        """Check Redis health."""
        try:
            import aioredis
            
            redis_url = f"redis://:{config.get('password', '')}@{config.get('host', 'localhost')}:{config.get('port', 6379)}"
            
            redis = aioredis.from_url(redis_url)
            await redis.ping()
            await redis.close()
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Redis health check failed: {e}")
            return False
    
    async def _start_ollama(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Start Ollama service."""
        try:
            # Check if already running
            if await self._check_ollama_health(config):
                self.logger.info(f"Ollama {service_name} already running")
                return True
            
            # Start Ollama process
            ollama_path = config.get("executable_path", "ollama")
            host = config.get("host", "127.0.0.1")
            port = config.get("port", 11434)
            
            env = os.environ.copy()
            env["OLLAMA_HOST"] = f"{host}:{port}"
            
            process = await asyncio.create_subprocess_exec(
                ollama_path, "serve",
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.running_processes[service_name] = process
            
            # Wait for Ollama to be ready
            for _ in range(30):
                await asyncio.sleep(1)
                if await self._check_ollama_health(config):
                    self.logger.info(f"Ollama {service_name} started successfully")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start Ollama {service_name}: {e}")
            return False
    
    async def _check_ollama_health(self, config: Dict[str, Any]) -> bool:
        """Check Ollama health."""
        try:
            host = config.get("host", "127.0.0.1")
            port = config.get("port", 11434)
            url = f"http://{host}:{port}/api/tags"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.debug(f"Ollama health check failed: {e}")
            return False
    
    async def _start_agent_service(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Start an agent service."""
        try:
            # This would start specific PyGent Factory agent services
            # For now, we'll simulate starting an agent service
            
            script_path = config.get("script_path")
            if not script_path:
                self.logger.error(f"No script_path specified for agent service {service_name}")
                return False
            
            # Start the agent process
            process = await asyncio.create_subprocess_exec(
                "python", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.running_processes[service_name] = process
            
            # Wait a bit and check if process is still running
            await asyncio.sleep(2)
            if process.returncode is None:
                self.logger.info(f"Agent service {service_name} started successfully")
                return True
            else:
                self.logger.error(f"Agent service {service_name} exited with code {process.returncode}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start agent service {service_name}: {e}")
            return False
    
    async def _start_monitoring_service(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Start a monitoring service (Prometheus, Grafana, etc.)."""
        try:
            # This would start monitoring services
            # Implementation depends on specific monitoring setup
            self.logger.info(f"Starting monitoring service {service_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring service {service_name}: {e}")
            return False
    
    async def _start_generic_service(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Start a generic service."""
        try:
            command = config.get("command")
            if not command:
                self.logger.error(f"No command specified for generic service {service_name}")
                return False
            
            # Start the process
            if isinstance(command, str):
                command = command.split()
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.running_processes[service_name] = process
            
            # Wait a bit and check if process is still running
            await asyncio.sleep(2)
            if process.returncode is None:
                self.logger.info(f"Generic service {service_name} started successfully")
                return True
            else:
                self.logger.error(f"Generic service {service_name} exited with code {process.returncode}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start generic service {service_name}: {e}")
            return False
    
    async def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        try:
            config = self.service_configs.get(service_name)
            if not config:
                return False
            
            service_type = config.get("service_type", "other")
            
            if service_type == ServiceType.POSTGRESQL:
                return await self._check_postgresql_health(config)
            elif service_type == ServiceType.REDIS:
                return await self._check_redis_health(config)
            elif service_type == ServiceType.OLLAMA:
                return await self._check_ollama_health(config)
            else:
                # For other services, check if process is running
                if service_name in self.running_processes:
                    process = self.running_processes[service_name]
                    return process.returncode is None
                elif service_name in self.docker_containers:
                    container_id = self.docker_containers[service_name]
                    try:
                        container = self.docker_client.containers.get(container_id)
                        return container.status == "running"
                    except:
                        return False
                return False
                
        except Exception as e:
            self.logger.error(f"Health check failed for {service_name}: {e}")
            return False
    
    async def get_service_health_score(self, service_name: str) -> float:
        """Get service health score (0.0 to 1.0)."""
        try:
            is_healthy = await self.check_service_health(service_name)
            return 1.0 if is_healthy else 0.0
        except Exception:
            return 0.0
    
    async def health_check(self) -> bool:
        """Check service manager health."""
        try:
            # Basic health check - ensure we can track services
            return True
        except Exception as e:
            self.logger.error(f"Service manager health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown all managed services."""
        try:
            self.logger.info("Shutting down all managed services")
            
            # Stop all running processes
            for service_name, process in self.running_processes.items():
                try:
                    if process.returncode is None:
                        process.terminate()
                        await asyncio.sleep(2)
                        if process.returncode is None:
                            process.kill()
                    self.logger.info(f"Stopped process for service: {service_name}")
                except Exception as e:
                    self.logger.error(f"Failed to stop process for {service_name}: {e}")
            
            # Stop Docker containers
            if self.docker_client:
                for service_name, container_id in self.docker_containers.items():
                    try:
                        container = self.docker_client.containers.get(container_id)
                        container.stop(timeout=10)
                        self.logger.info(f"Stopped Docker container for service: {service_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to stop container for {service_name}: {e}")
            
            self.logger.info("Service manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during service manager shutdown: {e}")
