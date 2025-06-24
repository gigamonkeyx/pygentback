"""
Service Orchestrator for PyGent Factory Startup Service
Dependency resolution, parallel startup optimization, and service lifecycle management.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, deque
import uuid

from ..models.schemas import (
    ServiceStatus, SequenceStatus, SystemHealthStatus,
    StartupRequest, StartupResponse, SystemStatusResponse
)
from ..utils.logging_config import orchestrator_logger
from .database import DatabaseManager
from .config_manager import ConfigurationManager
from .service_manager import ServiceManager


class DependencyResolver:
    """Resolves service dependencies and creates optimal startup order."""
    
    def __init__(self):
        self.logger = orchestrator_logger
    
    def resolve_dependencies(self, services: List[str], dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """
        Resolve service dependencies and return startup order with parallel groups.
        
        Returns:
            List of lists, where each inner list contains services that can start in parallel
        """
        try:
            # Build dependency graph
            graph = defaultdict(list)
            in_degree = defaultdict(int)
            
            # Initialize all services
            for service in services:
                in_degree[service] = 0
            
            # Build graph and calculate in-degrees
            for service, deps in dependencies.items():
                if service in services:
                    for dep in deps:
                        if dep in services:
                            graph[dep].append(service)
                            in_degree[service] += 1
            
            # Topological sort with parallel groups
            result = []
            queue = deque([service for service in services if in_degree[service] == 0])
            
            while queue:
                # All services in current queue can start in parallel
                current_level = list(queue)
                result.append(current_level)
                queue.clear()
                
                # Process all services in current level
                for service in current_level:
                    for dependent in graph[service]:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)
            
            # Check for circular dependencies
            total_resolved = sum(len(level) for level in result)
            if total_resolved != len(services):
                unresolved = set(services) - {s for level in result for s in level}
                raise ValueError(f"Circular dependency detected involving services: {unresolved}")
            
            self.logger.info(f"Dependency resolution completed: {len(result)} parallel groups")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to resolve dependencies: {e}")
            raise
    
    def validate_dependencies(self, services: List[str], dependencies: Dict[str, List[str]]) -> List[str]:
        """Validate dependency configuration and return any issues."""
        issues = []
        
        # Check for self-dependencies
        for service, deps in dependencies.items():
            if service in deps:
                issues.append(f"Service {service} depends on itself")
        
        # Check for missing services
        for service, deps in dependencies.items():
            if service not in services:
                issues.append(f"Dependency service {service} not in services list")
            for dep in deps:
                if dep not in services:
                    issues.append(f"Dependency {dep} for service {service} not in services list")
        
        return issues


class ServiceOrchestrator:
    """Main orchestrator for managing service startup, shutdown, and lifecycle."""
    
    def __init__(self, db_manager: DatabaseManager, config_manager: ConfigurationManager):
        self.db_manager = db_manager
        self.config_manager = config_manager
        self.service_manager = ServiceManager(config_manager)
        self.dependency_resolver = DependencyResolver()
        self.logger = orchestrator_logger
        
        # State tracking
        self.active_sequences: Dict[str, Dict[str, Any]] = {}
        self.service_states: Dict[str, ServiceStatus] = {}
        self.system_metrics: Dict[str, Any] = {}
        
        # Configuration
        self.default_timeout = 300  # seconds
        self.health_check_interval = 30  # seconds
        self.max_parallel_services = 10
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_collection_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """Initialize the service orchestrator."""
        try:
            self.logger.info("Initializing service orchestrator")
            
            # Initialize service manager
            await self.service_manager.initialize()
            
            # Load current system state
            await self._load_system_state()
            
            # Start background tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            self._metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
            
            self.logger.info("Service orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize service orchestrator: {e}")
            return False
    
    async def start_services(self, request: StartupRequest) -> StartupResponse:
        """Start services according to the startup request."""
        try:
            sequence_id = str(uuid.uuid4())
            
            # Determine services to start
            if request.profile_id:
                profile = await self.config_manager.get_profile(request.profile_id)
                if not profile:
                    raise ValueError(f"Configuration profile not found: {request.profile_id}")
                services = profile["startup_sequence"]
                dependencies = profile.get("dependencies", {})
            else:
                services = request.services
                dependencies = {}
            
            # Validate dependencies
            issues = self.dependency_resolver.validate_dependencies(services, dependencies)
            if issues:
                raise ValueError(f"Dependency validation failed: {'; '.join(issues)}")
            
            # Resolve startup order
            startup_groups = self.dependency_resolver.resolve_dependencies(services, dependencies)
            
            # Create startup sequence record
            sequence_data = {
                "sequence_name": f"startup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "services": services,
                "dependencies": dependencies,
                "environment": request.environment.value,
                "status": "running",
                "parallel_execution": request.parallel_execution,
                "timeout_seconds": request.timeout_seconds
            }
            
            await self.db_manager.create_startup_sequence(sequence_data)
            
            # Track active sequence
            self.active_sequences[sequence_id] = {
                "services": services,
                "startup_groups": startup_groups,
                "status": SequenceStatus.RUNNING,
                "started_at": datetime.utcnow(),
                "progress": {},
                "request": request
            }
            
            # Start services asynchronously
            asyncio.create_task(self._execute_startup_sequence(sequence_id))
            
            # Calculate estimated duration
            estimated_duration = len(startup_groups) * 30  # Rough estimate
            
            response = StartupResponse(
                sequence_id=sequence_id,
                status=SequenceStatus.RUNNING,
                message=f"Starting {len(services)} services in {len(startup_groups)} parallel groups",
                services=services,
                estimated_duration=estimated_duration,
                started_at=datetime.utcnow()
            )
            
            self.logger.info(f"Startup sequence initiated: {sequence_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to start services: {e}")
            raise
    
    async def _execute_startup_sequence(self, sequence_id: str):
        """Execute the startup sequence for given sequence ID."""
        try:
            sequence = self.active_sequences[sequence_id]
            startup_groups = sequence["startup_groups"]
            request = sequence["request"]
            
            self.logger.info(f"Executing startup sequence {sequence_id} with {len(startup_groups)} groups")
            
            # Execute each group in order
            for group_index, service_group in enumerate(startup_groups):
                self.logger.info(f"Starting group {group_index + 1}/{len(startup_groups)}: {service_group}")
                
                # Start all services in the group in parallel
                if request.parallel_execution and len(service_group) > 1:
                    tasks = [
                        self._start_single_service(service, sequence_id, request.environment.value)
                        for service in service_group
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Check for failures
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            service = service_group[i]
                            self.logger.error(f"Failed to start service {service}: {result}")
                            sequence["progress"][service] = {"status": "failed", "error": str(result)}
                        else:
                            service = service_group[i]
                            sequence["progress"][service] = {"status": "completed"}
                else:
                    # Sequential startup within group
                    for service in service_group:
                        try:
                            await self._start_single_service(service, sequence_id, request.environment.value)
                            sequence["progress"][service] = {"status": "completed"}
                        except Exception as e:
                            self.logger.error(f"Failed to start service {service}: {e}")
                            sequence["progress"][service] = {"status": "failed", "error": str(e)}
                
                # Wait for group to be healthy before proceeding
                await self._wait_for_group_health(service_group, timeout=request.timeout_seconds)
            
            # Update sequence status
            sequence["status"] = SequenceStatus.COMPLETED
            sequence["completed_at"] = datetime.utcnow()
            
            await self.db_manager.update_startup_sequence_status(
                sequence_id, "completed", sequence["progress"]
            )
            
            self.logger.info(f"Startup sequence {sequence_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Startup sequence {sequence_id} failed: {e}")
            
            # Update sequence status to failed
            if sequence_id in self.active_sequences:
                self.active_sequences[sequence_id]["status"] = SequenceStatus.FAILED
                self.active_sequences[sequence_id]["error"] = str(e)
                
                await self.db_manager.update_startup_sequence_status(
                    sequence_id, "failed", {"error": str(e)}
                )
    
    async def _start_single_service(self, service_name: str, sequence_id: str, environment: str) -> bool:
        """Start a single service."""
        try:
            self.logger.info(f"Starting service: {service_name}")
            
            # Update service status
            self.service_states[service_name] = ServiceStatus.STARTING
            
            # Get service configuration
            config = await self.config_manager.get_service_config(service_name, environment)
            if not config:
                raise ValueError(f"No configuration found for service {service_name}")
            
            # Start the service
            success = await self.service_manager.start_service(service_name, config)
            
            if success:
                self.service_states[service_name] = ServiceStatus.RUNNING
                self.logger.info(f"Service {service_name} started successfully")
                return True
            else:
                self.service_states[service_name] = ServiceStatus.ERROR
                raise RuntimeError(f"Failed to start service {service_name}")
                
        except Exception as e:
            self.service_states[service_name] = ServiceStatus.ERROR
            self.logger.error(f"Failed to start service {service_name}: {e}")
            raise
    
    async def _wait_for_group_health(self, services: List[str], timeout: int = 60):
        """Wait for all services in a group to become healthy."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_healthy = True
            
            for service in services:
                if not await self.service_manager.check_service_health(service):
                    all_healthy = False
                    break
            
            if all_healthy:
                self.logger.info(f"All services in group are healthy: {services}")
                return
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        self.logger.warning(f"Timeout waiting for group health: {services}")
    
    async def get_system_status(self) -> SystemStatusResponse:
        """Get comprehensive system status."""
        try:
            # Collect service statuses
            service_statuses = []
            for service_name, status in self.service_states.items():
                health_score = await self.service_manager.get_service_health_score(service_name)
                last_check = datetime.utcnow()  # In real implementation, track actual check times
                
                service_statuses.append({
                    "service_name": service_name,
                    "status": status,
                    "health_score": health_score,
                    "last_check": last_check,
                    "metrics": {}
                })
            
            # Calculate overall health
            if service_statuses:
                avg_health = sum(s["health_score"] for s in service_statuses) / len(service_statuses)
                if avg_health >= 0.8:
                    overall_status = SystemHealthStatus.HEALTHY
                elif avg_health >= 0.5:
                    overall_status = SystemHealthStatus.DEGRADED
                else:
                    overall_status = SystemHealthStatus.UNHEALTHY
            else:
                overall_status = SystemHealthStatus.UNKNOWN
                avg_health = 0.0
            
            # Get active sequences
            active_sequences = [seq_id for seq_id, seq in self.active_sequences.items() 
                             if seq["status"] == SequenceStatus.RUNNING]
            
            return SystemStatusResponse(
                overall_status=overall_status,
                health_score=avg_health,
                services=service_statuses,
                active_sequences=active_sequences,
                system_metrics=self.system_metrics,
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Perform health check on the orchestrator."""
        try:
            # Check database connectivity
            db_healthy = await self.db_manager.health_check()
            
            # Check service manager
            sm_healthy = await self.service_manager.health_check()
            
            return db_healthy and sm_healthy
            
        except Exception as e:
            self.logger.error(f"Orchestrator health check failed: {e}")
            return False
    
    async def _load_system_state(self):
        """Load current system state from database."""
        try:
            state = await self.db_manager.get_system_state()
            if state:
                self.service_states.update(state.get("services_status", {}))
                self.system_metrics.update(state.get("system_metrics", {}))
                self.logger.info("System state loaded from database")
        except Exception as e:
            self.logger.warning(f"Failed to load system state: {e}")
    
    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check all services
                for service_name in list(self.service_states.keys()):
                    try:
                        is_healthy = await self.service_manager.check_service_health(service_name)
                        if is_healthy:
                            if self.service_states[service_name] == ServiceStatus.ERROR:
                                self.service_states[service_name] = ServiceStatus.RUNNING
                        else:
                            if self.service_states[service_name] == ServiceStatus.RUNNING:
                                self.service_states[service_name] = ServiceStatus.ERROR
                    except Exception as e:
                        self.logger.warning(f"Health check failed for {service_name}: {e}")
                
                # Update system state in database
                await self._update_system_state()
                
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
    
    async def _metrics_collection_loop(self):
        """Background task for collecting system metrics."""
        while True:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute
                
                # Collect system metrics
                import psutil
                
                self.system_metrics.update({
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
    
    async def _update_system_state(self):
        """Update system state in database."""
        try:
            state_data = {
                "overall_status": "healthy",  # Calculate based on services
                "services_status": self.service_states,
                "system_metrics": self.system_metrics
            }
            
            await self.db_manager.update_system_state(state_data)
            
        except Exception as e:
            self.logger.warning(f"Failed to update system state: {e}")
    
    async def shutdown(self):
        """Shutdown the orchestrator and cleanup resources."""
        try:
            self.logger.info("Shutting down service orchestrator")
            
            # Cancel background tasks
            if self._health_check_task:
                self._health_check_task.cancel()
            if self._metrics_collection_task:
                self._metrics_collection_task.cancel()
            
            # Shutdown service manager
            await self.service_manager.shutdown()
            
            self.logger.info("Service orchestrator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during orchestrator shutdown: {e}")
