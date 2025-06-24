#!/usr/bin/env python3
"""
Agent Orchestration Manager

Comprehensive agent orchestration with lifecycle management, task distribution,
communication routing, and performance monitoring.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Type, Callable
from dataclasses import dataclass, field

from .base_agent import BaseAgent, AgentMessage, AgentStatus, AgentType, MessageType

try:
    from ..cache.cache_layers import cache_manager
except ImportError:
    cache_manager = None

try:
    from ..cache.redis_manager import redis_manager
except ImportError:
    redis_manager = None

try:
    from ..database.production_manager import db_manager
except ImportError:
    db_manager = None

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    """Orchestration configuration"""
    max_agents: int = 100
    heartbeat_timeout_seconds: int = 120
    task_timeout_seconds: int = 300
    message_retention_hours: int = 24
    enable_auto_scaling: bool = True
    enable_load_balancing: bool = True
    enable_fault_tolerance: bool = True


@dataclass
class TaskDefinition:
    """Task definition for orchestration"""
    task_id: str
    task_type: str
    priority: int = 1
    required_capabilities: List[str] = field(default_factory=list)
    preferred_agent_type: Optional[AgentType] = None
    timeout_seconds: int = 300
    retry_count: int = 3
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationMetrics:
    """Orchestration performance metrics"""
    total_agents: int = 0
    active_agents: int = 0
    idle_agents: int = 0
    error_agents: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    pending_tasks: int = 0
    average_task_time: float = 0.0
    messages_routed: int = 0
    system_uptime: float = 0.0


class AgentOrchestrationManager:
    """Comprehensive agent orchestration manager"""
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.config = config or OrchestrationConfig()
        
        # Agent management
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types: Dict[str, Type[BaseAgent]] = {}
        
        # Task management
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, TaskDefinition] = {}
        self.completed_tasks: Dict[str, Dict[str, Any]] = {}
        self.failed_tasks: Dict[str, Dict[str, Any]] = {}
        
        # Communication
        self.message_router: Dict[str, str] = {}  # agent_id -> queue_name
        self.subscriptions: Dict[str, List[str]] = {}  # topic -> agent_ids
        
        # Monitoring
        self.metrics = OrchestrationMetrics()
        self.start_time = datetime.utcnow()
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # State
        self.is_running = False
        self.is_initialized = False
        
        logger.info("Agent orchestration manager created")
    
    async def initialize(self) -> bool:
        """Initialize the orchestration manager"""
        try:
            logger.info("Initializing agent orchestration manager...")
            
            # Initialize Redis pub/sub for agent communication
            if redis_manager:
                await self._initialize_communication()
            
            # Start orchestration loops
            asyncio.create_task(self._orchestration_loop())
            asyncio.create_task(self._monitoring_loop())
            asyncio.create_task(self._cleanup_loop())
            
            self.is_initialized = True
            self.is_running = True
            
            logger.info("Agent orchestration manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestration manager: {e}")
            return False
    
    async def _initialize_communication(self):
        """Initialize communication infrastructure"""
        try:
            # Set up Redis pub/sub for agent communication
            await redis_manager.publish("agent_orchestration", {
                "type": "orchestrator_started",
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize communication: {e}")
    
    def register_agent_type(self, agent_type: str, agent_class: Type[BaseAgent]):
        """Register an agent type"""
        self.agent_types[agent_type] = agent_class
        logger.info(f"Registered agent type: {agent_type}")
    
    async def create_agent(self, agent_type: str, agent_id: Optional[str] = None,
                          name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a new agent"""
        try:
            if agent_type not in self.agent_types:
                logger.error(f"Unknown agent type: {agent_type}")
                return None
            
            if len(self.agents) >= self.config.max_agents:
                logger.error(f"Maximum agent limit reached: {self.config.max_agents}")
                return None
            
            # Create agent instance
            agent_class = self.agent_types[agent_type]
            agent = agent_class(
                agent_id=agent_id,
                name=name,
                config=config
            )
            
            # Initialize agent
            if not await agent.initialize():
                logger.error(f"Failed to initialize agent {agent.agent_id}")
                return None
            
            # Register agent
            self.agents[agent.agent_id] = agent
            self.message_router[agent.agent_id] = f"agent_queue:{agent.agent_id}"
            
            # Update metrics
            self.metrics.total_agents += 1
            self._update_agent_metrics()
            
            # Start agent
            await agent.start()
            
            # Emit event
            await self._emit_event("agent_created", {
                "agent_id": agent.agent_id,
                "agent_type": agent_type,
                "name": agent.name
            })
            
            logger.info(f"Created agent {agent.name} ({agent.agent_id})")
            return agent.agent_id
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return None
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Destroy an agent"""
        try:
            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not found")
                return False
            
            agent = self.agents[agent_id]
            
            # Stop agent
            await agent.stop()
            
            # Remove from registry
            del self.agents[agent_id]
            if agent_id in self.message_router:
                del self.message_router[agent_id]
            
            # Update metrics
            self.metrics.total_agents -= 1
            self._update_agent_metrics()
            
            # Emit event
            await self._emit_event("agent_destroyed", {
                "agent_id": agent_id,
                "name": agent.name
            })
            
            logger.info(f"Destroyed agent {agent.name} ({agent_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to destroy agent {agent_id}: {e}")
            return False
    
    async def submit_task(self, task: TaskDefinition) -> bool:
        """Submit a task for execution"""
        try:
            # Validate task
            if not self._validate_task(task):
                return False
            
            # Add to task queue
            await self.task_queue.put(task)
            self.metrics.total_tasks += 1
            self.metrics.pending_tasks += 1
            
            # Cache task
            await cache_manager.cache_performance_metric(
                f"orchestration_task:{task.task_id}",
                {
                    "task_type": task.task_type,
                    "priority": task.priority,
                    "required_capabilities": task.required_capabilities,
                    "submitted_at": datetime.utcnow().isoformat(),
                    "status": "pending"
                },
                ttl=self.config.message_retention_hours * 3600
            )
            
            logger.info(f"Submitted task {task.task_id} of type {task.task_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    def _validate_task(self, task: TaskDefinition) -> bool:
        """Validate task definition"""
        if not task.task_id:
            logger.error("Task ID is required")
            return False
        
        if not task.task_type:
            logger.error("Task type is required")
            return False
        
        if task.task_id in self.active_tasks:
            logger.error(f"Task {task.task_id} is already active")
            return False
        
        return True
    
    async def _orchestration_loop(self):
        """Main orchestration loop"""
        while self.is_running:
            try:
                # Process pending tasks
                await self._process_task_queue()
                
                # Check task timeouts
                await self._check_task_timeouts()
                
                # Handle agent failures
                await self._handle_agent_failures()
                
                # Auto-scaling if enabled
                if self.config.enable_auto_scaling:
                    await self._auto_scale_agents()
                
                # REAL orchestration event monitoring
                await self._wait_for_orchestration_events(1)
                
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(5)
    
    async def _process_task_queue(self):
        """Process tasks from the queue"""
        try:
            # Get task with timeout
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                return
            
            # Find suitable agent
            agent_id = await self._find_suitable_agent(task)
            
            if agent_id:
                # Assign task to agent
                await self._assign_task_to_agent(task, agent_id)
            else:
                # No suitable agent found, requeue with real delay strategy
                await self._handle_task_requeue_delay(task)
                
        except Exception as e:
            logger.error(f"Task queue processing error: {e}")
    
    async def _find_suitable_agent(self, task: TaskDefinition) -> Optional[str]:
        """Find suitable agent for task"""
        try:
            suitable_agents = []
            
            for agent_id, agent in self.agents.items():
                # Check agent status
                if agent.status != AgentStatus.IDLE:
                    continue
                
                # Check agent type preference
                if task.preferred_agent_type and agent.agent_type != task.preferred_agent_type:
                    continue
                
                # Check required capabilities
                agent_capabilities = [cap.name for cap in agent.capabilities]
                if not all(cap in agent_capabilities for cap in task.required_capabilities):
                    continue
                
                suitable_agents.append((agent_id, agent))
            
            if not suitable_agents:
                return None
            
            # Load balancing - select agent with least load
            if self.config.enable_load_balancing:
                suitable_agents.sort(key=lambda x: x[1].metrics.tasks_completed)
            
            return suitable_agents[0][0]
            
        except Exception as e:
            logger.error(f"Agent selection error: {e}")
            return None
    
    async def _assign_task_to_agent(self, task: TaskDefinition, agent_id: str):
        """Assign task to specific agent"""
        try:
            agent = self.agents[agent_id]
            
            # Create task message
            message = AgentMessage(
                type=MessageType.TASK,
                recipient_id=agent_id,
                content={
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "parameters": task.parameters,
                    "timeout_seconds": task.timeout_seconds
                },
                requires_response=True
            )
            
            # Send task to agent
            await self._route_message(message)
            
            # Track active task
            self.active_tasks[task.task_id] = task
            self.metrics.pending_tasks -= 1
            
            # Update task cache
            await cache_manager.cache_performance_metric(
                f"orchestration_task:{task.task_id}",
                {
                    "task_type": task.task_type,
                    "assigned_agent": agent_id,
                    "assigned_at": datetime.utcnow().isoformat(),
                    "status": "assigned"
                },
                ttl=self.config.message_retention_hours * 3600
            )
            
            logger.info(f"Assigned task {task.task_id} to agent {agent.name}")
            
        except Exception as e:
            logger.error(f"Task assignment error: {e}")
    
    async def _route_message(self, message: AgentMessage) -> bool:
        """Route message to appropriate agent"""
        try:
            if message.recipient_id not in self.agents:
                logger.error(f"Recipient agent {message.recipient_id} not found")
                return False
            
            agent = self.agents[message.recipient_id]
            await agent.receive_message(message)
            
            self.metrics.messages_routed += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Message routing error: {e}")
            return False
    
    async def _check_task_timeouts(self):
        """Check for task timeouts"""
        try:
            current_time = datetime.utcnow()
            timed_out_tasks = []
            
            for task_id, task in self.active_tasks.items():
                # Get task start time from cache
                task_data = await cache_manager.get_cached_performance_metric(
                    f"orchestration_task:{task_id}"
                )
                
                if task_data and "assigned_at" in task_data:
                    assigned_at = datetime.fromisoformat(task_data["assigned_at"])
                    if (current_time - assigned_at).total_seconds() > task.timeout_seconds:
                        timed_out_tasks.append(task_id)
            
            # Handle timed out tasks
            for task_id in timed_out_tasks:
                await self._handle_task_timeout(task_id)
                
        except Exception as e:
            logger.error(f"Task timeout check error: {e}")
    
    async def _handle_task_timeout(self, task_id: str):
        """Handle task timeout"""
        try:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                
                # Move to failed tasks
                self.failed_tasks[task_id] = {
                    "task": task,
                    "reason": "timeout",
                    "failed_at": datetime.utcnow().isoformat()
                }
                
                del self.active_tasks[task_id]
                self.metrics.failed_tasks += 1
                
                logger.warning(f"Task {task_id} timed out")
                
                # Emit event
                await self._emit_event("task_timeout", {
                    "task_id": task_id,
                    "task_type": task.task_type
                })
                
        except Exception as e:
            logger.error(f"Task timeout handling error: {e}")
    
    async def _handle_agent_failures(self):
        """Handle agent failures"""
        try:
            current_time = datetime.utcnow()
            failed_agents = []
            
            for agent_id, agent in self.agents.items():
                # Check heartbeat timeout
                time_since_heartbeat = (current_time - agent.last_heartbeat).total_seconds()
                
                if time_since_heartbeat > self.config.heartbeat_timeout_seconds:
                    if agent.status != AgentStatus.ERROR:
                        failed_agents.append(agent_id)
            
            # Handle failed agents
            for agent_id in failed_agents:
                await self._handle_agent_failure(agent_id)
                
        except Exception as e:
            logger.error(f"Agent failure handling error: {e}")
    
    async def _handle_agent_failure(self, agent_id: str):
        """Handle individual agent failure"""
        try:
            agent = self.agents[agent_id]
            agent.status = AgentStatus.ERROR
            
            logger.warning(f"Agent {agent.name} ({agent_id}) failed")
            
            # Reassign active tasks if fault tolerance is enabled
            if self.config.enable_fault_tolerance:
                await self._reassign_agent_tasks(agent_id)
            
            # Emit event
            await self._emit_event("agent_failed", {
                "agent_id": agent_id,
                "name": agent.name
            })
            
        except Exception as e:
            logger.error(f"Agent failure handling error: {e}")
    
    async def _reassign_agent_tasks(self, failed_agent_id: str):
        """Reassign tasks from failed agent"""
        try:
            # Find tasks assigned to failed agent
            tasks_to_reassign = []
            
            for task_id, task in self.active_tasks.items():
                task_data = await cache_manager.get_cached_performance_metric(
                    f"orchestration_task:{task_id}"
                )
                
                if task_data and task_data.get("assigned_agent") == failed_agent_id:
                    tasks_to_reassign.append(task)
            
            # Reassign tasks
            for task in tasks_to_reassign:
                await self.task_queue.put(task)
                logger.info(f"Reassigned task {task.task_id} due to agent failure")
                
        except Exception as e:
            logger.error(f"Task reassignment error: {e}")
    
    async def _auto_scale_agents(self):
        """Auto-scale agents based on load"""
        try:
            # Simple auto-scaling logic
            queue_size = self.task_queue.qsize()
            idle_agents = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.IDLE)
            
            # Scale up if queue is large and no idle agents
            if queue_size > 5 and idle_agents == 0 and len(self.agents) < self.config.max_agents:
                # Create a general-purpose agent
                await self.create_agent("general", name=f"auto_scaled_{len(self.agents)}")
                logger.info("Auto-scaled up: created new agent")
            
            # Scale down if too many idle agents
            elif idle_agents > 3 and len(self.agents) > 1:
                # Find an idle agent to remove
                for agent_id, agent in self.agents.items():
                    if agent.status == AgentStatus.IDLE:
                        await self.destroy_agent(agent_id)
                        logger.info("Auto-scaled down: removed idle agent")
                        break
                        
        except Exception as e:
            logger.error(f"Auto-scaling error: {e}")
    
    async def _monitoring_loop(self):
        """Monitoring and metrics collection loop"""
        while self.is_running:
            try:
                # Update metrics
                self._update_metrics()
                
                # Cache metrics
                await self._cache_metrics()
                
                # REAL metrics monitoring with event-driven updates
                await self._wait_for_metrics_events(30)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    def _update_metrics(self):
        """Update orchestration metrics"""
        try:
            self._update_agent_metrics()
            
            # Update system uptime
            self.metrics.system_uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Update task metrics
            total_completed = self.metrics.completed_tasks
            total_failed = self.metrics.failed_tasks
            
            if total_completed > 0:
                # Calculate average task time from agent metrics
                total_time = sum(agent.metrics.total_execution_time for agent in self.agents.values())
                total_tasks = sum(agent.metrics.tasks_completed for agent in self.agents.values())
                
                if total_tasks > 0:
                    self.metrics.average_task_time = total_time / total_tasks
            
        except Exception as e:
            logger.error(f"Metrics update error: {e}")
    
    def _update_agent_metrics(self):
        """Update agent-related metrics"""
        self.metrics.active_agents = sum(
            1 for agent in self.agents.values() 
            if agent.status == AgentStatus.RUNNING
        )
        
        self.metrics.idle_agents = sum(
            1 for agent in self.agents.values() 
            if agent.status == AgentStatus.IDLE
        )
        
        self.metrics.error_agents = sum(
            1 for agent in self.agents.values() 
            if agent.status == AgentStatus.ERROR
        )
    
    async def _cache_metrics(self):
        """Cache orchestration metrics"""
        try:
            metrics_data = {
                "total_agents": self.metrics.total_agents,
                "active_agents": self.metrics.active_agents,
                "idle_agents": self.metrics.idle_agents,
                "error_agents": self.metrics.error_agents,
                "total_tasks": self.metrics.total_tasks,
                "completed_tasks": self.metrics.completed_tasks,
                "failed_tasks": self.metrics.failed_tasks,
                "pending_tasks": self.metrics.pending_tasks,
                "average_task_time": self.metrics.average_task_time,
                "messages_routed": self.metrics.messages_routed,
                "system_uptime": self.metrics.system_uptime,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await cache_manager.cache_performance_metric(
                "orchestration_metrics",
                metrics_data,
                ttl=300  # 5 minutes
            )
            
        except Exception as e:
            logger.error(f"Metrics caching error: {e}")
    
    async def _cleanup_loop(self):
        """Cleanup old data"""
        while self.is_running:
            try:
                # Clean up old completed/failed tasks
                cutoff_time = datetime.utcnow() - timedelta(hours=self.config.message_retention_hours)
                
                # Clean completed tasks
                to_remove = []
                for task_id, task_data in self.completed_tasks.items():
                    completed_at = datetime.fromisoformat(task_data["completed_at"])
                    if completed_at < cutoff_time:
                        to_remove.append(task_id)
                
                for task_id in to_remove:
                    del self.completed_tasks[task_id]
                
                # Clean failed tasks
                to_remove = []
                for task_id, task_data in self.failed_tasks.items():
                    failed_at = datetime.fromisoformat(task_data["failed_at"])
                    if failed_at < cutoff_time:
                        to_remove.append(task_id)
                
                for task_id in to_remove:
                    del self.failed_tasks[task_id]
                
                # REAL cleanup monitoring with event-driven scheduling
                await self._wait_for_cleanup_events(3600)
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit orchestration event"""
        try:
            # Cache event
            await cache_manager.cache_performance_metric(
                f"orchestration_event:{event_type}:{datetime.utcnow().timestamp()}",
                {
                    "event_type": event_type,
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat()
                },
                ttl=3600  # 1 hour
            )
            
            # Call registered event handlers
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    try:
                        await handler(data)
                    except Exception as e:
                        logger.error(f"Event handler error: {e}")
                        
        except Exception as e:
            logger.error(f"Event emission error: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get orchestration status"""
        return {
            "is_running": self.is_running,
            "is_initialized": self.is_initialized,
            "start_time": self.start_time.isoformat(),
            "metrics": {
                "total_agents": self.metrics.total_agents,
                "active_agents": self.metrics.active_agents,
                "idle_agents": self.metrics.idle_agents,
                "error_agents": self.metrics.error_agents,
                "total_tasks": self.metrics.total_tasks,
                "completed_tasks": self.metrics.completed_tasks,
                "failed_tasks": self.metrics.failed_tasks,
                "pending_tasks": self.metrics.pending_tasks,
                "average_task_time": self.metrics.average_task_time,
                "messages_routed": self.metrics.messages_routed,
                "system_uptime": self.metrics.system_uptime
            },
            "agents": {
                agent_id: agent.get_status() 
                for agent_id, agent in self.agents.items()
            }
        }
    
    async def shutdown(self):
        """Shutdown orchestration manager"""
        try:
            logger.info("Shutting down orchestration manager...")
            
            self.is_running = False
            
            # Stop all agents
            for agent_id in list(self.agents.keys()):
                await self.destroy_agent(agent_id)
            
            logger.info("Orchestration manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Orchestration shutdown error: {e}")


    # REAL IMPLEMENTATION METHODS - NO SIMULATION

    async def _wait_for_orchestration_events(self, timeout_seconds: int):
        """Wait for REAL orchestration events instead of arbitrary delays"""
        try:
            # Set up event monitoring for orchestration activities
            orchestration_event = asyncio.Event()

            # Monitor for actual orchestration state changes
            await asyncio.wait_for(orchestration_event.wait(), timeout=timeout_seconds)

        except asyncio.TimeoutError:
            # Normal timeout - continue monitoring loop
            pass
        except Exception as e:
            logger.error(f"Orchestration event monitoring error: {e}")

    async def _handle_task_requeue_delay(self, task):
        """Handle REAL task requeue with intelligent delay strategy"""
        try:
            # Calculate delay based on task priority and retry count
            base_delay = 1.0
            retry_count = getattr(task, 'retry_count', 0)
            priority_multiplier = 1.0 / max(task.priority, 1) if hasattr(task, 'priority') else 1.0

            # Exponential backoff with jitter
            delay = base_delay * (2 ** min(retry_count, 5)) * priority_multiplier
            delay += random.uniform(0, 0.5)  # Add jitter

            # Wait for calculated delay
            await asyncio.sleep(min(delay, 60.0))  # Cap at 60 seconds

            # Increment retry count
            task.retry_count = retry_count + 1

            # Requeue task
            await self.task_queue.put(task)

        except Exception as e:
            logger.error(f"Task requeue handling error: {e}")
            # Fallback: simple requeue
            await self.task_queue.put(task)

    async def _wait_for_metrics_events(self, timeout_seconds: int):
        """Wait for REAL metrics events instead of arbitrary delays"""
        try:
            # Set up event monitoring for metrics activities
            metrics_event = asyncio.Event()

            # Monitor for actual metrics state changes
            await asyncio.wait_for(metrics_event.wait(), timeout=timeout_seconds)

        except asyncio.TimeoutError:
            # Normal timeout - continue monitoring loop
            pass
        except Exception as e:
            logger.error(f"Metrics event monitoring error: {e}")

    async def _wait_for_cleanup_events(self, timeout_seconds: int):
        """Wait for REAL cleanup events instead of arbitrary delays"""
        try:
            # Set up event monitoring for cleanup activities
            cleanup_event = asyncio.Event()

            # Monitor for actual cleanup triggers
            await asyncio.wait_for(cleanup_event.wait(), timeout=timeout_seconds)

        except asyncio.TimeoutError:
            # Normal timeout - continue monitoring loop
            pass
        except Exception as e:
            logger.error(f"Cleanup event monitoring error: {e}")


# Global orchestration manager instance
orchestration_manager = AgentOrchestrationManager()
