"""
Agent Manager and Pool

Management and pooling of agent instances for efficient resource utilization.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
from enum import Enum

from .base import ConfigurableAgent
from .registry import AgentFactory, AgentRegistry
from ..core import AgentStatus
from ..models import Task

logger = logging.getLogger(__name__)


class PoolStrategy(Enum):
    """Agent pool management strategies"""
    FIXED_SIZE = "fixed_size"
    DYNAMIC = "dynamic"
    ON_DEMAND = "on_demand"
    LOAD_BALANCED = "load_balanced"


class AgentPool:
    """
    Pool of agents for efficient resource management.
    """
    
    def __init__(self, pool_name: str, agent_type: str, factory: AgentFactory,
                 min_size: int = 1, max_size: int = 10, strategy: PoolStrategy = PoolStrategy.DYNAMIC):
        self.pool_name = pool_name
        self.agent_type = agent_type
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.strategy = strategy
        
        # Pool state
        self.agents: Dict[str, ConfigurableAgent] = {}
        self.available_agents: Set[str] = set()
        self.busy_agents: Set[str] = set()
        self.failed_agents: Set[str] = set()
        
        # Pool configuration
        self.config = {
            "idle_timeout_seconds": 300,  # 5 minutes
            "health_check_interval_seconds": 60,  # 1 minute
            "auto_scale_threshold": 0.8,  # Scale when 80% busy
            "scale_down_threshold": 0.3,  # Scale down when 30% busy
            "max_idle_agents": 3
        }
        
        # Statistics
        self.stats = {
            "agents_created": 0,
            "agents_destroyed": 0,
            "tasks_assigned": 0,
            "pool_utilization_history": []
        }
        
        # Pool management
        self.is_running = False
        self.management_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the agent pool"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Create initial agents
        await self._ensure_min_agents()
        
        # Start pool management task
        self.management_task = asyncio.create_task(self._pool_management_loop())
        
        logger.info(f"Started agent pool {self.pool_name} with {len(self.agents)} agents")
    
    async def stop(self):
        """Stop the agent pool"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel management task
        if self.management_task:
            self.management_task.cancel()
            try:
                await self.management_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown all agents
        for agent in list(self.agents.values()):
            await self._remove_agent(agent.agent_id)
        
        logger.info(f"Stopped agent pool {self.pool_name}")
    
    async def get_agent(self, timeout_seconds: float = 30.0) -> Optional[ConfigurableAgent]:
        """Get an available agent from the pool"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
            # Check for available agents
            if self.available_agents:
                agent_id = self.available_agents.pop()
                self.busy_agents.add(agent_id)
                agent = self.agents[agent_id]
                
                logger.debug(f"Assigned agent {agent_id} from pool {self.pool_name}")
                return agent
            
            # Try to create new agent if pool can grow
            if len(self.agents) < self.max_size:
                agent = await self._create_agent()
                if agent:
                    self.busy_agents.add(agent.agent_id)
                    logger.debug(f"Created and assigned new agent {agent.agent_id}")
                    return agent
            
            # Wait a bit before retrying
            await asyncio.sleep(0.1)
        
        logger.warning(f"No agents available in pool {self.pool_name} within timeout")
        return None
    
    async def return_agent(self, agent_id: str):
        """Return an agent to the pool"""
        if agent_id in self.busy_agents:
            self.busy_agents.remove(agent_id)
            
            # Check agent health before returning to available pool
            agent = self.agents.get(agent_id)
            if agent and agent.status == AgentStatus.IDLE:
                self.available_agents.add(agent_id)
                logger.debug(f"Returned agent {agent_id} to pool {self.pool_name}")
            else:
                # Agent is not healthy, remove it
                await self._remove_agent(agent_id)
                logger.warning(f"Removed unhealthy agent {agent_id} from pool {self.pool_name}")
    
    async def _ensure_min_agents(self):
        """Ensure minimum number of agents in pool"""
        while len(self.agents) < self.min_size:
            agent = await self._create_agent()
            if not agent:
                logger.error(f"Failed to create minimum agents for pool {self.pool_name}")
                break
    
    async def _create_agent(self) -> Optional[ConfigurableAgent]:
        """Create a new agent for the pool"""
        try:
            agent = self.factory.create_agent(self.agent_type)
            await agent.start()
            
            self.agents[agent.agent_id] = agent
            self.available_agents.add(agent.agent_id)
            self.stats["agents_created"] += 1
            
            logger.debug(f"Created agent {agent.agent_id} for pool {self.pool_name}")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent for pool {self.pool_name}: {e}")
            return None
    
    async def _remove_agent(self, agent_id: str):
        """Remove agent from pool"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Remove from all sets
            self.available_agents.discard(agent_id)
            self.busy_agents.discard(agent_id)
            self.failed_agents.discard(agent_id)
            
            # Shutdown agent
            try:
                await agent.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down agent {agent_id}: {e}")
            
            # Remove from pool
            del self.agents[agent_id]
            self.stats["agents_destroyed"] += 1
            
            logger.debug(f"Removed agent {agent_id} from pool {self.pool_name}")
    
    async def _pool_management_loop(self):
        """Main pool management loop"""
        while self.is_running:
            try:
                await self._manage_pool_size()
                await self._health_check_agents()
                await self._cleanup_idle_agents()
                await self._update_statistics()
                
                await asyncio.sleep(self.config["health_check_interval_seconds"])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pool management error for {self.pool_name}: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying
    
    async def _manage_pool_size(self):
        """Manage pool size based on strategy and load"""
        if self.strategy == PoolStrategy.FIXED_SIZE:
            return  # No dynamic sizing for fixed size pools
        
        total_agents = len(self.agents)
        busy_agents = len(self.busy_agents)
        
        if total_agents == 0:
            return
        
        utilization = busy_agents / total_agents
        
        # Scale up if utilization is high
        if (utilization > self.config["auto_scale_threshold"] and 
            total_agents < self.max_size):
            await self._create_agent()
            logger.info(f"Scaled up pool {self.pool_name} (utilization: {utilization:.2f})")
        
        # Scale down if utilization is low
        elif (utilization < self.config["scale_down_threshold"] and 
              total_agents > self.min_size and 
              len(self.available_agents) > self.config["max_idle_agents"]):
            # Remove one idle agent
            if self.available_agents:
                agent_id = self.available_agents.pop()
                await self._remove_agent(agent_id)
                logger.info(f"Scaled down pool {self.pool_name} (utilization: {utilization:.2f})")
    
    async def _health_check_agents(self):
        """Perform health checks on all agents"""
        unhealthy_agents = []
        
        for agent_id, agent in self.agents.items():
            if agent.status == AgentStatus.ERROR or agent.status == AgentStatus.OFFLINE:
                unhealthy_agents.append(agent_id)
            elif agent.status == AgentStatus.BUSY:
                # Check if agent has been busy too long
                if (agent.current_task and 
                    agent.current_task.started_at and
                    (datetime.utcnow() - agent.current_task.started_at).total_seconds() > 600):  # 10 minutes
                    logger.warning(f"Agent {agent_id} has been busy for too long")
                    unhealthy_agents.append(agent_id)
        
        # Remove unhealthy agents
        for agent_id in unhealthy_agents:
            self.failed_agents.add(agent_id)
            await self._remove_agent(agent_id)
    
    async def _cleanup_idle_agents(self):
        """Clean up agents that have been idle too long"""
        if len(self.available_agents) <= self.min_size:
            return  # Don't cleanup if at minimum size
        
        idle_timeout = timedelta(seconds=self.config["idle_timeout_seconds"])
        current_time = datetime.utcnow()
        
        agents_to_remove = []
        for agent_id in self.available_agents:
            agent = self.agents[agent_id]
            if (current_time - agent.metrics.last_activity) > idle_timeout:
                agents_to_remove.append(agent_id)
        
        # Remove idle agents (but keep minimum)
        for agent_id in agents_to_remove:
            if len(self.agents) > self.min_size:
                await self._remove_agent(agent_id)
                logger.debug(f"Removed idle agent {agent_id} from pool {self.pool_name}")
    
    async def _update_statistics(self):
        """Update pool statistics"""
        total_agents = len(self.agents)
        busy_agents = len(self.busy_agents)
        utilization = busy_agents / total_agents if total_agents > 0 else 0
        
        self.stats["pool_utilization_history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "total_agents": total_agents,
            "busy_agents": busy_agents,
            "available_agents": len(self.available_agents),
            "utilization": utilization
        })
        
        # Keep limited history
        if len(self.stats["pool_utilization_history"]) > 1000:
            self.stats["pool_utilization_history"] = self.stats["pool_utilization_history"][-500:]
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status"""
        total_agents = len(self.agents)
        return {
            "pool_name": self.pool_name,
            "agent_type": self.agent_type,
            "strategy": self.strategy.value,
            "total_agents": total_agents,
            "available_agents": len(self.available_agents),
            "busy_agents": len(self.busy_agents),
            "failed_agents": len(self.failed_agents),
            "utilization": len(self.busy_agents) / total_agents if total_agents > 0 else 0,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "is_running": self.is_running,
            "stats": self.stats.copy()
        }


class AgentManager:
    """
    High-level manager for multiple agent pools and individual agents.
    """
    
    def __init__(self, registry: AgentRegistry, factory: AgentFactory):
        self.registry = registry
        self.factory = factory
        
        # Pools and agents
        self.pools: Dict[str, AgentPool] = {}
        self.standalone_agents: Dict[str, ConfigurableAgent] = {}
        
        # Task assignment
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.assignment_callbacks: List[Callable] = []
        
        # Management state
        self.is_running = False
        self.assignment_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "total_tasks_assigned": 0,
            "assignment_failures": 0,
            "avg_assignment_time_ms": 0.0
        }
    
    async def start(self):
        """Start the agent manager"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start all pools
        for pool in self.pools.values():
            await pool.start()
        
        # Start task assignment loop
        self.assignment_task = asyncio.create_task(self._task_assignment_loop())
        
        logger.info("Agent manager started")
    
    async def stop(self):
        """Stop the agent manager"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel assignment task
        if self.assignment_task:
            self.assignment_task.cancel()
            try:
                await self.assignment_task
            except asyncio.CancelledError:
                pass
        
        # Stop all pools
        for pool in self.pools.values():
            await pool.stop()
        
        # Shutdown standalone agents
        for agent in self.standalone_agents.values():
            await agent.shutdown()
        
        logger.info("Agent manager stopped")
    
    def create_pool(self, pool_name: str, agent_type: str, min_size: int = 1, 
                   max_size: int = 10, strategy: PoolStrategy = PoolStrategy.DYNAMIC) -> AgentPool:
        """Create a new agent pool"""
        if pool_name in self.pools:
            raise ValueError(f"Pool {pool_name} already exists")
        
        pool = AgentPool(pool_name, agent_type, self.factory, min_size, max_size, strategy)
        self.pools[pool_name] = pool
        
        # Start pool if manager is running
        if self.is_running:
            asyncio.create_task(pool.start())
        
        logger.info(f"Created agent pool {pool_name}")
        return pool
    
    async def remove_pool(self, pool_name: str):
        """Remove an agent pool"""
        if pool_name in self.pools:
            pool = self.pools[pool_name]
            await pool.stop()
            del self.pools[pool_name]
            logger.info(f"Removed agent pool {pool_name}")
    
    async def assign_task(self, task: Task, preferred_pool: Optional[str] = None) -> bool:
        """Assign task to an appropriate agent"""
        try:
            start_time = datetime.utcnow()
            
            # Find suitable agent
            agent = await self._find_suitable_agent(task, preferred_pool)
            
            if not agent:
                logger.warning(f"No suitable agent found for task {task.task_id}")
                self.stats["assignment_failures"] += 1
                return False
            
            # Assign task to agent
            task.assigned_agent_id = agent.agent_id
            task.assigned_at = datetime.utcnow()
            
            # Send task to agent (simplified - in practice would use proper messaging)
            agent.current_task = task
            
            # Update statistics
            assignment_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_assignment_stats(assignment_time)
            
            # Call assignment callbacks
            for callback in self.assignment_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(task, agent)
                    else:
                        callback(task, agent)
                except Exception as e:
                    logger.warning(f"Assignment callback failed: {e}")
            
            logger.debug(f"Assigned task {task.task_id} to agent {agent.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Task assignment failed: {e}")
            self.stats["assignment_failures"] += 1
            return False
    
    async def _find_suitable_agent(self, task: Task, preferred_pool: Optional[str] = None) -> Optional[ConfigurableAgent]:
        """Find suitable agent for task"""
        # Try preferred pool first
        if preferred_pool and preferred_pool in self.pools:
            pool = self.pools[preferred_pool]
            agent = await pool.get_agent(timeout_seconds=5.0)
            if agent and self._agent_can_handle_task(agent, task):
                return agent
            elif agent:
                # Return agent to pool if not suitable
                await pool.return_agent(agent.agent_id)
        
        # Try other pools
        for pool_name, pool in self.pools.items():
            if pool_name == preferred_pool:
                continue  # Already tried
            
            # Check if pool has agents of suitable type
            if self._pool_can_handle_task(pool, task):
                agent = await pool.get_agent(timeout_seconds=1.0)
                if agent and self._agent_can_handle_task(agent, task):
                    return agent
                elif agent:
                    await pool.return_agent(agent.agent_id)
        
        # Try standalone agents
        for agent in self.standalone_agents.values():
            if (agent.status == AgentStatus.IDLE and 
                self._agent_can_handle_task(agent, task)):
                return agent
        
        return None
    
    def _pool_can_handle_task(self, pool: AgentPool, task: Task) -> bool:
        """Check if pool can handle task type"""
        # Get agent type capabilities from registry
        agent_config = self.registry.get_agent_config(pool.agent_type)
        if not agent_config:
            return False
        
        pool_capabilities = set(agent_config.get("default_capabilities", []))
        required_capabilities = set(task.required_capabilities)
        
        return required_capabilities.issubset(pool_capabilities)
    
    def _agent_can_handle_task(self, agent: ConfigurableAgent, task: Task) -> bool:
        """Check if agent can handle specific task"""
        for req_capability in task.required_capabilities:
            if not agent.has_capability(req_capability):
                return False
        return True
    
    async def _task_assignment_loop(self):
        """Main task assignment loop"""
        while self.is_running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Assign task
                success = await self.assign_task(task)
                
                if not success:
                    # Put task back in queue for retry (with limit)
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        await self.task_queue.put(task)
                
            except asyncio.TimeoutError:
                continue  # No tasks in queue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task assignment loop error: {e}")
    
    def _update_assignment_stats(self, assignment_time_ms: float):
        """Update assignment statistics"""
        self.stats["total_tasks_assigned"] += 1
        
        # Update average assignment time
        total_time = (self.stats["avg_assignment_time_ms"] * 
                     (self.stats["total_tasks_assigned"] - 1) + assignment_time_ms)
        self.stats["avg_assignment_time_ms"] = total_time / self.stats["total_tasks_assigned"]
    
    def add_assignment_callback(self, callback: Callable):
        """Add callback for task assignments"""
        self.assignment_callbacks.append(callback)
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get manager status"""
        pool_statuses = {name: pool.get_pool_status() for name, pool in self.pools.items()}
        
        return {
            "is_running": self.is_running,
            "total_pools": len(self.pools),
            "total_standalone_agents": len(self.standalone_agents),
            "task_queue_size": self.task_queue.qsize(),
            "stats": self.stats.copy(),
            "pools": pool_statuses
        }
