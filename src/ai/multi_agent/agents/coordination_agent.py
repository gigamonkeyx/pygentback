"""
Coordination Agent

Specialized agent for coordinating multi-agent workflows, task distribution,
and result aggregation in complex research operations.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Task definition for agent coordination"""
    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any]
    priority: int = 1
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class Agent:
    """Agent information for coordination"""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    status: str
    current_load: int = 0
    max_load: int = 5
    last_heartbeat: Optional[datetime] = None
    
    def is_available(self) -> bool:
        """Check if agent is available for new tasks"""
        return (
            self.status == "active" and 
            self.current_load < self.max_load and
            self.last_heartbeat and
            (datetime.utcnow() - self.last_heartbeat).total_seconds() < 60
        )
    
    def can_handle_task(self, task_type: str) -> bool:
        """Check if agent can handle specific task type"""
        return task_type in self.capabilities or "general" in self.capabilities


class CoordinationAgent:
    """
    Agent specialized in coordinating multi-agent workflows.
    
    Capabilities:
    - Task distribution and load balancing
    - Agent registration and management
    - Result aggregation and synthesis
    - Workflow orchestration
    - Performance monitoring
    """
    
    def __init__(self, agent_id: str = "coordination_agent"):
        self.agent_id = agent_id
        self.agent_type = "coordination"
        self.status = "initialized"
        self.capabilities = [
            "task_distribution",
            "agent_management",
            "result_aggregation",
            "workflow_orchestration",
            "performance_monitoring"
        ]
        
        # Coordination state
        self.registered_agents: Dict[str, Agent] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
        
        # Configuration
        self.config = {
            'max_concurrent_tasks': 20,
            'task_timeout_seconds': 300,
            'agent_heartbeat_interval': 30,
            'load_balancing_strategy': 'round_robin',  # round_robin, least_loaded, capability_based
            'retry_failed_tasks': True,
            'max_task_retries': 3
        }
        
        # Statistics
        self.stats = {
            'tasks_distributed': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'agents_managed': 0,
            'avg_task_completion_time_ms': 0.0,
            'coordination_sessions': 0
        }
        
        # Background tasks
        self._coordination_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        
        logger.info(f"CoordinationAgent {agent_id} initialized")
    
    async def start(self) -> bool:
        """Start the coordination agent"""
        try:
            self.status = "active"
            
            # Start background coordination tasks
            self._coordination_task = asyncio.create_task(self._coordination_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            logger.info(f"CoordinationAgent {self.agent_id} started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start CoordinationAgent {self.agent_id}: {e}")
            self.status = "error"
            return False
    
    async def stop(self) -> bool:
        """Stop the coordination agent"""
        try:
            self.status = "stopping"
            
            # Cancel background tasks
            if self._coordination_task:
                self._coordination_task.cancel()
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(
                self._coordination_task, self._heartbeat_task,
                return_exceptions=True
            )
            
            self.status = "stopped"
            logger.info(f"CoordinationAgent {self.agent_id} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop CoordinationAgent {self.agent_id}: {e}")
            return False
    
    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str]) -> bool:
        """Register an agent for coordination"""
        try:
            agent = Agent(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities,
                status="active",
                last_heartbeat=datetime.utcnow()
            )
            
            self.registered_agents[agent_id] = agent
            self.stats['agents_managed'] += 1
            
            logger.info(f"Registered agent {agent_id} with capabilities: {capabilities}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        try:
            if agent_id in self.registered_agents:
                # Cancel any tasks assigned to this agent
                for task in self.active_tasks.values():
                    if task.assigned_agent == agent_id:
                        task.status = TaskStatus.CANCELLED
                        task.error = f"Agent {agent_id} unregistered"
                        self._move_task_to_completed(task)
                
                del self.registered_agents[agent_id]
                logger.info(f"Unregistered agent {agent_id}")
                return True
            else:
                logger.warning(f"Agent {agent_id} not found for unregistration")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def coordinate_agents(self, agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Coordinate multiple agents for a complex task"""
        start_time = datetime.utcnow()
        
        try:
            coordination_id = f"coord_{int(start_time.timestamp())}"
            self.stats['coordination_sessions'] += 1
            
            # Register agents if not already registered
            for agent_info in agents:
                agent_id = agent_info.get('agent_id')
                if agent_id and agent_id not in self.registered_agents:
                    self.register_agent(
                        agent_id=agent_id,
                        agent_type=agent_info.get('agent_type', 'general'),
                        capabilities=agent_info.get('capabilities', ['general'])
                    )
            
            # Update agent heartbeats
            for agent_info in agents:
                agent_id = agent_info.get('agent_id')
                if agent_id in self.registered_agents:
                    self.registered_agents[agent_id].last_heartbeat = datetime.utcnow()
                    self.registered_agents[agent_id].status = "active"
            
            coordination_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = {
                'coordination_id': coordination_id,
                'success': True,
                'agents_coordinated': len(agents),
                'active_agents': len([a for a in self.registered_agents.values() if a.is_available()]),
                'coordination_time_ms': coordination_time,
                'timestamp': start_time.isoformat()
            }
            
            logger.debug(f"Coordinated {len(agents)} agents in {coordination_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Agent coordination failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agents_coordinated': 0,
                'timestamp': start_time.isoformat()
            }
    
    async def distribute_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[Task]]:
        """Distribute tasks among available agents"""
        try:
            task_objects = []
            
            # Convert task dictionaries to Task objects
            for i, task_data in enumerate(tasks):
                task = Task(
                    task_id=task_data.get('task_id', f"task_{i}_{int(datetime.utcnow().timestamp())}"),
                    task_type=task_data.get('task_type', 'general'),
                    description=task_data.get('description', 'No description'),
                    parameters=task_data.get('parameters', {}),
                    priority=task_data.get('priority', 1)
                )
                task_objects.append(task)
            
            # Add tasks to queue
            self.task_queue.extend(task_objects)
            
            # Distribute tasks using configured strategy
            distribution_result = await self._distribute_queued_tasks()
            
            self.stats['tasks_distributed'] += len(task_objects)
            
            logger.info(f"Distributed {len(task_objects)} tasks among agents")
            return distribution_result
            
        except Exception as e:
            logger.error(f"Task distribution failed: {e}")
            return {'error': str(e)}
    
    async def aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from multiple agents"""
        start_time = datetime.utcnow()
        
        try:
            aggregated = {
                'total_results': len(results),
                'successful_results': 0,
                'failed_results': 0,
                'aggregated_data': {},
                'summary': {},
                'metadata': {
                    'aggregation_time': start_time.isoformat(),
                    'coordinator_id': self.agent_id
                }
            }
            
            # Process each result
            for result in results:
                if result.get('success', False):
                    aggregated['successful_results'] += 1
                    
                    # Aggregate data by type
                    result_type = result.get('type', 'general')
                    if result_type not in aggregated['aggregated_data']:
                        aggregated['aggregated_data'][result_type] = []
                    
                    aggregated['aggregated_data'][result_type].append(result.get('data', {}))
                else:
                    aggregated['failed_results'] += 1
            
            # Generate summary
            aggregated['summary'] = {
                'success_rate': aggregated['successful_results'] / max(1, len(results)),
                'data_types': list(aggregated['aggregated_data'].keys()),
                'total_data_points': sum(len(data) for data in aggregated['aggregated_data'].values())
            }
            
            aggregation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            aggregated['metadata']['aggregation_time_ms'] = aggregation_time
            
            logger.debug(f"Aggregated {len(results)} results in {aggregation_time:.2f}ms")
            return aggregated
            
        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_results': len(results),
                'metadata': {'aggregation_time': start_time.isoformat()}
            }
    
    async def _coordination_loop(self):
        """Background coordination loop"""
        while self.status == "active":
            try:
                # Distribute queued tasks
                await self._distribute_queued_tasks()
                
                # Check task timeouts
                await self._check_task_timeouts()
                
                # Clean up completed tasks
                self._cleanup_old_tasks()
                
                # Wait before next iteration
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _heartbeat_loop(self):
        """Background heartbeat monitoring loop"""
        while self.status == "active":
            try:
                current_time = datetime.utcnow()
                
                # Check agent heartbeats
                for agent_id, agent in self.registered_agents.items():
                    if agent.last_heartbeat:
                        time_since_heartbeat = (current_time - agent.last_heartbeat).total_seconds()
                        if time_since_heartbeat > self.config['agent_heartbeat_interval'] * 2:
                            agent.status = "inactive"
                            logger.warning(f"Agent {agent_id} marked inactive due to missing heartbeat")
                
                await asyncio.sleep(self.config['agent_heartbeat_interval'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(10.0)
    
    async def _distribute_queued_tasks(self) -> Dict[str, List[Task]]:
        """Distribute tasks from queue to available agents"""
        distribution = {}
        
        while self.task_queue and len(self.active_tasks) < self.config['max_concurrent_tasks']:
            task = self.task_queue.pop(0)
            
            # Find suitable agent
            suitable_agent = self._find_suitable_agent(task)
            
            if suitable_agent:
                # Assign task to agent
                task.assigned_agent = suitable_agent.agent_id
                task.status = TaskStatus.ASSIGNED
                task.started_at = datetime.utcnow()
                
                # Update agent load
                suitable_agent.current_load += 1
                
                # Move to active tasks
                self.active_tasks[task.task_id] = task
                
                # Track distribution
                if suitable_agent.agent_id not in distribution:
                    distribution[suitable_agent.agent_id] = []
                distribution[suitable_agent.agent_id].append(task)
                
                logger.debug(f"Assigned task {task.task_id} to agent {suitable_agent.agent_id}")
            else:
                # No suitable agent available, put task back in queue
                self.task_queue.insert(0, task)
                break
        
        return distribution
    
    def _find_suitable_agent(self, task: Task) -> Optional[Agent]:
        """Find the most suitable agent for a task"""
        suitable_agents = [
            agent for agent in self.registered_agents.values()
            if agent.is_available() and agent.can_handle_task(task.task_type)
        ]
        
        if not suitable_agents:
            return None
        
        # Apply load balancing strategy
        if self.config['load_balancing_strategy'] == 'least_loaded':
            return min(suitable_agents, key=lambda a: a.current_load)
        elif self.config['load_balancing_strategy'] == 'capability_based':
            # Prefer agents with specific capabilities
            specific_agents = [a for a in suitable_agents if task.task_type in a.capabilities]
            return specific_agents[0] if specific_agents else suitable_agents[0]
        else:  # round_robin
            return suitable_agents[0]
    
    async def _check_task_timeouts(self):
        """Check for timed out tasks"""
        current_time = datetime.utcnow()
        timeout_seconds = self.config['task_timeout_seconds']
        
        timed_out_tasks = []
        for task in self.active_tasks.values():
            if task.started_at:
                elapsed = (current_time - task.started_at).total_seconds()
                if elapsed > timeout_seconds:
                    timed_out_tasks.append(task)
        
        for task in timed_out_tasks:
            task.status = TaskStatus.FAILED
            task.error = f"Task timed out after {timeout_seconds} seconds"
            task.completed_at = current_time
            
            # Reduce agent load
            if task.assigned_agent and task.assigned_agent in self.registered_agents:
                self.registered_agents[task.assigned_agent].current_load -= 1
            
            self._move_task_to_completed(task)
            logger.warning(f"Task {task.task_id} timed out")
    
    def _move_task_to_completed(self, task: Task):
        """Move task from active to completed"""
        if task.task_id in self.active_tasks:
            del self.active_tasks[task.task_id]
        
        self.completed_tasks[task.task_id] = task
        
        if task.status == TaskStatus.COMPLETED:
            self.stats['tasks_completed'] += 1
        elif task.status == TaskStatus.FAILED:
            self.stats['tasks_failed'] += 1
    
    def _cleanup_old_tasks(self):
        """Clean up old completed tasks"""
        current_time = datetime.utcnow()
        cutoff_time = current_time - datetime.timedelta(hours=1)  # Keep tasks for 1 hour
        
        old_task_ids = [
            task_id for task_id, task in self.completed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        
        for task_id in old_task_ids:
            del self.completed_tasks[task_id]
    
    def get_status(self) -> Dict[str, Any]:
        """Get coordination agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': self.status,
            'capabilities': self.capabilities,
            'registered_agents': len(self.registered_agents),
            'active_agents': len([a for a in self.registered_agents.values() if a.is_available()]),
            'active_tasks': len(self.active_tasks),
            'queued_tasks': len(self.task_queue),
            'completed_tasks': len(self.completed_tasks),
            'statistics': self.stats.copy(),
            'config': self.config.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'is_healthy': self.status == "active",
            'registered_agents': len(self.registered_agents),
            'active_agents': len([a for a in self.registered_agents.values() if a.is_available()]),
            'task_queue_size': len(self.task_queue),
            'active_tasks': len(self.active_tasks),
            'coordination_sessions': self.stats['coordination_sessions'],
            'last_check': datetime.utcnow().isoformat()
        }
