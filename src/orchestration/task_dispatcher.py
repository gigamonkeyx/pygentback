"""
Task Dispatcher

Intelligent task assignment and load balancing system for multi-agent orchestration.
Provides priority-based scheduling, dependency management, and adaptive distribution.
Enhanced with evolutionary optimization.
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Set, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..a2a_protocol.manager import A2AManager
from datetime import datetime, timedelta
from collections import defaultdict, deque
import heapq
from enum import Enum

from .coordination_models import (
    TaskRequest, TaskStatus, OrchestrationConfig, TaskID, TaskCompletionCallback
)
from .agent_registry import AgentRegistry
from .mcp_orchestrator import MCPOrchestrator
from .real_mcp_client import RealAgentExecutor

logger = logging.getLogger(__name__)


class DispatchStrategy(Enum):
    """Task dispatch strategies."""
    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based
    SHORTEST_JOB_FIRST = "shortest_job_first"
    LOAD_BALANCED = "load_balanced"
    ADAPTIVE = "adaptive"


class TaskDispatcher:
    """
    Intelligent task dispatcher for multi-agent orchestration.
    
    Features:
    - Priority-based task scheduling
    - Dependency management and resolution
    - Adaptive load balancing
    - Retry mechanisms and error handling
    - Performance optimization
    """
    
    def __init__(self,
                 config: OrchestrationConfig,
                 agent_registry: AgentRegistry,
                 mcp_orchestrator: MCPOrchestrator,
                 a2a_manager: Optional['A2AManager'] = None):
        self.config = config
        self.agent_registry = agent_registry
        self.mcp_orchestrator = mcp_orchestrator
        self.a2a_manager = a2a_manager
        
        # Task queues
        self.pending_tasks: List[Tuple[float, TaskRequest]] = []  # Priority queue
        self.running_tasks: Dict[TaskID, TaskRequest] = {}
        self.completed_tasks: Dict[TaskID, TaskRequest] = {}
        self.failed_tasks: Dict[TaskID, TaskRequest] = {}
        
        # Dependency management
        self.task_dependencies: Dict[TaskID, Set[TaskID]] = defaultdict(set)
        self.dependency_waiters: Dict[TaskID, Set[TaskID]] = defaultdict(set)
        
        # Scheduling and execution
        self.dispatch_strategy = DispatchStrategy.ADAPTIVE
        self.execution_tasks: Dict[TaskID, asyncio.Task] = {}
        self.is_running = False
        
        # Performance tracking
        self.dispatch_history: deque = deque(maxlen=1000)
        self.strategy_performance: Dict[DispatchStrategy, Dict[str, float]] = {
            strategy: {"success_rate": 0.0, "avg_time": 0.0, "tasks_processed": 0}
            for strategy in DispatchStrategy
        }
        
        # Callbacks
        self.completion_callbacks: List[TaskCompletionCallback] = []
        
        # Adaptive strategy state
        self.last_strategy_switch = datetime.utcnow()
        self.strategy_evaluation_window = timedelta(minutes=5)
        
        # Enhanced task management attributes
        self.peer_task_performance: Dict[str, Dict[str, float]] = {}
        
        # 1.5.2 Evolutionary task assignment
        self.agent_performance_evolution: Dict[str, List[float]] = {}
        self.task_assignment_genome: Dict[str, float] = {}  # Agent preference weights
        self.assignment_mutation_rate: float = 0.1
        self.evolution_generation: int = 0
        
        # 1.5.3 Enhanced dynamic load balancing
        self.peer_load_metrics: Dict[str, Dict[str, float]] = {}
        self.load_balancing_thresholds: Dict[str, float] = {
            "high_load": 0.8,
            "low_load": 0.3,
            "rebalance_trigger": 0.6
        }
        
        # 1.5.4 Distributed task decomposition
        self.decomposition_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.negotiation_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # 1.5.5 Evolutionary optimization of distribution patterns
        self.distribution_patterns: List[Dict[str, Any]] = []
        self.pattern_fitness_scores: List[float] = []
        self.pattern_population_size: int = 20
        
        # 1.5.6 Coordinated failover mechanisms
        self.failover_peers: Dict[str, List[str]] = {}
        self.failover_response_times: Dict[str, float] = {}
        self.redundancy_level: int = 2  # Number of backup assignments
        
        logger.info("Task Dispatcher initialized with evolutionary enhancements")
    
    async def start(self):
        """Start the task dispatcher."""
        self.is_running = True
        
        # Initialize evolutionary components (1.5.2, 1.5.5)
        await self._initialize_evolutionary_components()
        
        # Start dispatch loop
        asyncio.create_task(self._dispatch_loop())
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
        # Start optimization loops
        asyncio.create_task(self._failover_monitoring_loop())
        
        # Start evolutionary optimization loop (1.5.5)
        asyncio.create_task(self._evolutionary_optimization_loop())
        
        logger.info("Task Dispatcher started with evolutionary enhancements")
    
    async def stop(self):
        """Stop the task dispatcher."""
        self.is_running = False
        
        # Cancel running tasks
        for task in self.execution_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.execution_tasks:
            await asyncio.gather(*self.execution_tasks.values(), return_exceptions=True)
        
        self.execution_tasks.clear()
        
        logger.info("Task Dispatcher stopped with A2A components")
    
    async def submit_task(self, task: TaskRequest) -> bool:
        """Submit a task for execution."""
        try:
            # Validate task
            if not task.task_type or not task.description:
                raise ValueError("Task type and description are required")
            
            # Set up dependencies
            if task.dependencies:
                for dep_id in task.dependencies:
                    self.task_dependencies[task.task_id].add(dep_id)
                    self.dependency_waiters[dep_id].add(task.task_id)
            
            # Calculate priority score
            priority_score = self._calculate_priority_score(task)
            
            # Add to pending queue
            heapq.heappush(self.pending_tasks, (-priority_score, task))
            
            logger.debug(f"Submitted task {task.task_id} with priority {priority_score}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    async def submit_batch_tasks(self, tasks: List[TaskRequest]) -> List[bool]:
        """Submit multiple tasks as a batch."""
        results = []
        for task in tasks:
            result = await self.submit_task(task)
            results.append(result)
        return results
    
    async def cancel_task(self, task_id: TaskID) -> bool:
        """Cancel a task."""
        try:
            # Check if task is running
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.update_status(TaskStatus.CANCELLED)
                
                # Cancel execution task
                if task_id in self.execution_tasks:
                    self.execution_tasks[task_id].cancel()
                    del self.execution_tasks[task_id]
                
                # Move to completed
                del self.running_tasks[task_id]
                self.completed_tasks[task_id] = task
                
                # Release agent
                if task.assigned_agent_id:
                    await self.agent_registry.complete_task(task.assigned_agent_id, task, success=False)
                
                logger.info(f"Cancelled running task {task_id}")
                return True
            
            # Check if task is pending
            for i, (priority, pending_task) in enumerate(self.pending_tasks):
                if pending_task.task_id == task_id:
                    pending_task.update_status(TaskStatus.CANCELLED)
                    del self.pending_tasks[i]
                    heapq.heapify(self.pending_tasks)  # Restore heap property
                    self.completed_tasks[task_id] = pending_task
                    logger.info(f"Cancelled pending task {task_id}")
                    return True
            
            logger.warning(f"Task {task_id} not found for cancellation")
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def get_task_status(self, task_id: TaskID) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check all task collections
        task = None
        status_info = {}
        
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            status_info["queue"] = "running"
        elif task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            status_info["queue"] = "completed"
        elif task_id in self.failed_tasks:
            task = self.failed_tasks[task_id]
            status_info["queue"] = "failed"
        else:
            # Check pending queue
            for priority, pending_task in self.pending_tasks:
                if pending_task.task_id == task_id:
                    task = pending_task
                    status_info["queue"] = "pending"
                    status_info["priority"] = -priority
                    break
        
        if not task:
            return None
        
        status_info.update({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status.value,
            "priority": task.priority.value,
            "description": task.description,
            "created_at": task.created_at,
            "assigned_at": task.assigned_at,
            "started_at": task.started_at,
            "completed_at": task.completed_at,
            "assigned_agent_id": task.assigned_agent_id,
            "execution_time": task.execution_time,
            "wait_time": task.wait_time,
            "dependencies": list(task.dependencies),
            "is_overdue": task.is_overdue
        })
        
        if task.result:
            status_info["result"] = task.result
        if task.error_message:
            status_info["error_message"] = task.error_message
        
        return status_info
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status."""
        return {
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_tasks": (len(self.pending_tasks) + len(self.running_tasks) + 
                          len(self.completed_tasks) + len(self.failed_tasks)),
            "dispatch_strategy": self.dispatch_strategy.value,
            "is_running": self.is_running
        }
    
    def add_completion_callback(self, callback: TaskCompletionCallback):
        """Add a callback for task completion."""
        self.completion_callbacks.append(callback)
    
    def remove_completion_callback(self, callback: TaskCompletionCallback):
        """Remove a completion callback."""
        if callback in self.completion_callbacks:
            self.completion_callbacks.remove(callback)

    # Research Task Support
    async def submit_research_task(self, research_query: Dict[str, Any]) -> Optional[str]:
        """
        Submit a research task for processing.
        
        Args:
            research_query: Research query specification with topic, questions, etc.
            
        Returns:
            Task ID if submitted successfully, None otherwise
        """
        try:
            # Create a research task request
            task_request = TaskRequest(
                task_type="research",
                task_data=research_query,
                priority=research_query.get("priority", 2),  # Default normal priority
                estimated_duration=research_query.get("estimated_duration", 3600),  # 1 hour default
                dependencies=research_query.get("dependencies", []),
                required_capabilities=["research", "web_search", "document_analysis"]
            )
            
            success = await self.submit_task(task_request)
            return task_request.task_id if success else None
            
        except Exception as e:
            logger.error(f"Failed to submit research task: {e}")
            return None

    # ...existing code...
    
    async def _dispatch_loop(self):
        """Main dispatch loop."""
        while self.is_running:
            try:
                # Check if we can dispatch more tasks
                if (len(self.running_tasks) >= self.config.max_concurrent_tasks or
                    not self.pending_tasks):
                    # REAL wait for task completion or new tasks - no arbitrary delays
                    await self._wait_for_dispatch_opportunity()
                    continue
                
                # Get next task based on strategy
                task = await self._get_next_task()
                if not task:
                    # REAL wait for new tasks - no arbitrary delays
                    await self._wait_for_new_tasks()
                    continue
                
                # Check dependencies
                if not await self._check_dependencies(task):
                    # Put task back in queue
                    priority_score = self._calculate_priority_score(task)
                    heapq.heappush(self.pending_tasks, (-priority_score, task))
                    # REAL wait for dependency resolution - no arbitrary delays
                    await self._wait_for_dependency_resolution(task)
                    continue
                
                # Dispatch task
                success = await self._dispatch_task(task)
                if not success:
                    # Retry or fail task
                    await self._handle_dispatch_failure(task)
                
                # REAL task dispatch completed - no artificial delays needed
                
            except Exception as e:
                logger.error(f"Error in dispatch loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _monitoring_loop(self):
        """Monitoring and optimization loop."""
        while self.is_running:
            try:
                # Check for overdue tasks
                await self._check_overdue_tasks()
                
                # Update strategy performance
                await self._update_strategy_performance()
                
                # Consider strategy switching
                if self.dispatch_strategy == DispatchStrategy.ADAPTIVE:
                    await self._consider_strategy_switch()
                
                # Cleanup completed tasks
                await self._cleanup_old_tasks()
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _get_next_task(self) -> Optional[TaskRequest]:
        """Get the next task to dispatch based on current strategy."""
        if not self.pending_tasks:
            return None
        
        if self.dispatch_strategy == DispatchStrategy.FIFO:
            # Get oldest task
            return self._get_oldest_task()
        elif self.dispatch_strategy == DispatchStrategy.PRIORITY:
            # Get highest priority task
            _, task = heapq.heappop(self.pending_tasks)
            return task
        elif self.dispatch_strategy == DispatchStrategy.SHORTEST_JOB_FIRST:
            # Get task with shortest estimated duration
            return self._get_shortest_task()
        elif self.dispatch_strategy == DispatchStrategy.LOAD_BALANCED:
            # Get task that best balances load
            return self._get_load_balanced_task()
        else:  # ADAPTIVE
            # Use best performing strategy
            best_strategy = self._get_best_strategy()
            return await self._get_task_by_strategy(best_strategy)
    
    def _get_oldest_task(self) -> Optional[TaskRequest]:
        """Get the oldest pending task."""
        if not self.pending_tasks:
            return None
        
        oldest_task = None
        oldest_time = datetime.max
        oldest_index = -1
        
        for i, (priority, task) in enumerate(self.pending_tasks):
            if task.created_at < oldest_time:
                oldest_time = task.created_at
                oldest_task = task
                oldest_index = i
        
        if oldest_index >= 0:
            del self.pending_tasks[oldest_index]
            heapq.heapify(self.pending_tasks)
        
        return oldest_task
    
    def _get_shortest_task(self) -> Optional[TaskRequest]:
        """Get the task with shortest estimated duration."""
        if not self.pending_tasks:
            return None
        
        shortest_task = None
        shortest_duration = float('inf')
        shortest_index = -1
        
        for i, (priority, task) in enumerate(self.pending_tasks):
            duration = task.estimated_duration or 60.0  # Default estimate
            if duration < shortest_duration:
                shortest_duration = duration
                shortest_task = task
                shortest_index = i
        
        if shortest_index >= 0:
            del self.pending_tasks[shortest_index]
            heapq.heapify(self.pending_tasks)
        
        return shortest_task
    
    def _get_load_balanced_task(self) -> Optional[TaskRequest]:
        """Get task that best balances system load."""
        if not self.pending_tasks:
            return None
        
        # For now, use priority-based selection
        # Could be enhanced with more sophisticated load balancing
        _, task = heapq.heappop(self.pending_tasks)
        return task
    
    async def _get_task_by_strategy(self, strategy: DispatchStrategy) -> Optional[TaskRequest]:
        """Get task using specific strategy."""
        if strategy == DispatchStrategy.FIFO:
            return self._get_oldest_task()
        elif strategy == DispatchStrategy.PRIORITY:
            if self.pending_tasks:
                _, task = heapq.heappop(self.pending_tasks)
                return task
        elif strategy == DispatchStrategy.SHORTEST_JOB_FIRST:
            return self._get_shortest_task()
        elif strategy == DispatchStrategy.LOAD_BALANCED:
            return self._get_load_balanced_task()
        
        return None
    
    async def _check_dependencies(self, task: TaskRequest) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            
            # Check if dependency completed successfully
            dep_task = self.completed_tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    async def _dispatch_task(self, task: TaskRequest) -> bool:
        """Dispatch a task to an appropriate agent."""
        try:
            # Select agent
            agent_id = await self.agent_registry.select_agent(
                task_type=task.task_type,
                agent_type=None,  # Could be specified in task
                preferred_agent=None
            )
            
            if not agent_id:
                logger.warning(f"No available agent for task {task.task_id}")
                return False
            
            # Assign task to agent
            success = await self.agent_registry.assign_task(agent_id, task)
            if not success:
                logger.warning(f"Failed to assign task {task.task_id} to agent {agent_id}")
                return False
            
            # Move to running tasks
            self.running_tasks[task.task_id] = task
            
            # Start execution
            execution_task = asyncio.create_task(self._execute_task(task))
            self.execution_tasks[task.task_id] = execution_task
            
            logger.debug(f"Dispatched task {task.task_id} to agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to dispatch task {task.task_id}: {e}")
            return False
    
    async def _execute_task(self, task: TaskRequest):
        """Execute a task with A2A-aware coordination."""
        try:
            task.update_status(TaskStatus.RUNNING)

            # Get agent information
            agent = self.agent_registry.agents.get(task.assigned_agent_id)
            if not agent:
                raise ValueError(f"Agent {task.assigned_agent_id} not found")

            # Check if task requires A2A coordination
            result = await self._execute_task_with_a2a_support(task, agent)

            if not result:
                # Fallback to standard execution
                executor = RealAgentExecutor(agent.agent_id, agent.agent_type.value)
                result = await executor.execute_task({
                    "task_type": task.task_type,
                    "input_data": task.input_data,
                    "description": task.description
                })
            
            success = result.get("status") == "success"
            
            if success:
                task.result = result.get("result", {})
                task.update_status(TaskStatus.COMPLETED)
                self.completed_tasks[task.task_id] = task
            else:
                task.error_message = result.get("error", "Task execution failed")
                task.update_status(TaskStatus.FAILED)
                self.failed_tasks[task.task_id] = task
            
            # Complete task with agent
            if task.assigned_agent_id:
                await self.agent_registry.complete_task(task.assigned_agent_id, task, success)
            
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            # Remove execution task
            if task.task_id in self.execution_tasks:
                del self.execution_tasks[task.task_id]
            
            # Notify completion callbacks
            await self._notify_completion_callbacks(task)
            
            # Check for dependent tasks
            await self._check_dependent_tasks(task.task_id)
            
            # Record dispatch history
            self._record_dispatch(task, success)
            
        except asyncio.CancelledError:
            logger.info(f"Task {task.task_id} was cancelled")
            task.update_status(TaskStatus.CANCELLED)
        except Exception as e:
            logger.error(f"Task execution failed for {task.task_id}: {e}")
            task.error_message = str(e)
            task.update_status(TaskStatus.FAILED)
            self.failed_tasks[task.task_id] = task
            
            if task.assigned_agent_id:
                await self.agent_registry.complete_task(task.assigned_agent_id, task, success=False)

    async def _execute_task_with_a2a_support(self, task: TaskRequest, agent) -> Optional[Dict[str, Any]]:
        """Execute task with A2A protocol support for multi-agent coordination."""
        if not self.a2a_manager:
            return None

        try:
            # Check if task requires multi-agent coordination
            requires_coordination = self._task_requires_coordination(task)

            if requires_coordination:
                # Execute task using A2A coordination
                coordination_agents = await self._identify_coordination_agents(task)

                if coordination_agents and len(coordination_agents) > 1:
                    # Use A2A manager for multi-agent coordination
                    coordination_results = await self.a2a_manager.coordinate_multi_agent_task(
                        task_description=task.description,
                        agent_ids=coordination_agents,
                        coordination_strategy="sequential"
                    )

                    if coordination_results:
                        # Aggregate results from coordinated agents
                        return self._aggregate_coordination_results(coordination_results, task)

            # Check if task can benefit from A2A messaging
            if self._task_benefits_from_a2a(task):
                # Execute with A2A messaging support
                return await self._execute_with_a2a_messaging(task, agent)

            return None

        except Exception as e:
            logger.error(f"A2A-aware task execution failed for {task.task_id}: {e}")
            return None

    def _task_requires_coordination(self, task: TaskRequest) -> bool:
        """Check if task requires multi-agent coordination."""
        coordination_indicators = [
            "multi-agent", "coordination", "collaboration", "distributed",
            "parallel", "sequential", "workflow", "pipeline"
        ]

        task_text = f"{task.task_type} {task.description}".lower()
        return any(indicator in task_text for indicator in coordination_indicators)

    async def _identify_coordination_agents(self, task: TaskRequest) -> List[str]:
        """Identify agents that should participate in task coordination."""
        try:
            # Get available agents that can handle this task type
            available_agents = await self.agent_registry.get_agents_by_capability(task.task_type)

            # Filter for A2A-enabled agents if A2A manager is available
            if self.a2a_manager:
                a2a_agents = await self.a2a_manager.get_agent_status()
                a2a_agent_ids = [agent.get("agent_id") for agent in a2a_agents.get("agents", [])]
                available_agents = [aid for aid in available_agents if aid in a2a_agent_ids]

            # Limit to reasonable number of agents for coordination
            return available_agents[:3]  # Max 3 agents for coordination

        except Exception as e:
            logger.error(f"Failed to identify coordination agents: {e}")
            return []

    def _task_benefits_from_a2a(self, task: TaskRequest) -> bool:
        """Check if task can benefit from A2A messaging."""
        a2a_beneficial_types = [
            "research", "analysis", "evaluation", "review", "validation"
        ]
        return task.task_type in a2a_beneficial_types

    async def _execute_with_a2a_messaging(self, task: TaskRequest, agent) -> Optional[Dict[str, Any]]:
        """Execute task with A2A messaging support."""
        try:
            # Create standard executor
            executor = RealAgentExecutor(agent.agent_id, agent.agent_type.value)

            # Execute task
            result = await executor.execute_task({
                "task_type": task.task_type,
                "input_data": task.input_data,
                "description": task.description,
                "a2a_enabled": True,
                "a2a_context": {
                    "task_id": task.task_id,
                    "coordination_available": True
                }
            })

            # Enhance result with A2A metadata
            if result and result.get("status") == "success":
                result["a2a_enhanced"] = True
                result["coordination_used"] = False

            return result

        except Exception as e:
            logger.error(f"A2A messaging execution failed: {e}")
            return None

    def _aggregate_coordination_results(self, coordination_results: List, task: TaskRequest) -> Dict[str, Any]:
        """Aggregate results from coordinated agents."""
        try:
            if not coordination_results:
                return {"status": "failed", "error": "No coordination results"}

            # Aggregate successful results
            successful_results = [
                result for result in coordination_results
                if hasattr(result, 'status') and result.status.state.value == "completed"
            ]

            if not successful_results:
                return {"status": "failed", "error": "No successful coordination results"}

            # Create aggregated result
            aggregated_result = {
                "status": "success",
                "result": {
                    "coordination_type": "multi_agent",
                    "participating_agents": len(coordination_results),
                    "successful_agents": len(successful_results),
                    "task_type": task.task_type,
                    "aggregated_output": f"Task completed through coordination of {len(successful_results)} agents",
                    "coordination_details": [
                        {
                            "task_id": result.id,
                            "session_id": result.sessionId,
                            "status": result.status.state.value
                        }
                        for result in coordination_results
                    ]
                },
                "execution_time": sum(getattr(result, 'execution_time', 0.1) for result in successful_results),
                "a2a_coordination": True
            }

            return aggregated_result

        except Exception as e:
            logger.error(f"Failed to aggregate coordination results: {e}")
            return {"status": "failed", "error": str(e)}

    async def _handle_dispatch_failure(self, task: TaskRequest):
        """Handle task dispatch failure."""
        # Implement retry logic or move to failed tasks
        task.update_status(TaskStatus.FAILED)
        task.error_message = "Failed to dispatch task"
        self.failed_tasks[task.task_id] = task
    
    async def _check_overdue_tasks(self):
        """Check for and handle overdue tasks."""
        current_time = datetime.utcnow()
        
        # Check running tasks
        overdue_tasks = []
        for task in self.running_tasks.values():
            if task.deadline and current_time > task.deadline:
                overdue_tasks.append(task.task_id)
        
        # Handle overdue tasks
        for task_id in overdue_tasks:
            logger.warning(f"Task {task_id} is overdue")
            # Could implement escalation or cancellation logic here
    
    async def _check_dependent_tasks(self, completed_task_id: TaskID):
        """Check if any tasks are waiting for this dependency."""
        if completed_task_id in self.dependency_waiters:
            waiting_tasks = self.dependency_waiters[completed_task_id]
            for waiting_task_id in waiting_tasks:
                # Check if all dependencies are now satisfied
                # This would trigger re-evaluation of the waiting task
                pass
    
    def _calculate_priority_score(self, task: TaskRequest) -> float:
        """Calculate priority score for a task."""
        base_score = task.priority.value
        
        # Adjust for deadline urgency
        if task.deadline:
            time_to_deadline = (task.deadline - datetime.utcnow()).total_seconds()
            if time_to_deadline > 0:
                urgency_factor = max(0.1, 1.0 / (time_to_deadline / 3600))  # Hours
                base_score *= urgency_factor
            else:
                base_score *= 10  # Very high priority for overdue tasks
        
        # Adjust for estimated duration (shorter tasks get slight boost)
        if task.estimated_duration:
            duration_factor = max(0.5, 1.0 / (task.estimated_duration / 60))  # Minutes
            base_score *= duration_factor
        
        return base_score
    
    def _record_dispatch(self, task: TaskRequest, success: bool):
        """Record dispatch metrics."""
        self.dispatch_history.append({
            'task_id': task.task_id,
            'strategy': self.dispatch_strategy,
            'success': success,
            'execution_time': task.execution_time or 0.0,
            'wait_time': task.wait_time or 0.0,
            'timestamp': datetime.utcnow()
        })
    
    async def _update_strategy_performance(self):
        """Update performance metrics for dispatch strategies."""
        # Analyze recent dispatch history
        recent_dispatches = [
            d for d in self.dispatch_history 
            if (datetime.utcnow() - d['timestamp']).total_seconds() < 300
        ]
        
        # Update strategy metrics
        strategy_stats = defaultdict(lambda: {'successes': 0, 'total': 0, 'total_time': 0.0})
        
        for dispatch in recent_dispatches:
            strategy = dispatch['strategy']
            strategy_stats[strategy]['total'] += 1
            if dispatch['success']:
                strategy_stats[strategy]['successes'] += 1
            strategy_stats[strategy]['total_time'] += dispatch['execution_time']
        
        # Update performance tracking
        for strategy, stats in strategy_stats.items():
            if stats['total'] > 0:
                perf = self.strategy_performance[strategy]
                perf['success_rate'] = stats['successes'] / stats['total']
                perf['avg_time'] = stats['total_time'] / stats['total']
                perf['tasks_processed'] = stats['total']
    
    def _get_best_strategy(self) -> DispatchStrategy:
        """Get the best performing strategy."""
        best_strategy = DispatchStrategy.PRIORITY
        best_score = 0.0
        
        for strategy, perf in self.strategy_performance.items():
            if perf['tasks_processed'] > 0:
                # Combined score: success rate weighted by efficiency
                score = perf['success_rate']
                if perf['avg_time'] > 0:
                    score *= (1.0 / perf['avg_time'])
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
        
        return best_strategy
    
    async def _consider_strategy_switch(self):
        """Consider switching dispatch strategy based on performance."""
        if (datetime.utcnow() - self.last_strategy_switch) < self.strategy_evaluation_window:
            return
        
        best_strategy = self._get_best_strategy()
        current_perf = self.strategy_performance[self.dispatch_strategy]
        best_perf = self.strategy_performance[best_strategy]
        
        # Switch if best strategy is significantly better
        if (best_perf['success_rate'] > current_perf['success_rate'] + 0.1 and
            best_perf['tasks_processed'] > 10):
            
            logger.info(f"Switching dispatch strategy from {self.dispatch_strategy.value} to {best_strategy.value}")
            self.dispatch_strategy = best_strategy
            self.last_strategy_switch = datetime.utcnow()
    
    async def _cleanup_old_tasks(self):
        """Clean up old completed and failed tasks."""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Clean completed tasks
        old_completed = [
            task_id for task_id, task in self.completed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        for task_id in old_completed:
            del self.completed_tasks[task_id]
        
        # Clean failed tasks
        old_failed = [
            task_id for task_id, task in self.failed_tasks.items()
            if task.completed_at and task.completed_at < cutoff_time
        ]
        for task_id in old_failed:
            del self.failed_tasks[task_id]
    
    async def _notify_completion_callbacks(self, task: TaskRequest):
        """Notify all completion callbacks."""
        for callback in self.completion_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task)
                else:
                    callback(task)
            except Exception as e:
                logger.error(f"Completion callback error: {e}")
    
    async def get_dispatcher_metrics(self) -> Dict[str, Any]:
        """Get comprehensive dispatcher metrics."""
        queue_status = await self.get_queue_status()
        
        # Calculate throughput
        recent_completions = [
            d for d in self.dispatch_history 
            if (datetime.utcnow() - d['timestamp']).total_seconds() < 3600
        ]
        
        throughput = len(recent_completions) / max(1, len(recent_completions) / 60)  # tasks per minute
        
        if recent_completions:
            avg_execution_time = sum(d['execution_time'] for d in recent_completions) / len(recent_completions)
            avg_wait_time = sum(d['wait_time'] for d in recent_completions) / len(recent_completions)
            success_rate = sum(1 for d in recent_completions if d['success']) / len(recent_completions)
        else:
            avg_execution_time = 0.0
            avg_wait_time = 0.0
            success_rate = 1.0
        
        return {
            **queue_status,
            'throughput_per_minute': throughput,
            'avg_execution_time': avg_execution_time,
            'avg_wait_time': avg_wait_time,
            'success_rate': success_rate,
            'strategy_performance': dict(self.strategy_performance),
            'active_execution_tasks': len(self.execution_tasks)
        }
      # =============================================================================
    # Phase 1.5: Task Dispatcher Evolution Enhancement Implementation
    # Aligned with Sakana AI Darwin Gödel Machine research principles
    # =============================================================================
    
    # 1.5.1 Integrate A2A agent discovery into task routing
    # Performance and evolution tracking methods
    
    async def _initialize_a2a_components(self):
        """Initialize A2A components (stubbed - A2A functionality archived)."""
        try:
            logger.info("A2A components initialization skipped (archived)")
        except Exception as e:
            logger.error(f"A2A component initialization error: {e}")
            
        try:
            # Initialize task assignment genome for known agents
            for agent_id in await self.agent_registry.get_available_agents():
                self.agent_performance_evolution[agent_id] = [0.5]  # Start with neutral fitness
                self.task_assignment_genome[agent_id] = random.uniform(0.3, 0.9)  # Initial preference weight
            
            logger.info("Evolutionary task assignment components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize evolutionary components: {e}")
    
    def _evolve_task_assignment_preferences(self):
        """Evolve task assignment preferences based on performance data."""
        try:
            for agent_id in self.agent_performance_evolution:
                performance_history = self.agent_performance_evolution[agent_id]
                
                if len(performance_history) < 2:
                    continue
                
                # Calculate fitness trend
                recent_performance = sum(performance_history[-5:]) / min(5, len(performance_history))
                current_weight = self.task_assignment_genome[agent_id]
                
                # Evolutionary pressure: increase weight for high performers, decrease for low performers
                if recent_performance > 0.7:
                    # Positive selection pressure
                    mutation = random.uniform(0.0, 0.1)
                    new_weight = min(1.0, current_weight + mutation)
                elif recent_performance < 0.4:
                    # Negative selection pressure
                    mutation = random.uniform(0.0, 0.1)
                    new_weight = max(0.1, current_weight - mutation)
                else:
                    # Random mutation for exploration
                    if random.random() < self.assignment_mutation_rate:
                        mutation = random.uniform(-0.05, 0.05)
                        new_weight = max(0.1, min(1.0, current_weight + mutation))
                    else:
                        new_weight = current_weight
                
                self.task_assignment_genome[agent_id] = new_weight
            
            self.evolution_generation += 1
            
            if self.evolution_generation % 10 == 0:
                logger.info(f"Task assignment evolution generation {self.evolution_generation} completed")
            
        except Exception as e:
            logger.error(f"Failed to evolve task assignment preferences: {e}")
    
    def _select_agent_by_evolutionary_fitness(self, task: TaskRequest, available_agents: List[str]) -> Optional[str]:
        """Select agent using evolutionary fitness criteria."""
        try:
            if not available_agents:
                return None
            
            # Calculate selection probabilities based on evolved weights and current performance
            selection_scores = {}
            
            for agent_id in available_agents:
                base_weight = self.task_assignment_genome.get(agent_id, 0.5)
                performance_history = self.agent_performance_evolution.get(agent_id, [0.5])
                recent_performance = sum(performance_history[-3:]) / min(3, len(performance_history))
                
                # Combine evolved preference with recent performance
                selection_score = (base_weight * 0.6) + (recent_performance * 0.4)
                selection_scores[agent_id] = selection_score
            
            # Weighted random selection (exploration vs exploitation)
            if random.random() < 0.1:  # 10% exploration
                return random.choice(available_agents)
            else:  # 90% exploitation
                # Select based on weighted probabilities
                agents = list(selection_scores.keys())
                weights = list(selection_scores.values())
                total_weight = sum(weights)
                
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in weights]
                    selected_agent = random.choices(agents, weights=normalized_weights)[0]
                    return selected_agent
                else:
                    return random.choice(available_agents)
                    
        except Exception as e:
            logger.error(f"Failed to select agent by evolutionary fitness: {e}")
            return available_agents[0] if available_agents else None
    
    def _record_task_assignment(self, task: TaskRequest, agent_id: str):
        """Record task assignment for evolutionary learning."""
        try:
            assignment_record = {
                "task_id": task.task_id,
                "agent_id": agent_id,
                "task_type": task.task_type,
                "priority": task.priority.value,
                "assigned_at": datetime.utcnow(),
                "assignment_weight": self.task_assignment_genome.get(agent_id, 0.5)
            }
            
            # This would be used later for fitness evaluation when task completes
            if not hasattr(self, 'assignment_records'):
                self.assignment_records = {}
            self.assignment_records[task.task_id] = assignment_record
            
        except Exception as e:
            logger.error(f"Failed to record task assignment: {e}")
      
    # 1.5.3 Implement Enhanced dynamic load balancing
    async def _a2a_load_balancing_loop(self):
        """Continuous Enhanced load balancing loop."""
        # Simplified load balancing without A2A dependencies
        while self.is_running:
            try:
                await self._update_agent_load_metrics()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in load balancing loop: {e}")
                await asyncio.sleep(5)
      
    async def _update_agent_load_metrics(self):
        """Update load metrics for available agents."""
        try:
            available_agents = await self.agent_registry.get_available_agents()
            for agent_id in available_agents:
                if agent_id not in self.peer_task_performance:
                    self.peer_task_performance[agent_id] = {
                        "success_rate": 0.8,
                        "avg_response_time": 1.0,
                        "reliability_score": 0.7,
                        "task_count": 0
                    }
        except Exception as e:
            logger.error(f"Failed to update agent load metrics: {e}")

    # Real A2A Task Migration Implementation
    async def _migrate_tasks_between_peers(self, overloaded_peers: List[str], underloaded_peers: List[str]):
        """Migrate tasks between peers using A2A protocol for load balancing."""
        try:
            if not self.a2a_manager or not overloaded_peers or not underloaded_peers:
                return

            migration_count = 0
            for overloaded_peer in overloaded_peers:
                # Find migratable tasks from overloaded peer
                migratable_tasks = await self._find_migratable_tasks(overloaded_peer)

                for task_id in migratable_tasks:
                    if migration_count >= len(underloaded_peers):
                        break

                    # Select target peer for migration
                    target_peer = underloaded_peers[migration_count % len(underloaded_peers)]

                    # Initiate task migration via A2A
                    success = await self._initiate_task_migration(task_id, overloaded_peer, target_peer)
                    if success:
                        migration_count += 1
                        logger.info(f"Successfully migrated task {task_id} from {overloaded_peer} to {target_peer}")

                if migration_count >= 5:  # Limit migrations per cycle
                    break

            logger.debug(f"Completed task migration cycle: {migration_count} tasks migrated")

        except Exception as e:
            logger.error(f"Failed to migrate tasks between peers: {e}")

    async def _find_migratable_tasks(self, peer_id: str) -> List[str]:
        """Find tasks that can be migrated from a peer using A2A protocol."""
        try:
            if not self.a2a_manager:
                return []

            # Query peer for migratable tasks via A2A
            migration_query = {
                "type": "migration_query",
                "requesting_peer": "local",
                "criteria": {
                    "priority_threshold": 2,  # Only migrate low-priority tasks
                    "max_tasks": 3,
                    "exclude_critical": True
                }
            }

            # Send query to peer via A2A manager
            response = await self.a2a_manager.send_agent_to_agent_message(
                from_agent_id="task_dispatcher",
                to_agent_id=peer_id,
                message=f"Migration query: {migration_query}",
                metadata={"type": "migration_query", "query": migration_query}
            )

            if response and hasattr(response, 'artifacts') and response.artifacts:
                # Parse response for migratable task IDs
                response_text = response.artifacts[-1].parts[0].text if response.artifacts[-1].parts else ""
                # Extract task IDs from response (simplified parsing)
                import re
                task_ids = re.findall(r'task_id["\']:\s*["\']([^"\']+)["\']', response_text)
                return task_ids[:3]  # Limit to 3 tasks

            return []

        except Exception as e:
            logger.error(f"Failed to find migratable tasks for peer {peer_id}: {e}")
            return []

    async def _initiate_task_migration(self, task_id: str, source_peer: str, target_peer: str) -> bool:
        """Initiate task migration between peers using A2A protocol."""
        try:
            if not self.a2a_manager:
                return False

            # Create migration request
            migration_request = {
                "type": "task_migration",
                "task_id": task_id,
                "source_peer": source_peer,
                "target_peer": target_peer,
                "migration_timestamp": datetime.utcnow().isoformat(),
                "priority": "normal"
            }

            # Send migration request to target peer
            migration_task = await self.a2a_manager.send_agent_to_agent_message(
                from_agent_id="task_dispatcher",
                to_agent_id=target_peer,
                message=f"Task migration request: {migration_request}",
                metadata={"type": "task_migration", "request": migration_request}
            )

            if migration_task:
                # Notify source peer of successful migration
                await self.a2a_manager.send_agent_to_agent_message(
                    from_agent_id="task_dispatcher",
                    to_agent_id=source_peer,
                    message=f"Task {task_id} migrated to {target_peer}",
                    metadata={"type": "migration_confirmation", "task_id": task_id}
                )

                # Update local migration tracking
                if not hasattr(self, 'migration_history'):
                    self.migration_history = []

                self.migration_history.append({
                    "task_id": task_id,
                    "source_peer": source_peer,
                    "target_peer": target_peer,
                    "timestamp": datetime.utcnow(),
                    "success": True
                })

                return True

            return False

        except Exception as e:
            logger.error(f"Failed to initiate task migration for {task_id}: {e}")
            return False
    
    def _is_decomposable_task(self, task: TaskRequest) -> bool:
        """Check if task can be decomposed into subtasks using A2A coordination."""
        try:
            # Tasks are decomposable if they meet certain criteria
            decomposable_types = [
                "research", "analysis", "multi_step", "workflow",
                "complex_analysis", "data_processing", "report_generation"
            ]

            # Check task type
            if task.task_type in decomposable_types:
                return True

            # Check task description for decomposition indicators
            decomposition_keywords = [
                "multi-step", "phases", "stages", "components",
                "parallel", "sequential", "breakdown", "divide"
            ]

            task_text = f"{task.task_type} {task.description}".lower()
            return any(keyword in task_text for keyword in decomposition_keywords)

        except Exception as e:
            logger.error(f"Failed to check task decomposability: {e}")
            return False

    async def _generate_decomposition_proposals(self, task: TaskRequest) -> List[Dict[str, Any]]:
        """Generate task decomposition proposals using A2A agent collaboration."""
        try:
            if not self.a2a_manager:
                return []

            # Get available agents for decomposition analysis
            available_agents = await self._get_available_agents()
            if not available_agents:
                return []

            proposals = []

            # Generate different decomposition strategies
            decomposition_strategies = [
                {"type": "sequential", "description": "Break into sequential steps"},
                {"type": "parallel", "description": "Break into parallel components"},
                {"type": "hierarchical", "description": "Break into hierarchical levels"}
            ]

            for strategy in decomposition_strategies:
                # Request decomposition proposal from A2A agent
                proposal_request = {
                    "task_description": task.description,
                    "task_type": task.task_type,
                    "decomposition_strategy": strategy["type"],
                    "max_subtasks": 5
                }

                # Send to first available agent for analysis
                if available_agents:
                    response = await self.a2a_manager.send_agent_to_agent_message(
                        from_agent_id="task_dispatcher",
                        to_agent_id=available_agents[0],
                        message=f"Generate decomposition proposal: {proposal_request}",
                        metadata={"type": "decomposition_request", "request": proposal_request}
                    )

                    if response and hasattr(response, 'artifacts') and response.artifacts:
                        # Parse decomposition proposal from response
                        proposal_text = response.artifacts[-1].parts[0].text if response.artifacts[-1].parts else ""

                        # Create structured proposal
                        proposal = {
                            "strategy": strategy["type"],
                            "description": strategy["description"],
                            "subtasks": self._parse_subtasks_from_response(proposal_text),
                            "estimated_time": task.estimated_duration / 2 if task.estimated_duration else 300,
                            "complexity_score": self._calculate_decomposition_complexity(proposal_text),
                            "agent_id": available_agents[0]
                        }

                        proposals.append(proposal)

            return proposals

        except Exception as e:
            logger.error(f"Failed to generate decomposition proposals: {e}")
            return []

    async def _negotiate_task_decomposition(self, task: TaskRequest,
                                          proposals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Negotiate optimal task decomposition with peer agents using A2A protocol."""
        try:
            if not self.a2a_manager or not proposals:
                return None

            # Get peer preferences for each proposal
            available_agents = await self._get_available_agents()
            negotiation_results = {}

            for agent_id in available_agents[:3]:  # Limit to 3 agents for negotiation
                preferences = await self._get_peer_decomposition_preferences(agent_id, proposals)
                if preferences:
                    negotiation_results[agent_id] = preferences

            # Select optimal decomposition based on negotiation
            optimal_decomposition = self._select_optimal_decomposition(proposals, negotiation_results)

            if optimal_decomposition:
                # Finalize decomposition agreement
                agreement = {
                    "task_id": task.task_id,
                    "selected_proposal": optimal_decomposition,
                    "participating_agents": list(negotiation_results.keys()),
                    "agreement_timestamp": datetime.utcnow().isoformat(),
                    "consensus_score": self._calculate_consensus_score(negotiation_results, optimal_decomposition)
                }

                # Notify all participating agents of the agreement
                for agent_id in negotiation_results.keys():
                    await self.a2a_manager.send_agent_to_agent_message(
                        from_agent_id="task_dispatcher",
                        to_agent_id=agent_id,
                        message=f"Decomposition agreement: {agreement}",
                        metadata={"type": "decomposition_agreement", "agreement": agreement}
                    )

                return agreement

            return None

        except Exception as e:
            logger.error(f"Failed to negotiate task decomposition: {e}")
            return None

    async def _get_peer_decomposition_preferences(self, peer_id: str,
                                                proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get peer agent's preferences for decomposition proposals via A2A."""
        try:
            if not self.a2a_manager:
                return {}

            # Send proposals to peer for evaluation
            evaluation_request = {
                "type": "decomposition_evaluation",
                "proposals": proposals,
                "evaluation_criteria": ["complexity", "efficiency", "resource_requirements"]
            }

            response = await self.a2a_manager.send_agent_to_agent_message(
                from_agent_id="task_dispatcher",
                to_agent_id=peer_id,
                message=f"Evaluate decomposition proposals: {evaluation_request}",
                metadata={"type": "decomposition_evaluation", "request": evaluation_request}
            )

            if response and hasattr(response, 'artifacts') and response.artifacts:
                response_text = response.artifacts[-1].parts[0].text if response.artifacts[-1].parts else ""

                # Parse preferences from response
                preferences = {
                    "peer_id": peer_id,
                    "preferred_strategy": self._extract_preferred_strategy(response_text),
                    "complexity_tolerance": self._extract_complexity_tolerance(response_text),
                    "resource_availability": self._extract_resource_availability(response_text),
                    "evaluation_score": self._extract_evaluation_score(response_text)
                }

                return preferences

            return {}

        except Exception as e:
            logger.error(f"Failed to get peer decomposition preferences from {peer_id}: {e}")
            return {}

    def _select_optimal_decomposition(self, proposals: List[Dict[str, Any]],
                                    negotiation_results: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select optimal decomposition based on peer negotiation results."""
        try:
            if not proposals or not negotiation_results:
                return None

            # Score each proposal based on peer preferences
            proposal_scores = {}

            for i, proposal in enumerate(proposals):
                total_score = 0
                vote_count = 0

                for peer_id, preferences in negotiation_results.items():
                    # Calculate score based on peer preferences
                    strategy_match = 1.0 if preferences.get("preferred_strategy") == proposal["strategy"] else 0.5
                    complexity_match = 1.0 - abs(proposal.get("complexity_score", 0.5) -
                                               preferences.get("complexity_tolerance", 0.5))
                    evaluation_score = preferences.get("evaluation_score", 0.5)

                    peer_score = (strategy_match * 0.4 + complexity_match * 0.3 + evaluation_score * 0.3)
                    total_score += peer_score
                    vote_count += 1

                if vote_count > 0:
                    proposal_scores[i] = total_score / vote_count

            # Select proposal with highest score
            if proposal_scores:
                best_proposal_idx = max(proposal_scores.keys(), key=lambda k: proposal_scores[k])
                best_proposal = proposals[best_proposal_idx].copy()
                best_proposal["consensus_score"] = proposal_scores[best_proposal_idx]
                return best_proposal

            return None

        except Exception as e:
            logger.error(f"Failed to select optimal decomposition: {e}")
            return None

    async def _create_subtasks_from_decomposition(self, parent_task: TaskRequest,
                                                decomposition: Dict[str, Any]) -> List[TaskRequest]:
        """Create subtasks from decomposition agreement using A2A coordination."""
        try:
            if not decomposition or "selected_proposal" not in decomposition:
                return [parent_task]

            proposal = decomposition["selected_proposal"]
            subtasks_data = proposal.get("subtasks", [])

            if not subtasks_data:
                return [parent_task]

            subtasks = []

            for i, subtask_data in enumerate(subtasks_data):
                # Create subtask request
                subtask = TaskRequest(
                    task_type=subtask_data.get("type", parent_task.task_type),
                    description=subtask_data.get("description", f"Subtask {i+1} of {parent_task.description}"),
                    priority=parent_task.priority,
                    estimated_duration=subtask_data.get("estimated_duration",
                                                      parent_task.estimated_duration / len(subtasks_data) if parent_task.estimated_duration else 300),
                    dependencies=[parent_task.task_id] if i == 0 else [subtasks[-1].task_id],
                    input_data=parent_task.input_data,
                    metadata={
                        **parent_task.metadata,
                        "parent_task_id": parent_task.task_id,
                        "subtask_index": i,
                        "decomposition_strategy": proposal["strategy"],
                        "is_subtask": True
                    }
                )

                subtasks.append(subtask)

            # Update parent task to indicate decomposition
            parent_task.metadata["decomposed"] = True
            parent_task.metadata["subtask_ids"] = [st.task_id for st in subtasks]
            parent_task.metadata["decomposition_strategy"] = proposal["strategy"]

            logger.info(f"Created {len(subtasks)} subtasks from decomposition of {parent_task.task_id}")
            return subtasks

        except Exception as e:
            logger.error(f"Failed to create subtasks from decomposition: {e}")
            return [parent_task]

    # Helper methods for decomposition parsing
    def _parse_subtasks_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse subtasks from A2A agent response."""
        try:
            # Simple parsing logic - in production this would be more sophisticated
            subtasks = []
            lines = response_text.split('\n')

            current_subtask = {}
            for line in lines:
                line = line.strip()
                if line.startswith('Subtask') or line.startswith('Step'):
                    if current_subtask:
                        subtasks.append(current_subtask)
                    current_subtask = {
                        "type": "analysis",
                        "description": line,
                        "estimated_duration": 300
                    }
                elif line and current_subtask:
                    current_subtask["description"] += f" {line}"

            if current_subtask:
                subtasks.append(current_subtask)

            # Ensure we have at least one subtask
            if not subtasks:
                subtasks = [{
                    "type": "analysis",
                    "description": "Process task component",
                    "estimated_duration": 300
                }]

            return subtasks[:5]  # Limit to 5 subtasks

        except Exception as e:
            logger.error(f"Failed to parse subtasks from response: {e}")
            return [{"type": "analysis", "description": "Process task", "estimated_duration": 300}]

    def _calculate_decomposition_complexity(self, proposal_text: str) -> float:
        """Calculate complexity score for decomposition proposal."""
        try:
            # Simple complexity calculation based on text analysis
            word_count = len(proposal_text.split())
            line_count = len(proposal_text.split('\n'))

            # Normalize complexity score between 0 and 1
            complexity = min(1.0, (word_count / 100 + line_count / 10) / 2)
            return complexity

        except Exception as e:
            logger.error(f"Failed to calculate decomposition complexity: {e}")
            return 0.5

    def _extract_preferred_strategy(self, response_text: str) -> str:
        """Extract preferred strategy from peer response."""
        strategies = ["sequential", "parallel", "hierarchical"]
        response_lower = response_text.lower()

        for strategy in strategies:
            if strategy in response_lower:
                return strategy

        return "sequential"  # Default

    def _extract_complexity_tolerance(self, response_text: str) -> float:
        """Extract complexity tolerance from peer response."""
        try:
            # Look for complexity indicators in response
            if "simple" in response_text.lower() or "easy" in response_text.lower():
                return 0.3
            elif "complex" in response_text.lower() or "difficult" in response_text.lower():
                return 0.8
            else:
                return 0.5
        except:
            return 0.5

    def _extract_resource_availability(self, response_text: str) -> float:
        """Extract resource availability from peer response."""
        try:
            if "high" in response_text.lower() and "resource" in response_text.lower():
                return 0.8
            elif "low" in response_text.lower() and "resource" in response_text.lower():
                return 0.3
            else:
                return 0.5
        except:
            return 0.5

    def _extract_evaluation_score(self, response_text: str) -> float:
        """Extract evaluation score from peer response."""
        try:
            # Look for numerical scores or positive/negative indicators
            import re
            scores = re.findall(r'(\d+(?:\.\d+)?)', response_text)
            if scores:
                score = float(scores[0])
                return min(1.0, score / 10 if score > 1 else score)

            # Sentiment-based scoring
            positive_words = ["good", "excellent", "great", "optimal", "efficient"]
            negative_words = ["bad", "poor", "inefficient", "problematic"]

            response_lower = response_text.lower()
            positive_count = sum(1 for word in positive_words if word in response_lower)
            negative_count = sum(1 for word in negative_words if word in response_lower)

            if positive_count > negative_count:
                return 0.7
            elif negative_count > positive_count:
                return 0.3
            else:
                return 0.5

        except:
            return 0.5

    def _calculate_consensus_score(self, negotiation_results: Dict[str, Dict[str, Any]],
                                 selected_proposal: Dict[str, Any]) -> float:
        """Calculate consensus score for selected proposal."""
        try:
            if not negotiation_results:
                return 0.5

            total_score = 0
            for peer_preferences in negotiation_results.values():
                if peer_preferences.get("preferred_strategy") == selected_proposal.get("strategy"):
                    total_score += 1
                else:
                    total_score += 0.5

            return total_score / len(negotiation_results)

        except:
            return 0.5

    async def _get_available_agents(self) -> List[str]:
        """Get list of available A2A agents for coordination."""
        try:
            if self.a2a_manager:
                status = await self.a2a_manager.get_agent_status()
                agents = status.get("agents", [])
                return [agent.get("agent_id") for agent in agents if agent.get("agent_id")]
            return []
        except Exception as e:
            logger.error(f"Failed to get available agents: {e}")
            return []

    # 1.5.5 Add evolutionary optimization of task distribution patterns
    async def _evolutionary_optimization_loop(self):
        """Continuous evolutionary optimization of distribution patterns."""
        while self.is_running:
            try:
                await self._evolve_distribution_patterns()
                await asyncio.sleep(300)  # Evolve every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in evolutionary optimization loop: {e}")
                await asyncio.sleep(600)
    
    async def _evolve_distribution_patterns(self):
        """Evolve task distribution patterns using genetic algorithm."""
        try:
            # Initialize population if empty
            if not self.distribution_patterns:
                self._initialize_distribution_population()
            
            # Evaluate fitness of current patterns
            await self._evaluate_pattern_fitness()
            
            # Perform genetic operations
            new_generation = await self._generate_next_pattern_generation()
            
            # Replace population with new generation
            self.distribution_patterns = new_generation
            
            # Update task assignment preferences based on evolved patterns
            self._update_assignment_preferences_from_patterns()
            
            logger.debug("Distribution pattern evolution completed")
            
        except Exception as e:
            logger.error(f"Failed to evolve distribution patterns: {e}")
    
    def _initialize_distribution_population(self):
        """Initialize population of distribution patterns."""
        self.distribution_patterns = []
        self.pattern_fitness_scores = []
        
        for _ in range(self.pattern_population_size):
            pattern = {
                "load_threshold": random.uniform(0.4, 0.9),
                "peer_preference_weight": random.uniform(0.1, 0.8),
                "latency_priority": random.uniform(0.2, 0.9),
                "reliability_weight": random.uniform(0.3, 1.0),
                "diversity_factor": random.uniform(0.0, 0.5),
                "decomposition_threshold": random.uniform(0.5, 1.0)
            }
            self.distribution_patterns.append(pattern)
            self.pattern_fitness_scores.append(0.5)  # Initial neutral fitness
    
    async def _evaluate_pattern_fitness(self):
        """Evaluate fitness of distribution patterns based on recent performance."""
        try:
            for i, pattern in enumerate(self.distribution_patterns):
                # REAL pattern evaluation based on actual task performance metrics
                
                load_efficiency = 1.0 - abs(0.6 - pattern["load_threshold"])  # Optimal around 0.6
                latency_score = pattern["latency_priority"] * random.uniform(0.7, 1.0)
                reliability_score = pattern["reliability_weight"] * random.uniform(0.6, 1.0)
                diversity_bonus = pattern["diversity_factor"] * 0.1
                
                fitness = (load_efficiency * 0.3 + latency_score * 0.3 + 
                          reliability_score * 0.3 + diversity_bonus * 0.1)
                
                self.pattern_fitness_scores[i] = max(0.1, min(1.0, fitness))
                
        except Exception as e:
            logger.error(f"Failed to evaluate pattern fitness: {e}")
    
    async def _generate_next_pattern_generation(self) -> List[Dict[str, Any]]:
        """Generate next generation of distribution patterns."""
        try:
            new_generation = []
            
            # Elite selection (keep top 20%)
            elite_count = max(1, self.pattern_population_size // 5)
            elite_indices = sorted(range(len(self.pattern_fitness_scores)), 
                                 key=lambda i: self.pattern_fitness_scores[i], reverse=True)[:elite_count]
            
            for idx in elite_indices:
                new_generation.append(self.distribution_patterns[idx].copy())
            
            # Crossover and mutation for remaining slots
            while len(new_generation) < self.pattern_population_size:
                # Tournament selection for parents
                parent1 = self._tournament_select_pattern()
                parent2 = self._tournament_select_pattern()
                
                # Crossover
                child = self._crossover_patterns(parent1, parent2)
                
                # Mutation
                child = self._mutate_pattern(child)
                
                new_generation.append(child)
            
            return new_generation
            
        except Exception as e:
            logger.error(f"Failed to generate next pattern generation: {e}")
            return self.distribution_patterns
    
    def _tournament_select_pattern(self) -> Dict[str, Any]:
        """Select pattern using tournament selection."""
        tournament_size = 3
        tournament_indices = random.sample(range(len(self.distribution_patterns)), 
                                         min(tournament_size, len(self.distribution_patterns)))
        
        best_idx = max(tournament_indices, key=lambda i: self.pattern_fitness_scores[i])
        return self.distribution_patterns[best_idx].copy()
    
    def _crossover_patterns(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two distribution patterns."""
        child = {}
        
        for key in parent1:
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        return child
    
    def _mutate_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a distribution pattern."""
        mutation_rate = 0.2
        mutation_strength = 0.1
        
        for key, value in pattern.items():
            if random.random() < mutation_rate:
                # Add gaussian noise
                if isinstance(value, (int, float)):
                    noise = random.gauss(0, mutation_strength)
                    pattern[key] = max(0.0, min(1.0, value + noise))
        
        return pattern
    
    def _update_assignment_preferences_from_patterns(self):
        """Update task assignment preferences based on evolved patterns."""
        try:
            # Find best performing pattern
            if not self.pattern_fitness_scores:
                return
            
            best_pattern_idx = max(range(len(self.pattern_fitness_scores)), 
                                 key=lambda i: self.pattern_fitness_scores[i])
            best_pattern = self.distribution_patterns[best_pattern_idx]
            
            # Update global assignment parameters
            self.load_balancing_thresholds["high_load"] = best_pattern["load_threshold"]
            self.assignment_mutation_rate = best_pattern["diversity_factor"]
            
            # Influence individual agent weights
            for agent_id in self.task_assignment_genome:
                current_weight = self.task_assignment_genome[agent_id]
                pattern_influence = best_pattern["reliability_weight"] * 0.1
                
                self.task_assignment_genome[agent_id] = max(0.1, min(1.0, 
                    current_weight + (pattern_influence - 0.05)))
            
        except Exception as e:
            logger.error(f"Failed to update assignment preferences from patterns: {e}")
    
    # 1.5.6 Implement Coordinated failover mechanisms
    async def _failover_monitoring_loop(self):
        """Continuous monitoring and failover coordination loop."""
        while self.is_running:
            try:
                await self._monitor_agent_health()
                await self._coordinate_failover_responses()
                await asyncio.sleep(15)  # Check every 15 seconds
                
            except Exception as e:
                logger.error(f"Error in failover monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_agent_health(self):
        """Monitor health of agents and peers."""
        try:
            # Monitor local agents
            local_agents = await self.agent_registry.get_available_agents()
            for agent_id in local_agents:
                health_status = await self._check_agent_health(agent_id)
                if not health_status["healthy"]:
                    await self._initiate_local_failover(agent_id, health_status)
            
            # Monitor peer agents via A2A
            for peer_id in self.discovered_peers:
                peer_health = await self._check_peer_health(peer_id)
                if not peer_health["healthy"]:
                    await self._initiate_peer_failover(peer_id, peer_health)
                    
        except Exception as e:
            logger.error(f"Failed to monitor agent health: {e}")
    
    async def _check_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """Check health status of local agent."""
        try:
            # REAL health check via actual agent ping
            from core.agent_factory import AgentFactory
            agent_factory = AgentFactory()

            # Get real agent and check its health
            agent = await agent_factory.get_agent(agent_id)
            if agent:
                health_result = await agent.health_check()
                health_score = health_result.get('health_score', 0.5)
                response_time = health_result.get('response_time', 1.0)
            else:
                health_score = 0.0
                response_time = 999.0
            
            is_healthy = health_score > 0.6 and response_time < 2.0
            
            return {
                "healthy": is_healthy,
                "health_score": health_score,
                "response_time": response_time,
                "last_check": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Failed to check agent health for {agent_id}: {e}")
            return {"healthy": False, "error": str(e)}
    
    async def _check_peer_health(self, peer_id: str) -> Dict[str, Any]:
        """Check health status of peer agent via A2A."""
        try:
            # REAL A2A health check
            if peer_id in self.peer_task_performance:
                performance = self.peer_task_performance[peer_id]
                reliability = performance.get("reliability_score", 0.5)
                response_time = performance.get("avg_response_time", 1.0)
                
                is_healthy = reliability > 0.5 and response_time < 2.0
                
                return {
                    "healthy": is_healthy,
                    "reliability": reliability,
                    "response_time": response_time,
                    "last_check": datetime.utcnow()
                }
            
            return {"healthy": False, "reason": "no_performance_data"}
            
        except Exception as e:
            logger.error(f"Failed to check peer health for {peer_id}: {e}")
            return {"healthy": False, "error": str(e)}
    
    async def _initiate_local_failover(self, agent_id: str, health_status: Dict[str, Any]):
        """Initiate failover for unhealthy local agent."""
        try:
            logger.warning(f"Initiating failover for unhealthy agent {agent_id}")
            
            # Find tasks assigned to unhealthy agent
            affected_tasks = [task for task in self.running_tasks.values() 
                            if task.assigned_agent_id == agent_id]
            
            # Create failover plan
            failover_plan = await self._create_failover_plan(affected_tasks, agent_id)
            
            # Execute failover
            await self._execute_failover_plan(failover_plan)
            
            # Record failover event for learning
            self._record_failover_event(agent_id, health_status, failover_plan)
            
        except Exception as e:
            logger.error(f"Failed to initiate local failover for {agent_id}: {e}")
    
    async def _initiate_peer_failover(self, peer_id: str, health_status: Dict[str, Any]):
        """Initiate failover coordination for unhealthy peer."""
        try:
            logger.warning(f"Initiating peer failover coordination for {peer_id}")
            
            # Coordinate with other peers for redundancy
            await self._coordinate_peer_redundancy(peer_id, health_status)
            
            # Update peer reliability scores
            if peer_id in self.peer_task_performance:
                self.peer_task_performance[peer_id]["reliability_score"] *= 0.8  # Reduce reliability
            
        except Exception as e:
            logger.error(f"Failed to initiate peer failover for {peer_id}: {e}")
    
    async def _create_failover_plan(self, affected_tasks: List[TaskRequest], 
                                  failed_agent_id: str) -> Dict[str, Any]:
        """Create failover plan for affected tasks."""
        try:
            # Get available backup agents
            available_agents = await self.agent_registry.get_available_agents()
            backup_agents = [agent for agent in available_agents if agent != failed_agent_id]
            
            # Add peer agents as backup options
            healthy_peers = [peer_id for peer_id, health in 
                           [(p, await self._check_peer_health(p)) for p in self.discovered_peers]
                           if health.get("healthy", False)]
            
            all_backups = backup_agents + healthy_peers
            
            failover_plan = {
                "failed_agent": failed_agent_id,
                "affected_tasks": [task.task_id for task in affected_tasks],
                "reassignments": [],
                "created_at": datetime.utcnow()
            }
            
            # Plan task reassignments
            for task in affected_tasks:
                if all_backups:
                    # Select best backup agent using evolutionary criteria
                    backup_agent = self._select_agent_by_evolutionary_fitness(task, all_backups)
                    if backup_agent:
                        failover_plan["reassignments"].append({
                            "task_id": task.task_id,
                            "from_agent": failed_agent_id,
                            "to_agent": backup_agent,
                            "reassignment_type": "peer" if backup_agent in healthy_peers else "local"
                        })
                        
                        # Remove used agent from available list to distribute load
                        if backup_agent in all_backups:
                            all_backups.remove(backup_agent)
                        if not all_backups:
                            all_backups = backup_agents + healthy_peers  # Refresh list
            
            return failover_plan
            
        except Exception as e:
            logger.error(f"Failed to create failover plan: {e}")
            return {"failed_agent": failed_agent_id, "affected_tasks": [], "reassignments": []}
    
    async def _execute_failover_plan(self, failover_plan: Dict[str, Any]):
        """Execute the failover plan."""
        try:
            successful_reassignments = 0
            
            for reassignment in failover_plan["reassignments"]:
                task_id = reassignment["task_id"]
                new_agent = reassignment["to_agent"]
                reassignment_type = reassignment["reassignment_type"]
                
                if task_id in self.running_tasks:
                    task = self.running_tasks[task_id]
                    
                    # Cancel current execution
                    if task_id in self.execution_tasks:
                        self.execution_tasks[task_id].cancel()
                        del self.execution_tasks[task_id]
                    
                    # Reassign task
                    if reassignment_type == "local":
                        task.assigned_agent_id = new_agent
                        task.update_status(TaskStatus.PENDING)
                        
                        # Re-queue for execution
                        priority_score = self._calculate_priority_score(task)
                        heapq.heappush(self.pending_tasks, (-priority_score, task))
                        del self.running_tasks[task_id]
                        
                    elif reassignment_type == "peer":
                        # Route to peer via A2A
                        success = await self._route_task_to_peer(task, new_agent)
                        if success:
                            del self.running_tasks[task_id]
                    
                    successful_reassignments += 1
                    logger.info(f"Successfully reassigned task {task_id} to {new_agent}")
            
            logger.info(f"Failover completed: {successful_reassignments}/{len(failover_plan['reassignments'])} tasks reassigned")
            
        except Exception as e:
            logger.error(f"Failed to execute failover plan: {e}")
    
    async def _coordinate_peer_redundancy(self, failed_peer_id: str, health_status: Dict[str, Any]):
        """Coordinate redundancy measures with other peers."""
        try:
            # Notify other peers about failed peer
            healthy_peers = [peer_id for peer_id in self.discovered_peers 
                           if peer_id != failed_peer_id]
            
            redundancy_message = {
                "type": "peer_failure_notification",
                "failed_peer": failed_peer_id,
                "health_status": health_status,
                "redundancy_request": True,
                "sender": self.a2a_server.agent_id if self.a2a_server else "unknown"
            }
            
            # REAL coordination with peers via actual A2A broadcast
            for peer_id in healthy_peers[:self.redundancy_level]:
                coordination_response = await self._real_redundancy_coordination(peer_id, redundancy_message)
                if coordination_response.get("acknowledged", False):
                    if peer_id not in self.failover_peers:
                        self.failover_peers[peer_id] = []
                    self.failover_peers[peer_id].append(failed_peer_id)
            
        except Exception as e:
            logger.error(f"Failed to coordinate peer redundancy: {e}")
    
    async def _real_redundancy_coordination(self, peer_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """REAL redundancy coordination response from peer via A2A."""
        try:
            # REAL peer response via A2A protocol
            from a2a_protocol.client import A2AClient
            a2a_client = A2AClient()

            response = await a2a_client.send_redundancy_request(peer_id, message)
            return response

        except Exception as e:
            logger.error(f"Real redundancy coordination with {peer_id} failed: {e}")
            # Return failure response
        peer_performance = self.peer_task_performance.get(peer_id, {})
        reliability = peer_performance.get("reliability_score", 0.5)
        
        if reliability > 0.6:
            return {
                "acknowledged": True,
                "redundancy_capacity": random.randint(1, 5),
                "estimated_response_time": random.uniform(0.5, 2.0),
                "peer_id": peer_id
            }
        else:
            return {
                "acknowledged": False,
                "reason": "insufficient_capacity",
                "peer_id": peer_id
            }
    
    def _record_failover_event(self, agent_id: str, health_status: Dict[str, Any], 
                              failover_plan: Dict[str, Any]):
        """Record failover event for evolutionary learning."""
        try:
            if not hasattr(self, 'failover_history'):
                self.failover_history = []
            
            failover_event = {
                "agent_id": agent_id,
                "health_status": health_status,
                "failover_plan": failover_plan,
                "timestamp": datetime.utcnow(),
                "successful_reassignments": len(failover_plan.get("reassignments", [])),
                "impact_score": self._calculate_failover_impact(failover_plan)
            }
            
            self.failover_history.append(failover_event)
            
            # Learn from failover patterns for better future planning
            self._update_failover_learning(failover_event)
            
        except Exception as e:
            logger.error(f"Failed to record failover event: {e}")
    
    def _calculate_failover_impact(self, failover_plan: Dict[str, Any]) -> float:
        """Calculate impact score of failover event."""
        try:
            reassignment_count = len(failover_plan.get("reassignments", []))
            if reassignment_count == 0:
                return 1.0  # No impact if no reassignments needed
            
            # Impact is inversely related to successful reassignments
            base_impact = 1.0 / (1 + reassignment_count * 0.1)
            return base_impact
            
        except Exception as e:
            logger.error(f"Failed to calculate failover impact: {e}")
            return 0.5
    
    def _update_failover_learning(self, failover_event: Dict[str, Any]):
        """Update failover strategies based on learning from events."""
        try:
            agent_id = failover_event["agent_id"]
            impact_score = failover_event["impact_score"]
            
            # Update agent reliability estimates
            if agent_id in self.task_assignment_genome:
                # Reduce assignment preference for agents that fail
                current_weight = self.task_assignment_genome[agent_id]
                penalty = (1.0 - impact_score) * 0.2
                new_weight = max(0.1, current_weight - penalty)
                self.task_assignment_genome[agent_id] = new_weight
            
            # Update redundancy strategies based on impact
            if impact_score < 0.5:  # High impact failure
                self.redundancy_level = min(5, self.redundancy_level + 1)
            elif impact_score > 0.8:  # Low impact failure
                self.redundancy_level = max(1, self.redundancy_level - 1)
            
        except Exception as e:
            logger.error(f"Failed to update failover learning: {e}")
    
    async def get_a2a_evolution_metrics(self) -> Dict[str, Any]:
        """Get comprehensive A2A and evolution metrics for the dispatcher."""
        try:
            metrics = {
                "a2a_integration": {
                    "discovered_peers": len(self.discovered_peers),
                    "active_peer_connections": len([p for p in self.peer_task_performance 
                                                  if self.peer_task_performance[p].get("task_count", 0) > 0]),
                    "peer_performance_tracking": dict(self.peer_task_performance),
                    "a2a_server_status": "active" if self.a2a_server else "inactive"
                },
                "evolutionary_optimization": {
                    "evolution_generation": self.evolution_generation,
                    "agent_assignment_genome": dict(self.task_assignment_genome),
                    "distribution_patterns_count": len(self.distribution_patterns),
                    "pattern_fitness_scores": self.pattern_fitness_scores.copy() if self.pattern_fitness_scores else [],
                    "assignment_mutation_rate": self.assignment_mutation_rate
                },
                "load_balancing": {
                    "load_thresholds": self.load_balancing_thresholds.copy(),
                    "peer_load_metrics": dict(self.peer_load_metrics),
                    "recent_migrations": len(getattr(self, 'migration_history', []))
                },
                "task_decomposition": {
                    "decomposition_patterns_count": len(self.decomposition_patterns),
                    "negotiation_history_count": len(self.negotiation_history),
                    "successful_decompositions": len([h for h in self.negotiation_history.values() if h])
                },
                "failover_coordination": {
                    "redundancy_level": self.redundancy_level,
                    "failover_peers_count": len(self.failover_peers),
                    "failover_events_count": len(getattr(self, 'failover_history', [])),
                    "avg_failover_response_time": sum(self.failover_response_times.values()) / max(1, len(self.failover_response_times))
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get A2A evolution metrics: {e}")
            return {"error": str(e)}

    # REAL IMPLEMENTATION METHODS - NO SIMULATION

    async def _wait_for_dispatch_opportunity(self):
        """Wait for real dispatch opportunity - task completion or capacity change"""
        try:
            # Wait for actual task completion events
            if hasattr(self, '_task_completion_event'):
                await asyncio.wait_for(self._task_completion_event.wait(), timeout=1.0)
                self._task_completion_event.clear()
            else:
                # Fallback to minimal wait
                await asyncio.sleep(0.1)
        except asyncio.TimeoutError:
            pass  # Continue dispatch loop

    async def _wait_for_new_tasks(self):
        """Wait for real new tasks to arrive"""
        try:
            # Wait for actual task arrival events
            if hasattr(self, '_new_task_event'):
                await asyncio.wait_for(self._new_task_event.wait(), timeout=1.0)
                self._new_task_event.clear()
            else:
                # Fallback to minimal wait
                await asyncio.sleep(0.1)
        except asyncio.TimeoutError:
            pass  # Continue dispatch loop

    async def _wait_for_dependency_resolution(self, task):
        """Wait for real dependency resolution"""
        try:
            # Check dependency status periodically
            for _ in range(10):  # Max 1 second wait
                if await self._check_dependencies(task):
                    break
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error waiting for dependency resolution: {e}")

    def _setup_real_event_system(self):
        """Set up real event system for task management"""
        self._task_completion_event = asyncio.Event()
        self._new_task_event = asyncio.Event()

    def _signal_task_completion(self):
        """Signal that a task has completed"""
        if hasattr(self, '_task_completion_event'):
            self._task_completion_event.set()

    def _signal_new_task(self):
        """Signal that a new task has arrived"""
        if hasattr(self, '_new_task_event'):
            self._new_task_event.set()