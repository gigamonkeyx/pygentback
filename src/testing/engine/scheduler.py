"""
Test Scheduler

This module provides intelligent scheduling and orchestration of recipe tests
with load balancing, priority management, and resource optimization.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import heapq
import uuid

from ..core.framework import RecipeTestResult
from ..recipes.schema import RecipeDefinition


logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """A scheduled test task"""
    task_id: str
    recipe: RecipeDefinition
    test_function: Callable[[], Awaitable[RecipeTestResult]]
    priority: TaskPriority = TaskPriority.NORMAL
    scheduled_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[RecipeTestResult] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    timeout_seconds: int = 300
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        # Higher priority values come first, then earlier scheduled times
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.scheduled_at < other.scheduled_at


@dataclass
class SchedulerConfig:
    """Configuration for test scheduler"""
    max_concurrent_tasks: int = 5
    max_queue_size: int = 100
    default_timeout_seconds: int = 300
    retry_delay_seconds: int = 30
    enable_load_balancing: bool = True
    enable_priority_scheduling: bool = True
    resource_monitoring: bool = True
    max_memory_usage_mb: float = 4096.0
    max_cpu_usage_percent: float = 80.0


@dataclass
class SchedulerStats:
    """Scheduler statistics"""
    total_tasks_scheduled: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_tasks_cancelled: int = 0
    current_queue_size: int = 0
    current_running_tasks: int = 0
    average_execution_time_ms: float = 0.0
    average_queue_wait_time_ms: float = 0.0
    success_rate: float = 0.0
    throughput_tasks_per_minute: float = 0.0


class TestScheduler:
    """
    Intelligent scheduler for recipe test execution.
    
    Provides priority-based scheduling, load balancing, resource management,
    and dependency handling for optimal test execution.
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        
        # Task management
        self.task_queue: List[ScheduledTask] = []  # Priority queue
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: Dict[str, ScheduledTask] = {}
        self.failed_tasks: Dict[str, ScheduledTask] = {}
        
        # Scheduling state
        self.is_running = False
        self.scheduler_task: Optional[asyncio.Task] = None
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        # Statistics
        self.stats = SchedulerStats()
        self.start_time = datetime.utcnow()
        
        # Event callbacks
        self.on_task_started: Optional[Callable[[ScheduledTask], None]] = None
        self.on_task_completed: Optional[Callable[[ScheduledTask], None]] = None
        self.on_task_failed: Optional[Callable[[ScheduledTask], None]] = None
        self.on_queue_full: Optional[Callable[[int], None]] = None
        
        # Resource monitoring
        self.current_memory_usage = 0.0
        self.current_cpu_usage = 0.0
    
    async def start(self) -> None:
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        # Start scheduler loop
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("Test scheduler started")
    
    async def stop(self) -> None:
        """Stop the scheduler"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel scheduler task
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel running tasks
        for task in list(self.running_tasks.values()):
            await self._cancel_task(task)
        
        logger.info("Test scheduler stopped")
    
    def schedule_task(self, 
                     recipe: RecipeDefinition,
                     test_function: Callable[[], Awaitable[RecipeTestResult]],
                     priority: TaskPriority = TaskPriority.NORMAL,
                     timeout_seconds: Optional[int] = None,
                     dependencies: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Schedule a test task.
        
        Args:
            recipe: Recipe to test
            test_function: Async function that executes the test
            priority: Task priority
            timeout_seconds: Task timeout
            dependencies: List of task IDs this task depends on
            metadata: Additional task metadata
            
        Returns:
            Task ID
        """
        if len(self.task_queue) >= self.config.max_queue_size:
            if self.on_queue_full:
                self.on_queue_full(len(self.task_queue))
            raise RuntimeError(f"Task queue is full ({self.config.max_queue_size} tasks)")
        
        task_id = str(uuid.uuid4())
        
        task = ScheduledTask(
            task_id=task_id,
            recipe=recipe,
            test_function=test_function,
            priority=priority,
            timeout_seconds=timeout_seconds or self.config.default_timeout_seconds,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        # Add to priority queue
        heapq.heappush(self.task_queue, task)
        
        self.stats.total_tasks_scheduled += 1
        self.stats.current_queue_size = len(self.task_queue)
        
        logger.info(f"Scheduled task {task_id} for recipe {recipe.name} with priority {priority.name}")
        
        return task_id
    
    def schedule_multiple_tasks(self, 
                               recipes: List[RecipeDefinition],
                               test_function_factory: Callable[[RecipeDefinition], Callable[[], Awaitable[RecipeTestResult]]],
                               priority: TaskPriority = TaskPriority.NORMAL) -> List[str]:
        """
        Schedule multiple test tasks.
        
        Args:
            recipes: List of recipes to test
            test_function_factory: Function that creates test functions for recipes
            priority: Task priority
            
        Returns:
            List of task IDs
        """
        task_ids = []
        
        for recipe in recipes:
            test_function = test_function_factory(recipe)
            task_id = self.schedule_task(recipe, test_function, priority)
            task_ids.append(task_id)
        
        return task_ids
    
    async def execute_tests(self, test_tasks: List[Callable[[], Awaitable[RecipeTestResult]]]) -> List[RecipeTestResult]:
        """
        Execute a list of test tasks with scheduling.
        
        Args:
            test_tasks: List of test functions to execute
            
        Returns:
            List of test results
        """
        # Create dummy recipes for scheduling
        task_ids = []
        for i, test_task in enumerate(test_tasks):
            dummy_recipe = RecipeDefinition(name=f"test_task_{i}")
            task_id = self.schedule_task(dummy_recipe, test_task)
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        results = []
        for task_id in task_ids:
            result = await self.wait_for_task(task_id)
            if result:
                results.append(result)
        
        return results
    
    async def wait_for_task(self, task_id: str, timeout_seconds: Optional[int] = None) -> Optional[RecipeTestResult]:
        """
        Wait for a specific task to complete.
        
        Args:
            task_id: Task ID to wait for
            timeout_seconds: Maximum time to wait
            
        Returns:
            Test result or None if task failed/timed out
        """
        start_time = time.time()
        timeout = timeout_seconds or 600  # 10 minutes default
        
        while time.time() - start_time < timeout:
            # Check if task is completed
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id].result
            
            # Check if task failed
            if task_id in self.failed_tasks:
                logger.error(f"Task {task_id} failed: {self.failed_tasks[task_id].error}")
                return None
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        logger.error(f"Timeout waiting for task {task_id}")
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a scheduled or running task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if task was cancelled
        """
        # Check if task is in queue
        for i, task in enumerate(self.task_queue):
            if task.task_id == task_id:
                task.status = TaskStatus.CANCELLED
                self.task_queue.pop(i)
                heapq.heapify(self.task_queue)  # Restore heap property
                self.stats.total_tasks_cancelled += 1
                self.stats.current_queue_size = len(self.task_queue)
                logger.info(f"Cancelled queued task {task_id}")
                return True
        
        # Check if task is running
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            asyncio.create_task(self._cancel_task(task))
            return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        # Check running tasks
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task.status.value,
                "recipe_name": task.recipe.name,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "running_time_ms": int((datetime.utcnow() - task.started_at).total_seconds() * 1000) if task.started_at else 0
            }
        
        # Check completed tasks
        if task_id in self.completed_tasks:
            task = self.completed_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task.status.value,
                "recipe_name": task.recipe.name,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "execution_time_ms": int((task.completed_at - task.started_at).total_seconds() * 1000) if task.started_at and task.completed_at else 0,
                "success": task.result.success if task.result else False
            }
        
        # Check failed tasks
        if task_id in self.failed_tasks:
            task = self.failed_tasks[task_id]
            return {
                "task_id": task_id,
                "status": task.status.value,
                "recipe_name": task.recipe.name,
                "error": task.error,
                "retry_count": task.retry_count
            }
        
        # Check queue
        for task in self.task_queue:
            if task.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": task.status.value,
                    "recipe_name": task.recipe.name,
                    "scheduled_at": task.scheduled_at.isoformat(),
                    "queue_position": self.task_queue.index(task) + 1
                }
        
        return None
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get overall scheduler status"""
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Update throughput
        if uptime_seconds > 0:
            self.stats.throughput_tasks_per_minute = (self.stats.total_tasks_completed * 60) / uptime_seconds
        
        # Update success rate
        total_finished = self.stats.total_tasks_completed + self.stats.total_tasks_failed
        if total_finished > 0:
            self.stats.success_rate = self.stats.total_tasks_completed / total_finished
        
        return {
            "is_running": self.is_running,
            "uptime_seconds": uptime_seconds,
            "queue_size": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "statistics": {
                "total_scheduled": self.stats.total_tasks_scheduled,
                "total_completed": self.stats.total_tasks_completed,
                "total_failed": self.stats.total_tasks_failed,
                "success_rate": self.stats.success_rate,
                "throughput_per_minute": self.stats.throughput_tasks_per_minute,
                "average_execution_time_ms": self.stats.average_execution_time_ms
            },
            "resource_usage": {
                "memory_mb": self.current_memory_usage,
                "cpu_percent": self.current_cpu_usage
            }
        }
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        logger.info("Scheduler loop started")
        
        while self.is_running:
            try:
                # Process pending tasks
                await self._process_pending_tasks()
                
                # Update statistics
                self._update_statistics()
                
                # Resource monitoring
                if self.config.resource_monitoring:
                    await self._monitor_resources()
                
                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)  # Wait longer on error
        
        logger.info("Scheduler loop stopped")
    
    async def _process_pending_tasks(self) -> None:
        """Process pending tasks from the queue"""
        while (self.task_queue and 
               len(self.running_tasks) < self.config.max_concurrent_tasks and
               self._can_start_new_task()):
            
            # Get highest priority task
            task = heapq.heappop(self.task_queue)
            self.stats.current_queue_size = len(self.task_queue)
            
            # Check dependencies
            if not self._dependencies_satisfied(task):
                # Put task back in queue
                heapq.heappush(self.task_queue, task)
                break
            
            # Start task execution
            await self._start_task(task)
    
    def _dependencies_satisfied(self, task: ScheduledTask) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def _can_start_new_task(self) -> bool:
        """Check if we can start a new task based on resource constraints"""
        if not self.config.resource_monitoring:
            return True
        
        # Check memory usage
        if self.current_memory_usage > self.config.max_memory_usage_mb:
            return False
        
        # Check CPU usage
        if self.current_cpu_usage > self.config.max_cpu_usage_percent:
            return False
        
        return True
    
    async def _start_task(self, task: ScheduledTask) -> None:
        """Start executing a task"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        self.running_tasks[task.task_id] = task
        self.stats.current_running_tasks = len(self.running_tasks)
        
        # Notify callback
        if self.on_task_started:
            self.on_task_started(task)
        
        logger.info(f"Starting task {task.task_id} for recipe {task.recipe.name}")
        
        # Execute task asynchronously
        asyncio.create_task(self._execute_task(task))
    
    async def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a single task"""
        async with self.semaphore:
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    task.test_function(),
                    timeout=task.timeout_seconds
                )
                
                # Task completed successfully
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
                
                # Move to completed tasks
                self.running_tasks.pop(task.task_id, None)
                self.completed_tasks[task.task_id] = task
                
                self.stats.total_tasks_completed += 1
                self.stats.current_running_tasks = len(self.running_tasks)
                
                # Notify callback
                if self.on_task_completed:
                    self.on_task_completed(task)
                
                logger.info(f"Task {task.task_id} completed successfully")
                
            except asyncio.TimeoutError:
                await self._handle_task_failure(task, "Task timed out")
                
            except Exception as e:
                await self._handle_task_failure(task, str(e))
    
    async def _handle_task_failure(self, task: ScheduledTask, error: str) -> None:
        """Handle task failure with retry logic"""
        task.error = error
        task.retry_count += 1
        
        logger.warning(f"Task {task.task_id} failed: {error} (retry {task.retry_count}/{task.max_retries})")
        
        # Check if we should retry
        if task.retry_count <= task.max_retries:
            # Schedule retry
            task.status = TaskStatus.PENDING
            task.scheduled_at = datetime.utcnow() + timedelta(seconds=self.config.retry_delay_seconds)
            
            # Remove from running tasks and add back to queue
            self.running_tasks.pop(task.task_id, None)
            heapq.heappush(self.task_queue, task)
            self.stats.current_queue_size = len(self.task_queue)
            
            logger.info(f"Scheduled retry for task {task.task_id}")
        else:
            # Task failed permanently
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            
            # Move to failed tasks
            self.running_tasks.pop(task.task_id, None)
            self.failed_tasks[task.task_id] = task
            
            self.stats.total_tasks_failed += 1
            
            # Notify callback
            if self.on_task_failed:
                self.on_task_failed(task)
            
            logger.error(f"Task {task.task_id} failed permanently after {task.retry_count} retries")
        
        self.stats.current_running_tasks = len(self.running_tasks)
    
    async def _cancel_task(self, task: ScheduledTask) -> None:
        """Cancel a running task"""
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        
        # Remove from running tasks
        self.running_tasks.pop(task.task_id, None)
        self.stats.total_tasks_cancelled += 1
        self.stats.current_running_tasks = len(self.running_tasks)
        
        logger.info(f"Cancelled task {task.task_id}")
    
    def _update_statistics(self) -> None:
        """Update scheduler statistics"""
        # Calculate average execution time
        if self.completed_tasks:
            total_time = 0
            count = 0
            for task in self.completed_tasks.values():
                if task.started_at and task.completed_at:
                    execution_time = (task.completed_at - task.started_at).total_seconds() * 1000
                    total_time += execution_time
                    count += 1
            
            if count > 0:
                self.stats.average_execution_time_ms = total_time / count
    
    async def _monitor_resources(self) -> None:
        """Monitor system resources"""
        try:
            import psutil
            
            # Get current process
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            self.current_memory_usage = memory_info.rss / 1024 / 1024  # MB
            
            # CPU usage
            self.current_cpu_usage = process.cpu_percent()
            
        except ImportError:
            # psutil not available, skip monitoring
            pass
        except Exception as e:
            logger.warning(f"Resource monitoring failed: {e}")
