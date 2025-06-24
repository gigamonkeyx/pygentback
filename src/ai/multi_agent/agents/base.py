"""
Base Agent Classes

Core agent implementations and base classes for specialized agents.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from ..core import BaseAgent, AgentCapability, AgentStatus
from ..models import Task, TaskResult, Message

logger = logging.getLogger(__name__)


class SpecializedAgent(BaseAgent):
    """
    Base class for specialized agents with common functionality.
    """
    
    def __init__(self, agent_id: str, name: str, agent_type: str, capabilities: List[AgentCapability]):
        super().__init__(agent_id, name, capabilities)
        self.agent_type = agent_type
        
        # Specialized agent configuration
        self.performance_thresholds = {
            'max_execution_time_ms': 30000,
            'max_memory_usage_mb': 512,
            'min_success_rate': 0.8
        }
        
        # Task queue for this agent
        self.task_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        
        # Performance tracking
        self.execution_times: List[float] = []
        self.last_performance_check = datetime.utcnow()
    
    async def initialize(self) -> bool:
        """Initialize the specialized agent"""
        try:
            # Perform agent-specific initialization
            await self._initialize_agent_specific()
            
            # Validate capabilities
            if not await self._validate_capabilities():
                logger.error(f"Agent {self.name} capability validation failed")
                return False
            
            # Setup monitoring
            await self._setup_monitoring()
            
            logger.info(f"Specialized agent {self.name} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Agent {self.name} initialization failed: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the specialized agent"""
        try:
            # Complete current task if any
            if self.current_task:
                logger.info(f"Agent {self.name} completing current task before shutdown")
                await asyncio.sleep(1.0)  # Give time for task completion
            
            # Perform agent-specific cleanup
            await self._cleanup_agent_specific()
            
            # Clear queues
            while not self.task_queue.empty():
                try:
                    self.task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            logger.info(f"Specialized agent {self.name} shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"Agent {self.name} shutdown failed: {e}")
            return False
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a task with performance tracking"""
        start_time = time.time()
        
        try:
            # Validate task
            if not await self._validate_task(task):
                raise ValueError(f"Task validation failed for {task.task_id}")
            
            # Update status
            task.started_at = datetime.utcnow()
            self.status = AgentStatus.BUSY
            
            # Execute agent-specific task logic
            result_data = await self._execute_task_specific(task)
            
            # Calculate execution time
            execution_time = (time.time() - start_time) * 1000
            self.execution_times.append(execution_time)
            
            # Update metrics
            self._update_performance_metrics(execution_time)
            
            # Create successful result
            result = TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=result_data,
                execution_time_ms=execution_time,
                resource_usage=await self._get_resource_usage(),
                metadata={'agent_type': self.agent_type}
            )
            
            logger.debug(f"Agent {self.name} completed task {task.task_id} in {execution_time:.1f}ms")
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            # Create failed result
            result = TaskResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time,
                metadata={'agent_type': self.agent_type}
            )
            
            logger.error(f"Agent {self.name} failed task {task.task_id}: {e}")
            return result
        
        finally:
            self.status = AgentStatus.IDLE
    
    @abstractmethod
    async def _initialize_agent_specific(self):
        """Agent-specific initialization logic"""
        pass
    
    @abstractmethod
    async def _cleanup_agent_specific(self):
        """Agent-specific cleanup logic"""
        pass
    
    @abstractmethod
    async def _execute_task_specific(self, task: Task) -> Any:
        """Agent-specific task execution logic"""
        pass
    
    async def _validate_capabilities(self) -> bool:
        """Validate agent capabilities"""
        for capability in self.capabilities:
            if not await self._test_capability(capability):
                logger.warning(f"Capability {capability.name} validation failed")
                return False
        return True
    
    async def _test_capability(self, capability: AgentCapability) -> bool:
        """Test a specific capability"""
        # Default implementation - override in specialized agents
        return True
    
    async def _validate_task(self, task: Task) -> bool:
        """Validate task before execution"""
        # Check if agent has required capabilities
        for req_capability in task.required_capabilities:
            if not self.has_capability(req_capability):
                logger.warning(f"Agent {self.name} missing capability: {req_capability}")
                return False
        
        # Check task constraints
        for constraint in task.constraints:
            if not await self._check_constraint(constraint):
                logger.warning(f"Task constraint not met: {constraint.constraint_type}")
                return False
        
        return True
    
    async def _check_constraint(self, constraint) -> bool:
        """Check if task constraint is satisfied"""
        # Default implementation - override for specific constraints
        return True
    
    async def _setup_monitoring(self):
        """Setup agent monitoring"""
        # Start periodic performance checks
        asyncio.create_task(self._performance_monitor())
    
    async def _performance_monitor(self):
        """Monitor agent performance"""
        while self.status != AgentStatus.OFFLINE:
            try:
                await asyncio.sleep(30.0)  # Check every 30 seconds
                await self._check_performance()
            except Exception as e:
                logger.error(f"Performance monitoring error for {self.name}: {e}")
    
    async def _check_performance(self):
        """Check agent performance against thresholds"""
        # Check execution time
        if self.execution_times:
            avg_time = sum(self.execution_times[-10:]) / min(len(self.execution_times), 10)
            if avg_time > self.performance_thresholds['max_execution_time_ms']:
                logger.warning(f"Agent {self.name} execution time above threshold: {avg_time:.1f}ms")
        
        # Check success rate
        if self.metrics.success_rate < self.performance_thresholds['min_success_rate']:
            logger.warning(f"Agent {self.name} success rate below threshold: {self.metrics.success_rate:.2f}")
        
        # Check memory usage
        memory_usage = await self._get_memory_usage()
        if memory_usage > self.performance_thresholds['max_memory_usage_mb']:
            logger.warning(f"Agent {self.name} memory usage above threshold: {memory_usage:.1f}MB")
    
    def _update_performance_metrics(self, execution_time: float):
        """Update performance metrics"""
        # Update average execution time
        total_time = self.metrics.avg_execution_time_ms * self.metrics.tasks_completed + execution_time
        self.metrics.avg_execution_time_ms = total_time / (self.metrics.tasks_completed + 1)
        
        # Limit execution time history
        if len(self.execution_times) > 100:
            self.execution_times = self.execution_times[-50:]
    
    async def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        return {
            'memory_mb': await self._get_memory_usage(),
            'cpu_percent': await self._get_cpu_usage()
        }
    
    async def _get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        # Simplified implementation - in practice would use psutil or similar
        return 128.0  # Default value
    
    async def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        # Simplified implementation - in practice would use psutil or similar
        return 25.0  # Default value
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'agent_type': self.agent_type,
            'status': self.status.value,
            'tasks_completed': self.metrics.tasks_completed,
            'tasks_failed': self.metrics.tasks_failed,
            'success_rate': self.metrics.success_rate,
            'avg_execution_time_ms': self.metrics.avg_execution_time_ms,
            'recent_execution_times': self.execution_times[-10:] if self.execution_times else [],
            'capabilities': [cap.name for cap in self.capabilities],
            'uptime_seconds': self.metrics.uptime_seconds
        }


class TaskProcessingMixin:
    """
    Mixin for common task processing functionality.
    """
    
    async def preprocess_task_data(self, task: Task) -> Any:
        """Preprocess task input data"""
        if task.input_data is None:
            return None
        
        # Basic data validation and preprocessing
        if isinstance(task.input_data, dict):
            return self._preprocess_dict_data(task.input_data)
        elif isinstance(task.input_data, str):
            return self._preprocess_string_data(task.input_data)
        elif isinstance(task.input_data, list):
            return self._preprocess_list_data(task.input_data)
        else:
            return task.input_data
    
    def _preprocess_dict_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess dictionary data"""
        # Remove None values and empty strings
        cleaned_data = {}
        for key, value in data.items():
            if value is not None and value != "":
                cleaned_data[key] = value
        return cleaned_data
    
    def _preprocess_string_data(self, data: str) -> str:
        """Preprocess string data"""
        # Basic string cleaning
        return data.strip()
    
    def _preprocess_list_data(self, data: List[Any]) -> List[Any]:
        """Preprocess list data"""
        # Remove None values
        return [item for item in data if item is not None]
    
    async def postprocess_result_data(self, result_data: Any, task: Task) -> Any:
        """Postprocess result data"""
        if result_data is None:
            return None
        
        # Format result based on expected output type
        expected_type = task.expected_output_type.lower()
        
        if expected_type == "json" and not isinstance(result_data, (dict, list)):
            return {"result": result_data}
        elif expected_type == "string" and not isinstance(result_data, str):
            return str(result_data)
        elif expected_type == "list" and not isinstance(result_data, list):
            return [result_data]
        
        return result_data


class ErrorHandlingMixin:
    """
    Mixin for common error handling functionality.
    """
    
    async def handle_task_error(self, task: Task, error: Exception) -> TaskResult:
        """Handle task execution error"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log error with context
        logger.error(f"Task {task.task_id} failed with {error_type}: {error_message}")
        
        # Determine if task should be retried
        should_retry = await self._should_retry_task(task, error)
        
        # Create error result
        result = TaskResult(
            task_id=task.task_id,
            success=False,
            error_message=f"{error_type}: {error_message}",
            metadata={
                'error_type': error_type,
                'should_retry': should_retry,
                'retry_count': task.retry_count
            }
        )
        
        return result
    
    async def _should_retry_task(self, task: Task, error: Exception) -> bool:
        """Determine if task should be retried"""
        # Don't retry if max retries reached
        if task.retry_count >= task.max_retries:
            return False
        
        # Don't retry for certain error types
        non_retryable_errors = [ValueError, TypeError, AttributeError]
        if type(error) in non_retryable_errors:
            return False
        
        # Retry for network errors, timeouts, etc.
        retryable_errors = [ConnectionError, TimeoutError, asyncio.TimeoutError]
        if type(error) in retryable_errors:
            return True
        
        # Default: retry for unknown errors
        return True
    
    def log_performance_warning(self, metric_name: str, value: float, threshold: float):
        """Log performance warning"""
        logger.warning(
            f"Agent {getattr(self, 'name', 'unknown')} performance warning: "
            f"{metric_name} = {value:.2f} (threshold: {threshold:.2f})"
        )
    
    def log_capability_error(self, capability_name: str, error: str):
        """Log capability error"""
        logger.error(
            f"Agent {getattr(self, 'name', 'unknown')} capability error: "
            f"{capability_name} - {error}"
        )


class CommunicationMixin:
    """
    Mixin for agent communication functionality.
    """
    
    async def send_status_update(self, status_data: Dict[str, Any]):
        """Send status update to coordinator"""
        if hasattr(self, 'coordinator') and self.coordinator:
            message = Message(
                sender_id=getattr(self, 'agent_id', 'unknown'),
                recipient_id='coordinator',
                message_type='status_update',
                payload=status_data
            )
            # In a real implementation, would send through communication hub
            logger.debug(f"Status update sent: {status_data}")
    
    async def request_resource(self, resource_type: str, amount: float) -> bool:
        """Request resource allocation"""
        if hasattr(self, 'coordinator') and self.coordinator:
            message = Message(
                sender_id=getattr(self, 'agent_id', 'unknown'),
                recipient_id='coordinator',
                message_type='resource_request',
                payload={
                    'resource_type': resource_type,
                    'amount': amount
                }
            )
            # In a real implementation, would send through communication hub
            logger.debug(f"Resource request sent: {resource_type} = {amount}")
            return True
        return False
    
    async def notify_task_completion(self, task_id: str, result: TaskResult):
        """Notify coordinator of task completion"""
        if hasattr(self, 'coordinator') and self.coordinator:
            await self.coordinator.receive_task_result(
                getattr(self, 'agent_id', 'unknown'), 
                result
            )
    
    async def notify_error(self, error_type: str, error_message: str):
        """Notify coordinator of error"""
        if hasattr(self, 'coordinator') and self.coordinator:
            message = Message(
                sender_id=getattr(self, 'agent_id', 'unknown'),
                recipient_id='coordinator',
                message_type='error_notification',
                payload={
                    'error_type': error_type,
                    'error_message': error_message,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            # In a real implementation, would send through communication hub
            logger.debug(f"Error notification sent: {error_type}")


class ConfigurableAgent(SpecializedAgent, TaskProcessingMixin, ErrorHandlingMixin, CommunicationMixin):
    """
    Configurable agent that combines all mixins for maximum functionality.
    """
    
    def __init__(self, agent_id: str, name: str, agent_type: str, capabilities: List[AgentCapability]):
        super().__init__(agent_id, name, agent_type, capabilities)
        
        # Configuration options
        self.config_schema = {}
        self.default_config = {}
    
    async def _initialize_agent_specific(self):
        """Initialize configurable agent"""
        # Load configuration
        await self._load_configuration()
        
        # Validate configuration
        if not await self._validate_configuration():
            raise ValueError("Configuration validation failed")
        
        logger.info(f"Configurable agent {self.name} initialized with config")
    
    async def _cleanup_agent_specific(self):
        """Cleanup configurable agent"""
        # Save any state if needed
        await self._save_state()
        
        logger.info(f"Configurable agent {self.name} cleanup completed")
    
    async def _execute_task_specific(self, task: Task) -> Any:
        """Execute task with full processing pipeline"""
        # Preprocess input data
        processed_input = await self.preprocess_task_data(task)
        
        # Execute core task logic (to be implemented by subclasses)
        result_data = await self._execute_core_logic(processed_input, task)
        
        # Postprocess result data
        final_result = await self.postprocess_result_data(result_data, task)
        
        return final_result
    
    async def _execute_core_logic(self, input_data: Any, task: Task) -> Any:
        """Core task execution logic - override in subclasses"""
        # Default implementation
        return {"status": "completed", "input_received": input_data is not None}
    
    async def _load_configuration(self):
        """Load agent configuration"""
        # Default implementation - override for specific config sources
        self.config.update(self.default_config)
    
    async def _validate_configuration(self) -> bool:
        """Validate agent configuration"""
        # Default implementation - override for specific validation
        return True
    
    async def _save_state(self):
        """Save agent state"""
        # Default implementation - override for state persistence
        pass
