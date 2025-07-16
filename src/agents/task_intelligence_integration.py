"""
Task Intelligence System Integration Layer

Integrates the Task Intelligence System with existing PyGent Factory infrastructure:
- TaskDispatcher integration for intelligent task routing
- A2A Protocol integration for agent coordination
- MCP Server integration for context gathering
- Database integration for persistence
- Real-time monitoring and analytics
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .supervisor_agent import TaskIntelligenceSystem, MetaSupervisorAgent
from ..orchestration.task_dispatcher import TaskDispatcher
from ..orchestration.coordination_models import TaskRequest, TaskStatus
from ..a2a_protocol.manager import A2AManager
from ..orchestration.mcp_orchestrator import MCPOrchestrator
from ..database.models import Task, Agent
from ..database.production_manager import db_manager

logger = logging.getLogger(__name__)


class TaskIntelligenceIntegration:
    """
    Integration layer for Task Intelligence System with PyGent Factory infrastructure
    """
    
    def __init__(self, 
                 task_dispatcher: TaskDispatcher,
                 a2a_manager: A2AManager,
                 mcp_orchestrator: MCPOrchestrator):
        self.task_dispatcher = task_dispatcher
        self.a2a_manager = a2a_manager
        self.mcp_orchestrator = mcp_orchestrator
        
        # Task Intelligence components
        self.task_intelligence = TaskIntelligenceSystem(
            mcp_manager=mcp_orchestrator,
            a2a_manager=a2a_manager
        )
        self.meta_supervisor = MetaSupervisorAgent(
            mcp_manager=mcp_orchestrator,
            a2a_manager=a2a_manager
        )
        
        # Integration state
        self.active_intelligent_tasks: Dict[str, str] = {}  # task_id -> intelligence_id
        self.task_performance_tracking: Dict[str, Dict[str, Any]] = {}
        self.integration_metrics = {
            "tasks_processed": 0,
            "intelligence_success_rate": 0.0,
            "average_improvement": 0.0,
            "pattern_applications": 0
        }
        
        # Configuration
        self.enable_intelligent_routing = True
        self.enable_pattern_learning = True
        self.enable_failure_analysis = True
        self.complexity_threshold = 5  # Tasks above this use Task Intelligence
        
    async def initialize(self) -> bool:
        """Initialize the integration layer"""
        try:
            logger.info("Initializing Task Intelligence Integration...")
            
            # Hook into TaskDispatcher for intelligent routing
            await self._setup_task_dispatcher_integration()
            
            # Setup database persistence for patterns and analytics
            await self._setup_database_integration()
            
            # Start monitoring and optimization loops
            asyncio.create_task(self._monitoring_loop())
            asyncio.create_task(self._pattern_optimization_loop())
            
            logger.info("Task Intelligence Integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Task Intelligence Integration: {e}")
            return False
    
    async def _setup_task_dispatcher_integration(self):
        """Integrate with TaskDispatcher for intelligent task routing"""
        
        # Store original dispatch method
        original_dispatch = self.task_dispatcher._dispatch_task
        
        async def intelligent_dispatch_wrapper(task: TaskRequest) -> bool:
            """Enhanced dispatch with Task Intelligence"""
            
            try:
                # Analyze task complexity
                complexity = await self._analyze_task_complexity(task)
                
                if complexity >= self.complexity_threshold and self.enable_intelligent_routing:
                    # Use Task Intelligence System
                    return await self._dispatch_with_intelligence(task, complexity)
                else:
                    # Use standard dispatch
                    return await original_dispatch(task)
                    
            except Exception as e:
                logger.error(f"Intelligent dispatch failed for task {task.task_id}: {e}")
                # Fallback to standard dispatch
                return await original_dispatch(task)
        
        # Replace dispatch method
        self.task_dispatcher._dispatch_task = intelligent_dispatch_wrapper
        
        logger.info("TaskDispatcher integration completed")
    
    async def _analyze_task_complexity(self, task: TaskRequest) -> int:
        """Analyze task complexity to determine if Task Intelligence is needed"""
        
        complexity_factors = 0
        
        # Description complexity
        word_count = len(task.description.split())
        if word_count > 50:
            complexity_factors += 3
        elif word_count > 20:
            complexity_factors += 2
        elif word_count > 10:
            complexity_factors += 1
        
        # Dependencies
        complexity_factors += len(task.dependencies)
        
        # Task type complexity
        complex_types = ["multi_agent", "integration", "research", "complex_analysis"]
        if task.task_type in complex_types:
            complexity_factors += 2
        
        # Input data complexity
        if task.input_data and isinstance(task.input_data, dict):
            complexity_factors += min(2, len(task.input_data) // 3)
        
        return min(10, complexity_factors)
    
    async def _dispatch_with_intelligence(self, task: TaskRequest, complexity: int) -> bool:
        """Dispatch task using Task Intelligence System"""
        
        try:
            # Convert TaskRequest to Task Intelligence format
            task_context = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "complexity": complexity,
                "dependencies": task.dependencies,
                "input_data": task.input_data,
                "priority": getattr(task, 'priority', 5)
            }
            
            # Check if task requires meta-coordination (multiple domains)
            if complexity >= 8 or len(task.dependencies) > 3:
                # Use MetaSupervisorAgent for complex workflows
                workflow_id = await self.meta_supervisor.coordinate_complex_task(
                    task.description, task_context
                )
                self.active_intelligent_tasks[task.task_id] = f"meta_{workflow_id}"
                
                # Start monitoring workflow
                asyncio.create_task(self._monitor_meta_workflow(task, workflow_id))
                
            else:
                # Use standard Task Intelligence System
                intelligence_id = await self.task_intelligence.create_task_intelligence(
                    task.description, task_context
                )
                self.active_intelligent_tasks[task.task_id] = intelligence_id
                
                # Start monitoring task
                asyncio.create_task(self._monitor_intelligent_task(task, intelligence_id))
            
            # Update task status
            task.update_status(TaskStatus.RUNNING)
            
            # Record performance tracking
            self.task_performance_tracking[task.task_id] = {
                "start_time": datetime.utcnow(),
                "complexity": complexity,
                "intelligence_type": "meta" if complexity >= 8 else "standard",
                "success": None,
                "completion_time": None,
                "quality_score": None
            }
            
            logger.info(f"Dispatched task {task.task_id} with Task Intelligence (complexity: {complexity})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to dispatch task {task.task_id} with intelligence: {e}")
            return False
    
    async def _monitor_intelligent_task(self, task: TaskRequest, intelligence_id: str):
        """Monitor Task Intelligence execution"""
        
        try:
            start_time = datetime.utcnow()
            
            # Monitor until completion
            while True:
                # Check if task is complete
                if intelligence_id in self.task_intelligence.progress_ledgers:
                    progress_ledger = self.task_intelligence.progress_ledgers[intelligence_id]
                    
                    if progress_ledger.is_task_complete():
                        # Task completed successfully
                        completion_time = datetime.utcnow()
                        execution_time = (completion_time - start_time).total_seconds()
                        
                        # Calculate quality score
                        quality_score = await self._calculate_quality_score(intelligence_id)
                        
                        # Update task status
                        task.update_status(TaskStatus.COMPLETED)
                        
                        # Record pattern for learning
                        if self.enable_pattern_learning:
                            pattern_id = await self.task_intelligence.record_workflow_pattern(
                                intelligence_id, True, execution_time, quality_score
                            )
                            if pattern_id:
                                self.integration_metrics["pattern_applications"] += 1
                        
                        # Update performance tracking
                        await self._update_performance_tracking(task.task_id, True, execution_time, quality_score)
                        
                        logger.info(f"Task {task.task_id} completed successfully with quality score {quality_score:.2f}")
                        break
                
                # Check for timeout (2 hours max)
                elapsed = datetime.utcnow() - start_time
                if elapsed.total_seconds() > 7200:
                    await self._handle_task_timeout(task, intelligence_id)
                    break
                
                # Brief pause
                await asyncio.sleep(10)
                
        except Exception as e:
            logger.error(f"Error monitoring intelligent task {task.task_id}: {e}")
            await self._handle_task_failure(task, intelligence_id, str(e))
    
    async def _monitor_meta_workflow(self, task: TaskRequest, workflow_id: str):
        """Monitor MetaSupervisorAgent workflow execution"""
        
        try:
            start_time = datetime.utcnow()
            
            # Monitor until completion
            while True:
                workflow_status = self.meta_supervisor.get_workflow_status(workflow_id)
                
                if workflow_status and workflow_status["status"] in ["completed", "failed"]:
                    completion_time = datetime.utcnow()
                    execution_time = (completion_time - start_time).total_seconds()
                    
                    success = workflow_status["status"] == "completed"
                    
                    if success:
                        task.update_status(TaskStatus.COMPLETED)
                        quality_score = 0.9  # High quality for successful meta-coordination
                    else:
                        task.update_status(TaskStatus.FAILED)
                        quality_score = 0.0
                        
                        # Analyze failure for learning
                        if self.enable_failure_analysis:
                            await self._analyze_workflow_failure(workflow_id, workflow_status)
                    
                    # Update performance tracking
                    await self._update_performance_tracking(task.task_id, success, execution_time, quality_score)
                    
                    logger.info(f"Meta-workflow {workflow_id} for task {task.task_id} {'completed' if success else 'failed'}")
                    break
                
                # Check for timeout
                elapsed = datetime.utcnow() - start_time
                if elapsed.total_seconds() > 10800:  # 3 hours for complex workflows
                    await self._handle_workflow_timeout(task, workflow_id)
                    break
                
                await asyncio.sleep(15)
                
        except Exception as e:
            logger.error(f"Error monitoring meta-workflow {workflow_id}: {e}")
            await self._handle_task_failure(task, f"meta_{workflow_id}", str(e))

    async def _calculate_quality_score(self, intelligence_id: str) -> float:
        """Calculate quality score for completed task"""

        try:
            if intelligence_id not in self.task_intelligence.progress_ledgers:
                return 0.5  # Default score

            progress_ledger = self.task_intelligence.progress_ledgers[intelligence_id]

            # Calculate quality based on multiple factors
            completion_rate = len([s for s in progress_ledger.step_status.values() if s == "completed"]) / max(1, len(progress_ledger.step_status))
            stall_penalty = max(0, 1.0 - (progress_ledger.stall_count * 0.1))

            # Time efficiency (bonus for completing faster than estimated)
            time_efficiency = 1.0  # Default
            if intelligence_id in self.task_intelligence.task_ledgers:
                task_ledger = self.task_intelligence.task_ledgers[intelligence_id]
                estimated_time = sum(step.get("estimated_time", 15) for step in task_ledger.current_plan)
                actual_time = (datetime.utcnow() - progress_ledger.last_progress_time).total_seconds() / 60

                if estimated_time > 0:
                    time_efficiency = min(1.2, estimated_time / max(1, actual_time))

            quality_score = (completion_rate * 0.5 + stall_penalty * 0.3 + time_efficiency * 0.2)
            return min(1.0, quality_score)

        except Exception as e:
            logger.error(f"Failed to calculate quality score for {intelligence_id}: {e}")
            return 0.5

    async def _update_performance_tracking(self, task_id: str, success: bool, execution_time: float, quality_score: float):
        """Update performance tracking metrics"""

        if task_id in self.task_performance_tracking:
            tracking = self.task_performance_tracking[task_id]
            tracking["success"] = success
            tracking["completion_time"] = execution_time
            tracking["quality_score"] = quality_score

            # Update integration metrics
            self.integration_metrics["tasks_processed"] += 1

            # Update success rate (exponential moving average)
            alpha = 0.1
            current_success_rate = self.integration_metrics["intelligence_success_rate"]
            new_success = 1.0 if success else 0.0
            self.integration_metrics["intelligence_success_rate"] = current_success_rate * (1 - alpha) + new_success * alpha

            # Update average improvement (quality score as improvement metric)
            current_improvement = self.integration_metrics["average_improvement"]
            self.integration_metrics["average_improvement"] = current_improvement * (1 - alpha) + quality_score * alpha

    async def _handle_task_timeout(self, task: TaskRequest, intelligence_id: str):
        """Handle task timeout"""

        logger.warning(f"Task {task.task_id} timed out")

        task.update_status(TaskStatus.FAILED)

        # Record failure pattern
        if self.enable_failure_analysis:
            failure_details = {
                "type": "timeout",
                "root_cause": "execution_timeout",
                "task_context": {
                    "task_type": task.task_type,
                    "complexity": self.task_performance_tracking.get(task.task_id, {}).get("complexity", 0)
                }
            }

            await self.task_intelligence.record_failure_pattern(intelligence_id, failure_details)

        # Update performance tracking
        await self._update_performance_tracking(task.task_id, False, 7200, 0.0)

    async def _handle_task_failure(self, task: TaskRequest, intelligence_id: str, error: str):
        """Handle task failure"""

        logger.error(f"Task {task.task_id} failed: {error}")

        task.update_status(TaskStatus.FAILED)

        # Record failure pattern
        if self.enable_failure_analysis:
            failure_details = {
                "type": "execution_error",
                "root_cause": error,
                "task_context": {
                    "task_type": task.task_type,
                    "complexity": self.task_performance_tracking.get(task.task_id, {}).get("complexity", 0)
                }
            }

            # Extract intelligence_id if it's a meta workflow
            actual_intelligence_id = intelligence_id.replace("meta_", "") if intelligence_id.startswith("meta_") else intelligence_id
            await self.task_intelligence.record_failure_pattern(actual_intelligence_id, failure_details)

        # Update performance tracking
        execution_time = (datetime.utcnow() - self.task_performance_tracking.get(task.task_id, {}).get("start_time", datetime.utcnow())).total_seconds()
        await self._update_performance_tracking(task.task_id, False, execution_time, 0.0)

    async def _handle_workflow_timeout(self, task: TaskRequest, workflow_id: str):
        """Handle meta-workflow timeout"""

        logger.warning(f"Meta-workflow {workflow_id} for task {task.task_id} timed out")

        task.update_status(TaskStatus.FAILED)

        # Update performance tracking
        await self._update_performance_tracking(task.task_id, False, 10800, 0.0)

    async def _analyze_workflow_failure(self, workflow_id: str, workflow_status: Dict[str, Any]):
        """Analyze meta-workflow failure for learning"""

        try:
            failure_details = {
                "type": "meta_workflow_failure",
                "root_cause": workflow_status.get("error", "unknown"),
                "workflow_context": {
                    "workflow_id": workflow_id,
                    "completion_status": workflow_status.get("completion_status", {}),
                    "supervisor_tasks": workflow_status.get("supervisor_tasks", {})
                }
            }

            # Record failure pattern (use workflow_id as intelligence_id)
            await self.task_intelligence.record_failure_pattern(workflow_id, failure_details)

        except Exception as e:
            logger.error(f"Failed to analyze workflow failure for {workflow_id}: {e}")

    async def _setup_database_integration(self):
        """Setup database integration for persistence"""

        try:
            # This would create tables for pattern storage if needed
            # For now, we'll use in-memory storage with periodic persistence

            # Start periodic persistence task
            asyncio.create_task(self._periodic_persistence_loop())

            logger.info("Database integration setup completed")

        except Exception as e:
            logger.error(f"Failed to setup database integration: {e}")

    async def _periodic_persistence_loop(self):
        """Periodically persist patterns and metrics to database"""

        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Persist workflow patterns
                await self._persist_workflow_patterns()

                # Persist integration metrics
                await self._persist_integration_metrics()

            except Exception as e:
                logger.error(f"Periodic persistence failed: {e}")

    async def _persist_workflow_patterns(self):
        """Persist workflow patterns to database"""

        try:
            # This would save patterns to database
            # For now, just log the count
            pattern_count = len(self.task_intelligence.workflow_patterns)
            failure_count = len(self.task_intelligence.failure_patterns)

            logger.debug(f"Persisting {pattern_count} workflow patterns and {failure_count} failure patterns")

        except Exception as e:
            logger.error(f"Failed to persist workflow patterns: {e}")

    async def _persist_integration_metrics(self):
        """Persist integration metrics to database"""

        try:
            # This would save metrics to database
            logger.debug(f"Persisting integration metrics: {self.integration_metrics}")

        except Exception as e:
            logger.error(f"Failed to persist integration metrics: {e}")

    async def _monitoring_loop(self):
        """Main monitoring loop for integration health"""

        while True:
            try:
                await asyncio.sleep(60)  # Every minute

                # Monitor active intelligent tasks
                await self._monitor_active_tasks()

                # Check system health
                await self._check_system_health()

                # Generate optimization suggestions
                await self._generate_optimization_suggestions()

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

    async def _monitor_active_tasks(self):
        """Monitor active intelligent tasks"""

        try:
            active_count = len(self.active_intelligent_tasks)

            if active_count > 0:
                logger.debug(f"Monitoring {active_count} active intelligent tasks")

                # Check for stalled tasks
                current_time = datetime.utcnow()
                stalled_tasks = []

                for task_id, intelligence_id in self.active_intelligent_tasks.items():
                    if task_id in self.task_performance_tracking:
                        start_time = self.task_performance_tracking[task_id]["start_time"]
                        elapsed = (current_time - start_time).total_seconds()

                        # Consider task stalled if running for more than 1 hour without completion
                        if elapsed > 3600:
                            stalled_tasks.append(task_id)

                if stalled_tasks:
                    logger.warning(f"Found {len(stalled_tasks)} potentially stalled tasks")

        except Exception as e:
            logger.error(f"Failed to monitor active tasks: {e}")

    async def _check_system_health(self):
        """Check overall system health"""

        try:
            # Check success rate
            success_rate = self.integration_metrics["intelligence_success_rate"]
            if success_rate < 0.7:
                logger.warning(f"Task Intelligence success rate is low: {success_rate:.2f}")

            # Check average improvement
            avg_improvement = self.integration_metrics["average_improvement"]
            if avg_improvement < 0.6:
                logger.warning(f"Average task improvement is low: {avg_improvement:.2f}")

            # Check pattern application rate
            tasks_processed = self.integration_metrics["tasks_processed"]
            pattern_applications = self.integration_metrics["pattern_applications"]

            if tasks_processed > 10 and pattern_applications / tasks_processed < 0.3:
                logger.info("Low pattern application rate - consider adjusting pattern matching threshold")

        except Exception as e:
            logger.error(f"System health check failed: {e}")

    async def _generate_optimization_suggestions(self):
        """Generate optimization suggestions based on performance"""

        try:
            # This would analyze performance data and generate suggestions
            # For now, just run pattern library optimization
            await self.task_intelligence.optimize_pattern_library()

        except Exception as e:
            logger.error(f"Failed to generate optimization suggestions: {e}")

    async def _pattern_optimization_loop(self):
        """Pattern optimization loop"""

        while True:
            try:
                await asyncio.sleep(1800)  # Every 30 minutes

                # Optimize pattern library
                await self.task_intelligence.optimize_pattern_library()

                # Update pattern matching threshold based on performance
                await self._adjust_pattern_matching_threshold()

            except Exception as e:
                logger.error(f"Pattern optimization loop error: {e}")

    async def _adjust_pattern_matching_threshold(self):
        """Adjust pattern matching threshold based on performance"""

        try:
            success_rate = self.integration_metrics["intelligence_success_rate"]

            # If success rate is high, we can be more selective with patterns
            if success_rate > 0.85:
                self.task_intelligence.pattern_matching_threshold = min(0.8, self.task_intelligence.pattern_matching_threshold + 0.05)
            # If success rate is low, be less selective
            elif success_rate < 0.65:
                self.task_intelligence.pattern_matching_threshold = max(0.4, self.task_intelligence.pattern_matching_threshold - 0.05)

        except Exception as e:
            logger.error(f"Failed to adjust pattern matching threshold: {e}")

    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and metrics"""

        return {
            "active_intelligent_tasks": len(self.active_intelligent_tasks),
            "integration_metrics": self.integration_metrics.copy(),
            "task_intelligence_stats": self.task_intelligence.get_question_generation_stats(),
            "pattern_analytics": self.task_intelligence.get_pattern_analytics(),
            "meta_supervisor_performance": self.meta_supervisor.get_supervisor_performance(),
            "configuration": {
                "enable_intelligent_routing": self.enable_intelligent_routing,
                "enable_pattern_learning": self.enable_pattern_learning,
                "enable_failure_analysis": self.enable_failure_analysis,
                "complexity_threshold": self.complexity_threshold,
                "pattern_matching_threshold": self.task_intelligence.pattern_matching_threshold
            }
        }
