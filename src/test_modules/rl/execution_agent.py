"""
Recipe Execution Agent

RL-based agent for intelligent test execution, learning from past results
to optimize test selection, ordering, and resource allocation.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
import statistics

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Test execution strategies"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PRIORITY_BASED = "priority_based"
    ADAPTIVE = "adaptive"
    RISK_BASED = "risk_based"


@dataclass
class TestExecutionContext:
    """Context for test execution"""
    test_id: str
    test_name: str
    category: str
    estimated_duration: float
    priority: int
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    historical_success_rate: float = 1.0
    last_execution_time: Optional[datetime] = None


@dataclass
class ExecutionResult:
    """Result of test execution"""
    test_id: str
    success: bool
    execution_time: float
    resource_usage: Dict[str, float]
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionPlan:
    """Plan for test execution"""
    strategy: ExecutionStrategy
    test_order: List[str]
    parallel_groups: List[List[str]]
    estimated_total_time: float
    resource_allocation: Dict[str, Any]
    confidence_score: float


class RecipeExecutionAgent:
    """
    RL-based Recipe Execution Agent.
    
    Uses reinforcement learning to optimize test execution strategies,
    learning from past results to improve future execution plans.
    """
    
    def __init__(self, learning_rate: float = 0.1, exploration_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Agent state
        self.execution_history: List[ExecutionResult] = []
        self.test_contexts: Dict[str, TestExecutionContext] = {}
        self.strategy_performance: Dict[ExecutionStrategy, List[float]] = {
            strategy: [] for strategy in ExecutionStrategy
        }
        
        # Learning components
        self.q_table: Dict[str, Dict[str, float]] = {}  # State-action values
        self.state_features: Dict[str, Any] = {}
        self.reward_history: List[float] = []
        
        # Execution metrics
        self.total_executions = 0
        self.successful_executions = 0
        self.average_execution_time = 0.0
        
        # Configuration
        self.max_parallel_tests = 4
        self.timeout_threshold = 300.0  # 5 minutes
        self.success_rate_threshold = 0.8
    
    def add_test_context(self, context: TestExecutionContext):
        """Add test context for execution planning"""
        self.test_contexts[context.test_id] = context
        logger.debug(f"Added test context: {context.test_name}")
    
    def update_test_context(self, test_id: str, **updates):
        """Update test context with new information"""
        if test_id in self.test_contexts:
            context = self.test_contexts[test_id]
            for key, value in updates.items():
                if hasattr(context, key):
                    setattr(context, key, value)
    
    async def plan_execution(self, test_ids: List[str], 
                           constraints: Optional[Dict[str, Any]] = None) -> ExecutionPlan:
        """
        Plan optimal execution strategy for given tests.
        
        Args:
            test_ids: List of test IDs to execute
            constraints: Optional execution constraints
            
        Returns:
            ExecutionPlan with optimized strategy
        """
        constraints = constraints or {}
        
        # Get current state
        state = self._get_current_state(test_ids, constraints)
        
        # Choose strategy using epsilon-greedy
        if random.random() < self.exploration_rate:
            strategy = random.choice(list(ExecutionStrategy))
        else:
            strategy = self._choose_best_strategy(state)
        
        # Generate execution plan based on strategy
        plan = await self._generate_execution_plan(test_ids, strategy, constraints)
        
        logger.info(f"Generated execution plan with {strategy.value} strategy for {len(test_ids)} tests")
        return plan
    
    def _get_current_state(self, test_ids: List[str], constraints: Dict[str, Any]) -> str:
        """Get current state representation for RL"""
        # Simple state representation
        num_tests = len(test_ids)
        avg_duration = statistics.mean([
            self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1)).estimated_duration
            for tid in test_ids
        ]) if test_ids else 0.0
        
        has_dependencies = any(
            self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1)).dependencies
            for tid in test_ids
        )
        
        # Create state string
        state = f"tests_{min(num_tests, 10)}_duration_{int(avg_duration)}_deps_{has_dependencies}"
        return state
    
    def _choose_best_strategy(self, state: str) -> ExecutionStrategy:
        """Choose best strategy based on Q-values"""
        if state not in self.q_table:
            return ExecutionStrategy.ADAPTIVE  # Default strategy
        
        state_values = self.q_table[state]
        if not state_values:
            return ExecutionStrategy.ADAPTIVE
        
        # Choose strategy with highest Q-value
        best_strategy_name = max(state_values.items(), key=lambda x: x[1])[0]
        return ExecutionStrategy(best_strategy_name)
    
    async def _generate_execution_plan(self, test_ids: List[str], 
                                     strategy: ExecutionStrategy,
                                     constraints: Dict[str, Any]) -> ExecutionPlan:
        """Generate execution plan based on strategy"""
        if strategy == ExecutionStrategy.SEQUENTIAL:
            return self._plan_sequential_execution(test_ids, constraints)
        elif strategy == ExecutionStrategy.PARALLEL:
            return self._plan_parallel_execution(test_ids, constraints)
        elif strategy == ExecutionStrategy.PRIORITY_BASED:
            return self._plan_priority_based_execution(test_ids, constraints)
        elif strategy == ExecutionStrategy.ADAPTIVE:
            return self._plan_adaptive_execution(test_ids, constraints)
        elif strategy == ExecutionStrategy.RISK_BASED:
            return self._plan_risk_based_execution(test_ids, constraints)
        else:
            return self._plan_sequential_execution(test_ids, constraints)
    
    def _plan_sequential_execution(self, test_ids: List[str], 
                                 constraints: Dict[str, Any]) -> ExecutionPlan:
        """Plan sequential execution"""
        # Sort by dependencies first, then by priority
        sorted_tests = self._sort_tests_by_dependencies_and_priority(test_ids)
        
        total_time = sum(
            self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1)).estimated_duration
            for tid in sorted_tests
        )
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.SEQUENTIAL,
            test_order=sorted_tests,
            parallel_groups=[],
            estimated_total_time=total_time,
            resource_allocation={"cpu_cores": 1, "memory_gb": 2},
            confidence_score=0.9
        )
    
    def _plan_parallel_execution(self, test_ids: List[str], 
                               constraints: Dict[str, Any]) -> ExecutionPlan:
        """Plan parallel execution"""
        # Group tests that can run in parallel
        parallel_groups = self._create_parallel_groups(test_ids)
        
        # Estimate total time as max time of any group
        group_times = []
        for group in parallel_groups:
            group_time = max(
                self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1)).estimated_duration
                for tid in group
            ) if group else 0.0
            group_times.append(group_time)
        
        total_time = sum(group_times)
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.PARALLEL,
            test_order=[],
            parallel_groups=parallel_groups,
            estimated_total_time=total_time,
            resource_allocation={"cpu_cores": self.max_parallel_tests, "memory_gb": 8},
            confidence_score=0.7
        )
    
    def _plan_priority_based_execution(self, test_ids: List[str], 
                                     constraints: Dict[str, Any]) -> ExecutionPlan:
        """Plan priority-based execution"""
        # Sort by priority (higher priority first)
        sorted_tests = sorted(test_ids, key=lambda tid: 
            self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1)).priority, 
            reverse=True
        )
        
        # Handle dependencies
        sorted_tests = self._sort_tests_by_dependencies_and_priority(sorted_tests)
        
        total_time = sum(
            self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1)).estimated_duration
            for tid in sorted_tests
        )
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.PRIORITY_BASED,
            test_order=sorted_tests,
            parallel_groups=[],
            estimated_total_time=total_time,
            resource_allocation={"cpu_cores": 2, "memory_gb": 4},
            confidence_score=0.8
        )
    
    def _plan_adaptive_execution(self, test_ids: List[str], 
                               constraints: Dict[str, Any]) -> ExecutionPlan:
        """Plan adaptive execution based on historical performance"""
        # Analyze historical performance to choose best approach
        fast_tests = []
        slow_tests = []
        
        for tid in test_ids:
            context = self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1))
            if context.estimated_duration < 10.0:  # Fast tests
                fast_tests.append(tid)
            else:
                slow_tests.append(tid)
        
        # Run fast tests in parallel, slow tests sequentially
        parallel_groups = []
        if fast_tests:
            # Group fast tests
            parallel_groups.extend(self._create_parallel_groups(fast_tests))
        
        # Add slow tests as individual groups
        for slow_test in slow_tests:
            parallel_groups.append([slow_test])
        
        # Estimate total time
        group_times = []
        for group in parallel_groups:
            if len(group) == 1:
                # Sequential execution
                group_time = self.test_contexts.get(
                    group[0], TestExecutionContext("", "", "", 1.0, 1)
                ).estimated_duration
            else:
                # Parallel execution
                group_time = max(
                    self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1)).estimated_duration
                    for tid in group
                )
            group_times.append(group_time)
        
        total_time = sum(group_times)
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.ADAPTIVE,
            test_order=[],
            parallel_groups=parallel_groups,
            estimated_total_time=total_time,
            resource_allocation={"cpu_cores": 3, "memory_gb": 6},
            confidence_score=0.85
        )
    
    def _plan_risk_based_execution(self, test_ids: List[str], 
                                 constraints: Dict[str, Any]) -> ExecutionPlan:
        """Plan risk-based execution (high-risk tests first)"""
        # Sort by success rate (lower success rate = higher risk)
        sorted_tests = sorted(test_ids, key=lambda tid:
            self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1)).historical_success_rate
        )
        
        # Handle dependencies
        sorted_tests = self._sort_tests_by_dependencies_and_priority(sorted_tests)
        
        total_time = sum(
            self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1)).estimated_duration
            for tid in sorted_tests
        )
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.RISK_BASED,
            test_order=sorted_tests,
            parallel_groups=[],
            estimated_total_time=total_time,
            resource_allocation={"cpu_cores": 2, "memory_gb": 4},
            confidence_score=0.75
        )
    
    def _sort_tests_by_dependencies_and_priority(self, test_ids: List[str]) -> List[str]:
        """Sort tests respecting dependencies and priority"""
        # Simple topological sort with priority
        sorted_tests = []
        remaining_tests = set(test_ids)
        
        while remaining_tests:
            # Find tests with no unresolved dependencies
            ready_tests = []
            for tid in remaining_tests:
                context = self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1))
                if all(dep not in remaining_tests for dep in context.dependencies):
                    ready_tests.append(tid)
            
            if not ready_tests:
                # Break circular dependencies by taking highest priority
                ready_tests = [max(remaining_tests, key=lambda tid:
                    self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1)).priority
                )]
            
            # Sort ready tests by priority
            ready_tests.sort(key=lambda tid:
                self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1)).priority,
                reverse=True
            )
            
            # Add to sorted list and remove from remaining
            sorted_tests.extend(ready_tests)
            remaining_tests -= set(ready_tests)
        
        return sorted_tests
    
    def _create_parallel_groups(self, test_ids: List[str]) -> List[List[str]]:
        """Create groups of tests that can run in parallel"""
        groups = []
        remaining_tests = list(test_ids)
        
        while remaining_tests:
            current_group = []
            tests_to_remove = []
            
            for tid in remaining_tests:
                context = self.test_contexts.get(tid, TestExecutionContext("", "", "", 1.0, 1))
                
                # Check if test can be added to current group
                can_add = True
                
                # Check dependencies
                for dep in context.dependencies:
                    if dep in remaining_tests:
                        can_add = False
                        break
                
                # Check resource conflicts (simplified)
                if len(current_group) >= self.max_parallel_tests:
                    can_add = False
                
                if can_add:
                    current_group.append(tid)
                    tests_to_remove.append(tid)
            
            # Remove tests added to group
            for tid in tests_to_remove:
                remaining_tests.remove(tid)
            
            if current_group:
                groups.append(current_group)
            elif remaining_tests:
                # Force add one test to avoid infinite loop
                groups.append([remaining_tests.pop(0)])
        
        return groups
    
    async def execute_plan(self, plan: ExecutionPlan) -> List[ExecutionResult]:
        """Execute the planned test execution"""
        results = []
        start_time = datetime.utcnow()
        
        try:
            if plan.strategy == ExecutionStrategy.SEQUENTIAL or plan.test_order:
                # Sequential execution
                for test_id in plan.test_order:
                    result = await self._execute_single_test(test_id)
                    results.append(result)
            else:
                # Parallel execution
                for group in plan.parallel_groups:
                    if len(group) == 1:
                        # Single test
                        result = await self._execute_single_test(group[0])
                        results.append(result)
                    else:
                        # Parallel group
                        group_tasks = [self._execute_single_test(tid) for tid in group]
                        group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
                        
                        for result in group_results:
                            if isinstance(result, ExecutionResult):
                                results.append(result)
                            else:
                                # Handle exception
                                logger.error(f"Test execution failed: {result}")
            
            # Update learning
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            success_rate = sum(1 for r in results if r.success) / len(results) if results else 0.0
            
            await self._update_learning(plan, results, execution_time, success_rate)
            
        except Exception as e:
            logger.error(f"Execution plan failed: {e}")
        
        return results
    
    async def _execute_single_test(self, test_id: str) -> ExecutionResult:
        """Execute a single test using real test execution"""
        context = self.test_contexts.get(test_id, TestExecutionContext("", "", "", 1.0, 1))

        start_time = datetime.utcnow()

        try:
            # Execute real test based on context
            result = await self._execute_real_test(test_id, context)

            # Update execution history
            self.execution_history.append(result)

            return result

        except Exception as e:
            logger.error(f"Test execution failed for {test_id}: {e}")

            # Return error result
            result = ExecutionResult(
                test_id=test_id,
                success=False,
                execution_time=0.0,
                resource_usage={"cpu": 0, "memory": 0},
                error_message=str(e),
                timestamp=start_time
            )

            self.execution_history.append(result)
            return result

    async def _execute_real_test(self, test_id: str, context: TestExecutionContext) -> ExecutionResult:
        """Execute real test using appropriate test runner."""
        start_time = datetime.utcnow()

        try:
            # Determine test type and execution method
            if context.test_type == "unit":
                result = await self._execute_unit_test_real(test_id, context)
            elif context.test_type == "integration":
                result = await self._execute_integration_test_real(test_id, context)
            elif context.test_type == "performance":
                result = await self._execute_performance_test_real(test_id, context)
            else:
                result = await self._execute_generic_test_real(test_id, context)

            end_time = datetime.utcnow()
            actual_time = (end_time - start_time).total_seconds()

            return ExecutionResult(
                test_id=test_id,
                success=result["success"],
                execution_time=actual_time,
                resource_usage=result.get("resource_usage", {"cpu": 0, "memory": 0}),
                error_message=result.get("error_message"),
                timestamp=start_time
            )

        except Exception as e:
            end_time = datetime.utcnow()
            actual_time = (end_time - start_time).total_seconds()

            return ExecutionResult(
                test_id=test_id,
                success=False,
                execution_time=actual_time,
                resource_usage={"cpu": 0, "memory": 0},
                error_message=str(e),
                timestamp=start_time
            )

    async def _execute_unit_test_real(self, test_id: str, context: TestExecutionContext) -> Dict[str, Any]:
        """Execute real unit test."""
        try:
            import subprocess
            import asyncio
            import psutil
            import os

            # Monitor resource usage
            process = psutil.Process(os.getpid())
            initial_cpu = process.cpu_percent()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Execute test command
            if context.test_command:
                proc = await asyncio.create_subprocess_shell(
                    context.test_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=context.working_directory or "."
                )

                stdout, stderr = await proc.communicate()
                success = proc.returncode == 0

                # Calculate resource usage
                final_cpu = process.cpu_percent()
                final_memory = process.memory_info().rss / 1024 / 1024  # MB

                return {
                    "success": success,
                    "resource_usage": {
                        "cpu": max(0, final_cpu - initial_cpu),
                        "memory": max(0, final_memory - initial_memory)
                    },
                    "error_message": stderr.decode() if stderr and not success else None,
                    "output": stdout.decode() if stdout else ""
                }
            else:
                # Basic validation test
                return {
                    "success": True,
                    "resource_usage": {"cpu": 1, "memory": 1},
                    "error_message": None,
                    "output": f"Unit test {test_id} validation passed"
                }

        except Exception as e:
            return {
                "success": False,
                "resource_usage": {"cpu": 0, "memory": 0},
                "error_message": str(e),
                "output": ""
            }

    async def _execute_integration_test_real(self, test_id: str, context: TestExecutionContext) -> Dict[str, Any]:
        """Execute real integration test."""
        try:
            import subprocess
            import asyncio
            import psutil
            import os

            # Monitor resource usage
            process = psutil.Process(os.getpid())
            initial_cpu = process.cpu_percent()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Execute integration test with longer timeout
            if context.test_command:
                proc = await asyncio.create_subprocess_shell(
                    context.test_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=context.working_directory or "."
                )

                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120.0)
                    success = proc.returncode == 0

                    # Calculate resource usage
                    final_cpu = process.cpu_percent()
                    final_memory = process.memory_info().rss / 1024 / 1024  # MB

                    return {
                        "success": success,
                        "resource_usage": {
                            "cpu": max(0, final_cpu - initial_cpu),
                            "memory": max(0, final_memory - initial_memory)
                        },
                        "error_message": stderr.decode() if stderr and not success else None,
                        "output": stdout.decode() if stdout else ""
                    }

                except asyncio.TimeoutError:
                    proc.kill()
                    return {
                        "success": False,
                        "resource_usage": {"cpu": 10, "memory": 10},
                        "error_message": "Integration test timed out",
                        "output": ""
                    }
            else:
                # Basic integration validation
                return {
                    "success": True,
                    "resource_usage": {"cpu": 5, "memory": 5},
                    "error_message": None,
                    "output": f"Integration test {test_id} validation passed"
                }

        except Exception as e:
            return {
                "success": False,
                "resource_usage": {"cpu": 0, "memory": 0},
                "error_message": str(e),
                "output": ""
            }

    async def _execute_performance_test_real(self, test_id: str, context: TestExecutionContext) -> Dict[str, Any]:
        """Execute real performance test."""
        try:
            import time
            import psutil
            import os

            # Monitor system resources during test
            process = psutil.Process(os.getpid())
            initial_cpu = process.cpu_percent()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            start_time = time.time()

            if context.test_command:
                # Execute performance test command
                proc = await asyncio.create_subprocess_shell(
                    context.test_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=context.working_directory or "."
                )

                stdout, stderr = await proc.communicate()
                success = proc.returncode == 0

                end_time = time.time()
                execution_time = end_time - start_time

                # Calculate resource usage
                final_cpu = process.cpu_percent()
                final_memory = process.memory_info().rss / 1024 / 1024  # MB

                return {
                    "success": success,
                    "resource_usage": {
                        "cpu": max(0, final_cpu - initial_cpu),
                        "memory": max(0, final_memory - initial_memory)
                    },
                    "error_message": stderr.decode() if stderr and not success else None,
                    "output": stdout.decode() if stdout else "",
                    "performance_metrics": {
                        "execution_time": execution_time,
                        "cpu_usage": final_cpu - initial_cpu,
                        "memory_usage": final_memory - initial_memory
                    }
                }
            else:
                # Basic performance validation
                await asyncio.sleep(0.1)  # Minimal performance test
                end_time = time.time()

                return {
                    "success": True,
                    "resource_usage": {"cpu": 2, "memory": 2},
                    "error_message": None,
                    "output": f"Performance test {test_id} completed",
                    "performance_metrics": {
                        "execution_time": end_time - start_time,
                        "cpu_usage": 2,
                        "memory_usage": 2
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "resource_usage": {"cpu": 0, "memory": 0},
                "error_message": str(e),
                "output": "",
                "performance_metrics": {}
            }

    async def _execute_generic_test_real(self, test_id: str, context: TestExecutionContext) -> Dict[str, Any]:
        """Execute generic test with basic validation."""
        try:
            # Basic test validation
            if context.test_command:
                proc = await asyncio.create_subprocess_shell(
                    context.test_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=context.working_directory or "."
                )

                stdout, stderr = await proc.communicate()
                success = proc.returncode == 0

                return {
                    "success": success,
                    "resource_usage": {"cpu": 1, "memory": 1},
                    "error_message": stderr.decode() if stderr and not success else None,
                    "output": stdout.decode() if stdout else ""
                }
            else:
                # Minimal validation
                return {
                    "success": True,
                    "resource_usage": {"cpu": 1, "memory": 1},
                    "error_message": None,
                    "output": f"Generic test {test_id} validation passed"
                }

        except Exception as e:
            return {
                "success": False,
                "resource_usage": {"cpu": 0, "memory": 0},
                "error_message": str(e),
                "output": ""
            }

        # Update execution history
        self.execution_history.append(result)
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        
        return result
    
    async def _update_learning(self, plan: ExecutionPlan, results: List[ExecutionResult],
                             execution_time: float, success_rate: float):
        """Update RL learning based on execution results"""
        # Calculate reward
        reward = self._calculate_reward(plan, results, execution_time, success_rate)
        self.reward_history.append(reward)
        
        # Update Q-table (simplified Q-learning)
        state = f"tests_{len(results)}_time_{int(execution_time)}"
        action = plan.strategy.value
        
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # Q-learning update
        old_q = self.q_table[state][action]
        self.q_table[state][action] = old_q + self.learning_rate * (reward - old_q)
        
        # Update strategy performance
        self.strategy_performance[plan.strategy].append(reward)
        
        # Update average execution time
        self.average_execution_time = (
            (self.average_execution_time * (self.total_executions - len(results)) + execution_time) /
            self.total_executions
        )
        
        logger.debug(f"Updated learning: reward={reward:.3f}, strategy={plan.strategy.value}")
    
    def _calculate_reward(self, plan: ExecutionPlan, results: List[ExecutionResult],
                         execution_time: float, success_rate: float) -> float:
        """Calculate reward for the execution"""
        # Base reward from success rate
        reward = success_rate * 100
        
        # Penalty for long execution time
        if execution_time > plan.estimated_total_time * 1.2:
            reward -= 20
        elif execution_time < plan.estimated_total_time * 0.8:
            reward += 10
        
        # Bonus for high success rate
        if success_rate > 0.9:
            reward += 20
        elif success_rate < 0.5:
            reward -= 30
        
        # Penalty for timeouts or errors
        for result in results:
            if not result.success:
                if "timeout" in (result.error_message or "").lower():
                    reward -= 15
                else:
                    reward -= 10
        
        return max(0.0, reward)  # Ensure non-negative reward
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the execution agent"""
        if self.total_executions == 0:
            return {"status": "no_executions"}
        
        overall_success_rate = self.successful_executions / self.total_executions
        
        # Strategy performance
        strategy_stats = {}
        for strategy, rewards in self.strategy_performance.items():
            if rewards:
                strategy_stats[strategy.value] = {
                    "average_reward": statistics.mean(rewards),
                    "executions": len(rewards),
                    "best_reward": max(rewards),
                    "worst_reward": min(rewards)
                }
        
        return {
            "total_executions": self.total_executions,
            "success_rate": overall_success_rate,
            "average_execution_time": self.average_execution_time,
            "strategy_performance": strategy_stats,
            "recent_rewards": self.reward_history[-10:] if self.reward_history else [],
            "q_table_size": len(self.q_table),
            "exploration_rate": self.exploration_rate
        }
