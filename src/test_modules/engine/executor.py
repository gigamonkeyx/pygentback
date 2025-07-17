"""
Test Execution Engine

This module provides the core test execution engine for running Agent + MCP
recipe tests with comprehensive monitoring, profiling, and result collection.
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import psutil
import os

from ..recipes.schema import RecipeDefinition, RecipeStep
from ..engine.profiler import PerformanceProfiler
from ...core.agent_factory import AgentFactory
from ...mcp.server.manager import MCPServerManager
from ...core.agent.base import BaseAgent

# Local RecipeTestResult to avoid circular imports
@dataclass
class RecipeTestResult:
    """Local recipe test result to avoid circular imports"""
    recipe_id: str
    recipe_name: str
    success: bool
    score: float = 0.0
    execution_time_ms: int = 0
    memory_usage_mb: float = 0.0
    error_message: Optional[str] = None
    detailed_results: Dict[str, Any] = field(default_factory=dict)
    test_scenarios_passed: int = 0
    test_scenarios_total: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)


logger = logging.getLogger(__name__)


@dataclass
class StepExecutionResult:
    """Result of executing a single recipe step"""
    step_id: str
    step_name: str
    success: bool
    execution_time_ms: int
    memory_usage_mb: float
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    agent_actions: List[str] = field(default_factory=list)


@dataclass
class RecipeExecutionContext:
    """Context for recipe execution"""
    recipe: RecipeDefinition
    agent: BaseAgent
    mcp_manager: MCPServerManager
    profiler: Optional[PerformanceProfiler]
    test_scenarios: List[Dict[str, Any]]
    execution_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    step_results: List[StepExecutionResult] = field(default_factory=list)
    shared_data: Dict[str, Any] = field(default_factory=dict)


class TestExecutor:
    """
    Core test execution engine for Agent + MCP recipes.
    
    Executes recipe tests with comprehensive monitoring, error handling,
    and performance profiling.
    """
    
    def __init__(self, 
                 max_concurrent_executions: int = 3,
                 default_timeout_seconds: int = 300):
        self.max_concurrent_executions = max_concurrent_executions
        self.default_timeout_seconds = default_timeout_seconds
        
        # Execution state
        self.active_executions: Dict[str, RecipeExecutionContext] = {}
        self.execution_semaphore = asyncio.Semaphore(max_concurrent_executions)
        
        # Components
        self.agent_factory: Optional[AgentFactory] = None
        self.mcp_manager: Optional[MCPServerManager] = None
        
        # Execution statistics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        
        # Resource monitoring
        self.process = psutil.Process(os.getpid())
    
    async def initialize(self, 
                        agent_factory: AgentFactory,
                        mcp_manager: MCPServerManager) -> None:
        """Initialize the test executor"""
        self.agent_factory = agent_factory
        self.mcp_manager = mcp_manager
        
        logger.info("Test Executor initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the test executor"""
        # Cancel active executions
        for execution_id in list(self.active_executions.keys()):
            await self._cancel_execution(execution_id)
        
        logger.info("Test Executor shutdown complete")
    
    async def execute_recipe_test(self, 
                                recipe: RecipeDefinition,
                                scenarios: List[Dict[str, Any]],
                                profiler: Optional[PerformanceProfiler] = None) -> RecipeTestResult:
        """
        Execute a complete recipe test with scenarios.
        
        Args:
            recipe: Recipe to test
            scenarios: Test scenarios to execute
            profiler: Optional performance profiler
            
        Returns:
            Complete test result
        """
        execution_id = f"exec_{recipe.id}_{int(time.time())}"
        
        async with self.execution_semaphore:
            try:
                logger.info(f"Starting recipe test: {recipe.name} (ID: {execution_id})")
                
                # Create agent for recipe
                agent = await self._create_agent_for_recipe(recipe)
                if not agent:
                    return self._create_failed_result(recipe, "Failed to create agent")
                
                # Create execution context
                context = RecipeExecutionContext(
                    recipe=recipe,
                    agent=agent,
                    mcp_manager=self.mcp_manager,
                    profiler=profiler,
                    test_scenarios=scenarios,
                    execution_id=execution_id
                )
                
                self.active_executions[execution_id] = context
                
                # Execute recipe with timeout
                try:
                    result = await asyncio.wait_for(
                        self._execute_recipe_with_scenarios(context),
                        timeout=self.default_timeout_seconds
                    )
                    
                    self.successful_executions += 1
                    logger.info(f"Recipe test completed successfully: {recipe.name}")
                    
                except asyncio.TimeoutError:
                    result = self._create_failed_result(recipe, "Execution timeout")
                    self.failed_executions += 1
                    logger.error(f"Recipe test timed out: {recipe.name}")
                
                except Exception as e:
                    result = self._create_failed_result(recipe, f"Execution error: {str(e)}")
                    self.failed_executions += 1
                    logger.error(f"Recipe test failed: {recipe.name} - {e}")
                
                finally:
                    # Cleanup
                    self.active_executions.pop(execution_id, None)
                    await self._cleanup_agent(agent)
                
                self.total_executions += 1
                return result
            
            except Exception as e:
                logger.error(f"Critical error in recipe execution: {e}")
                return self._create_failed_result(recipe, f"Critical error: {str(e)}")
    
    async def _execute_recipe_with_scenarios(self, context: RecipeExecutionContext) -> RecipeTestResult:
        """Execute recipe with all test scenarios"""
        recipe = context.recipe
        scenarios = context.test_scenarios
        
        if not scenarios:
            # Execute recipe without specific scenarios
            return await self._execute_recipe_steps(context, {})
        
        # Execute recipe with each scenario
        scenario_results = []
        total_execution_time = 0
        total_memory_usage = 0.0
        
        for i, scenario in enumerate(scenarios):
            logger.debug(f"Executing scenario {i+1}/{len(scenarios)} for recipe {recipe.name}")
            
            # Reset shared data for each scenario
            context.shared_data = scenario.get("input_data", {})
            context.step_results = []
            
            try:
                scenario_result = await self._execute_recipe_steps(context, scenario)
                scenario_results.append(scenario_result)
                
                total_execution_time += scenario_result.execution_time_ms
                total_memory_usage += scenario_result.memory_usage_mb
            
            except Exception as e:
                logger.error(f"Scenario {i+1} failed for recipe {recipe.name}: {e}")
                failed_result = self._create_failed_result(recipe, f"Scenario {i+1} failed: {str(e)}")
                scenario_results.append(failed_result)
        
        # Aggregate scenario results
        successful_scenarios = sum(1 for r in scenario_results if r.success)
        total_scenarios = len(scenario_results)
        
        # Calculate overall success and score
        success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0.0
        overall_success = success_rate >= recipe.validation_criteria.success_threshold
        
        # Calculate composite score
        score = self._calculate_composite_score(scenario_results, recipe)
        
        # Create aggregated result
        return RecipeTestResult(
            recipe_id=recipe.id,
            recipe_name=recipe.name,
            success=overall_success,
            score=score,
            execution_time_ms=total_execution_time // len(scenarios) if scenarios else 0,
            memory_usage_mb=total_memory_usage / len(scenarios) if scenarios else 0.0,
            test_scenarios_passed=successful_scenarios,
            test_scenarios_total=total_scenarios,
            detailed_results={
                "scenario_results": [r.__dict__ for r in scenario_results],
                "step_results": [r.__dict__ for r in context.step_results]
            },
            performance_metrics=self._extract_performance_metrics(scenario_results),
            quality_metrics=self._extract_quality_metrics(scenario_results, recipe)
        )
    
    async def _execute_recipe_steps(self, 
                                  context: RecipeExecutionContext,
                                  scenario: Dict[str, Any]) -> RecipeTestResult:
        """Execute all steps in a recipe for a single scenario"""
        recipe = context.recipe
        agent = context.agent
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        step_results = []
        execution_success = True
        error_message = None
        
        try:
            # Execute steps in order
            for step in recipe.steps:
                step_result = await self._execute_recipe_step(context, step, scenario)
                step_results.append(step_result)
                
                # Store step result in context
                context.step_results.append(step_result)
                
                # Update shared data with step output
                context.shared_data.update(step_result.output_data)
                
                # Check if critical step failed
                if not step_result.success and step.critical:
                    execution_success = False
                    error_message = f"Critical step failed: {step_result.error_message}"
                    break
        
        except Exception as e:
            execution_success = False
            error_message = f"Recipe execution error: {str(e)}"
            logger.error(f"Recipe step execution failed: {e}")
        
        # Calculate execution metrics
        execution_time_ms = int((time.time() - start_time) * 1000)
        memory_usage_mb = self._get_memory_usage() - start_memory
        
        # Validate results against criteria
        validation_success = self._validate_recipe_results(recipe, step_results, context.shared_data)
        
        # Overall success
        overall_success = execution_success and validation_success
        
        # Calculate score
        score = self._calculate_step_score(step_results, recipe, overall_success)
        
        return RecipeTestResult(
            recipe_id=recipe.id,
            recipe_name=recipe.name,
            success=overall_success,
            score=score,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb,
            error_message=error_message,
            detailed_results={
                "step_results": [r.__dict__ for r in step_results],
                "shared_data": context.shared_data,
                "validation_success": validation_success
            },
            test_scenarios_passed=1 if overall_success else 0,
            test_scenarios_total=1
        )
    
    async def _execute_recipe_step(self, 
                                 context: RecipeExecutionContext,
                                 step: RecipeStep,
                                 scenario: Dict[str, Any]) -> StepExecutionResult:
        """Execute a single recipe step"""
        logger.debug(f"Executing step: {step.name}")
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Prepare step input data
            step_input = {
                **context.shared_data,
                **step.input_data,
                **scenario.get("step_inputs", {}).get(step.step_id, {})
            }
            
            # Execute MCP tools if specified
            tool_results = []
            if step.mcp_tools:
                for tool_name in step.mcp_tools:
                    tool_result = await self._execute_mcp_tool(
                        context, tool_name, step_input
                    )
                    tool_results.append(tool_result)
            
            # Execute agent action
            agent_result = await self._execute_agent_action(
                context, step.agent_action, step_input, tool_results
            )
            
            # Combine results
            output_data = {
                **agent_result.get("output", {}),
                "tool_results": tool_results
            }
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            memory_usage_mb = self._get_memory_usage() - start_memory
            
            return StepExecutionResult(
                step_id=step.step_id,
                step_name=step.name,
                success=True,
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_usage_mb,
                output_data=output_data,
                tool_calls=tool_results,
                agent_actions=[step.agent_action]
            )
        
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            memory_usage_mb = self._get_memory_usage() - start_memory
            
            logger.error(f"Step execution failed: {step.name} - {e}")
            
            return StepExecutionResult(
                step_id=step.step_id,
                step_name=step.name,
                success=False,
                execution_time_ms=execution_time_ms,
                memory_usage_mb=memory_usage_mb,
                error_message=str(e)
            )
    
    async def _execute_mcp_tool(self, 
                              context: RecipeExecutionContext,
                              tool_name: str,
                              input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool"""
        try:
            # Find the appropriate MCP server for this tool
            server_name = None
            for mcp_req in context.recipe.mcp_requirements:
                if tool_name in mcp_req.required_capabilities or tool_name == mcp_req.tool_name:
                    server_name = mcp_req.server_name
                    break
            
            if not server_name:
                raise ValueError(f"No MCP server found for tool: {tool_name}")
            
            # Execute tool via MCP manager
            if context.mcp_manager:
                result = await context.mcp_manager.call_tool(
                    server_name=server_name,
                    tool_name=tool_name,
                    arguments=input_data
                )
                
                return {
                    "tool_name": tool_name,
                    "server_name": server_name,
                    "success": True,
                    "result": result,
                    "input": input_data
                }
            else:
                raise ValueError("MCP manager not available")
        
        except Exception as e:
            logger.error(f"MCP tool execution failed: {tool_name} - {e}")
            return {
                "tool_name": tool_name,
                "server_name": server_name,
                "success": False,
                "error": str(e),
                "input": input_data
            }
    
    async def _execute_agent_action(self, 
                                  context: RecipeExecutionContext,
                                  action: str,
                                  input_data: Dict[str, Any],
                                  tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute an agent action"""
        try:
            agent = context.agent
            
            # Prepare agent input with tool results
            agent_input = {
                **input_data,
                "tool_results": tool_results,
                "action": action
            }
            
            # Execute agent action (simplified - would be more sophisticated in practice)
            if hasattr(agent, 'execute_action'):
                result = await agent.execute_action(action, agent_input)
            else:
                # Fallback to basic agent interaction
                prompt = f"Execute action: {action} with data: {json.dumps(agent_input, indent=2)}"
                result = await agent.process_message(prompt)
            
            return {
                "action": action,
                "success": True,
                "output": result if isinstance(result, dict) else {"response": str(result)},
                "input": agent_input
            }
        
        except Exception as e:
            logger.error(f"Agent action execution failed: {action} - {e}")
            return {
                "action": action,
                "success": False,
                "error": str(e),
                "input": input_data
            }
    
    async def _create_agent_for_recipe(self, recipe: RecipeDefinition) -> Optional[BaseAgent]:
        """Create an agent instance for recipe execution"""
        try:
            if not self.agent_factory:
                raise ValueError("Agent factory not available")
            
            # Get the first agent requirement (simplified)
            if not recipe.agent_requirements:
                raise ValueError("Recipe has no agent requirements")
            
            agent_req = recipe.agent_requirements[0]
            
            # Create agent with specified configuration
            agent_config = {
                "agent_type": agent_req.agent_type,
                "capabilities": [cap.value for cap in agent_req.required_capabilities],
                "memory_limit": agent_req.memory_limit_mb,
                "max_execution_time": agent_req.max_execution_time,
                **agent_req.configuration
            }
            
            agent = await self.agent_factory.create_agent(agent_config)
            return agent
        
        except Exception as e:
            logger.error(f"Failed to create agent for recipe {recipe.name}: {e}")
            return None
    
    async def _cleanup_agent(self, agent: BaseAgent) -> None:
        """Cleanup agent resources"""
        try:
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
        except Exception as e:
            logger.warning(f"Agent cleanup failed: {e}")
    
    def _validate_recipe_results(self, 
                               recipe: RecipeDefinition,
                               step_results: List[StepExecutionResult],
                               shared_data: Dict[str, Any]) -> bool:
        """Validate recipe results against criteria"""
        try:
            criteria = recipe.validation_criteria
            
            # Check required outputs
            for required_output in criteria.required_outputs:
                if required_output not in shared_data:
                    logger.warning(f"Required output missing: {required_output}")
                    return False
            
            # Check performance budget
            total_execution_time = sum(r.execution_time_ms for r in step_results)
            if total_execution_time > criteria.performance_budget_ms:
                logger.warning(f"Performance budget exceeded: {total_execution_time}ms > {criteria.performance_budget_ms}ms")
                return False
            
            # Check memory budget
            max_memory_usage = max((r.memory_usage_mb for r in step_results), default=0.0)
            if max_memory_usage > criteria.memory_budget_mb:
                logger.warning(f"Memory budget exceeded: {max_memory_usage}MB > {criteria.memory_budget_mb}MB")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Result validation failed: {e}")
            return False
    
    def _calculate_step_score(self, 
                            step_results: List[StepExecutionResult],
                            recipe: RecipeDefinition,
                            overall_success: bool) -> float:
        """Calculate score for recipe execution"""
        if not step_results:
            return 0.0
        
        # Base score from success rate
        successful_steps = sum(1 for r in step_results if r.success)
        success_score = successful_steps / len(step_results)
        
        # Performance score (inverse of execution time relative to budget)
        total_time = sum(r.execution_time_ms for r in step_results)
        budget_time = recipe.validation_criteria.performance_budget_ms
        performance_score = max(0.0, 1.0 - (total_time / budget_time)) if budget_time > 0 else 1.0
        
        # Memory efficiency score
        max_memory = max((r.memory_usage_mb for r in step_results), default=0.0)
        budget_memory = recipe.validation_criteria.memory_budget_mb
        memory_score = max(0.0, 1.0 - (max_memory / budget_memory)) if budget_memory > 0 else 1.0
        
        # Weighted composite score
        composite_score = (
            success_score * 0.5 +
            performance_score * 0.3 +
            memory_score * 0.2
        )
        
        # Bonus for overall success
        if overall_success:
            composite_score = min(1.0, composite_score + 0.1)
        
        return composite_score
    
    def _calculate_composite_score(self, 
                                 scenario_results: List[RecipeTestResult],
                                 recipe: RecipeDefinition) -> float:
        """Calculate composite score from multiple scenario results"""
        if not scenario_results:
            return 0.0
        
        # Average scores across scenarios
        avg_score = sum(r.score for r in scenario_results) / len(scenario_results)
        
        # Success rate bonus
        success_rate = sum(1 for r in scenario_results if r.success) / len(scenario_results)
        success_bonus = success_rate * 0.1
        
        return min(1.0, avg_score + success_bonus)
    
    def _extract_performance_metrics(self, results: List[RecipeTestResult]) -> Dict[str, float]:
        """Extract performance metrics from results"""
        if not results:
            return {}
        
        execution_times = [r.execution_time_ms for r in results]
        memory_usages = [r.memory_usage_mb for r in results]
        
        return {
            "avg_execution_time_ms": sum(execution_times) / len(execution_times),
            "max_execution_time_ms": max(execution_times),
            "min_execution_time_ms": min(execution_times),
            "avg_memory_usage_mb": sum(memory_usages) / len(memory_usages),
            "max_memory_usage_mb": max(memory_usages),
            "success_rate": sum(1 for r in results if r.success) / len(results)
        }
    
    def _extract_quality_metrics(self, 
                                results: List[RecipeTestResult],
                                recipe: RecipeDefinition) -> Dict[str, float]:
        """Extract quality metrics from results"""
        if not results:
            return {}
        
        return {
            "consistency": self._calculate_consistency(results),
            "reliability": sum(1 for r in results if r.success) / len(results),
            "efficiency": self._calculate_efficiency(results, recipe),
            "completeness": self._calculate_completeness(results)
        }
    
    def _calculate_consistency(self, results: List[RecipeTestResult]) -> float:
        """Calculate consistency score across results"""
        if len(results) < 2:
            return 1.0
        
        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        
        # Lower variance = higher consistency
        return max(0.0, 1.0 - variance)
    
    def _calculate_efficiency(self, 
                            results: List[RecipeTestResult],
                            recipe: RecipeDefinition) -> float:
        """Calculate efficiency score"""
        if not results:
            return 0.0
        
        avg_time = sum(r.execution_time_ms for r in results) / len(results)
        budget_time = recipe.validation_criteria.performance_budget_ms
        
        return max(0.0, 1.0 - (avg_time / budget_time)) if budget_time > 0 else 1.0
    
    def _calculate_completeness(self, results: List[RecipeTestResult]) -> float:
        """Calculate completeness score"""
        if not results:
            return 0.0
        
        # Check if all scenarios completed (not just succeeded)
        completed_scenarios = sum(r.test_scenarios_total for r in results)
        total_scenarios = len(results)
        
        return completed_scenarios / total_scenarios if total_scenarios > 0 else 0.0
    
    def _create_failed_result(self, recipe: RecipeDefinition, error_message: str) -> RecipeTestResult:
        """Create a failed test result"""
        return RecipeTestResult(
            recipe_id=recipe.id,
            recipe_name=recipe.name,
            success=False,
            score=0.0,
            execution_time_ms=0,
            memory_usage_mb=0.0,
            error_message=error_message,
            test_scenarios_passed=0,
            test_scenarios_total=1
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    async def _cancel_execution(self, execution_id: str) -> None:
        """Cancel an active execution"""
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            logger.info(f"Cancelling execution: {execution_id}")
            
            # Cleanup agent
            await self._cleanup_agent(context.agent)
            
            # Remove from active executions
            self.active_executions.pop(execution_id, None)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.successful_executions / self.total_executions if self.total_executions > 0 else 0.0,
            "active_executions": len(self.active_executions),
            "max_concurrent_executions": self.max_concurrent_executions
        }
