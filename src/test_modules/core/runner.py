"""
Recipe Test Runner

This module provides the main test runner for executing Agent + MCP recipe tests
with comprehensive orchestration, monitoring, and result collection.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from ..recipes.schema import RecipeDefinition
from ..core.framework import RecipeTestResult, TestSession
from ..engine.executor import TestExecutor
from ..scenarios.generator import TestScenarioGenerator
from ..ml.predictor import RecipeSuccessPredictor


logger = logging.getLogger(__name__)


@dataclass
class TestRunConfig:
    """Configuration for test run"""
    max_concurrent_tests: int = 3
    timeout_seconds: int = 300
    retry_failed_tests: bool = True
    max_retries: int = 2
    collect_performance_data: bool = True
    enable_ml_prediction: bool = True
    scenario_categories: Optional[List[str]] = None
    custom_scenarios: Optional[List[Dict[str, Any]]] = None


@dataclass
class TestRunResult:
    """Result of a test run"""
    run_id: str
    session: TestSession
    recipe_results: List[RecipeTestResult] = field(default_factory=list)
    execution_time_ms: int = 0
    success_rate: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    ml_predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class RecipeTestRunner:
    """
    Main test runner for Agent + MCP recipe tests.
    
    Orchestrates test execution with scenario generation, ML prediction,
    performance monitoring, and comprehensive result collection.
    """
    
    def __init__(self, 
                 test_executor: TestExecutor,
                 scenario_generator: TestScenarioGenerator,
                 ml_predictor: Optional[RecipeSuccessPredictor] = None):
        self.test_executor = test_executor
        self.scenario_generator = scenario_generator
        self.ml_predictor = ml_predictor
        
        # State
        self.active_runs: Dict[str, TestRunResult] = {}
        self.completed_runs: List[TestRunResult] = []
        
        # Statistics
        self.total_runs = 0
        self.total_tests_executed = 0
        self.total_execution_time_ms = 0
        
        # Event callbacks
        self.on_run_started: Optional[Callable[[str], None]] = None
        self.on_run_completed: Optional[Callable[[TestRunResult], None]] = None
        self.on_test_started: Optional[Callable[[str, str], None]] = None
        self.on_test_completed: Optional[Callable[[str, RecipeTestResult], None]] = None
    
    async def run_single_recipe(self, 
                               recipe: RecipeDefinition,
                               config: Optional[TestRunConfig] = None) -> TestRunResult:
        """
        Run tests for a single recipe.
        
        Args:
            recipe: Recipe to test
            config: Test run configuration
            
        Returns:
            Test run result
        """
        config = config or TestRunConfig()
        run_id = f"single_{recipe.id}_{int(time.time())}"
        
        logger.info(f"Starting single recipe test run: {recipe.name} (ID: {run_id})")
        
        # Create test session
        session = TestSession(
            session_id=run_id,
            name=f"Single Recipe Test: {recipe.name}",
            description=f"Test execution for recipe {recipe.name}"
        )
        
        # Create run result
        run_result = TestRunResult(
            run_id=run_id,
            session=session
        )
        
        self.active_runs[run_id] = run_result
        
        try:
            # Notify run started
            if self.on_run_started:
                self.on_run_started(run_id)
            
            start_time = time.time()
            
            # Generate ML prediction if enabled
            if config.enable_ml_prediction and self.ml_predictor:
                try:
                    prediction = await self.ml_predictor.predict_success(recipe)
                    run_result.ml_predictions[recipe.id] = {
                        "success_probability": prediction.success_probability,
                        "confidence": prediction.confidence,
                        "execution_time_prediction": prediction.execution_time_prediction,
                        "memory_usage_prediction": prediction.memory_usage_prediction,
                        "risk_factors": prediction.risk_factors,
                        "recommendations": prediction.recommendations
                    }
                    logger.info(f"ML prediction for {recipe.name}: {prediction.success_probability:.2%} success probability")
                except Exception as e:
                    logger.warning(f"ML prediction failed for {recipe.name}: {e}")
            
            # Generate test scenarios
            scenarios = await self._generate_scenarios(recipe, config)
            logger.info(f"Generated {len(scenarios)} test scenarios for {recipe.name}")
            
            # Execute test
            if self.on_test_started:
                self.on_test_started(run_id, recipe.id)
            
            test_result = await self.test_executor.execute_recipe_test(
                recipe=recipe,
                scenarios=scenarios
            )
            
            # Add to results
            run_result.recipe_results.append(test_result)
            session.results.append(test_result)
            session.recipes_tested.append(recipe.id)
            
            # Notify test completed
            if self.on_test_completed:
                self.on_test_completed(run_id, test_result)
            
            # Calculate metrics
            run_result.execution_time_ms = int((time.time() - start_time) * 1000)
            run_result.success_rate = 1.0 if test_result.success else 0.0
            run_result.performance_metrics = {
                "execution_time_ms": test_result.execution_time_ms,
                "memory_usage_mb": test_result.memory_usage_mb,
                "scenarios_passed": test_result.test_scenarios_passed,
                "scenarios_total": test_result.test_scenarios_total
            }
            
            # Update session
            session.completed_at = datetime.utcnow()
            session.status = "completed"
            session.success_rate = run_result.success_rate
            session.average_score = test_result.score
            session.total_execution_time_ms = run_result.execution_time_ms
            
            logger.info(f"Single recipe test completed: {recipe.name} (Success: {test_result.success})")
            
        except Exception as e:
            error_msg = f"Test run failed: {str(e)}"
            logger.error(error_msg)
            run_result.errors.append(error_msg)
            session.status = "failed"
            session.completed_at = datetime.utcnow()
        
        finally:
            # Move to completed runs
            self.active_runs.pop(run_id, None)
            self.completed_runs.append(run_result)
            
            # Update statistics
            self.total_runs += 1
            self.total_tests_executed += len(run_result.recipe_results)
            self.total_execution_time_ms += run_result.execution_time_ms
            
            # Notify run completed
            if self.on_run_completed:
                self.on_run_completed(run_result)
        
        return run_result
    
    async def run_multiple_recipes(self, 
                                  recipes: List[RecipeDefinition],
                                  config: Optional[TestRunConfig] = None) -> TestRunResult:
        """
        Run tests for multiple recipes.
        
        Args:
            recipes: List of recipes to test
            config: Test run configuration
            
        Returns:
            Test run result
        """
        config = config or TestRunConfig()
        run_id = f"multi_{len(recipes)}_{int(time.time())}"
        
        logger.info(f"Starting multi-recipe test run: {len(recipes)} recipes (ID: {run_id})")
        
        # Create test session
        session = TestSession(
            session_id=run_id,
            name=f"Multi-Recipe Test: {len(recipes)} recipes",
            description=f"Test execution for {len(recipes)} recipes"
        )
        
        # Create run result
        run_result = TestRunResult(
            run_id=run_id,
            session=session
        )
        
        self.active_runs[run_id] = run_result
        
        try:
            # Notify run started
            if self.on_run_started:
                self.on_run_started(run_id)
            
            start_time = time.time()
            
            # Generate ML predictions if enabled
            if config.enable_ml_prediction and self.ml_predictor:
                await self._generate_ml_predictions(recipes, run_result, config)
            
            # Execute tests with concurrency control
            test_results = await self._execute_recipes_concurrent(recipes, run_result, config)
            
            # Calculate overall metrics
            run_result.execution_time_ms = int((time.time() - start_time) * 1000)
            run_result.success_rate = sum(1 for r in test_results if r.success) / len(test_results) if test_results else 0.0
            
            # Performance metrics
            if test_results:
                run_result.performance_metrics = {
                    "total_execution_time_ms": sum(r.execution_time_ms for r in test_results),
                    "average_execution_time_ms": sum(r.execution_time_ms for r in test_results) / len(test_results),
                    "total_memory_usage_mb": sum(r.memory_usage_mb for r in test_results),
                    "average_memory_usage_mb": sum(r.memory_usage_mb for r in test_results) / len(test_results),
                    "total_scenarios": sum(r.test_scenarios_total for r in test_results),
                    "passed_scenarios": sum(r.test_scenarios_passed for r in test_results),
                    "average_score": sum(r.score for r in test_results) / len(test_results)
                }
            
            # Update session
            session.completed_at = datetime.utcnow()
            session.status = "completed"
            session.success_rate = run_result.success_rate
            session.average_score = run_result.performance_metrics.get("average_score", 0.0)
            session.total_execution_time_ms = run_result.execution_time_ms
            
            logger.info(f"Multi-recipe test completed: {len(test_results)} tests (Success rate: {run_result.success_rate:.2%})")
            
        except Exception as e:
            error_msg = f"Multi-recipe test run failed: {str(e)}"
            logger.error(error_msg)
            run_result.errors.append(error_msg)
            session.status = "failed"
            session.completed_at = datetime.utcnow()
        
        finally:
            # Move to completed runs
            self.active_runs.pop(run_id, None)
            self.completed_runs.append(run_result)
            
            # Update statistics
            self.total_runs += 1
            self.total_tests_executed += len(run_result.recipe_results)
            self.total_execution_time_ms += run_result.execution_time_ms
            
            # Notify run completed
            if self.on_run_completed:
                self.on_run_completed(run_result)
        
        return run_result
    
    async def _generate_scenarios(self, 
                                recipe: RecipeDefinition,
                                config: TestRunConfig) -> List[Dict[str, Any]]:
        """Generate test scenarios for a recipe"""
        if config.custom_scenarios:
            return config.custom_scenarios
        
        return await self.scenario_generator.generate_scenarios_for_recipe(
            recipe=recipe,
            categories=config.scenario_categories,
            max_scenarios=10
        )
    
    async def _generate_ml_predictions(self, 
                                     recipes: List[RecipeDefinition],
                                     run_result: TestRunResult,
                                     config: TestRunConfig) -> None:
        """Generate ML predictions for multiple recipes"""
        for recipe in recipes:
            try:
                prediction = await self.ml_predictor.predict_success(recipe)
                run_result.ml_predictions[recipe.id] = {
                    "success_probability": prediction.success_probability,
                    "confidence": prediction.confidence,
                    "execution_time_prediction": prediction.execution_time_prediction,
                    "memory_usage_prediction": prediction.memory_usage_prediction,
                    "risk_factors": prediction.risk_factors,
                    "recommendations": prediction.recommendations
                }
            except Exception as e:
                logger.warning(f"ML prediction failed for {recipe.name}: {e}")
    
    async def _execute_recipes_concurrent(self, 
                                        recipes: List[RecipeDefinition],
                                        run_result: TestRunResult,
                                        config: TestRunConfig) -> List[RecipeTestResult]:
        """Execute multiple recipes with concurrency control"""
        semaphore = asyncio.Semaphore(config.max_concurrent_tests)
        
        async def execute_single_recipe(recipe: RecipeDefinition) -> RecipeTestResult:
            async with semaphore:
                try:
                    # Notify test started
                    if self.on_test_started:
                        self.on_test_started(run_result.run_id, recipe.id)
                    
                    # Generate scenarios
                    scenarios = await self._generate_scenarios(recipe, config)
                    
                    # Execute test
                    result = await self.test_executor.execute_recipe_test(
                        recipe=recipe,
                        scenarios=scenarios
                    )
                    
                    # Add to session
                    run_result.session.results.append(result)
                    run_result.session.recipes_tested.append(recipe.id)
                    run_result.recipe_results.append(result)
                    
                    # Notify test completed
                    if self.on_test_completed:
                        self.on_test_completed(run_result.run_id, result)
                    
                    return result
                
                except Exception as e:
                    logger.error(f"Failed to execute recipe {recipe.name}: {e}")
                    # Create failed result
                    failed_result = RecipeTestResult(
                        recipe_id=recipe.id,
                        recipe_name=recipe.name,
                        success=False,
                        score=0.0,
                        execution_time_ms=0,
                        memory_usage_mb=0.0,
                        error_message=str(e)
                    )
                    run_result.recipe_results.append(failed_result)
                    return failed_result
        
        # Execute all recipes concurrently
        tasks = [execute_single_recipe(recipe) for recipe in recipes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = [r for r in results if isinstance(r, RecipeTestResult)]
        return valid_results
    
    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a test run"""
        if run_id in self.active_runs:
            run_result = self.active_runs[run_id]
            return {
                "run_id": run_id,
                "status": "running",
                "session": run_result.session.__dict__,
                "tests_completed": len(run_result.recipe_results),
                "current_success_rate": run_result.success_rate,
                "errors": run_result.errors
            }
        
        # Check completed runs
        for run_result in self.completed_runs:
            if run_result.run_id == run_id:
                return {
                    "run_id": run_id,
                    "status": run_result.session.status,
                    "session": run_result.session.__dict__,
                    "tests_completed": len(run_result.recipe_results),
                    "success_rate": run_result.success_rate,
                    "execution_time_ms": run_result.execution_time_ms,
                    "performance_metrics": run_result.performance_metrics,
                    "errors": run_result.errors
                }
        
        return None
    
    def get_runner_statistics(self) -> Dict[str, Any]:
        """Get test runner statistics"""
        active_runs = len(self.active_runs)
        completed_runs = len(self.completed_runs)
        
        # Calculate success rates
        if self.completed_runs:
            overall_success_rate = sum(r.success_rate for r in self.completed_runs) / len(self.completed_runs)
            avg_execution_time = sum(r.execution_time_ms for r in self.completed_runs) / len(self.completed_runs)
        else:
            overall_success_rate = 0.0
            avg_execution_time = 0.0
        
        return {
            "total_runs": self.total_runs,
            "active_runs": active_runs,
            "completed_runs": completed_runs,
            "total_tests_executed": self.total_tests_executed,
            "overall_success_rate": overall_success_rate,
            "average_execution_time_ms": avg_execution_time,
            "total_execution_time_ms": self.total_execution_time_ms
        }
