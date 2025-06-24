"""
Recipe Testing Framework Core

This module provides the main testing framework for Agent + MCP recipe combinations,
orchestrating discovery, testing, scoring, and validation of recipes.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..recipes.schema import RecipeDefinition, RecipeStatus
from ..recipes.registry import RecipeRegistry
from ..recipes.generator import RecipeGenerator
from ..recipes.validator import RecipeValidator
from ..mcp.pool_manager import MCPServerPoolManager
from ..engine.executor import TestExecutor
from ..engine.scheduler import TestScheduler
from ..engine.profiler import PerformanceProfiler
from ..ml.predictor import RecipeSuccessPredictor
from ..ml.optimizer import RecipeOptimizer
from ..scenarios.generator import TestScenarioGenerator
from ...core.agent_factory import AgentFactory

try:
    from ...mcp.server.manager import MCPServerManager
except ImportError:
    try:
        from ...mcp.server_registry import MCPServerManager
    except ImportError:
        logger.error("MCPServerManager not found in expected locations")
        raise ImportError("MCPServerManager is required for testing framework")


logger = logging.getLogger(__name__)


@dataclass
class RecipeTestResult:
    """Result of testing a recipe"""
    recipe_id: str
    recipe_name: str
    success: bool
    score: float
    execution_time_ms: int
    memory_usage_mb: float
    error_message: Optional[str] = None
    detailed_results: Dict[str, Any] = field(default_factory=dict)
    test_scenarios_passed: int = 0
    test_scenarios_total: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TestSession:
    """A testing session with multiple recipes"""
    session_id: str
    name: str
    description: str
    recipes_tested: List[str] = field(default_factory=list)
    results: List[RecipeTestResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, cancelled
    total_execution_time_ms: int = 0
    success_rate: float = 0.0
    average_score: float = 0.0


class RecipeTestingFramework:
    """
    Main testing framework for Agent + MCP recipe combinations.
    
    Orchestrates the entire testing process including recipe discovery,
    MCP server management, test execution, scoring, and optimization.
    """
    
    def __init__(self, 
                 data_dir: str = "./data/recipe_testing",
                 max_concurrent_tests: int = 5,
                 enable_ml_optimization: bool = True):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent_tests = max_concurrent_tests
        self.enable_ml_optimization = enable_ml_optimization
        
        # Core components
        self.recipe_registry = RecipeRegistry(str(self.data_dir / "recipes"))
        self.recipe_generator = RecipeGenerator()
        self.recipe_validator = RecipeValidator()
        self.mcp_pool_manager = MCPServerPoolManager(str(self.data_dir / "mcp_pool"))
        
        # Testing components
        self.test_executor = TestExecutor()
        self.test_scheduler = TestScheduler(max_concurrent_tests)
        self.performance_profiler = PerformanceProfiler()
        self.scenario_generator = TestScenarioGenerator()
        
        # ML/AI components
        self.success_predictor = RecipeSuccessPredictor() if enable_ml_optimization else None
        self.recipe_optimizer = RecipeOptimizer() if enable_ml_optimization else None
        
        # External dependencies
        self.agent_factory: Optional[AgentFactory] = None
        self.mcp_manager: Optional[MCPServerManager] = None
        
        # State
        self.active_sessions: Dict[str, TestSession] = {}
        self.test_history: List[RecipeTestResult] = []
        self._initialized = False
    
    async def initialize(self, 
                        agent_factory: AgentFactory,
                        mcp_manager: MCPServerManager) -> None:
        """Initialize the testing framework"""
        if self._initialized:
            return
        
        self.agent_factory = agent_factory
        self.mcp_manager = mcp_manager
        
        # Initialize components
        await self.recipe_registry.initialize()
        await self.mcp_pool_manager.initialize(mcp_manager)
        await self.test_executor.initialize(agent_factory, mcp_manager)
        
        if self.success_predictor:
            await self.success_predictor.initialize()
        if self.recipe_optimizer:
            await self.recipe_optimizer.initialize()
        
        # Load test history
        await self._load_test_history()
        
        self._initialized = True
        logger.info("Recipe Testing Framework initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the testing framework"""
        if not self._initialized:
            return
        
        # Cancel active sessions
        for session in self.active_sessions.values():
            if session.status == "running":
                session.status = "cancelled"
                session.completed_at = datetime.utcnow()
        
        # Shutdown components
        await self.mcp_pool_manager.shutdown()
        await self.test_executor.shutdown()
        
        # Save test history
        await self._save_test_history()
        
        self._initialized = False
        logger.info("Recipe Testing Framework shutdown complete")
    
    async def setup_mcp_ecosystem(self, 
                                 categories: Optional[List[str]] = None,
                                 max_servers_per_category: int = 5) -> Dict[str, List[str]]:
        """
        Set up the MCP server ecosystem for testing.
        
        Args:
            categories: Categories of servers to install
            max_servers_per_category: Maximum servers per category
            
        Returns:
            Dict mapping categories to installed server names
        """
        logger.info("Setting up MCP ecosystem for testing")
        
        # Discover and install MCP servers
        installation_results = await self.mcp_pool_manager.discover_and_install_servers(
            categories=categories,
            max_servers_per_category=max_servers_per_category
        )
        
        # Generate recipes for installed servers
        await self._generate_recipes_for_servers()
        
        logger.info(f"MCP ecosystem setup complete: {installation_results}")
        return installation_results
    
    async def generate_comprehensive_recipes(self, 
                                           agent_types: Optional[List[str]] = None,
                                           max_recipes_per_combination: int = 3) -> List[RecipeDefinition]:
        """
        Generate comprehensive recipe combinations.
        
        Args:
            agent_types: Agent types to include, or None for all
            max_recipes_per_combination: Maximum recipes per agent+MCP combination
            
        Returns:
            List of generated recipes
        """
        logger.info("Generating comprehensive recipe combinations")
        
        # Get available MCP servers
        pool_status = await self.mcp_pool_manager.get_pool_status()
        available_servers = list(self.mcp_pool_manager.server_instances.keys())
        
        if not available_servers:
            logger.warning("No MCP servers available for recipe generation")
            return []
        
        # Generate recipes
        recipes = await self.recipe_generator.generate_comprehensive_recipes(
            agent_types=agent_types or ["research", "code", "conversation"],
            mcp_servers=available_servers,
            max_recipes_per_combination=max_recipes_per_combination
        )
        
        # Validate and register recipes
        validated_recipes = []
        for recipe in recipes:
            validation_issues = self.recipe_validator.validate_recipe(recipe)
            if not validation_issues:
                await self.recipe_registry.register_recipe(recipe)
                validated_recipes.append(recipe)
            else:
                logger.warning(f"Recipe validation failed for {recipe.name}: {validation_issues}")
        
        logger.info(f"Generated and validated {len(validated_recipes)} recipes")
        return validated_recipes
    
    async def run_comprehensive_test_suite(self, 
                                         recipe_filters: Optional[Dict[str, Any]] = None,
                                         test_categories: Optional[List[str]] = None) -> TestSession:
        """
        Run comprehensive testing of recipe combinations.
        
        Args:
            recipe_filters: Filters for recipe selection
            test_categories: Categories of tests to run
            
        Returns:
            TestSession with results
        """
        session_id = f"comprehensive_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        session = TestSession(
            session_id=session_id,
            name="Comprehensive Recipe Test Suite",
            description="Full testing of Agent + MCP recipe combinations"
        )
        
        self.active_sessions[session_id] = session
        
        try:
            # Get recipes to test
            recipes = await self.recipe_registry.find_recipes(filters=recipe_filters)
            if not recipes:
                logger.warning("No recipes found matching filters")
                session.status = "completed"
                session.completed_at = datetime.utcnow()
                return session
            
            logger.info(f"Starting comprehensive test suite with {len(recipes)} recipes")
            
            # Start required MCP servers
            await self._start_required_servers(recipes)
            
            # Generate test scenarios
            test_scenarios = await self._generate_test_scenarios(recipes, test_categories)
            
            # Execute tests
            test_results = await self._execute_recipe_tests(recipes, test_scenarios, session)
            
            # Process results
            session.results = test_results
            session.success_rate = sum(1 for r in test_results if r.success) / len(test_results)
            session.average_score = sum(r.score for r in test_results) / len(test_results)
            session.total_execution_time_ms = sum(r.execution_time_ms for r in test_results)
            session.status = "completed"
            session.completed_at = datetime.utcnow()
            
            # Update test history
            self.test_history.extend(test_results)
            
            # ML optimization if enabled
            if self.enable_ml_optimization and self.recipe_optimizer:
                await self._optimize_recipes_from_results(test_results)
            
            logger.info(f"Comprehensive test suite completed: {session.success_rate:.2%} success rate")
            
        except Exception as e:
            logger.error(f"Comprehensive test suite failed: {e}")
            session.status = "failed"
            session.completed_at = datetime.utcnow()
        
        return session
    
    async def test_single_recipe(self, 
                               recipe_id: str,
                               custom_scenarios: Optional[List[Dict[str, Any]]] = None) -> RecipeTestResult:
        """
        Test a single recipe with detailed analysis.
        
        Args:
            recipe_id: ID of recipe to test
            custom_scenarios: Custom test scenarios to use
            
        Returns:
            Detailed test result
        """
        recipe = await self.recipe_registry.get_recipe(recipe_id)
        if not recipe:
            raise ValueError(f"Recipe not found: {recipe_id}")
        
        logger.info(f"Testing single recipe: {recipe.name}")
        
        # Start required servers
        await self._start_required_servers([recipe])
        
        # Generate or use custom scenarios
        if custom_scenarios:
            scenarios = custom_scenarios
        else:
            scenarios = await self.scenario_generator.generate_scenarios_for_recipe(recipe)
        
        # Execute test
        result = await self.test_executor.execute_recipe_test(
            recipe=recipe,
            scenarios=scenarios,
            profiler=self.performance_profiler
        )
        
        # Add to history
        self.test_history.append(result)
        
        logger.info(f"Single recipe test completed: {result.success} (score: {result.score:.2f})")
        return result
    
    async def predict_recipe_success(self, recipe: RecipeDefinition) -> Dict[str, float]:
        """
        Predict success probability for a recipe using ML.
        
        Args:
            recipe: Recipe to analyze
            
        Returns:
            Dict with success probability and confidence metrics
        """
        if not self.success_predictor:
            return {"success_probability": 0.5, "confidence": 0.0}
        
        return await self.success_predictor.predict_success(recipe)
    
    async def optimize_recipe(self, recipe_id: str) -> Optional[RecipeDefinition]:
        """
        Optimize a recipe using ML-based optimization.
        
        Args:
            recipe_id: ID of recipe to optimize
            
        Returns:
            Optimized recipe or None if optimization failed
        """
        if not self.recipe_optimizer:
            logger.warning("Recipe optimization not enabled")
            return None
        
        recipe = await self.recipe_registry.get_recipe(recipe_id)
        if not recipe:
            return None
        
        # Get historical results for this recipe
        historical_results = [r for r in self.test_history if r.recipe_id == recipe_id]
        
        # Optimize recipe
        optimized_recipe = await self.recipe_optimizer.optimize_recipe(recipe, historical_results)
        
        if optimized_recipe:
            # Register optimized version
            optimized_recipe.version = f"{recipe.version}.opt"
            await self.recipe_registry.register_recipe(optimized_recipe)
            logger.info(f"Recipe optimized: {recipe.name} -> {optimized_recipe.name}")
        
        return optimized_recipe
    
    async def get_test_analytics(self) -> Dict[str, Any]:
        """Get comprehensive test analytics"""
        if not self.test_history:
            return {"message": "No test history available"}
        
        # Basic statistics
        total_tests = len(self.test_history)
        successful_tests = sum(1 for r in self.test_history if r.success)
        success_rate = successful_tests / total_tests
        
        # Performance statistics
        avg_execution_time = sum(r.execution_time_ms for r in self.test_history) / total_tests
        avg_score = sum(r.score for r in self.test_history) / total_tests
        avg_memory_usage = sum(r.memory_usage_mb for r in self.test_history) / total_tests
        
        # Recipe statistics
        recipe_stats = {}
        for result in self.test_history:
            if result.recipe_name not in recipe_stats:
                recipe_stats[result.recipe_name] = {"tests": 0, "successes": 0, "avg_score": 0.0}
            
            stats = recipe_stats[result.recipe_name]
            stats["tests"] += 1
            if result.success:
                stats["successes"] += 1
            stats["avg_score"] = (stats["avg_score"] * (stats["tests"] - 1) + result.score) / stats["tests"]
        
        # Calculate success rates for recipes
        for stats in recipe_stats.values():
            stats["success_rate"] = stats["successes"] / stats["tests"]
        
        return {
            "total_tests": total_tests,
            "success_rate": success_rate,
            "average_execution_time_ms": avg_execution_time,
            "average_score": avg_score,
            "average_memory_usage_mb": avg_memory_usage,
            "recipe_statistics": recipe_stats,
            "active_sessions": len(self.active_sessions),
            "mcp_pool_status": await self.mcp_pool_manager.get_pool_status()
        }
    
    async def _generate_recipes_for_servers(self) -> None:
        """Generate recipes for installed MCP servers"""
        pool_status = await self.mcp_pool_manager.get_pool_status()
        server_names = list(self.mcp_pool_manager.server_instances.keys())
        
        if server_names:
            recipes = await self.recipe_generator.generate_recipes_for_servers(server_names)
            for recipe in recipes:
                await self.recipe_registry.register_recipe(recipe)
            
            logger.info(f"Generated {len(recipes)} recipes for installed servers")
    
    async def _start_required_servers(self, recipes: List[RecipeDefinition]) -> None:
        """Start MCP servers required by recipes"""
        required_servers = set()
        for recipe in recipes:
            required_servers.update(recipe.get_required_servers())
        
        for server_name in required_servers:
            if server_name in self.mcp_pool_manager.server_instances:
                await self.mcp_pool_manager.start_server(server_name)
    
    async def _generate_test_scenarios(self, 
                                     recipes: List[RecipeDefinition],
                                     test_categories: Optional[List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate test scenarios for recipes"""
        scenarios = {}
        for recipe in recipes:
            recipe_scenarios = await self.scenario_generator.generate_scenarios_for_recipe(
                recipe, 
                categories=test_categories
            )
            scenarios[recipe.id] = recipe_scenarios
        
        return scenarios
    
    async def _execute_recipe_tests(self, 
                                  recipes: List[RecipeDefinition],
                                  scenarios: Dict[str, List[Dict[str, Any]]],
                                  session: TestSession) -> List[RecipeTestResult]:
        """Execute tests for multiple recipes"""
        test_tasks = []
        
        for recipe in recipes:
            recipe_scenarios = scenarios.get(recipe.id, [])
            task = self.test_executor.execute_recipe_test(
                recipe=recipe,
                scenarios=recipe_scenarios,
                profiler=self.performance_profiler
            )
            test_tasks.append(task)
        
        # Execute tests with concurrency control
        results = await self.test_scheduler.execute_tests(test_tasks)
        
        return results
    
    async def _optimize_recipes_from_results(self, results: List[RecipeTestResult]) -> None:
        """Optimize recipes based on test results"""
        if not self.recipe_optimizer:
            return
        
        # Group results by recipe
        recipe_results = {}
        for result in results:
            if result.recipe_id not in recipe_results:
                recipe_results[result.recipe_id] = []
            recipe_results[result.recipe_id].append(result)
        
        # Optimize recipes with poor performance
        for recipe_id, recipe_results_list in recipe_results.items():
            avg_score = sum(r.score for r in recipe_results_list) / len(recipe_results_list)
            success_rate = sum(1 for r in recipe_results_list if r.success) / len(recipe_results_list)
            
            if avg_score < 0.7 or success_rate < 0.8:
                await self.optimize_recipe(recipe_id)
    
    async def _load_test_history(self) -> None:
        """Load test history from disk"""
        try:
            history_file = self.data_dir / "test_history.json"
            if history_file.exists():
                # Implementation would load and deserialize test history
                logger.info("Test history loaded from disk")
        except Exception as e:
            logger.error(f"Failed to load test history: {e}")
    
    async def _save_test_history(self) -> None:
        """Save test history to disk"""
        try:
            history_file = self.data_dir / "test_history.json"
            # Implementation would serialize and save test history
            logger.info("Test history saved to disk")
        except Exception as e:
            logger.error(f"Failed to save test history: {e}")
