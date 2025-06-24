"""
Agent Evaluation System

This module provides comprehensive evaluation capabilities for agents in PyGent Factory,
including test case management, benchmark execution, performance metrics,
and automated evaluation reporting.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import statistics

from ..core.agent import BaseAgent
from ..core.agent_factory import AgentFactory
from ..database.connection import get_database_manager
from ..config.settings import Settings


logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class TestCategory(Enum):
    """Test categories"""
    FUNCTIONALITY = "functionality"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    SECURITY = "security"
    INTEGRATION = "integration"
    STRESS = "stress"
    REGRESSION = "regression"


class TestDifficulty(Enum):
    """Test difficulty levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class TestCase:
    """Represents a test case for agent evaluation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: TestCategory = TestCategory.FUNCTIONALITY
    difficulty: TestDifficulty = TestDifficulty.BASIC
    input_data: Dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[Dict[str, Any]] = None
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    max_retries: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test case to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "evaluation_criteria": self.evaluation_criteria,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class TestResult:
    """Represents the result of a test execution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    test_case_id: str = ""
    agent_id: str = ""
    status: TestStatus = TestStatus.PENDING
    score: Optional[float] = None
    max_score: float = 100.0
    actual_output: Optional[Dict[str, Any]] = None
    execution_time_ms: int = 0
    memory_usage_mb: Optional[float] = None
    error_message: Optional[str] = None
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary"""
        return {
            "id": self.id,
            "test_case_id": self.test_case_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "score": self.score,
            "max_score": self.max_score,
            "actual_output": self.actual_output,
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "error_message": self.error_message,
            "detailed_metrics": self.detailed_metrics,
            "retry_count": self.retry_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }


@dataclass
class EvaluationSuite:
    """Collection of test cases for comprehensive evaluation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    test_cases: List[TestCase] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the suite"""
        self.test_cases.append(test_case)
    
    def get_test_cases_by_category(self, category: TestCategory) -> List[TestCase]:
        """Get test cases by category"""
        return [tc for tc in self.test_cases if tc.category == category]
    
    def get_test_cases_by_difficulty(self, difficulty: TestDifficulty) -> List[TestCase]:
        """Get test cases by difficulty"""
        return [tc for tc in self.test_cases if tc.difficulty == difficulty]


class TestEvaluator:
    """Evaluates test results against expected outcomes"""
    
    def __init__(self):
        self.evaluation_functions: Dict[str, Callable] = {
            "exact_match": self._exact_match,
            "fuzzy_match": self._fuzzy_match,
            "numeric_range": self._numeric_range,
            "contains": self._contains,
            "custom": self._custom_evaluation
        }
    
    def evaluate_result(self, test_case: TestCase, actual_output: Dict[str, Any]) -> float:
        """
        Evaluate test result and return score (0-100).
        
        Args:
            test_case: The test case being evaluated
            actual_output: The actual output from the agent
            
        Returns:
            float: Score between 0 and 100
        """
        if not test_case.expected_output:
            # If no expected output, use custom evaluation criteria
            return self._evaluate_with_criteria(test_case, actual_output)
        
        evaluation_method = test_case.evaluation_criteria.get("method", "exact_match")
        evaluation_func = self.evaluation_functions.get(evaluation_method, self._exact_match)
        
        try:
            return evaluation_func(test_case, actual_output)
        except Exception as e:
            logger.error(f"Evaluation error for test {test_case.id}: {str(e)}")
            return 0.0
    
    def _exact_match(self, test_case: TestCase, actual_output: Dict[str, Any]) -> float:
        """Exact match evaluation"""
        if actual_output == test_case.expected_output:
            return 100.0
        return 0.0
    
    def _fuzzy_match(self, test_case: TestCase, actual_output: Dict[str, Any]) -> float:
        """Fuzzy match evaluation with partial credit"""
        expected = test_case.expected_output
        if not expected:
            return 0.0
        
        total_fields = len(expected)
        if total_fields == 0:
            return 100.0
        
        matching_fields = 0
        for key, expected_value in expected.items():
            if key in actual_output and actual_output[key] == expected_value:
                matching_fields += 1
        
        return (matching_fields / total_fields) * 100.0
    
    def _numeric_range(self, test_case: TestCase, actual_output: Dict[str, Any]) -> float:
        """Numeric range evaluation"""
        criteria = test_case.evaluation_criteria
        field = criteria.get("field")
        min_value = criteria.get("min_value")
        max_value = criteria.get("max_value")
        
        if not field or field not in actual_output:
            return 0.0
        
        actual_value = actual_output[field]
        
        try:
            actual_value = float(actual_value)
            if min_value is not None and actual_value < min_value:
                return 0.0
            if max_value is not None and actual_value > max_value:
                return 0.0
            return 100.0
        except (ValueError, TypeError):
            return 0.0
    
    def _contains(self, test_case: TestCase, actual_output: Dict[str, Any]) -> float:
        """Contains evaluation - check if output contains expected elements"""
        expected = test_case.expected_output
        if not expected:
            return 0.0
        
        score = 0.0
        total_checks = 0
        
        for key, expected_value in expected.items():
            total_checks += 1
            if key in actual_output:
                actual_value = str(actual_output[key]).lower()
                expected_str = str(expected_value).lower()
                if expected_str in actual_value:
                    score += 1
        
        return (score / max(1, total_checks)) * 100.0
    
    def _custom_evaluation(self, test_case: TestCase, actual_output: Dict[str, Any]) -> float:
        """Custom evaluation using provided criteria"""
        return self._evaluate_with_criteria(test_case, actual_output)
    
    def _evaluate_with_criteria(self, test_case: TestCase, actual_output: Dict[str, Any]) -> float:
        """Evaluate using custom criteria"""
        criteria = test_case.evaluation_criteria
        
        # Basic scoring based on output presence and structure
        if not actual_output:
            return 0.0
        
        # Check for required fields
        required_fields = criteria.get("required_fields", [])
        if required_fields:
            present_fields = sum(1 for field in required_fields if field in actual_output)
            field_score = (present_fields / len(required_fields)) * 50.0
        else:
            field_score = 50.0
        
        # Check for non-empty values
        non_empty_score = 0.0
        if actual_output:
            non_empty_values = sum(1 for v in actual_output.values() if v is not None and str(v).strip())
            total_values = len(actual_output)
            if total_values > 0:
                non_empty_score = (non_empty_values / total_values) * 50.0
        
        return field_score + non_empty_score


class TestRunner:
    """
    Main test runner for agent evaluation.
    
    Executes test cases against agents and collects results
    with comprehensive metrics and reporting.
    """
    
    def __init__(self, settings: Settings, agent_factory: AgentFactory):
        self.settings = settings
        self.agent_factory = agent_factory
        self.evaluator = TestEvaluator()
        self.test_suites: Dict[str, EvaluationSuite] = {}
        self.test_results: List[TestResult] = []
        self._load_default_test_suites()
    
    def _load_default_test_suites(self):
        """Load default test suites"""
        # Basic functionality test suite
        basic_suite = EvaluationSuite(
            name="Basic Functionality",
            description="Basic agent functionality tests"
        )
        
        # Simple echo test
        echo_test = TestCase(
            name="Echo Test",
            description="Test agent's ability to echo input",
            category=TestCategory.FUNCTIONALITY,
            difficulty=TestDifficulty.BASIC,
            input_data={"message": "Hello, World!"},
            expected_output={"response": "Hello, World!"},
            evaluation_criteria={"method": "contains", "required_fields": ["response"]}
        )
        basic_suite.add_test_case(echo_test)
        
        # Capability test
        capability_test = TestCase(
            name="Capability Listing",
            description="Test agent's ability to list its capabilities",
            category=TestCategory.FUNCTIONALITY,
            difficulty=TestDifficulty.BASIC,
            input_data={"action": "list_capabilities"},
            evaluation_criteria={"method": "custom", "required_fields": ["capabilities"]}
        )
        basic_suite.add_test_case(capability_test)
        
        self.test_suites["basic"] = basic_suite
    
    async def run_test_case(self, agent: BaseAgent, test_case: TestCase) -> TestResult:
        """
        Run a single test case against an agent.
        
        Args:
            agent: The agent to test
            test_case: The test case to execute
            
        Returns:
            TestResult: The test execution result
        """
        result = TestResult(
            test_case_id=test_case.id,
            agent_id=agent.agent_id,
            started_at=datetime.utcnow()
        )
        
        try:
            result.status = TestStatus.RUNNING
            
            # Execute test with timeout
            start_time = time.time()
            
            try:
                # Execute the test case
                actual_output = await asyncio.wait_for(
                    self._execute_test_case(agent, test_case),
                    timeout=test_case.timeout_seconds
                )
                
                result.actual_output = actual_output
                result.execution_time_ms = int((time.time() - start_time) * 1000)
                
                # Evaluate result
                result.score = self.evaluator.evaluate_result(test_case, actual_output)
                result.status = TestStatus.PASSED if result.score > 0 else TestStatus.FAILED
                
            except asyncio.TimeoutError:
                result.status = TestStatus.TIMEOUT
                result.error_message = f"Test timed out after {test_case.timeout_seconds} seconds"
                result.score = 0.0
            
        except Exception as e:
            result.status = TestStatus.ERROR
            result.error_message = str(e)
            result.score = 0.0
            logger.error(f"Test execution error: {str(e)}")
        
        finally:
            result.completed_at = datetime.utcnow()
            if result.execution_time_ms == 0 and result.started_at:
                result.execution_time_ms = int(
                    (result.completed_at - result.started_at).total_seconds() * 1000
                )
        
        return result
    
    async def _execute_test_case(self, agent: BaseAgent, test_case: TestCase) -> Dict[str, Any]:
        """Execute a test case against an agent"""
        input_data = test_case.input_data
        
        # Determine how to execute based on input data
        if "capability" in input_data:
            # Execute specific capability
            capability = input_data["capability"]
            params = input_data.get("parameters", {})
            result = await agent.execute_capability(capability, params)
            return {"result": result}
        
        elif "message" in input_data:
            # Send message to agent
            from ..core.agent import AgentMessage, MessageType
            message = AgentMessage(
                type=MessageType.REQUEST,
                sender="test_runner",
                recipient=agent.agent_id,
                content=input_data
            )
            response = await agent.process_message(message)
            return response.content
        
        elif "action" in input_data:
            # Execute specific action
            action = input_data["action"]
            
            if action == "list_capabilities":
                capabilities = agent.get_capabilities()
                return {
                    "capabilities": [cap.name for cap in capabilities]
                }
            elif action == "get_status":
                return agent.get_status()
        
        # Default: return input as echo
        return {"response": input_data.get("message", "No response")}
    
    async def run_test_suite(self, agent: BaseAgent, suite_name: str) -> List[TestResult]:
        """
        Run a complete test suite against an agent.
        
        Args:
            agent: The agent to test
            suite_name: Name of the test suite to run
            
        Returns:
            List[TestResult]: List of test results
        """
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite not found: {suite_name}")
        
        suite = self.test_suites[suite_name]
        results = []
        
        logger.info(f"Running test suite '{suite.name}' against agent {agent.agent_id}")
        
        for test_case in suite.test_cases:
            logger.info(f"Running test: {test_case.name}")
            
            result = await self.run_test_case(agent, test_case)
            results.append(result)
            self.test_results.append(result)
            
            logger.info(f"Test completed: {test_case.name} - Status: {result.status.value}, Score: {result.score}")
        
        return results
    
    async def run_benchmark(self, agent_id: str, suite_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark against an agent.
        
        Args:
            agent_id: ID of the agent to benchmark
            suite_names: List of test suite names to run (default: all)
            
        Returns:
            Dict[str, Any]: Benchmark report
        """
        # Get agent
        agent = await self.agent_factory.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent not found: {agent_id}")
        
        # Determine which suites to run
        if suite_names is None:
            suite_names = list(self.test_suites.keys())
        
        benchmark_start = datetime.utcnow()
        all_results = []
        
        # Run each test suite
        for suite_name in suite_names:
            if suite_name in self.test_suites:
                suite_results = await self.run_test_suite(agent, suite_name)
                all_results.extend(suite_results)
        
        benchmark_end = datetime.utcnow()
        
        # Generate benchmark report
        report = self._generate_benchmark_report(agent, all_results, benchmark_start, benchmark_end)
        
        logger.info(f"Benchmark completed for agent {agent_id}: {report['summary']['overall_score']:.1f}% overall score")
        
        return report
    
    def _generate_benchmark_report(self, agent: BaseAgent, results: List[TestResult], 
                                 start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        if not results:
            return {"error": "No test results available"}
        
        # Calculate summary statistics
        scores = [r.score for r in results if r.score is not None]
        execution_times = [r.execution_time_ms for r in results if r.execution_time_ms > 0]
        
        passed_tests = [r for r in results if r.status == TestStatus.PASSED]
        failed_tests = [r for r in results if r.status == TestStatus.FAILED]
        error_tests = [r for r in results if r.status == TestStatus.ERROR]
        timeout_tests = [r for r in results if r.status == TestStatus.TIMEOUT]
        
        summary = {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "agent_type": agent.type,
            "total_tests": len(results),
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "error_tests": len(error_tests),
            "timeout_tests": len(timeout_tests),
            "success_rate": (len(passed_tests) / len(results)) * 100 if results else 0,
            "overall_score": statistics.mean(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "avg_execution_time_ms": statistics.mean(execution_times) if execution_times else 0,
            "total_execution_time_ms": sum(execution_times),
            "benchmark_duration_seconds": (end_time - start_time).total_seconds()
        }
        
        # Category breakdown
        category_stats = {}
        for category in TestCategory:
            category_results = [r for r in results 
                              if any(tc.category == category for tc in self._get_all_test_cases() 
                                   if tc.id == r.test_case_id)]
            if category_results:
                category_scores = [r.score for r in category_results if r.score is not None]
                category_stats[category.value] = {
                    "total_tests": len(category_results),
                    "passed_tests": len([r for r in category_results if r.status == TestStatus.PASSED]),
                    "average_score": statistics.mean(category_scores) if category_scores else 0
                }
        
        return {
            "summary": summary,
            "category_breakdown": category_stats,
            "detailed_results": [r.to_dict() for r in results],
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _get_all_test_cases(self) -> List[TestCase]:
        """Get all test cases from all suites"""
        all_test_cases = []
        for suite in self.test_suites.values():
            all_test_cases.extend(suite.test_cases)
        return all_test_cases
    
    def add_test_suite(self, suite: EvaluationSuite) -> None:
        """Add a test suite"""
        self.test_suites[suite.name.lower().replace(" ", "_")] = suite
    
    def get_test_results(self, agent_id: Optional[str] = None) -> List[TestResult]:
        """Get test results, optionally filtered by agent ID"""
        if agent_id:
            return [r for r in self.test_results if r.agent_id == agent_id]
        return self.test_results.copy()
    
    def get_test_suites(self) -> List[str]:
        """Get list of available test suite names"""
        return list(self.test_suites.keys())
