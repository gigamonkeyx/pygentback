"""
Test helper functions and utilities.
"""

import time
import asyncio
import psutil
import random
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


def create_test_recipe(name: str = "test_recipe", complexity: int = 5) -> Dict[str, Any]:
    """Create a test recipe with specified parameters."""
    return {
        "name": name,
        "description": f"Test recipe: {name}",
        "category": "testing",
        "difficulty": "intermediate" if complexity <= 5 else "advanced",
        "steps": [
            {
                "step": i,
                "action": f"action_{i}",
                "description": f"Test step {i}",
                "parameters": {"timeout": 30 + i * 10}
            }
            for i in range(1, complexity + 1)
        ],
        "requirements": {
            "mcp_servers": ["test-server"],
            "capabilities": ["testing"],
            "resources": {"cpu": complexity, "memory": f"{complexity * 512}MB"}
        },
        "metadata": {
            "complexity": complexity,
            "estimated_duration": complexity * 60,
            "created_at": datetime.utcnow().isoformat()
        }
    }


def create_mock_training_data(size: int = 100, features: int = 10) -> Dict[str, Any]:
    """Create mock training data for predictive model testing."""
    import numpy as np
    
    np.random.seed(42)  # For reproducible results in tests
    
    # Generate random features
    X = np.random.randn(size, features)
    
    # Generate target values with some correlation to features
    y = np.random.randn(size) + np.sum(X[:, :3], axis=1) * 0.5
    
    return {
        "features": X.tolist(),
        "targets": y.tolist(),
        "feature_names": [f"feature_{i}" for i in range(features)],
        "size": size,
        "metadata": {
            "created_at": datetime.utcnow().isoformat(),
            "type": "mock_training_data",
            "features_count": features,
            "samples_count": size
        }
    }


def create_test_agent(agent_type: str = "testing", capabilities: List[str] = None) -> Dict[str, Any]:
    """Create a test agent configuration."""
    if capabilities is None:
        capabilities = ["test_execution", "result_analysis"]
    
    return {
        "agent_id": f"test_agent_{random.randint(1000, 9999)}",
        "agent_type": agent_type,
        "name": f"Test {agent_type.title()} Agent",
        "capabilities": capabilities,
        "configuration": {
            "max_concurrent_tasks": random.randint(3, 8),
            "timeout_seconds": random.randint(60, 300),
            "retry_attempts": random.randint(1, 5)
        },
        "resources": {
            "cpu_cores": random.randint(1, 4),
            "memory_mb": random.randint(512, 2048),
            "storage_mb": random.randint(256, 1024)
        },
        "status": "active",
        "created_at": datetime.utcnow().isoformat()
    }


def create_test_workflow(name: str = "test_workflow", num_steps: int = 3) -> Dict[str, Any]:
    """Create a test workflow definition."""
    steps = []
    
    for i in range(num_steps):
        step = {
            "step_id": f"step_{i+1}",
            "name": f"Test Step {i+1}",
            "component_type": random.choice(["nlp_system", "multi_agent", "predictive_optimization"]),
            "action": f"test_action_{i+1}",
            "parameters": {"test_param": f"value_{i+1}"},
            "required": i < 2,  # First two steps required
            "timeout_seconds": 60.0 + i * 30
        }
        
        if i > 0:
            step["dependencies"] = [f"step_{i}"]
        
        steps.append(step)
    
    return {
        "name": name,
        "description": f"Test workflow: {name}",
        "version": "1.0",
        "steps": steps,
        "execution_mode": "sequential",
        "timeout_seconds": 300.0,
        "metadata": {
            "created_at": datetime.utcnow().isoformat(),
            "test_workflow": True
        }
    }


def assert_prediction_valid(prediction: Dict[str, Any], expected_keys: List[str] = None):
    """Assert that a prediction result is valid."""
    if expected_keys is None:
        expected_keys = ["predicted_value", "confidence", "model_name"]
    
    for key in expected_keys:
        assert key in prediction, f"Missing key '{key}' in prediction"
    
    assert isinstance(prediction["predicted_value"], (int, float)), "predicted_value must be numeric"
    assert 0.0 <= prediction["confidence"] <= 1.0, "confidence must be between 0 and 1"
    assert isinstance(prediction["model_name"], str), "model_name must be string"


def assert_workflow_success(result: Dict[str, Any]):
    """Assert that a workflow execution was successful."""
    assert "execution_id" in result, "Missing execution_id in workflow result"
    assert "success" in result, "Missing success flag in workflow result"
    assert result["success"] is True, f"Workflow failed: {result.get('error_message', 'Unknown error')}"
    assert "results" in result, "Missing results in workflow result"


def measure_performance(func, *args, **kwargs):
    """Measure performance of a function execution."""
    process = psutil.Process()
    
    # Measure before execution
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_cpu = process.cpu_percent()
    
    # Execute function
    if asyncio.iscoroutinefunction(func):
        result = asyncio.run(func(*args, **kwargs))
    else:
        result = func(*args, **kwargs)
    
    # Measure after execution
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    end_cpu = process.cpu_percent()
    
    return {
        "result": result,
        "execution_time": end_time - start_time,
        "memory_usage": end_memory - start_memory,
        "cpu_usage": end_cpu - start_cpu
    }


async def measure_async_performance(coro):
    """Measure performance of an async function."""
    process = psutil.Process()
    
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024
    
    result = await coro
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024
    
    return {
        "result": result,
        "execution_time": end_time - start_time,
        "memory_usage": end_memory - start_memory
    }


def validate_component_health(health_result: Dict[str, Any]):
    """Validate component health check result."""
    assert "status" in health_result, "Missing status in health result"
    assert "score" in health_result, "Missing score in health result"
    
    assert health_result["status"] in ["healthy", "degraded", "unhealthy", "error"], \
        f"Invalid health status: {health_result['status']}"
    
    assert 0.0 <= health_result["score"] <= 1.0, \
        f"Health score must be between 0 and 1, got: {health_result['score']}"


def create_synthetic_training_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Create synthetic training data for ML model testing."""
    training_data = []
    
    for i in range(num_samples):
        sample = {
            "features": {
                "recipe_complexity": random.uniform(1.0, 10.0),
                "resource_allocation": random.uniform(0.5, 5.0),
                "parallel_tasks": random.randint(1, 10),
                "data_size": random.uniform(10.0, 1000.0)
            },
            "target": random.uniform(50.0, 100.0),  # Performance score
            "metadata": {
                "sample_id": i,
                "created_at": datetime.utcnow().isoformat()
            }
        }
        training_data.append(sample)
    
    return training_data


def create_synthetic_test_results(num_results: int = 10) -> List[Dict[str, Any]]:
    """Create synthetic test results for validation testing."""
    results = []
    
    for i in range(num_results):
        result = {
            "test_id": f"test_{i:03d}",
            "recipe_name": f"test_recipe_{i}",
            "execution_time": random.uniform(10.0, 300.0),
            "success": random.choice([True, True, True, False]),  # 75% success rate
            "results": {
                "tests_run": random.randint(5, 20),
                "tests_passed": random.randint(3, 18),
                "coverage": random.uniform(60.0, 95.0)
            },
            "performance_metrics": {
                "cpu_usage": random.uniform(20.0, 80.0),
                "memory_usage": random.uniform(100.0, 1000.0),
                "network_usage": random.uniform(1.0, 50.0)
            },
            "timestamp": (datetime.utcnow() - timedelta(hours=random.randint(0, 24))).isoformat()
        }
        
        # Add some errors for failed tests
        if not result["success"]:
            result["errors"] = [
                {
                    "step": random.randint(1, 5),
                    "error_type": random.choice(["ValidationError", "TimeoutError", "ResourceError"]),
                    "message": f"Test error in step {random.randint(1, 5)}",
                    "severity": random.choice(["warning", "error", "critical"])
                }
            ]
        
        results.append(result)
    
    return results


def wait_for_condition(condition_func, timeout: float = 10.0, interval: float = 0.1):
    """Wait for a condition to become true."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        time.sleep(interval)
    
    return False


async def wait_for_async_condition(condition_func, timeout: float = 10.0, interval: float = 0.1):
    """Wait for an async condition to become true."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if await condition_func():
            return True
        await asyncio.sleep(interval)
    
    return False


def compare_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any], tolerance: float = 0.001) -> bool:
    """Compare two dictionaries with tolerance for numeric values."""
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    
    for key in dict1.keys():
        val1, val2 = dict1[key], dict2[key]
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if abs(val1 - val2) > tolerance:
                return False
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if not compare_dictionaries(val1, val2, tolerance):
                return False
        elif val1 != val2:
            return False
    
    return True


def generate_test_scenarios(scenario_type: str = "basic") -> List[Dict[str, Any]]:
    """Generate test scenarios for different testing needs."""
    scenarios = {
        "basic": [
            {"name": "simple_recipe", "complexity": 3, "expected_success": True},
            {"name": "medium_recipe", "complexity": 5, "expected_success": True},
            {"name": "complex_recipe", "complexity": 8, "expected_success": False}
        ],
        "performance": [
            {"name": "fast_execution", "max_time": 5.0, "load": "light"},
            {"name": "normal_execution", "max_time": 30.0, "load": "medium"},
            {"name": "stress_test", "max_time": 120.0, "load": "heavy"}
        ],
        "integration": [
            {"name": "nlp_multiagent", "components": ["nlp", "multiagent"]},
            {"name": "predictive_optimization", "components": ["predictive", "genetic"]},
            {"name": "full_pipeline", "components": ["nlp", "multiagent", "predictive", "integration"]}
        ]
    }
    
    return scenarios.get(scenario_type, scenarios["basic"])
