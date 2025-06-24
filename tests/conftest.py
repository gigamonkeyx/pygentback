"""
Pytest configuration and shared fixtures for PyGent Factory tests.
"""

import pytest
import asyncio
import tempfile
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock
from datetime import datetime

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Test data and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_recipe_data():
    """Sample recipe data for testing."""
    return {
        "name": "Test Recipe",
        "description": "A test recipe for validation",
        "category": "testing",
        "difficulty": "intermediate",
        "steps": [
            {
                "step": 1,
                "action": "initialize",
                "description": "Setup test environment",
                "parameters": {"timeout": 30}
            },
            {
                "step": 2,
                "action": "execute",
                "description": "Run test logic",
                "parameters": {"parallel": True}
            },
            {
                "step": 3,
                "action": "validate",
                "description": "Validate results",
                "parameters": {"strict": True}
            }
        ],
        "requirements": {
            "mcp_servers": ["test-server"],
            "capabilities": ["testing", "validation"],
            "resources": {"cpu": 2, "memory": "1GB"}
        },
        "expected_outputs": ["test_results", "validation_report"],
        "metadata": {
            "author": "test_user",
            "version": "1.0",
            "tags": ["test", "validation"]
        }
    }


@pytest.fixture
def sample_test_results():
    """Sample test results for testing."""
    return {
        "test_id": "test_001",
        "recipe_name": "Test Recipe",
        "execution_time": 45.2,
        "success": True,
        "results": {
            "tests_run": 10,
            "tests_passed": 8,
            "tests_failed": 2,
            "coverage": 85.5
        },
        "errors": [
            {
                "step": 2,
                "error_type": "ValidationError",
                "message": "Invalid parameter value",
                "severity": "warning"
            }
        ],
        "performance_metrics": {
            "cpu_usage": 45.2,
            "memory_usage": 512.0,
            "network_usage": 10.5
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@pytest.fixture
def mock_nlp_processor():
    """Mock NLP processor for testing."""
    processor = Mock()
    processor.parse_recipe = AsyncMock(return_value={
        "parsed_steps": [
            {"action": "initialize", "parameters": {}},
            {"action": "execute", "parameters": {}},
            {"action": "validate", "parameters": {}}
        ],
        "complexity": 5,
        "estimated_duration": 300
    })
    processor.interpret_results = AsyncMock(return_value={
        "summary": "Test completed successfully",
        "insights": ["Good performance", "Minor issues detected"],
        "recommendations": ["Optimize step 2", "Add error handling"]
    })
    return processor


@pytest.fixture
def mock_agent_coordinator():
    """Mock agent coordinator for testing."""
    coordinator = Mock()
    coordinator.start = AsyncMock()
    coordinator.shutdown = AsyncMock()
    coordinator.create_agent = AsyncMock(return_value=Mock(agent_id="test_agent_001"))
    coordinator.coordinate_agents = AsyncMock(return_value={
        "coordination_id": "coord_001",
        "success": True,
        "results": {"agent_1": "completed", "agent_2": "completed"}
    })
    coordinator.get_system_status = Mock(return_value={
        "is_running": True,
        "agent_count": 3,
        "active_workflows": 1
    })
    return coordinator


@pytest.fixture
def mock_predictive_engine():
    """Mock predictive engine for testing."""
    engine = Mock()
    engine.start = AsyncMock()
    engine.stop = AsyncMock()
    engine.predict = AsyncMock(return_value={
        "predicted_value": 85.5,
        "confidence": 0.92,
        "model_name": "test_predictor"
    })
    engine.optimize = AsyncMock(return_value={
        "optimal_parameters": {"param1": 10, "param2": 0.5},
        "optimal_value": 95.2,
        "iterations": 50
    })
    engine.get_engine_status = Mock(return_value={
        "is_running": True,
        "total_models": 3,
        "trained_models": 2
    })
    return engine


@pytest.fixture
def mock_integration_engine():
    """Mock integration engine for testing."""
    engine = Mock()
    engine.start = AsyncMock()
    engine.shutdown = AsyncMock()
    engine.register_component = Mock()
    engine.execute_workflow = AsyncMock(return_value={
        "execution_id": "exec_001",
        "success": True,
        "results": {"step_1": "completed", "step_2": "completed"}
    })
    engine.get_system_status = Mock(return_value={
        "integration_status": "ready",
        "total_components": 5,
        "active_components": 4
    })
    return engine


@pytest.fixture
def sample_workflow_definition():
    """Sample workflow definition for testing."""
    return {
        "name": "test_workflow",
        "description": "Test workflow for validation",
        "steps": [
            {
                "step_id": "parse_recipe",
                "component_type": "nlp_system",
                "action": "parse_recipe",
                "parameters": {"recipe_text": "test recipe"},
                "required": True
            },
            {
                "step_id": "predict_performance",
                "component_type": "predictive_optimization",
                "action": "predict_performance",
                "parameters": {"recipe_data": "parsed_recipe"},
                "dependencies": ["parse_recipe"],
                "required": False
            },
            {
                "step_id": "execute_tests",
                "component_type": "multi_agent",
                "action": "execute_tests",
                "parameters": {"test_suite": "comprehensive"},
                "dependencies": ["parse_recipe"],
                "required": True
            }
        ]
    }


@pytest.fixture
def sample_agent_config():
    """Sample agent configuration for testing."""
    return {
        "agent_id": "test_agent_001",
        "agent_type": "testing",
        "name": "Test Agent",
        "capabilities": ["test_execution", "result_analysis"],
        "configuration": {
            "max_concurrent_tasks": 5,
            "timeout_seconds": 300,
            "retry_attempts": 3
        },
        "resources": {
            "cpu_cores": 2,
            "memory_mb": 1024,
            "storage_mb": 512
        }
    }


@pytest.fixture
def sample_prediction_data():
    """Sample prediction data for testing."""
    return {
        "input_features": {
            "recipe_complexity": 5.0,
            "resource_allocation": 2.0,
            "parallel_tasks": 3.0,
            "data_size": 100.0
        },
        "target_value": 85.5,
        "model_type": "performance_predictor",
        "training_data": [
            {"features": {"complexity": 3.0, "resources": 1.0}, "target": 75.0},
            {"features": {"complexity": 7.0, "resources": 3.0}, "target": 90.0},
            {"features": {"complexity": 5.0, "resources": 2.0}, "target": 82.0}
        ]
    }


@pytest.fixture
def sample_optimization_config():
    """Sample optimization configuration for testing."""
    return {
        "algorithm": "genetic",
        "parameter_space": {
            "param1": {"type": "continuous", "min": 0.0, "max": 10.0},
            "param2": {"type": "integer", "min": 1, "max": 100},
            "param3": {"type": "discrete", "values": ["option1", "option2", "option3"]}
        },
        "objective_function": "maximize_performance",
        "constraints": {
            "max_iterations": 100,
            "convergence_threshold": 0.001,
            "timeout_seconds": 300
        }
    }


# Performance testing fixtures
@pytest.fixture
def performance_benchmark():
    """Performance benchmark configuration."""
    return {
        "max_execution_time": 5.0,  # seconds
        "max_memory_usage": 100.0,  # MB
        "min_throughput": 10.0,     # operations per second
        "max_error_rate": 0.05      # 5% error rate
    }


# Mock external services
@pytest.fixture
def mock_external_services():
    """Mock external services for testing."""
    return {
        "mcp_server": Mock(),
        "vector_store": Mock(),
        "embedding_service": Mock(),
        "monitoring_service": Mock()
    }


# Test categories
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Async test utilities
@pytest.fixture
async def async_test_client():
    """Async test client for API testing."""
    # This would be implemented when we have API endpoints
    pass


# Database fixtures (if needed)
@pytest.fixture
def test_database():
    """Test database fixture."""
    # This would be implemented when we have database components
    pass


# Logging configuration for tests
@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logging for tests."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # Reduce noise in tests


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "fast: marks tests as fast")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "nlp: marks tests as NLP tests")
    config.addinivalue_line("markers", "multiagent: marks tests as multi-agent tests")
    config.addinivalue_line("markers", "predictive: marks tests as predictive tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to async tests by default
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

        # Add fast marker to non-async tests
        if not asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.fast)
