"""
Additional test fixtures for PyGent Factory tests.
"""

import pytest
from typing import Dict, Any, List


@pytest.fixture
def sample_recipe_config():
    """Sample recipe configuration for testing."""
    return {
        "name": "Sample Recipe",
        "description": "A sample recipe for testing purposes",
        "category": "testing",
        "difficulty": "intermediate",
        "steps": [
            {
                "step": 1,
                "action": "initialize",
                "description": "Initialize the test environment",
                "parameters": {"timeout": 30}
            },
            {
                "step": 2,
                "action": "execute",
                "description": "Execute the main logic",
                "parameters": {"parallel": True}
            },
            {
                "step": 3,
                "action": "validate",
                "description": "Validate the results",
                "parameters": {"strict": True}
            }
        ],
        "requirements": {
            "mcp_servers": ["test-server"],
            "capabilities": ["testing", "validation"],
            "resources": {"cpu": 2, "memory": "1GB"}
        }
    }


@pytest.fixture
def sample_agent_pool():
    """Sample agent pool for testing."""
    return [
        {
            "agent_id": "agent_001",
            "agent_type": "testing",
            "capabilities": ["test_execution", "validation"],
            "status": "active"
        },
        {
            "agent_id": "agent_002", 
            "agent_type": "analysis",
            "capabilities": ["data_analysis", "reporting"],
            "status": "active"
        },
        {
            "agent_id": "agent_003",
            "agent_type": "optimization",
            "capabilities": ["parameter_tuning", "performance_optimization"],
            "status": "idle"
        }
    ]


@pytest.fixture
def sample_workflow_config():
    """Sample workflow configuration for testing."""
    return {
        "name": "Test Workflow",
        "description": "A test workflow for validation",
        "execution_mode": "sequential",
        "steps": [
            {
                "step_id": "step_1",
                "component_type": "nlp_system",
                "action": "parse_recipe",
                "parameters": {"recipe_text": "test recipe"},
                "required": True
            },
            {
                "step_id": "step_2",
                "component_type": "multi_agent",
                "action": "execute_tests",
                "parameters": {"test_suite": "basic"},
                "dependencies": ["step_1"],
                "required": True
            }
        ]
    }


@pytest.fixture
def sample_performance_metrics():
    """Sample performance metrics for testing."""
    return {
        "execution_time": 125.5,
        "cpu_usage": 65.2,
        "memory_usage": 1024.0,
        "network_usage": 15.8,
        "success_rate": 0.92,
        "throughput": 45.6,
        "error_count": 2,
        "warning_count": 5
    }


@pytest.fixture
def sample_optimization_results():
    """Sample optimization results for testing."""
    return {
        "optimal_parameters": {
            "learning_rate": 0.01,
            "batch_size": 64,
            "num_epochs": 100
        },
        "optimal_value": 95.8,
        "iterations": 150,
        "convergence_time": 45.2,
        "function_evaluations": 750,
        "improvement": 12.3
    }


# Export all fixtures
__all__ = [
    'sample_recipe_config',
    'sample_agent_pool', 
    'sample_workflow_config',
    'sample_performance_metrics',
    'sample_optimization_results'
]
