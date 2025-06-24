"""
Mock data generators for PyGent Factory tests.
"""

import random
import string
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


def generate_mock_training_data(size: int = 100) -> List[Dict[str, Any]]:
    """Generate mock training data for AI models."""
    return [
        {
            "id": i,
            "input": f"test_input_{i}",
            "output": f"test_output_{i}",
            "features": [random.uniform(0, 1) for _ in range(10)],
            "label": random.choice(["positive", "negative", "neutral"]),
            "timestamp": datetime.utcnow() - timedelta(days=random.randint(0, 30))
        }
        for i in range(size)
    ]


def generate_mock_test_results(test_count: int = 50) -> List[Dict[str, Any]]:
    """Generate mock test results."""
    return [
        {
            "test_id": f"test_{i:04d}",
            "status": random.choice(["passed", "failed", "skipped"]),
            "duration": random.uniform(0.1, 5.0),
            "message": f"Test {i} result message",
            "assertions": {
                "total": random.randint(1, 10),
                "passed": lambda total: random.randint(0, total),
                "failed": lambda total, passed: total - passed
            },
            "timestamp": datetime.utcnow()
        }
        for i in range(test_count)
    ]


def generate_mock_agent_responses(count: int = 20) -> List[Dict[str, Any]]:
    """Generate mock agent responses."""
    return [
        {
            "agent_id": f"agent_{random.randint(1000, 9999)}",
            "response_id": f"resp_{i:06d}",
            "query": f"test query {i}",
            "response": f"test response {i}",
            "confidence": random.uniform(0.6, 1.0),
            "processing_time": random.uniform(0.05, 2.0),
            "metadata": {
                "model": random.choice(["gpt-4", "claude-3", "llama-2"]),
                "tokens_used": random.randint(50, 500),
                "temperature": random.uniform(0.1, 0.9)
            },
            "timestamp": datetime.utcnow()
        }
        for i in range(count)
    ]


def generate_mock_mcp_server_data(server_count: int = 5) -> List[Dict[str, Any]]:
    """Generate mock MCP server data."""
    return [
        {
            "server_id": f"mcp_server_{i}",
            "name": f"Test MCP Server {i}",
            "url": f"http://localhost:{8000 + i}",
            "status": random.choice(["active", "inactive", "error"]),
            "capabilities": random.sample([
                "text_processing", "image_analysis", "data_transformation",
                "api_integration", "file_handling", "database_access"
            ], random.randint(2, 4)),
            "health": {
                "uptime": random.uniform(0.8, 1.0),
                "response_time": random.uniform(10, 100),
                "error_rate": random.uniform(0, 0.05)
            },
            "metadata": {
                "version": f"1.{random.randint(0, 9)}.{random.randint(0, 9)}",
                "last_updated": datetime.utcnow() - timedelta(days=random.randint(0, 7))
            }
        }
        for i in range(server_count)
    ]


def generate_mock_workflow_data(workflow_count: int = 10) -> List[Dict[str, Any]]:
    """Generate mock workflow execution data."""
    return [
        {
            "workflow_id": f"workflow_{i:04d}",
            "name": f"Test Workflow {i}",
            "status": random.choice(["pending", "running", "completed", "failed"]),
            "steps": [
                {
                    "step_id": f"step_{j}",
                    "name": f"Step {j}",
                    "status": random.choice(["pending", "running", "completed", "failed"]),
                    "duration": random.uniform(1, 60)
                }
                for j in range(random.randint(3, 8))
            ],
            "start_time": datetime.utcnow() - timedelta(minutes=random.randint(1, 120)),
            "end_time": datetime.utcnow() if random.choice([True, False]) else None,
            "metrics": {
                "total_duration": random.uniform(30, 300),
                "steps_completed": random.randint(1, 8),
                "success_rate": random.uniform(0.7, 1.0)
            }
        }
        for i in range(workflow_count)
    ]


def generate_mock_reasoning_data(count: int = 15) -> List[Dict[str, Any]]:
    """Generate mock reasoning/thought process data."""
    return [
        {
            "reasoning_id": f"reasoning_{i:04d}",
            "query": f"test reasoning query {i}",
            "thought_tree": {
                "root": f"Initial thought {i}",
                "branches": [
                    {
                        "branch_id": f"branch_{j}",
                        "thought": f"Branch thought {j}",
                        "confidence": random.uniform(0.5, 0.9),
                        "children": [
                            f"child_thought_{k}" for k in range(random.randint(1, 3))
                        ]
                    }
                    for j in range(random.randint(2, 5))
                ]
            },
            "final_answer": f"Final reasoning result {i}",
            "confidence": random.uniform(0.6, 0.95),
            "processing_steps": random.randint(5, 15),
            "timestamp": datetime.utcnow()
        }
        for i in range(count)
    ]


def generate_random_string(length: int = 10) -> str:
    """Generate a random string of specified length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_mock_api_response(endpoint: str = "test", success: bool = True) -> Dict[str, Any]:
    """Generate a mock API response."""
    if success:
        return {
            "status": "success",
            "code": 200,
            "data": {
                "endpoint": endpoint,
                "result": f"Mock result for {endpoint}",
                "timestamp": datetime.utcnow().isoformat(),
                "id": generate_random_string(8)
            },
            "message": "Request processed successfully"
        }
    else:
        return {
            "status": "error",
            "code": random.choice([400, 404, 500]),
            "error": {
                "type": "MockError",
                "message": f"Mock error for {endpoint}",
                "details": "This is a generated error for testing"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
