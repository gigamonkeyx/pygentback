"""
Basic functionality tests to verify test infrastructure.
"""

import pytest
import asyncio
from tests.utils.helpers import (
    create_test_recipe, create_test_agent, create_test_workflow,
    create_synthetic_training_data, assert_prediction_valid
)


class TestBasicFunctionality:
    """Basic tests to verify test infrastructure works."""
    
    def test_create_test_recipe(self):
        """Test recipe creation utility."""
        recipe = create_test_recipe("test_recipe", complexity=5)
        
        assert recipe["name"] == "test_recipe"
        assert recipe["metadata"]["complexity"] == 5
        assert len(recipe["steps"]) == 5
        assert "requirements" in recipe
    
    def test_create_test_agent(self):
        """Test agent creation utility."""
        agent = create_test_agent("testing", ["test_execution"])
        
        assert agent["agent_type"] == "testing"
        assert "test_execution" in agent["capabilities"]
        assert "agent_id" in agent
        assert "configuration" in agent
    
    def test_create_test_workflow(self):
        """Test workflow creation utility."""
        workflow = create_test_workflow("test_workflow", num_steps=3)
        
        assert workflow["name"] == "test_workflow"
        assert len(workflow["steps"]) == 3
        assert workflow["steps"][0]["step_id"] == "step_1"
    
    def test_create_synthetic_training_data(self):
        """Test synthetic training data generation."""
        data = create_synthetic_training_data(num_samples=10)
        
        assert len(data) == 10
        assert all("features" in sample for sample in data)
        assert all("target" in sample for sample in data)
    
    def test_assert_prediction_valid(self):
        """Test prediction validation utility."""
        valid_prediction = {
            "predicted_value": 85.5,
            "confidence": 0.92,
            "model_name": "test_model"
        }
        
        # Should not raise any exception
        assert_prediction_valid(valid_prediction)
        
        # Test with invalid prediction
        invalid_prediction = {
            "predicted_value": "invalid",
            "confidence": 1.5,  # > 1.0
            "model_name": 123   # Not string
        }
        
        with pytest.raises(AssertionError):
            assert_prediction_valid(invalid_prediction)
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async test functionality."""
        async def sample_async_function():
            await asyncio.sleep(0.01)
            return "async_result"
        
        result = await sample_async_function()
        assert result == "async_result"


@pytest.mark.fast
class TestFastTests:
    """Fast tests that should run quickly."""
    
    def test_fast_operation(self):
        """Test that runs quickly."""
        result = 2 + 2
        assert result == 4
    
    def test_another_fast_operation(self):
        """Another fast test."""
        data = {"key": "value"}
        assert data["key"] == "value"


@pytest.mark.slow
class TestSlowTests:
    """Slow tests that take more time."""
    
    @pytest.mark.asyncio
    async def test_slow_operation(self):
        """Test that takes some time."""
        await asyncio.sleep(0.1)  # Simulate slow operation
        assert True
