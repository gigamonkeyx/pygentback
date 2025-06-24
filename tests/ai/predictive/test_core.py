"""
Tests for Predictive Optimization core components.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.ai.predictive.core import PredictiveEngine, OptimizationEngine, RecommendationEngine
from src.ai.predictive.models import Prediction, Optimization, Recommendation, PredictionType
from tests.utils.helpers import create_mock_training_data, assert_prediction_valid


class TestPredictiveEngine:
    """Test cases for PredictiveEngine."""
    
    @pytest.fixture
    def predictive_engine(self):
        """Create predictive engine instance."""
        return PredictiveEngine()
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_engine_initialization(self, predictive_engine):
        """Test predictive engine initialization."""
        assert predictive_engine.is_running is False
        
        await predictive_engine.start()
        assert predictive_engine.is_running is True
        
        status = predictive_engine.get_engine_status()
        assert status["is_running"] is True
        assert "total_models" in status
        assert "trained_models" in status
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_model_registration(self, predictive_engine):
        """Test predictive model registration."""
        await predictive_engine.start()
        
        # Register performance predictor
        model_config = {
            "model_name": "test_performance_predictor",
            "model_type": "regression",
            "prediction_type": "performance"
        }
        
        success = await predictive_engine.register_model(model_config)
        assert success is True
        
        # Verify model is registered
        models = predictive_engine.get_registered_models()
        assert "test_performance_predictor" in models
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_model_training(self, predictive_engine):
        """Test model training functionality."""
        await predictive_engine.start()
        
        # Register model
        model_config = {
            "model_name": "test_predictor",
            "model_type": "regression",
            "prediction_type": "performance"
        }
        await predictive_engine.register_model(model_config)
        
        # Create training data
        training_data = create_mock_training_data(num_samples=50)
        
        # Train model
        training_result = await predictive_engine.train_model(
            "test_predictor", training_data
        )
        
        assert training_result["success"] is True
        assert "training_metrics" in training_result
        assert training_result["samples_trained"] == 50
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_prediction_generation(self, predictive_engine, sample_prediction_data):
        """Test prediction generation."""
        await predictive_engine.start()
        
        # Register and train model
        model_config = {
            "model_name": "test_predictor",
            "model_type": "regression",
            "prediction_type": "performance"
        }
        await predictive_engine.register_model(model_config)
        
        training_data = sample_prediction_data["training_data"]
        await predictive_engine.train_model("test_predictor", training_data)
        
        # Generate prediction
        input_features = sample_prediction_data["input_features"]
        prediction = await predictive_engine.predict(
            "test_predictor", input_features
        )
        
        assert isinstance(prediction, Prediction)
        assert prediction.model_name == "test_predictor"
        assert prediction.prediction_type == PredictionType.PERFORMANCE
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.predicted_value is not None
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_model_evaluation(self, predictive_engine):
        """Test model evaluation."""
        await predictive_engine.start()
        
        # Register and train model
        model_config = {
            "model_name": "eval_predictor",
            "model_type": "regression",
            "prediction_type": "performance"
        }
        await predictive_engine.register_model(model_config)
        
        training_data = create_mock_training_data(num_samples=100)
        await predictive_engine.train_model("eval_predictor", training_data)
        
        # Create test data
        test_data = create_mock_training_data(num_samples=20)
        
        # Evaluate model
        evaluation_result = await predictive_engine.evaluate_model(
            "eval_predictor", test_data
        )
        
        assert "metrics" in evaluation_result
        assert "mean_absolute_error" in evaluation_result["metrics"]
        assert "r_squared" in evaluation_result["metrics"]
        assert evaluation_result["test_samples"] == 20
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_batch_predictions(self, predictive_engine):
        """Test batch prediction functionality."""
        await predictive_engine.start()
        
        # Setup model
        model_config = {
            "model_name": "batch_predictor",
            "model_type": "regression",
            "prediction_type": "performance"
        }
        await predictive_engine.register_model(model_config)
        
        training_data = create_mock_training_data(num_samples=50)
        await predictive_engine.train_model("batch_predictor", training_data)
        
        # Create batch input
        batch_inputs = [
            {"recipe_complexity": 3.0, "resource_allocation": 1.5},
            {"recipe_complexity": 7.0, "resource_allocation": 3.0},
            {"recipe_complexity": 5.0, "resource_allocation": 2.0}
        ]
        
        # Generate batch predictions
        predictions = await predictive_engine.predict_batch(
            "batch_predictor", batch_inputs
        )
        
        assert len(predictions) == 3
        for prediction in predictions:
            assert isinstance(prediction, Prediction)
            assert_prediction_valid(prediction.to_dict())


class TestOptimizationEngine:
    """Test cases for OptimizationEngine."""
    
    @pytest.fixture
    def optimization_engine(self):
        """Create optimization engine instance."""
        return OptimizationEngine()
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_engine_initialization(self, optimization_engine):
        """Test optimization engine initialization."""
        await optimization_engine.start()
        assert optimization_engine.is_running is True
        
        status = optimization_engine.get_engine_status()
        assert status["is_running"] is True
        assert "available_optimizers" in status
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_genetic_optimization(self, optimization_engine, sample_optimization_config):
        """Test genetic algorithm optimization."""
        await optimization_engine.start()
        
        # Define objective function
        def objective_function(params):
            # Simple quadratic function with maximum at x=5, y=3
            x = params["param1"]
            y = params["param2"]
            return -(x - 5)**2 - (y - 3)**2 + 100
        
        # Configure optimization
        config = sample_optimization_config.copy()
        config["algorithm"] = "genetic"
        config["parameter_space"] = {
            "param1": {"type": "continuous", "min": 0.0, "max": 10.0},
            "param2": {"type": "continuous", "min": 0.0, "max": 6.0}
        }
        
        # Run optimization
        result = await optimization_engine.optimize(
            objective_function, config
        )
        
        assert result.success is True
        assert "param1" in result.optimal_parameters
        assert "param2" in result.optimal_parameters
        assert result.optimal_value > 90  # Should find near-optimal solution
        assert result.iterations > 0
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_bayesian_optimization(self, optimization_engine):
        """Test Bayesian optimization."""
        await optimization_engine.start()
        
        # Define objective function
        def objective_function(params):
            x = params["x"]
            return -(x - 2)**2 + 10  # Maximum at x=2
        
        config = {
            "algorithm": "bayesian",
            "parameter_space": {
                "x": {"type": "continuous", "min": -5.0, "max": 5.0}
            },
            "constraints": {
                "max_iterations": 20,
                "convergence_threshold": 0.01
            }
        }
        
        result = await optimization_engine.optimize(objective_function, config)
        
        assert result.success is True
        assert abs(result.optimal_parameters["x"] - 2.0) < 1.0  # Close to optimum
        assert result.optimal_value > 8.0  # Near maximum value
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_multi_objective_optimization(self, optimization_engine):
        """Test multi-objective optimization."""
        await optimization_engine.start()
        
        # Define multi-objective function
        def multi_objective_function(params):
            x = params["x"]
            y = params["y"]
            # Two conflicting objectives
            obj1 = x**2 + y**2  # Minimize distance from origin
            obj2 = (x - 5)**2 + (y - 5)**2  # Minimize distance from (5,5)
            return {"objective_1": -obj1, "objective_2": -obj2}
        
        config = {
            "algorithm": "genetic",
            "parameter_space": {
                "x": {"type": "continuous", "min": 0.0, "max": 5.0},
                "y": {"type": "continuous", "min": 0.0, "max": 5.0}
            },
            "multi_objective": True,
            "constraints": {"max_iterations": 50}
        }
        
        result = await optimization_engine.optimize(multi_objective_function, config)
        
        assert result.success is True
        assert "pareto_front" in result.metadata
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_constrained_optimization(self, optimization_engine):
        """Test optimization with constraints."""
        await optimization_engine.start()
        
        def objective_function(params):
            x = params["x"]
            y = params["y"]
            return x + y  # Maximize x + y
        
        def constraint_function(params):
            x = params["x"]
            y = params["y"]
            return x**2 + y**2 <= 4  # Circle constraint
        
        config = {
            "algorithm": "genetic",
            "parameter_space": {
                "x": {"type": "continuous", "min": -3.0, "max": 3.0},
                "y": {"type": "continuous", "min": -3.0, "max": 3.0}
            },
            "constraints": {
                "constraint_functions": [constraint_function],
                "max_iterations": 100
            }
        }
        
        result = await optimization_engine.optimize(objective_function, config)
        
        assert result.success is True
        # Optimal solution should be near (√2, √2) ≈ (1.41, 1.41)
        x_opt = result.optimal_parameters["x"]
        y_opt = result.optimal_parameters["y"]
        assert x_opt**2 + y_opt**2 <= 4.1  # Satisfies constraint with tolerance


class TestRecommendationEngine:
    """Test cases for RecommendationEngine."""
    
    @pytest.fixture
    def recommendation_engine(self):
        """Create recommendation engine instance."""
        return RecommendationEngine()
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_engine_initialization(self, recommendation_engine):
        """Test recommendation engine initialization."""
        await recommendation_engine.start()
        assert recommendation_engine.is_running is True
        
        status = recommendation_engine.get_engine_status()
        assert status["is_running"] is True
        assert "recommendation_models" in status
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_performance_recommendations(self, recommendation_engine):
        """Test performance improvement recommendations."""
        await recommendation_engine.start()
        
        # Performance data indicating bottlenecks
        performance_data = {
            "execution_time": 300.0,  # 5 minutes - slow
            "cpu_usage": 95.0,        # High CPU usage
            "memory_usage": 7500.0,   # High memory usage
            "success_rate": 0.75,     # Low success rate
            "error_types": ["TimeoutError", "MemoryError"],
            "recipe_complexity": 8
        }
        
        recommendations = await recommendation_engine.generate_performance_recommendations(
            performance_data
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend optimizations for high resource usage
        rec_text = " ".join([r.description for r in recommendations]).lower()
        assert any(keyword in rec_text for keyword in 
                  ["cpu", "memory", "timeout", "optimize", "reduce"])
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_parameter_recommendations(self, recommendation_engine):
        """Test parameter optimization recommendations."""
        await recommendation_engine.start()
        
        # Current parameters with suboptimal performance
        current_params = {
            "batch_size": 10,         # Too small
            "timeout": 30,            # Too short
            "parallel_workers": 1,    # Not utilizing parallelism
            "retry_attempts": 1       # Too few retries
        }
        
        performance_history = [
            {"params": current_params, "performance": 65.0},
            {"params": {"batch_size": 50, "timeout": 60}, "performance": 85.0},
            {"params": {"parallel_workers": 4}, "performance": 78.0}
        ]
        
        recommendations = await recommendation_engine.generate_parameter_recommendations(
            current_params, performance_history
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend increasing batch_size, timeout, parallel_workers
        param_recommendations = [r for r in recommendations 
                               if r.recommendation_type.value == "parameter_tuning"]
        assert len(param_recommendations) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_architecture_recommendations(self, recommendation_engine):
        """Test architecture improvement recommendations."""
        await recommendation_engine.start()
        
        # System architecture data
        architecture_data = {
            "components": ["single_agent", "sequential_processing"],
            "bottlenecks": ["data_loading", "computation"],
            "scalability_issues": ["memory_constraints", "cpu_bound"],
            "current_performance": 60.0,
            "target_performance": 90.0
        }
        
        recommendations = await recommendation_engine.generate_architecture_recommendations(
            architecture_data
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend multi-agent, parallel processing, etc.
        arch_recommendations = [r for r in recommendations 
                              if r.recommendation_type.value == "architecture_change"]
        assert len(arch_recommendations) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.predictive
    async def test_recommendation_ranking(self, recommendation_engine):
        """Test recommendation ranking and prioritization."""
        await recommendation_engine.start()
        
        # Generate multiple recommendations
        performance_data = {
            "execution_time": 200.0,
            "cpu_usage": 80.0,
            "memory_usage": 6000.0,
            "success_rate": 0.85
        }
        
        recommendations = await recommendation_engine.generate_performance_recommendations(
            performance_data
        )
        
        # Rank recommendations
        ranked_recommendations = await recommendation_engine.rank_recommendations(
            recommendations, criteria=["impact", "feasibility", "cost"]
        )
        
        assert len(ranked_recommendations) == len(recommendations)
        
        # Verify ranking (higher ranked items should have higher scores)
        for i in range(len(ranked_recommendations) - 1):
            current_score = ranked_recommendations[i].confidence
            next_score = ranked_recommendations[i + 1].confidence
            assert current_score >= next_score


@pytest.mark.predictive
@pytest.mark.integration
class TestPredictiveIntegration:
    """Integration tests for predictive optimization components."""
    
    @pytest.mark.asyncio
    async def test_full_predictive_optimization_pipeline(self):
        """Test complete predictive optimization pipeline."""
        # Initialize engines
        predictive_engine = PredictiveEngine()
        optimization_engine = OptimizationEngine()
        recommendation_engine = RecommendationEngine()
        
        await predictive_engine.start()
        await optimization_engine.start()
        await recommendation_engine.start()
        
        # 1. Train predictive model
        model_config = {
            "model_name": "recipe_performance_predictor",
            "model_type": "regression",
            "prediction_type": "performance"
        }
        await predictive_engine.register_model(model_config)
        
        training_data = create_mock_training_data(num_samples=100)
        training_result = await predictive_engine.train_model(
            "recipe_performance_predictor", training_data
        )
        assert training_result["success"] is True
        
        # 2. Use model in optimization objective function
        async def performance_objective(params):
            prediction = await predictive_engine.predict(
                "recipe_performance_predictor", params
            )
            return prediction.predicted_value
        
        # 3. Optimize parameters
        optimization_config = {
            "algorithm": "genetic",
            "parameter_space": {
                "recipe_complexity": {"type": "continuous", "min": 1.0, "max": 10.0},
                "resource_allocation": {"type": "continuous", "min": 0.5, "max": 5.0},
                "parallel_tasks": {"type": "integer", "min": 1, "max": 10}
            },
            "constraints": {"max_iterations": 50}
        }
        
        optimization_result = await optimization_engine.optimize(
            performance_objective, optimization_config
        )
        assert optimization_result.success is True
        
        # 4. Generate recommendations based on optimization
        performance_data = {
            "current_parameters": optimization_result.optimal_parameters,
            "predicted_performance": optimization_result.optimal_value,
            "optimization_history": optimization_result.metadata.get("convergence_history", [])
        }
        
        recommendations = await recommendation_engine.generate_performance_recommendations(
            performance_data
        )
        assert len(recommendations) > 0
        
        # Cleanup
        await predictive_engine.stop()
        await optimization_engine.stop()
        await recommendation_engine.stop()
