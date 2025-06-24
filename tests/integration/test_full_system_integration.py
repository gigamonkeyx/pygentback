"""
Full System Integration Tests

Tests for complete PyGent Factory system integration across all AI components.
"""

import pytest
import asyncio
from datetime import datetime

from src.integration.core import IntegrationEngine, OrchestrationEngine, WorkflowOrchestrator
from src.integration.adapters import (
    GeneticAlgorithmAdapter, NeuralSearchAdapter, MultiAgentAdapter,
    NLPAdapter, PredictiveAdapter
)
from tests.utils.helpers import create_test_recipe, create_test_workflow, assert_workflow_success


@pytest.mark.integration
@pytest.mark.e2e
class TestFullSystemIntegration:
    """End-to-end integration tests for the complete PyGent Factory system."""
    
    @pytest.fixture
    async def full_system(self):
        """Set up complete PyGent Factory system."""
        # Initialize core engines
        integration_engine = IntegrationEngine()
        orchestration_engine = OrchestrationEngine(integration_engine)
        workflow_orchestrator = WorkflowOrchestrator(integration_engine, orchestration_engine)
        
        # Register all AI component adapters BEFORE starting
        from src.integration.core import ComponentInfo, ComponentType
        
        adapters_config = [
            (ComponentType.GENETIC_ALGORITHM, GeneticAlgorithmAdapter(), "genetic_algorithm"),
            (ComponentType.NEURAL_SEARCH, NeuralSearchAdapter(), "neural_search"),
            (ComponentType.MULTI_AGENT, MultiAgentAdapter(), "multi_agent"),
            (ComponentType.NLP_SYSTEM, NLPAdapter(), "nlp_system"),
            (ComponentType.PREDICTIVE_OPTIMIZATION, PredictiveAdapter(), "predictive_optimization")
        ]
        
        for component_type, adapter, component_id in adapters_config:
            component_info = ComponentInfo(
                component_id=component_id,
                component_type=component_type,
                name=f"{component_type.value.title()} System",
                version="1.0",
                capabilities=["integration_test", "full_system"]
            )
            
            integration_engine.register_component(component_info, adapter)
        
        # NOW start the engine to initialize registered components
        await integration_engine.start()
        
        yield {
            "integration_engine": integration_engine,
            "orchestration_engine": orchestration_engine,
            "workflow_orchestrator": workflow_orchestrator
        }
        
        # Cleanup
        await integration_engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_complete_recipe_lifecycle(self, full_system):
        """Test complete recipe lifecycle from creation to optimization."""
        orchestrator = full_system["workflow_orchestrator"]
        
        # Recipe text for processing
        recipe_text = """
        Recipe: Advanced Data Pipeline
        Description: Complete data processing pipeline with ML optimization
        
        Steps:
        1. Extract data from multiple sources (APIs, databases, files)
        2. Clean and validate data using pandas and custom rules
        3. Apply feature engineering and transformations
        4. Train machine learning models for prediction
        5. Optimize model hyperparameters using genetic algorithms
        6. Deploy model with monitoring and alerting
        7. Generate comprehensive reports and documentation
        
        Requirements:
        - Python 3.9+
        - pandas, scikit-learn, tensorflow
        - PostgreSQL database
        - Redis for caching
        - Docker for deployment
        
        Expected Performance:
        - Processing time: < 2 hours
        - Accuracy: > 95%
        - Throughput: 1000 records/minute
        """
        
        # Execute recipe optimization workflow
        optimization_config = {
            "objective_function": "maximize_performance",
            "parameter_space": {
                "batch_size": {"type": "integer", "min": 32, "max": 512},
                "learning_rate": {"type": "continuous", "min": 0.001, "max": 0.1},
                "num_epochs": {"type": "integer", "min": 10, "max": 100}
            }
        }
        
        result = await orchestrator.optimize_recipe(recipe_text, optimization_config)
        
        # Verify successful execution
        assert_workflow_success(result.to_dict())
        
        # Verify all steps completed
        expected_steps = ["parse_recipe", "predict_performance", "optimize_parameters", "validate_optimized_recipe"]
        for step in expected_steps:
            assert step in result.results
            assert result.results[step]["success"] is True
    
    @pytest.mark.asyncio
    async def test_multi_component_coordination(self, full_system):
        """Test coordination between multiple AI components."""
        integration_engine = full_system["integration_engine"]
        
        # Create complex workflow involving all components
        complex_workflow = {
            "name": "multi_component_coordination",
            "description": "Workflow testing coordination between all AI components",
            "steps": [
                {
                    "step_id": "nlp_analysis",
                    "component_type": "nlp_system",
                    "action": "parse_recipe",
                    "parameters": {"recipe_text": "Complex AI recipe for testing"},
                    "required": True
                },
                {
                    "step_id": "performance_prediction",
                    "component_type": "predictive_optimization",
                    "action": "predict_performance",
                    "parameters": {"recipe_data": "nlp_analysis_result"},
                    "dependencies": ["nlp_analysis"],
                    "required": True
                },
                {
                    "step_id": "genetic_optimization",
                    "component_type": "genetic_algorithm",
                    "action": "optimize",
                    "parameters": {"objective": "performance", "constraints": {}},
                    "dependencies": ["performance_prediction"],
                    "required": True
                },
                {
                    "step_id": "neural_architecture_search",
                    "component_type": "neural_search",
                    "action": "search_architecture",
                    "parameters": {"search_space": {}, "constraints": {}},
                    "dependencies": ["genetic_optimization"],
                    "required": False
                },
                {
                    "step_id": "multi_agent_validation",
                    "component_type": "multi_agent",
                    "action": "execute_tests",
                    "parameters": {"test_suite": "comprehensive"},
                    "dependencies": ["neural_architecture_search"],
                    "required": True
                }
            ]
        }
        
        from src.integration.models import WorkflowDefinition
        workflow_def = WorkflowDefinition(
            name=complex_workflow["name"],
            description=complex_workflow["description"],
            steps=complex_workflow["steps"]
        )
        
        # Execute workflow
        result = await integration_engine.execute_workflow(workflow_def)
        
        # Verify coordination success
        assert result.success is True
        assert len(result.results) == 5
        
        # Verify dependency execution order
        step_order = list(result.results.keys())
        assert step_order.index("nlp_analysis") < step_order.index("performance_prediction")
        assert step_order.index("performance_prediction") < step_order.index("genetic_optimization")
        assert step_order.index("genetic_optimization") < step_order.index("multi_agent_validation")
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring(self, full_system):
        """Test system-wide health monitoring."""
        integration_engine = full_system["integration_engine"]
        
        # Get initial system status
        initial_status = integration_engine.get_system_status()
        
        assert initial_status["integration_status"] == "ready"
        assert initial_status["total_components"] == 5
        assert initial_status["active_components"] >= 4  # Allow for some initialization delays
        
        # Trigger health checks
        await integration_engine._perform_health_checks()
        
        # Verify health monitoring
        updated_status = integration_engine.get_system_status()
        
        for component_id, component_status in updated_status["components"].items():
            assert "health_score" in component_status
            assert 0.0 <= component_status["health_score"] <= 1.0
            assert "last_health_check" in component_status
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, full_system):
        """Test system error recovery and resilience."""
        integration_engine = full_system["integration_engine"]
        
        # Create workflow with intentional failure
        failing_workflow = {
            "name": "resilience_test",
            "description": "Test system resilience with failing components",
            "steps": [
                {
                    "step_id": "successful_step",
                    "component_type": "nlp_system",
                    "action": "parse_recipe",
                    "parameters": {"recipe_text": "Simple recipe"},
                    "required": True
                },
                {
                    "step_id": "failing_step",
                    "component_type": "genetic_algorithm",
                    "action": "nonexistent_action",  # This will fail
                    "parameters": {},
                    "dependencies": ["successful_step"],
                    "required": False  # Non-required step
                },
                {
                    "step_id": "recovery_step",
                    "component_type": "multi_agent",
                    "action": "execute_tests",
                    "parameters": {"test_suite": "basic"},
                    "dependencies": ["successful_step"],  # Depends on successful step, not failing one
                    "required": True
                }
            ]
        }
        
        from src.integration.models import WorkflowDefinition
        workflow_def = WorkflowDefinition(
            name=failing_workflow["name"],
            description=failing_workflow["description"],
            steps=failing_workflow["steps"]
        )
        
        # Execute workflow
        result = await integration_engine.execute_workflow(workflow_def)
        
        # Verify partial success (successful and recovery steps should complete)
        assert "successful_step" in result.results
        assert result.results["successful_step"]["success"] is True
        
        assert "recovery_step" in result.results
        assert result.results["recovery_step"]["success"] is True
        
        # Failing step should be recorded but not prevent overall workflow
        assert "failing_step" in result.results
        assert result.results["failing_step"]["success"] is False
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, full_system):
        """Test system performance under concurrent load."""
        integration_engine = full_system["integration_engine"]
        
        # Create multiple concurrent workflows
        workflows = []
        for i in range(5):  # 5 concurrent workflows
            workflow = {
                "name": f"load_test_workflow_{i}",
                "description": f"Load test workflow {i}",
                "steps": [
                    {
                        "step_id": f"nlp_step_{i}",
                        "component_type": "nlp_system",
                        "action": "parse_recipe",
                        "parameters": {"recipe_text": f"Load test recipe {i}"},
                        "required": True
                    },
                    {
                        "step_id": f"prediction_step_{i}",
                        "component_type": "predictive_optimization",
                        "action": "predict_performance",
                        "parameters": {"recipe_data": f"recipe_{i}"},
                        "dependencies": [f"nlp_step_{i}"],
                        "required": True
                    }
                ]
            }
            workflows.append(workflow)
        
        # Execute workflows concurrently
        from src.integration.models import WorkflowDefinition
        
        tasks = []
        for workflow in workflows:
            workflow_def = WorkflowDefinition(
                name=workflow["name"],
                description=workflow["description"],
                steps=workflow["steps"]
            )
            task = asyncio.create_task(integration_engine.execute_workflow(workflow_def))
            tasks.append(task)
        
        # Wait for all workflows to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all workflows completed successfully
        successful_workflows = 0
        for result in results:
            if not isinstance(result, Exception) and result.success:
                successful_workflows += 1
        
        # At least 80% should succeed under load
        success_rate = successful_workflows / len(workflows)
        assert success_rate >= 0.8, f"Success rate {success_rate} below threshold"
    
    @pytest.mark.asyncio
    async def test_comprehensive_testing_workflow(self, full_system):
        """Test comprehensive testing workflow with all components."""
        orchestrator = full_system["workflow_orchestrator"]
        
        # Execute comprehensive testing workflow
        test_config = {
            "test_suite": [
                "unit_tests",
                "integration_tests", 
                "performance_tests",
                "security_tests"
            ],
            "environment": "testing",
            "parallel": True,
            "coverage_threshold": 0.8,
            "performance_threshold": 0.9
        }
        
        result = await orchestrator.run_comprehensive_testing(test_config)
        
        # Verify comprehensive testing success
        assert_workflow_success(result.to_dict())
        
        # Verify all testing steps completed
        expected_steps = [
            "setup_test_environment",
            "execute_tests", 
            "analyze_results",
            "generate_recommendations"
        ]
        
        for step in expected_steps:
            assert step in result.results
            if step != "generate_recommendations":  # This step is optional
                assert result.results[step]["success"] is True


@pytest.mark.integration
@pytest.mark.performance
class TestSystemPerformance:
    """Performance tests for the integrated system."""
    
    @pytest.mark.asyncio
    async def test_workflow_execution_performance(self, full_system):
        """Test workflow execution performance metrics."""
        integration_engine = full_system["integration_engine"]
        
        # Create performance test workflow
        workflow = {
            "name": "performance_test",
            "description": "Performance measurement workflow",
            "steps": [
                {
                    "step_id": "perf_step_1",
                    "component_type": "nlp_system",
                    "action": "parse_recipe",
                    "parameters": {"recipe_text": "Performance test recipe"},
                    "required": True
                }
            ]
        }
        
        from src.integration.models import WorkflowDefinition
        workflow_def = WorkflowDefinition(
            name=workflow["name"],
            description=workflow["description"],
            steps=workflow["steps"]
        )
        
        # Measure execution time
        start_time = asyncio.get_event_loop().time()
        result = await integration_engine.execute_workflow(workflow_def)
        end_time = asyncio.get_event_loop().time()
        
        execution_time = end_time - start_time
        
        # Verify performance
        assert result.success is True
        assert execution_time < 10.0  # Should complete within 10 seconds
        assert result.execution_time_seconds < 10.0
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, full_system):
        """Test memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        integration_engine = full_system["integration_engine"]
        
        # Execute multiple workflows to test memory usage
        for i in range(10):
            workflow = {
                "name": f"memory_test_{i}",
                "description": f"Memory test workflow {i}",
                "steps": [
                    {
                        "step_id": f"memory_step_{i}",
                        "component_type": "predictive_optimization",
                        "action": "predict_performance",
                        "parameters": {"recipe_data": {"complexity": i}},
                        "required": True
                    }
                ]
            }
            
            from src.integration.models import WorkflowDefinition
            workflow_def = WorkflowDefinition(
                name=workflow["name"],
                description=workflow["description"],
                steps=workflow["steps"]
            )
            
            await integration_engine.execute_workflow(workflow_def)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 10 workflows)
        assert memory_increase < 100, f"Memory increase {memory_increase}MB too high"
