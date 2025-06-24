"""
Tests for Integration core components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.integration.core import IntegrationEngine, OrchestrationEngine, WorkflowOrchestrator
from src.integration.models import IntegrationConfig, WorkflowDefinition, ExecutionContext, IntegrationResult
from tests.utils.helpers import create_test_workflow, assert_workflow_success


class TestIntegrationEngine:
    """Test cases for IntegrationEngine."""
    
    @pytest.fixture
    def integration_engine(self):
        """Create integration engine instance."""
        return IntegrationEngine()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_engine_initialization(self, integration_engine):
        """Test integration engine initialization."""
        assert integration_engine.status.value == "initializing"
        
        await integration_engine.start()
        assert integration_engine.status.value == "ready"
        
        status = integration_engine.get_system_status()
        assert status["integration_status"] == "ready"
        assert "total_components" in status
        assert "active_components" in status
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_component_registration(self, integration_engine):
        """Test component registration."""
        await integration_engine.start()
        
        # Create mock component info and adapter
        from src.integration.core import ComponentInfo, ComponentType
        
        component_info = ComponentInfo(
            component_id="test_component_001",
            component_type=ComponentType.NLP_SYSTEM,
            name="Test NLP Component",
            version="1.0",
            capabilities=["text_processing", "recipe_parsing"]
        )
        
        mock_adapter = Mock()
        mock_adapter.initialize = AsyncMock(return_value=True)
        mock_adapter.health_check = AsyncMock(return_value={
            "status": "healthy", "score": 1.0
        })
        
        # Register component
        integration_engine.register_component(component_info, mock_adapter)
        
        # Verify registration
        status = integration_engine.get_system_status()
        assert status["total_components"] == 1
        assert "test_component_001" in status["components"]
        assert status["components"]["test_component_001"]["name"] == "Test NLP Component"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_execution(self, integration_engine, sample_workflow_definition):
        """Test workflow execution."""
        # Register mock components BEFORE starting engine
        from src.integration.core import ComponentInfo, ComponentType
        
        # NLP component
        nlp_info = ComponentInfo(
            component_id="nlp_001",
            component_type=ComponentType.NLP_SYSTEM,
            name="NLP System",
            version="1.0"
        )
        nlp_adapter = Mock()
        nlp_adapter.initialize = AsyncMock()
        nlp_adapter.health_check = AsyncMock(return_value={
            "status": "healthy", "score": 1.0
        })
        nlp_adapter.parse_recipe = AsyncMock(return_value={
            "success": True,
            "result": {"parsed_steps": [], "complexity": 5}
        })
        
        # Multi-agent component
        agent_info = ComponentInfo(
            component_id="multiagent_001",
            component_type=ComponentType.MULTI_AGENT,
            name="Multi-Agent System",
            version="1.0"
        )
        agent_adapter = Mock()
        agent_adapter.initialize = AsyncMock()
        agent_adapter.health_check = AsyncMock(return_value={
            "status": "healthy", "score": 1.0
        })
        agent_adapter.execute_tests = AsyncMock(return_value={
            "success": True,
            "result": {"tests_passed": 8, "tests_failed": 2}
        })
        
        # Predictive optimization component
        pred_info = ComponentInfo(
            component_id="predictive_001",
            component_type=ComponentType.PREDICTIVE_OPTIMIZATION,
            name="Predictive Optimization",
            version="1.0"
        )
        pred_adapter = Mock()
        pred_adapter.initialize = AsyncMock()
        pred_adapter.health_check = AsyncMock(return_value={
            "status": "healthy", "score": 1.0
        })
        pred_adapter.predict_performance = AsyncMock(return_value={
            "success": True,
            "result": {"performance_score": 0.85, "optimization_suggestions": []}
        })
        
        integration_engine.register_component(nlp_info, nlp_adapter)
        integration_engine.register_component(agent_info, agent_adapter)
        integration_engine.register_component(pred_info, pred_adapter)
        
        # NOW start the engine to initialize the registered components
        await integration_engine.start()
        
        # Create workflow definition
        workflow_def = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow execution",
            steps=sample_workflow_definition["steps"]
        )
        
        # Execute workflow
        result = await integration_engine.execute_workflow(workflow_def)
        
        assert_workflow_success(result.to_dict())
        assert result.workflow_name == "test_workflow"
        assert len(result.results) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_health_monitoring(self, integration_engine):
        """Test component health monitoring."""
        await integration_engine.start()
        
        # Register component with health check
        from src.integration.core import ComponentInfo, ComponentType
        
        component_info = ComponentInfo(
            component_id="monitored_component",
            component_type=ComponentType.PREDICTIVE_OPTIMIZATION,
            name="Monitored Component",
            version="1.0"
        )
        
        mock_adapter = Mock()
        mock_adapter.initialize = AsyncMock()
        mock_adapter.health_check = AsyncMock(return_value={
            "status": "healthy",
            "score": 0.95,
            "details": {"cpu_usage": 45.2, "memory_usage": 512.0}
        })
        
        integration_engine.register_component(component_info, mock_adapter)
        
        # Trigger health check
        await integration_engine._perform_health_checks()
        
        # Verify health status
        status = integration_engine.get_system_status()
        component_status = status["components"]["monitored_component"]
        assert component_status["health_score"] == 0.95
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_handling(self, integration_engine):
        """Test error handling in workflow execution."""
        await integration_engine.start()
        
        # Register component that will fail
        from src.integration.core import ComponentInfo, ComponentType
        
        failing_info = ComponentInfo(
            component_id="failing_component",
            component_type=ComponentType.GENETIC_ALGORITHM,
            name="Failing Component",
            version="1.0"
        )
        
        failing_adapter = Mock()
        failing_adapter.initialize = AsyncMock()
        failing_adapter.optimize = AsyncMock(side_effect=Exception("Optimization failed"))
        
        integration_engine.register_component(failing_info, failing_adapter)
        
        # Create workflow with failing step
        workflow_def = WorkflowDefinition(
            name="failing_workflow",
            description="Workflow with failing step",
            steps=[
                {
                    "step_id": "failing_step",
                    "component_type": "genetic_algorithm",
                    "action": "optimize",
                    "parameters": {},
                    "required": True
                }
            ]
        )
        
        # Execute workflow
        result = await integration_engine.execute_workflow(workflow_def)
        
        assert result.success is False
        assert result.error_message is not None
        assert "Optimization failed" in result.error_message
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_engine_shutdown(self, integration_engine):
        """Test integration engine shutdown."""
        await integration_engine.start()
        
        # Register components
        from src.integration.core import ComponentInfo, ComponentType
        
        for i in range(3):
            info = ComponentInfo(
                component_id=f"component_{i}",
                component_type=ComponentType.NLP_SYSTEM,
                name=f"Component {i}",
                version="1.0"
            )
            adapter = Mock()
            adapter.initialize = AsyncMock()
            adapter.shutdown = AsyncMock()
            
            integration_engine.register_component(info, adapter)
        
        # Shutdown engine
        await integration_engine.shutdown()
        assert integration_engine.status.value == "shutdown"


class TestOrchestrationEngine:
    """Test cases for OrchestrationEngine."""
    
    @pytest.fixture
    def orchestration_engine(self, integration_engine):
        """Create orchestration engine instance."""
        return OrchestrationEngine(integration_engine)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workflow_template_registration(self, orchestration_engine):
        """Test workflow template registration."""
        # Create template
        template = WorkflowDefinition(
            name="test_template",
            description="Test workflow template",
            steps=[
                {
                    "step_id": "template_step",
                    "component_type": "nlp_system",
                    "action": "parse_recipe",
                    "parameters": {"recipe_text": "${recipe_input}"}
                }
            ]
        )
        
        # Register template
        orchestration_engine.register_workflow_template(template)
        
        # Verify registration
        status = orchestration_engine.get_orchestration_status()
        assert "test_template" in status["template_names"]
        assert status["registered_templates"] == 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_template_execution(self, orchestration_engine):
        """Test workflow template execution."""
        # Setup integration engine
        await orchestration_engine.integration_engine.start()
        
        # Register mock NLP component
        from src.integration.core import ComponentInfo, ComponentType
        
        nlp_info = ComponentInfo(
            component_id="nlp_template_test",
            component_type=ComponentType.NLP_SYSTEM,
            name="NLP for Template",
            version="1.0"
        )
        nlp_adapter = Mock()
        nlp_adapter.initialize = AsyncMock()
        nlp_adapter.parse_recipe = AsyncMock(return_value={
            "success": True,
            "result": {"parsed_recipe": "test result"}
        })
        
        orchestration_engine.integration_engine.register_component(nlp_info, nlp_adapter)
        
        # Register template
        template = WorkflowDefinition(
            name="parsing_template",
            description="Recipe parsing template",
            steps=[
                {
                    "step_id": "parse_step",
                    "component_type": "nlp_system",
                    "action": "parse_recipe",
                    "parameters": {"recipe_text": "${recipe_input}"}
                }
            ]
        )
        orchestration_engine.register_workflow_template(template)
        
        # Execute template with parameters
        parameters = {"recipe_input": "Test recipe text"}
        result = await orchestration_engine.execute_template("parsing_template", parameters)
        
        assert result.success is True
        assert result.workflow_name.startswith("parsing_template_instance")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_parameter_substitution(self, orchestration_engine):
        """Test parameter substitution in templates."""
        # Create template with parameters
        template = WorkflowDefinition(
            name="parameterized_template",
            description="Template with parameters",
            steps=[
                {
                    "step_id": "param_step",
                    "component_type": "predictive_optimization",
                    "action": "predict",
                    "parameters": {
                        "model_name": "${model}",
                        "input_data": "${data}",
                        "threshold": "${threshold}"
                    }
                }
            ]
        )
        
        orchestration_engine.register_workflow_template(template)
        
        # Test parameter substitution
        parameters = {
            "model": "performance_predictor",
            "data": {"complexity": 5},
            "threshold": 0.8
        }
        
        # Get instantiated workflow (without execution)
        instantiated = orchestration_engine._instantiate_workflow(template, parameters)
        
        step_params = instantiated.steps[0]["parameters"]
        assert step_params["model_name"] == "performance_predictor"
        assert step_params["input_data"] == {"complexity": 5}
        assert step_params["threshold"] == 0.8


class TestWorkflowOrchestrator:
    """Test cases for WorkflowOrchestrator."""
    
    @pytest.fixture
    def workflow_orchestrator(self):
        """Create workflow orchestrator instance."""
        integration_engine = IntegrationEngine()
        orchestration_engine = OrchestrationEngine(integration_engine)
        return WorkflowOrchestrator(integration_engine, orchestration_engine)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_recipe_optimization_workflow(self, workflow_orchestrator):
        """Test recipe optimization workflow."""
        # Setup engines
        await workflow_orchestrator.integration_engine.start()
        
        # Register mock components
        from src.integration.core import ComponentInfo, ComponentType
        
        # NLP component
        nlp_info = ComponentInfo(
            component_id="nlp_orchestrator",
            component_type=ComponentType.NLP_SYSTEM,
            name="NLP System",
            version="1.0"
        )
        nlp_adapter = Mock()
        nlp_adapter.initialize = AsyncMock()
        nlp_adapter.parse_recipe = AsyncMock(return_value={
            "success": True,
            "result": {"complexity": 5, "steps": []}
        })
        
        # Predictive component
        pred_info = ComponentInfo(
            component_id="predictive_orchestrator",
            component_type=ComponentType.PREDICTIVE_OPTIMIZATION,
            name="Predictive System",
            version="1.0"
        )
        pred_adapter = Mock()
        pred_adapter.initialize = AsyncMock()
        pred_adapter.predict_performance = AsyncMock(return_value={
            "success": True,
            "result": {"predicted_performance": 85.5}
        })
        
        # Genetic algorithm component
        ga_info = ComponentInfo(
            component_id="ga_orchestrator",
            component_type=ComponentType.GENETIC_ALGORITHM,
            name="Genetic Algorithm",
            version="1.0"
        )
        ga_adapter = Mock()
        ga_adapter.initialize = AsyncMock()
        ga_adapter.optimize = AsyncMock(return_value={
            "success": True,
            "result": {"optimal_parameters": {"param1": 5.0}}
        })
        
        # Multi-agent component
        agent_info = ComponentInfo(
            component_id="agent_orchestrator",
            component_type=ComponentType.MULTI_AGENT,
            name="Multi-Agent System",
            version="1.0"
        )
        agent_adapter = Mock()
        agent_adapter.initialize = AsyncMock()
        agent_adapter.validate_recipe = AsyncMock(return_value={
            "success": True,
            "result": {"validation_passed": True}
        })
        
        # Register all components
        for info, adapter in [(nlp_info, nlp_adapter), (pred_info, pred_adapter),
                             (ga_info, ga_adapter), (agent_info, agent_adapter)]:
            workflow_orchestrator.integration_engine.register_component(info, adapter)
        
        # Execute recipe optimization workflow
        recipe_text = "Test recipe for optimization"
        optimization_config = {
            "objective_function": "maximize_performance",
            "parameter_space": {"param1": {"type": "continuous", "min": 0, "max": 10}}
        }
        
        result = await workflow_orchestrator.optimize_recipe(recipe_text, optimization_config)
        
        assert result.success is True
        assert "parse_recipe" in result.results
        assert "optimize_parameters" in result.results
        assert "validate_optimized_recipe" in result.results
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_comprehensive_testing_workflow(self, workflow_orchestrator):
        """Test comprehensive testing workflow."""
        # Setup engines
        await workflow_orchestrator.integration_engine.start()
        
        # Register mock components
        from src.integration.core import ComponentInfo, ComponentType
        
        # Multi-agent component
        agent_info = ComponentInfo(
            component_id="agent_testing",
            component_type=ComponentType.MULTI_AGENT,
            name="Multi-Agent Testing",
            version="1.0"
        )
        agent_adapter = Mock()
        agent_adapter.initialize = AsyncMock()
        agent_adapter.setup_environment = AsyncMock(return_value={
            "success": True,
            "result": {"environment_ready": True}
        })
        agent_adapter.execute_tests = AsyncMock(return_value={
            "success": True,
            "result": {"tests_run": 10, "tests_passed": 8}
        })
        
        # NLP component
        nlp_info = ComponentInfo(
            component_id="nlp_testing",
            component_type=ComponentType.NLP_SYSTEM,
            name="NLP Testing",
            version="1.0"
        )
        nlp_adapter = Mock()
        nlp_adapter.initialize = AsyncMock()
        nlp_adapter.interpret_test_results = AsyncMock(return_value={
            "success": True,
            "result": {"summary": "Tests completed", "insights": []}
        })
        
        # Predictive component
        pred_info = ComponentInfo(
            component_id="predictive_testing",
            component_type=ComponentType.PREDICTIVE_OPTIMIZATION,
            name="Predictive Testing",
            version="1.0"
        )
        pred_adapter = Mock()
        pred_adapter.initialize = AsyncMock()
        pred_adapter.generate_recommendations = AsyncMock(return_value={
            "success": True,
            "result": {"recommendations": ["Optimize performance"]}
        })
        
        # Register components
        for info, adapter in [(agent_info, agent_adapter), (nlp_info, nlp_adapter), 
                             (pred_info, pred_adapter)]:
            workflow_orchestrator.integration_engine.register_component(info, adapter)
        
        # Execute comprehensive testing workflow
        test_config = {
            "test_suite": ["unit_tests", "integration_tests", "performance_tests"],
            "environment": "testing",
            "parallel": True
        }
        
        result = await workflow_orchestrator.run_comprehensive_testing(test_config)
        
        assert result.success is True
        assert "setup_test_environment" in result.results
        assert "execute_tests" in result.results
        assert "analyze_results" in result.results
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_available_workflows(self, workflow_orchestrator):
        """Test getting available workflows."""
        workflows = workflow_orchestrator.get_available_workflows()
        
        assert isinstance(workflows, list)
        assert "recipe_optimization" in workflows
        assert "comprehensive_testing" in workflows


@pytest.mark.integration
@pytest.mark.workflow_orchestration
class TestIntegrationWorkflows:
    """Integration tests for complete workflow orchestration."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_integration_workflow(self):
        """Test complete end-to-end integration workflow."""
        # Initialize all components
        integration_engine = IntegrationEngine()
        orchestration_engine = OrchestrationEngine(integration_engine)
        workflow_orchestrator = WorkflowOrchestrator(integration_engine, orchestration_engine)
        
        await integration_engine.start()
        
        # Register all AI component adapters
        from src.integration.adapters import (
            GeneticAlgorithmAdapter, NeuralSearchAdapter, MultiAgentAdapter,
            NLPAdapter, PredictiveAdapter
        )
        from src.integration.core import ComponentInfo, ComponentType
        
        adapters = [
            (ComponentType.GENETIC_ALGORITHM, GeneticAlgorithmAdapter()),
            (ComponentType.MULTI_AGENT, MultiAgentAdapter()),
            (ComponentType.NLP_SYSTEM, NLPAdapter()),
            (ComponentType.PREDICTIVE_OPTIMIZATION, PredictiveAdapter())
        ]
        
        for component_type, adapter in adapters:
            component_info = ComponentInfo(
                component_id=f"{component_type.value}_integration",
                component_type=component_type,
                name=f"{component_type.value.title()} Component",
                version="1.0",
                capabilities=["integration_test"]
            )
            
            integration_engine.register_component(component_info, adapter)
        
        # Create complex multi-component workflow
        complex_workflow = WorkflowDefinition(
            name="complex_integration_workflow",
            description="Complex workflow testing all components",
            steps=[
                {
                    "step_id": "parse_requirements",
                    "component_type": "nlp_system",
                    "action": "parse_recipe",
                    "parameters": {"recipe_text": "Complex AI recipe"},
                    "required": True
                },
                {
                    "step_id": "predict_performance",
                    "component_type": "predictive_optimization",
                    "action": "predict_performance",
                    "parameters": {"recipe_data": "parsed_requirements"},
                    "dependencies": ["parse_requirements"],
                    "required": False
                },
                {
                    "step_id": "optimize_parameters",
                    "component_type": "genetic_algorithm",
                    "action": "optimize",
                    "parameters": {"objective": "performance"},
                    "dependencies": ["predict_performance"],
                    "required": True
                },
                {
                    "step_id": "validate_solution",
                    "component_type": "multi_agent",
                    "action": "execute_tests",
                    "parameters": {"test_suite": "validation"},
                    "dependencies": ["optimize_parameters"],
                    "required": True
                }
            ]
        )
        
        # Execute complex workflow
        result = await integration_engine.execute_workflow(complex_workflow)
        
        # Verify successful execution
        assert result.success is True
        assert len(result.results) == 4
        assert all(step_id in result.results for step_id in 
                  ["parse_requirements", "predict_performance", "optimize_parameters", "validate_solution"])
        
        # Verify system health
        system_status = integration_engine.get_system_status()
        assert system_status["integration_status"] == "ready"
        assert system_status["active_components"] >= 4
        
        # Cleanup
        await integration_engine.shutdown()
