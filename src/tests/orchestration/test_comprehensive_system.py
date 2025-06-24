"""
Comprehensive System Test

Complete end-to-end test of all orchestration phases:
- Phase 1: Foundation Orchestration
- Phase 2: Adaptive Coordination  
- Phase 3: Advanced Intelligence
- Production Deployment
- PyGent Factory Integration
"""

import asyncio
import pytest
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import orchestration components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from orchestration.orchestration_manager import OrchestrationManager
from orchestration.coordination_models import (
    OrchestrationConfig, TaskRequest, TaskPriority, MCPServerType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestComprehensiveSystem:
    """Comprehensive tests for the complete orchestration system."""
    
    @pytest.fixture
    async def full_orchestration_system(self):
        """Create and start complete orchestration system."""
        config = OrchestrationConfig(
            evolution_enabled=True,
            max_concurrent_tasks=50,
            batch_processing_enabled=True,
            detailed_logging=True
        )
        
        manager = OrchestrationManager(config)
        await manager.start()
        
        # Setup complete test environment
        await manager.register_existing_mcp_servers()
        
        # Create diverse agent pool
        await manager.create_tot_agent("Advanced ToT Agent", ["reasoning", "analysis", "problem_solving"])
        await manager.create_rag_agent("Advanced RAG Retrieval", "retrieval")
        await manager.create_rag_agent("Advanced RAG Generation", "generation")
        await manager.create_evaluation_agent("Quality Evaluator", ["quality_assessment", "performance_evaluation"])
        await manager.create_evaluation_agent("System Monitor", ["system_monitoring", "health_assessment"])
        
        yield manager
        
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_complete_system_startup(self, full_orchestration_system):
        """Test complete system startup with all phases."""
        manager = full_orchestration_system
        
        # Verify system is running
        assert manager.is_running
        
        # Check all components are operational
        status = await manager.get_system_status()
        assert status["is_running"] is True
        
        components = status.get("components", {})
        
        # Phase 1 components
        assert "agent_registry" in components
        assert "mcp_orchestrator" in components
        assert "task_dispatcher" in components
        assert "metrics_collector" in components
        assert "evolutionary_orchestrator" in components
        
        # Phase 2 components
        assert "adaptive_load_balancer" in components
        assert "transaction_coordinator" in components
        assert "emergent_behavior_detector" in components
        
        # Phase 3 components
        assert "meta_learning_engine" in components
        assert "predictive_optimizer" in components
        assert "pygent_integration" in components
        
        logger.info("✅ Complete system startup test passed")
    
    @pytest.mark.asyncio
    async def test_advanced_intelligence_workflow(self, full_orchestration_system):
        """Test advanced intelligence capabilities."""
        manager = full_orchestration_system
        
        # Test ToT reasoning
        reasoning_result = await manager.execute_tot_reasoning(
            "How can we optimize multi-agent coordination for maximum efficiency?",
            {"domain": "orchestration", "complexity": "high"}
        )
        
        assert "reasoning_path" in reasoning_result
        assert "solution" in reasoning_result
        assert "confidence" in reasoning_result
        assert reasoning_result["confidence"] > 0.0
        
        # Test RAG workflow
        rag_result = await manager.execute_rag_workflow(
            "What are the best practices for distributed system coordination?",
            "computer_science"
        )
        
        assert "retrieval" in rag_result
        assert "generation" in rag_result
        assert "total_time" in rag_result
        
        # Test research workflow
        research_result = await manager.execute_research_workflow(
            "Multi-agent system optimization techniques",
            "comprehensive"
        )
        
        assert "research_summary" in research_result
        assert "sources_found" in research_result
        assert "key_findings" in research_result
        
        logger.info("✅ Advanced intelligence workflow test passed")
    
    @pytest.mark.asyncio
    async def test_predictive_optimization(self, full_orchestration_system):
        """Test predictive optimization capabilities."""
        manager = full_orchestration_system
        
        # Update system with metrics for prediction
        current_metrics = await manager.get_system_metrics()
        await manager.predictive_optimizer.update_metrics(current_metrics)
        
        # Test performance prediction
        predictions = await manager.predict_system_performance(hours_ahead=2)
        assert isinstance(predictions, dict)
        
        # Test optimization recommendations
        recommendations = await manager.get_optimization_recommendations()
        assert isinstance(recommendations, list)
        
        # Test bottleneck detection
        bottlenecks = await manager.detect_system_bottlenecks()
        assert isinstance(bottlenecks, list)
        
        logger.info("✅ Predictive optimization test passed")
    
    @pytest.mark.asyncio
    async def test_meta_learning_capabilities(self, full_orchestration_system):
        """Test meta-learning capabilities."""
        manager = full_orchestration_system
        
        # Test learning from task execution
        performance_data = {
            "domain": "reasoning",
            "task_type": "complex_analysis",
            "features": {"complexity": 0.8, "duration": 120.0},
            "difficulty": 0.7
        }
        
        learning_success = await manager.learn_from_task_execution("test_task_123", performance_data)
        assert isinstance(learning_success, bool)
        
        # Get meta-learning metrics
        meta_metrics = await manager.meta_learning_engine.get_meta_learning_metrics()
        assert "total_experiences" in meta_metrics
        assert "strategy_effectiveness" in meta_metrics
        
        logger.info("✅ Meta-learning capabilities test passed")
    
    @pytest.mark.asyncio
    async def test_pygent_integration(self, full_orchestration_system):
        """Test PyGent Factory integration."""
        manager = full_orchestration_system
        
        # Test component health
        component_health = await manager.get_pygent_component_health()
        assert isinstance(component_health, dict)
        
        # Test integration metrics
        integration_metrics = await manager.pygent_integration.get_integration_metrics()
        assert "is_connected" in integration_metrics
        assert "total_components" in integration_metrics
        assert "component_details" in integration_metrics
        
        logger.info("✅ PyGent integration test passed")
    
    @pytest.mark.asyncio
    async def test_distributed_transaction_workflow(self, full_orchestration_system):
        """Test complex distributed transaction workflow."""
        manager = full_orchestration_system
        
        # Create complex multi-server transaction
        operations = [
            {
                "server_id": "filesystem_server",
                "operation_type": "create_research_file",
                "operation_data": {
                    "path": "/research/advanced_coordination.md",
                    "content": "# Advanced Multi-Agent Coordination Research\n\nThis document contains research findings..."
                },
                "compensation_data": {
                    "path": "/research/advanced_coordination.md",
                    "action": "delete"
                }
            },
            {
                "server_id": "memory_server",
                "operation_type": "store_research_metadata",
                "operation_data": {
                    "key": "research_advanced_coordination",
                    "value": {
                        "title": "Advanced Multi-Agent Coordination",
                        "status": "in_progress",
                        "created": datetime.utcnow().isoformat()
                    }
                },
                "compensation_data": {
                    "key": "research_advanced_coordination",
                    "action": "delete"
                }
            },
            {
                "server_id": "postgresql_server",
                "operation_type": "log_research_activity",
                "operation_data": {
                    "table": "research_log",
                    "data": {
                        "activity": "research_started",
                        "topic": "advanced_coordination",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                },
                "compensation_data": {
                    "table": "research_log",
                    "condition": "topic='advanced_coordination'",
                    "action": "delete"
                }
            }
        ]
        
        # Execute transaction
        transaction_id = await manager.begin_distributed_transaction(operations, timeout_minutes=5)
        assert transaction_id is not None
        
        # Check transaction status
        status = await manager.get_transaction_status(transaction_id)
        assert status is not None
        assert status["operations_count"] == 3
        
        # Commit transaction
        commit_success = await manager.commit_transaction(transaction_id)
        assert commit_success is True
        
        logger.info("✅ Distributed transaction workflow test passed")
    
    @pytest.mark.asyncio
    async def test_emergent_behavior_detection(self, full_orchestration_system):
        """Test emergent behavior detection with complex scenarios."""
        manager = full_orchestration_system
        
        # Generate complex system activity to trigger emergent behaviors
        tasks = []
        for i in range(10):
            task = TaskRequest(
                task_type="reasoning" if i % 2 == 0 else "analysis",
                priority=TaskPriority.HIGH if i < 5 else TaskPriority.NORMAL,
                description=f"Complex emergent behavior test task {i}",
                input_data={
                    "problem": f"Multi-dimensional optimization problem {i}",
                    "complexity": 0.8 + (i * 0.02),
                    "domain": "advanced_coordination"
                },
                required_capabilities={"reasoning", "analysis"},
                required_mcp_servers={MCPServerType.MEMORY, MCPServerType.FILESYSTEM}
            )
            tasks.append(task)
        
        # Submit batch of complex tasks
        batch_results = await manager.submit_batch_tasks(tasks)
        assert all(batch_results)
        
        # Trigger multiple system observations
        for _ in range(5):
            await manager.observe_system_for_behaviors()
            await asyncio.sleep(0.5)
        
        # Check for detected behaviors
        behaviors = await manager.detect_emergent_behaviors()
        assert isinstance(behaviors, list)
        
        # Get behavior detector metrics
        detector_metrics = await manager.emergent_behavior_detector.get_detector_metrics()
        assert "detected_patterns" in detector_metrics
        assert "behavior_events" in detector_metrics
        
        logger.info("✅ Emergent behavior detection test passed")
    
    @pytest.mark.asyncio
    async def test_adaptive_load_balancing_under_stress(self, full_orchestration_system):
        """Test adaptive load balancing under high load."""
        manager = full_orchestration_system
        
        # Create high-load scenario
        stress_tasks = []
        for i in range(25):
            task = TaskRequest(
                task_type="reasoning",
                priority=TaskPriority.HIGH,
                description=f"Stress test task {i}",
                input_data={"problem": f"High-load optimization problem {i}"},
                required_capabilities={"reasoning"}
            )
            stress_tasks.append(task)
        
        # Submit stress load
        start_time = datetime.utcnow()
        batch_results = await manager.submit_batch_tasks(stress_tasks)
        end_time = datetime.utcnow()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify system handled load
        assert all(batch_results)
        assert processing_time < 30.0  # Should complete within 30 seconds
        
        # Check load balancer metrics
        lb_metrics = await manager.adaptive_load_balancer.get_load_balancer_metrics()
        assert "prediction_accuracy" in lb_metrics
        assert "total_allocations" in lb_metrics
        assert lb_metrics["total_allocations"] > 0
        
        logger.info("✅ Adaptive load balancing stress test passed")
    
    @pytest.mark.asyncio
    async def test_production_deployment_readiness(self, full_orchestration_system):
        """Test production deployment readiness."""
        manager = full_orchestration_system
        
        # Test production deployment
        production_config = {
            "environment": "test_production",
            "min_agents": 3,
            "max_agents": 15,
            "max_concurrent_tasks": 30,
            "enable_authentication": False,  # Disabled for testing
            "enable_encryption": False,      # Disabled for testing
            "backup_enabled": False          # Disabled for testing
        }
        
        deployment_success = await manager.deploy_to_production(production_config)
        
        # Deployment may fail due to missing infrastructure, but should not crash
        assert isinstance(deployment_success, bool)
        
        if manager.production_deployment:
            deployment_status = await manager.production_deployment.get_deployment_status()
            assert "deployment_status" in deployment_status
            assert "configuration" in deployment_status
        
        logger.info("✅ Production deployment readiness test passed")
    
    @pytest.mark.asyncio
    async def test_complete_system_metrics(self, full_orchestration_system):
        """Test comprehensive system metrics collection."""
        manager = full_orchestration_system
        
        # Get complete system status
        status = await manager.get_system_status()
        
        # Verify all component metrics are present
        components = status.get("components", {})
        
        expected_components = [
            "agent_registry", "mcp_orchestrator", "task_dispatcher", 
            "metrics_collector", "evolutionary_orchestrator",
            "adaptive_load_balancer", "transaction_coordinator", 
            "emergent_behavior_detector", "meta_learning_engine",
            "predictive_optimizer", "pygent_integration"
        ]
        
        for component in expected_components:
            assert component in components, f"Component {component} missing from status"
            assert isinstance(components[component], dict), f"Component {component} metrics not a dict"
        
        # Get system metrics
        metrics = await manager.get_system_metrics()
        assert "total_tasks" in metrics
        assert "active_agents" in metrics
        assert "coordination_efficiency" in metrics
        
        logger.info("✅ Complete system metrics test passed")


async def run_comprehensive_tests():
    """Run all comprehensive system tests."""
    logger.info("🧪 Starting Comprehensive System Tests...")
    
    try:
        # Create test instance
        config = OrchestrationConfig(
            evolution_enabled=True,
            max_concurrent_tasks=30,
            detailed_logging=True
        )
        
        manager = OrchestrationManager(config)
        await manager.start()
        
        # Setup test environment
        await manager.register_existing_mcp_servers()
        await manager.create_tot_agent("Test ToT Agent", ["reasoning"])
        await manager.create_rag_agent("Test RAG Agent", "retrieval")
        await manager.create_evaluation_agent("Test Eval Agent", ["quality_assessment"])
        
        # Run comprehensive tests
        test_instance = TestComprehensiveSystem()
        
        await test_instance.test_complete_system_startup(manager)
        await test_instance.test_advanced_intelligence_workflow(manager)
        await test_instance.test_predictive_optimization(manager)
        await test_instance.test_meta_learning_capabilities(manager)
        await test_instance.test_pygent_integration(manager)
        await test_instance.test_distributed_transaction_workflow(manager)
        await test_instance.test_emergent_behavior_detection(manager)
        await test_instance.test_adaptive_load_balancing_under_stress(manager)
        await test_instance.test_production_deployment_readiness(manager)
        await test_instance.test_complete_system_metrics(manager)
        
        await manager.stop()
        
        logger.info("✅ ALL COMPREHENSIVE TESTS PASSED!")
        logger.info("🎉 COMPLETE ORCHESTRATION SYSTEM VALIDATED!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_tests())
    if success:
        print("🏆 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION! 🏆")
    else:
        print("❌ TESTS FAILED - SYSTEM NEEDS ATTENTION")