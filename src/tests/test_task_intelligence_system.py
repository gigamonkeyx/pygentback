"""
Comprehensive Test Suite for Task Intelligence System

Tests all components of the Task Intelligence System:
- Dual-loop orchestrator functionality
- Enhanced task analysis engine
- Context intelligence system
- Meta-coordination layer
- Teaching agent framework
- Dynamic question generation
- Pattern learning and optimization
- Integration with PyGent Factory infrastructure
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.agents.supervisor_agent import (
    TaskIntelligenceSystem, MetaSupervisorAgent, TaskLedger, ProgressLedger,
    TaskType, TaskComplexity, TaskAnalysis, QualityScore
)
from src.agents.task_intelligence_integration import TaskIntelligenceIntegration
from src.orchestration.coordination_models import TaskRequest, TaskStatus


class TestTaskIntelligenceSystem:
    """Test suite for core Task Intelligence System functionality"""
    
    @pytest.fixture
    async def task_intelligence(self):
        """Create Task Intelligence System instance for testing"""
        mock_mcp_manager = Mock()
        mock_a2a_manager = Mock()
        
        system = TaskIntelligenceSystem(
            mcp_manager=mock_mcp_manager,
            a2a_manager=mock_a2a_manager
        )
        return system
    
    @pytest.fixture
    def sample_task_description(self):
        """Sample task description for testing"""
        return "Create a Vue.js component for user authentication with database integration and testing"
    
    @pytest.mark.asyncio
    async def test_dual_loop_orchestrator_creation(self, task_intelligence, sample_task_description):
        """Test dual-loop orchestrator task creation"""
        
        # Test task intelligence creation
        task_id = await task_intelligence.create_task_intelligence(sample_task_description)
        
        assert task_id is not None
        assert task_id in task_intelligence.task_ledgers
        assert task_id in task_intelligence.progress_ledgers
        
        # Verify Task Ledger (outer loop)
        task_ledger = task_intelligence.task_ledgers[task_id]
        assert task_ledger.original_request == sample_task_description
        assert len(task_ledger.facts) > 0
        assert len(task_ledger.current_plan) > 0
        
        # Verify Progress Ledger (inner loop)
        progress_ledger = task_intelligence.progress_ledgers[task_id]
        assert progress_ledger.task_id == task_id
        assert progress_ledger.total_steps > 0
    
    @pytest.mark.asyncio
    async def test_enhanced_task_analysis(self, task_intelligence, sample_task_description):
        """Test enhanced task analysis engine"""
        
        # Test task analysis
        analysis = await task_intelligence.analyze_task(sample_task_description)
        
        assert isinstance(analysis, TaskAnalysis)
        assert analysis.task_type in [TaskType.UI_CREATION, TaskType.CODING]
        assert isinstance(analysis.complexity, int)
        assert analysis.complexity >= 1 and analysis.complexity <= 10
        assert len(analysis.required_capabilities) > 0
        assert len(analysis.success_criteria) > 0
    
    @pytest.mark.asyncio
    async def test_enhanced_complexity_assessment(self, task_intelligence):
        """Test enhanced complexity assessment"""
        
        # Test simple task
        simple_task = "Create a button"
        complexity, confidence = task_intelligence._assess_complexity_enhanced(simple_task)
        assert complexity == TaskComplexity.SIMPLE
        assert confidence > 0.8
        
        # Test complex task
        complex_task = "Build a distributed microservice architecture with multiple databases, API gateways, and real-time monitoring"
        complexity, confidence = task_intelligence._assess_complexity_enhanced(complex_task)
        assert complexity in [TaskComplexity.COMPLEX, TaskComplexity.ENTERPRISE]
        assert confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_requirement_extraction(self, task_intelligence, sample_task_description):
        """Test enhanced requirement extraction"""
        
        requirements = task_intelligence._extract_requirements_enhanced(sample_task_description)
        
        assert "frontend_development" in requirements
        assert "database_operations" in requirements
        assert "testing" in requirements
        assert len(requirements) >= 3
    
    @pytest.mark.asyncio
    async def test_context_intelligence_system(self, task_intelligence, sample_task_description):
        """Test context intelligence system"""
        
        # Create task ledger
        task_id = "test_task_123"
        ledger = TaskLedger(task_id=task_id, original_request=sample_task_description)
        
        # Test context gathering
        await task_intelligence._gather_context(ledger, {"initial": "test_context"})
        
        assert len(ledger.facts) > 0
        assert len(ledger.context_sources) > 0
        assert any("Initial context" in fact for fact in ledger.facts)
    
    @pytest.mark.asyncio
    async def test_pattern_similarity_calculation(self, task_intelligence):
        """Test pattern similarity calculation"""
        
        # Create test pattern
        pattern = {
            "task_characteristics": {
                "original_request": "Create Vue component with authentication",
                "requirements": ["frontend_development", "security"],
                "complexity": 5
            }
        }
        
        # Test similarity calculation
        similarity = await task_intelligence._calculate_pattern_similarity(
            "Build Vue.js authentication component",
            ["frontend_development", "security"],
            pattern
        )
        
        assert similarity > 0.5  # Should be similar
        assert similarity <= 1.0
    
    @pytest.mark.asyncio
    async def test_workflow_pattern_recording(self, task_intelligence, sample_task_description):
        """Test workflow pattern recording"""
        
        # Create task intelligence
        task_id = await task_intelligence.create_task_intelligence(sample_task_description)
        
        # Record pattern
        pattern_id = await task_intelligence.record_workflow_pattern(
            task_id, success=True, execution_time=300, quality_score=0.85
        )
        
        assert pattern_id != ""
        assert pattern_id in task_intelligence.workflow_patterns
        
        pattern = task_intelligence.workflow_patterns[pattern_id]
        assert pattern["performance_metrics"]["success"] == True
        assert pattern["performance_metrics"]["quality_score"] == 0.85
    
    @pytest.mark.asyncio
    async def test_failure_pattern_recording(self, task_intelligence, sample_task_description):
        """Test failure pattern recording"""
        
        # Create task intelligence
        task_id = await task_intelligence.create_task_intelligence(sample_task_description)
        
        # Record failure pattern
        failure_details = {
            "type": "execution_stall",
            "root_cause": "timeout",
            "context": {"complexity": 7}
        }
        
        failure_id = await task_intelligence.record_failure_pattern(task_id, failure_details)
        
        assert failure_id != ""
        assert failure_id in task_intelligence.failure_patterns
        
        failure_pattern = task_intelligence.failure_patterns[failure_id]
        assert failure_pattern["failure_type"] == "execution_stall"
        assert len(failure_pattern["prevention_suggestions"]) > 0


class TestMetaSupervisorAgent:
    """Test suite for MetaSupervisorAgent functionality"""
    
    @pytest.fixture
    async def meta_supervisor(self):
        """Create MetaSupervisorAgent instance for testing"""
        mock_mcp_manager = Mock()
        mock_a2a_manager = Mock()
        
        return MetaSupervisorAgent(
            mcp_manager=mock_mcp_manager,
            a2a_manager=mock_a2a_manager
        )
    
    @pytest.mark.asyncio
    async def test_workflow_requirements_analysis(self, meta_supervisor):
        """Test workflow requirements analysis"""
        
        complex_task = "Build a full-stack application with Vue.js frontend, Node.js backend, PostgreSQL database, and deploy to production"
        
        analysis = await meta_supervisor._analyze_workflow_requirements(complex_task, {})
        
        assert analysis["requires_multiple_supervisors"] == True
        assert len(analysis["supervisor_requirements"]) >= 3
        assert analysis["estimated_supervisors"] > 1
        assert len(analysis["task_decomposition"]) > 0
    
    @pytest.mark.asyncio
    async def test_coordination_plan_creation(self, meta_supervisor):
        """Test coordination plan creation"""
        
        workflow_analysis = {
            "requires_multiple_supervisors": True,
            "supervisor_requirements": ["frontend", "backend", "deployment"],
            "task_decomposition": [
                {"domain": "frontend", "task": "Build UI", "priority": 3, "dependencies": []},
                {"domain": "backend", "task": "Build API", "priority": 2, "dependencies": []},
                {"domain": "deployment", "task": "Deploy app", "priority": 5, "dependencies": ["frontend", "backend"]}
            ]
        }
        
        plan = await meta_supervisor._create_coordination_plan(workflow_analysis)
        
        assert plan["execution_strategy"] in ["sequential", "hybrid"]
        assert len(plan["supervisor_assignments"]) == 3
        assert "dependencies" in plan
        assert len(plan["success_criteria"]) > 0


class TestDynamicQuestionGeneration:
    """Test suite for Dynamic Question Generation Framework"""
    
    @pytest.fixture
    async def task_intelligence(self):
        """Create Task Intelligence System with question generation enabled"""
        mock_mcp_manager = Mock()
        mock_a2a_manager = Mock()
        
        system = TaskIntelligenceSystem(
            mcp_manager=mock_mcp_manager,
            a2a_manager=mock_a2a_manager
        )
        system.question_generation_enabled = True
        return system
    
    @pytest.mark.asyncio
    async def test_context_gap_analysis(self, task_intelligence):
        """Test context gap analysis"""
        
        # Create task ledger with ambiguous request
        task_id = "test_ambiguous"
        ledger = TaskLedger(task_id=task_id, original_request="Maybe create something with unclear requirements")
        
        # Analyze missing context
        missing_context = await task_intelligence._analyze_missing_context(ledger, 0.3)
        
        assert len(missing_context) > 0
        assert any(gap["type"] == "requirement_ambiguity" for gap in missing_context)
    
    @pytest.mark.asyncio
    async def test_question_generation_for_gaps(self, task_intelligence):
        """Test question generation for context gaps"""
        
        context_gap = {
            "type": "requirement_ambiguity",
            "severity": "high",
            "description": "Task contains ambiguous requirements",
            "keywords": ["maybe", "unclear"]
        }
        
        ledger = TaskLedger(task_id="test", original_request="Test task")
        questions = await task_intelligence._generate_questions_for_gap(context_gap, ledger)
        
        assert len(questions) > 0
        assert all("question" in q for q in questions)
        assert all("priority" in q for q in questions)
        assert any(q["priority"] == "high" for q in questions)
    
    @pytest.mark.asyncio
    async def test_question_filtering_by_expertise(self, task_intelligence):
        """Test question filtering based on user expertise"""
        
        questions = [
            {"question": "Basic question", "priority": "low", "type": "basic"},
            {"question": "Technical question", "priority": "high", "type": "technical"},
            {"question": "Advanced question", "priority": "medium", "type": "specification"}
        ]
        
        # Test expert filtering
        task_intelligence.user_expertise_level = "expert"
        filtered = task_intelligence._filter_questions_by_expertise(questions)
        assert len(filtered) <= len(questions)
        assert all(q["priority"] == "high" or q["type"] in ["technical", "specification"] for q in filtered)
    
    @pytest.mark.asyncio
    async def test_human_response_processing(self, task_intelligence):
        """Test processing of human responses"""
        
        # Create task with questions
        task_id = await task_intelligence.create_task_intelligence("Ambiguous task with unclear requirements")
        
        # Simulate human responses
        responses = {
            "q1": "I need a Vue.js component for user authentication",
            "q2": "It should integrate with PostgreSQL database",
            "q3": "Success means users can login and logout securely"
        }
        
        success = await task_intelligence.process_human_responses(task_id, responses)
        
        assert success == True
        
        # Check that context was updated
        ledger = task_intelligence.task_ledgers[task_id]
        assert any("Human clarification" in fact for fact in ledger.facts)
        assert "technical_specification_provided" in ledger.requirements


class TestTeachingAgentFramework:
    """Test suite for Teaching Agent Framework"""
    
    @pytest.fixture
    async def task_intelligence(self):
        """Create Task Intelligence System for teaching tests"""
        mock_mcp_manager = Mock()
        mock_a2a_manager = Mock()
        
        return TaskIntelligenceSystem(
            mcp_manager=mock_mcp_manager,
            a2a_manager=mock_a2a_manager
        )
    
    @pytest.mark.asyncio
    async def test_failure_pattern_analysis(self, task_intelligence):
        """Test failure pattern analysis"""
        
        failure_details = {
            "error": "Task execution stalled after timeout",
            "task_context": {
                "task_type": "coding",
                "complexity": 7
            }
        }
        
        pattern = await task_intelligence._analyze_failure_pattern("agent_123", failure_details)
        
        assert pattern["failure_type"] == "execution_stall"
        assert pattern["teachable_moment"] == True
        assert pattern["frequency"] == 1
    
    @pytest.mark.asyncio
    async def test_teaching_intervention_generation(self, task_intelligence):
        """Test teaching intervention generation"""
        
        failure_pattern = {
            "failure_type": "quality_failure",
            "frequency": 2,
            "teachable_moment": True
        }
        
        intervention = await task_intelligence._generate_teaching_intervention("agent_123", failure_pattern)
        
        assert intervention["strategy"] != "none"
        assert len(intervention["actions"]) > 0
        assert intervention["expected_improvement"] > 0
        assert intervention["confidence"] > 0
    
    @pytest.mark.asyncio
    async def test_teaching_effectiveness_tracking(self, task_intelligence):
        """Test teaching effectiveness tracking"""
        
        agent_id = "agent_123"
        task_id = "task_456"
        
        # Record teaching session
        teaching_result = {
            "teaching_applied": True,
            "teaching_strategy": "feedback_loop",
            "improvement_predicted": 0.4
        }
        
        failure_pattern = {"failure_type": "quality_failure"}
        
        await task_intelligence._record_teaching_session(agent_id, task_id, teaching_result, failure_pattern)
        
        # Update effectiveness
        await task_intelligence.update_teaching_effectiveness(agent_id, task_id, 0.6)
        
        assert agent_id in task_intelligence.agent_learning_history
        assert agent_id in task_intelligence.feedback_effectiveness
        assert task_intelligence.feedback_effectiveness[agent_id] > 0


class TestTaskIntelligenceIntegration:
    """Test suite for Task Intelligence Integration with PyGent Factory"""
    
    @pytest.fixture
    async def integration_system(self):
        """Create integration system for testing"""
        mock_task_dispatcher = Mock()
        mock_a2a_manager = Mock()
        mock_mcp_orchestrator = Mock()
        
        integration = TaskIntelligenceIntegration(
            task_dispatcher=mock_task_dispatcher,
            a2a_manager=mock_a2a_manager,
            mcp_orchestrator=mock_mcp_orchestrator
        )
        
        return integration
    
    @pytest.mark.asyncio
    async def test_task_complexity_analysis(self, integration_system):
        """Test task complexity analysis for routing decisions"""
        
        # Simple task
        simple_task = TaskRequest(
            task_id="simple_123",
            task_type="simple",
            description="Create a button",
            dependencies=[]
        )
        
        complexity = await integration_system._analyze_task_complexity(simple_task)
        assert complexity < 5
        
        # Complex task
        complex_task = TaskRequest(
            task_id="complex_123",
            task_type="multi_agent",
            description="Build a distributed system with multiple microservices, databases, and real-time monitoring across multiple cloud providers",
            dependencies=["dep1", "dep2", "dep3"]
        )
        
        complexity = await integration_system._analyze_task_complexity(complex_task)
        assert complexity >= 5
    
    @pytest.mark.asyncio
    async def test_quality_score_calculation(self, integration_system):
        """Test quality score calculation"""
        
        # Mock progress ledger
        mock_progress_ledger = Mock()
        mock_progress_ledger.step_status = {"step1": "completed", "step2": "completed", "step3": "failed"}
        mock_progress_ledger.stall_count = 1
        mock_progress_ledger.last_progress_time = datetime.utcnow() - timedelta(minutes=10)
        
        integration_system.task_intelligence.progress_ledgers["test_id"] = mock_progress_ledger
        
        quality_score = await integration_system._calculate_quality_score("test_id")
        
        assert 0.0 <= quality_score <= 1.0
        assert quality_score < 1.0  # Should be penalized for failed step and stall
    
    @pytest.mark.asyncio
    async def test_integration_metrics_tracking(self, integration_system):
        """Test integration metrics tracking"""
        
        # Test performance tracking update
        await integration_system._update_performance_tracking(
            "task_123", success=True, execution_time=300, quality_score=0.85
        )
        
        assert integration_system.integration_metrics["tasks_processed"] == 1
        assert integration_system.integration_metrics["intelligence_success_rate"] > 0
        assert integration_system.integration_metrics["average_improvement"] > 0


@pytest.mark.asyncio
async def test_end_to_end_task_intelligence_workflow():
    """End-to-end test of complete Task Intelligence workflow"""
    
    # Create system components
    mock_mcp_manager = Mock()
    mock_a2a_manager = Mock()
    
    task_intelligence = TaskIntelligenceSystem(
        mcp_manager=mock_mcp_manager,
        a2a_manager=mock_a2a_manager
    )
    
    # Test complete workflow
    task_description = "Create a comprehensive user management system with Vue.js frontend, FastAPI backend, PostgreSQL database, and comprehensive testing"
    
    # 1. Create task intelligence
    task_id = await task_intelligence.create_task_intelligence(task_description)
    assert task_id is not None
    
    # 2. Verify dual-loop creation
    assert task_id in task_intelligence.task_ledgers
    assert task_id in task_intelligence.progress_ledgers
    
    # 3. Check task analysis
    task_ledger = task_intelligence.task_ledgers[task_id]
    assert len(task_ledger.requirements) > 0
    assert len(task_ledger.current_plan) > 0
    
    # 4. Test pattern recording
    pattern_id = await task_intelligence.record_workflow_pattern(
        task_id, success=True, execution_time=1800, quality_score=0.9
    )
    assert pattern_id != ""
    
    # 5. Test pattern analytics
    analytics = task_intelligence.get_pattern_analytics()
    assert analytics["workflow_patterns"]["total_patterns"] > 0
    
    # 6. Test pattern library optimization
    await task_intelligence.optimize_pattern_library()
    
    print("✅ End-to-end Task Intelligence workflow test completed successfully")


if __name__ == "__main__":
    # Run end-to-end test
    asyncio.run(test_end_to_end_task_intelligence_workflow())
