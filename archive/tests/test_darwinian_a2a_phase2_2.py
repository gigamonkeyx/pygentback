"""
Test suite for Phase 2.2: Collaborative Self-Improvement
Tests the collaborative self-improvement implementation with A2A coordination.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.orchestration.collaborative_self_improvement import (
    CollaborativeSelfImprovement,
    ImprovementProposal,
    CollaborativeTask,
    CodeGenerationRequest,
    TestingSuite,
    ImprovementType,
    ValidationStatus
)


class TestCollaborativeSelfImprovement:
    """Test collaborative self-improvement implementation."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Mock orchestrator for testing."""
        orchestrator = Mock()
        orchestrator.agent_id = "test_agent"
        orchestrator.a2a_server = Mock()
        orchestrator.a2a_server.broadcast_evolution_event = AsyncMock()
        orchestrator.a2a_server.request_peer_analysis = AsyncMock()
        orchestrator.get_peer_agents = AsyncMock(return_value=[
            {"id": "agent1", "capabilities": ["problem_solving", "validation"]},
            {"id": "agent2", "capabilities": ["code_generation", "testing"]},
            {"id": "agent3", "capabilities": ["validation", "expertise"]}
        ])
        return orchestrator
    
    @pytest.fixture
    def csi(self, mock_orchestrator):
        """Create collaborative self-improvement instance."""
        return CollaborativeSelfImprovement(
            orchestrator=mock_orchestrator,
            improvement_threshold=0.1
        )
    
    def test_csi_initialization(self, csi):
        """Test CSI initialization."""
        assert csi.improvement_threshold == 0.1
        assert len(csi.proposals) == 0
        assert len(csi.collaborative_tasks) == 0
        assert len(csi.code_requests) == 0
        assert len(csi.testing_suites) == 0
        assert len(csi.improvement_history) == 0
    
    @pytest.mark.asyncio
    async def test_start_improvement_session(self, csi, mock_orchestrator):
        """Test starting improvement session."""
        session_id = await csi.start_improvement_session()
        
        assert session_id.startswith("improvement_")
        mock_orchestrator.a2a_server.broadcast_evolution_event.assert_called_once()
        
        # Verify broadcast content
        call_args = mock_orchestrator.a2a_server.broadcast_evolution_event.call_args[0][0]
        assert call_args['event_type'] == 'improvement_session_start'
        assert call_args['session_id'] == session_id
        assert call_args['coordinator'] == "test_agent"
    
    @pytest.mark.asyncio
    async def test_collaborative_problem_decomposition(self, csi, mock_orchestrator):
        """Test A2A-enabled collaborative problem decomposition."""
        problem = "Optimize database query performance for large datasets"
        
        # Mock peer analysis responses
        mock_orchestrator.a2a_server.request_peer_analysis = AsyncMock(
            return_value={
                'improvements': [
                    {'subtask': 'Index optimization', 'complexity': 0.3, 'dependencies': []},
                    {'subtask': 'Query rewriting', 'complexity': 0.4, 'dependencies': [0]}
                ]
            }
        )
        
        task = await csi.collaborative_problem_decomposition(problem)
        
        assert isinstance(task, CollaborativeTask)
        assert task.problem_description == problem
        assert task.coordinator_id == "test_agent"
        assert len(task.participants) > 0
        assert len(task.decomposition) > 0
        assert task.status == "active"
        assert task.id in csi.collaborative_tasks
    
    @pytest.mark.asyncio
    async def test_generate_improvement_proposal(self, csi, mock_orchestrator):
        """Test distributed improvement proposal generation."""
        # Mock peer analysis for expert input
        mock_orchestrator.a2a_server.request_peer_analysis = AsyncMock(
            return_value={
                'additional_improvements': {
                    'caching_layer': {
                        'type': 'redis_integration',
                        'expected_improvement': '40%'
                    }
                }
            }
        )
        
        analysis_data = {
            'current_performance': {'response_time': 0.8, 'throughput': 200},
            'bottlenecks': ['database_queries', 'network_latency']
        }
        
        proposal = await csi.generate_improvement_proposal(
            ImprovementType.PERFORMANCE_TUNING,
            "database_service",
            analysis_data
        )
        
        assert isinstance(proposal, ImprovementProposal)
        assert proposal.improvement_type == ImprovementType.PERFORMANCE_TUNING
        assert proposal.target_component == "database_service"
        assert proposal.proposer_id == "test_agent"
        assert proposal.status == ValidationStatus.PENDING
        assert len(proposal.expected_benefits) > 0
        assert proposal.id in csi.proposals
    
    @pytest.mark.asyncio
    async def test_validate_improvement_proposal(self, csi, mock_orchestrator):
        """Test A2A-coordinated improvement validation."""
        # Create test proposal
        proposal = ImprovementProposal(
            id="test_proposal",
            proposer_id="test_agent",
            improvement_type=ImprovementType.CODE_OPTIMIZATION,
            title="Test Improvement",
            description="Test improvement description",
            target_component="test_component",
            proposed_changes={"optimization": "test"},
            expected_benefits=["faster execution"],
            risk_assessment={"deployment_risk": "low"},
            validation_requirements=["unit_tests"],
            metadata={},
            timestamp=asyncio.get_event_loop().time()
        )
        
        # Mock validation responses
        mock_orchestrator.a2a_server.request_peer_analysis = AsyncMock(
            side_effect=[
                {
                    'validation_score': 0.8,
                    'tests_run': 10,
                    'tests_passed': 9,
                    'performance_impact': {'response_time_change': -0.1},
                    'risk_score': 0.2
                },
                {
                    'validation_score': 0.9,
                    'tests_run': 15,
                    'tests_passed': 14,
                    'performance_impact': {'response_time_change': -0.15},
                    'risk_score': 0.1
                }
            ]
        )
        
        validation_result = await csi.validate_improvement_proposal(proposal)
        
        assert 'proposal_id' in validation_result
        assert validation_result['proposal_id'] == "test_proposal"
        assert 'consensus_score' in validation_result
        assert validation_result['consensus_score'] > 0.8
        assert validation_result['recommendation'] == 'approve'
        assert proposal.status == ValidationStatus.VALIDATED
    
    @pytest.mark.asyncio
    async def test_collaborative_code_generation_parallel(self, csi, mock_orchestrator):
        """Test collaborative code generation in parallel mode."""
        request = CodeGenerationRequest(
            id="test_request",
            requester_id="test_agent",
            specification="Create a function to sort a list efficiently",
            requirements=["O(n log n) complexity", "in-place sorting"],
            constraints=["no external libraries"],
            target_language="python",
            quality_criteria=["performance", "readability"],
            collaboration_mode="parallel",
            participants=["agent1", "agent2"]
        )
        
        # Mock code generation responses
        mock_orchestrator.a2a_server.request_peer_analysis = AsyncMock(
            side_effect=[
                {
                    'generated_code': 'def sort_list(arr):\n    return sorted(arr)',
                    'quality_score': 0.7
                },
                {
                    'generated_code': 'def sort_list(arr):\n    arr.sort()\n    return arr',
                    'quality_score': 0.9
                }
            ]
        )
        
        result = await csi.collaborative_code_generation(request)
        
        assert 'request_id' in result
        assert result['request_id'] == "test_request"
        assert 'code_variants' in result
        assert len(result['code_variants']) == 2
        assert 'consensus_code' in result
        assert result['consensus_code'] is not None
        assert request.id in csi.code_requests
    
    @pytest.mark.asyncio
    async def test_collaborative_code_generation_sequential(self, csi, mock_orchestrator):
        """Test collaborative code generation in sequential mode."""
        request = CodeGenerationRequest(
            id="test_request_seq",
            requester_id="test_agent",
            specification="Create a hash function",
            requirements=["collision resistant"],
            constraints=["no crypto libraries"],
            target_language="python",
            quality_criteria=["security"],
            collaboration_mode="sequential",
            participants=["agent1", "agent2"]
        )
        
        # Mock refinement responses
        mock_orchestrator.a2a_server.request_peer_analysis = AsyncMock(
            side_effect=[
                {
                    'refined_code': 'def hash_func(data):\n    # Improved version 1\n    return hash(data)'
                },
                {
                    'refined_code': 'def hash_func(data):\n    # Final optimized version\n    return hash(str(data))'
                }
            ]
        )
        
        result = await csi.collaborative_code_generation(request)
        
        assert result['request_id'] == "test_request_seq"
        assert 'consensus_code' in result
        assert "Final optimized version" in result['consensus_code']
    
    @pytest.mark.asyncio
    async def test_create_distributed_testing_network(self, csi, mock_orchestrator):
        """Test distributed testing and validation networks."""
        test_suite = TestingSuite(
            id="test_suite_1",
            name="Performance Test Suite",
            test_cases=[
                {'name': 'test_response_time', 'type': 'performance'},
                {'name': 'test_throughput', 'type': 'load'},
                {'name': 'test_accuracy', 'type': 'functional'},
                {'name': 'test_memory_usage', 'type': 'resource'}
            ],
            coverage_requirements={'code': 0.8, 'branch': 0.7},
            performance_benchmarks={'response_time': 0.1, 'throughput': 1000},
            security_checks=['input_validation', 'sql_injection'],
            compatibility_matrix={'python': ['3.8', '3.9', '3.10']},
            execution_agents=["agent1", "agent2"]
        )
        
        # Mock test execution responses
        mock_orchestrator.a2a_server.request_peer_analysis = AsyncMock(
            side_effect=[
                {
                    'test_results': {'passed': 2, 'failed': 0, 'total': 2},
                    'coverage': {'code': 0.85, 'branch': 0.75},
                    'performance': {'response_time': 0.08, 'throughput': 1100}
                },
                {
                    'test_results': {'passed': 2, 'failed': 0, 'total': 2},
                    'coverage': {'code': 0.82, 'branch': 0.72},
                    'performance': {'response_time': 0.09, 'throughput': 1050}
                }
            ]
        )
        
        result = await csi.create_distributed_testing_network(test_suite)
        
        assert result['suite_id'] == "test_suite_1"
        assert 'test_results' in result
        assert 'overall_coverage' in result
        assert result['overall_coverage'] >= 0.8
        assert result['overall_status'] == 'passed'
        assert test_suite.id in csi.testing_suites
    
    @pytest.mark.asyncio
    async def test_implement_improvement_adoption_gradual(self, csi, mock_orchestrator):
        """Test A2A-enabled improvement adoption with gradual strategy."""
        proposal = ImprovementProposal(
            id="test_adoption",
            proposer_id="test_agent",
            improvement_type=ImprovementType.ALGORITHM_ENHANCEMENT,
            title="Test Adoption",
            description="Test adoption description",
            target_component="test_component",
            proposed_changes={"enhancement": "test"},
            expected_benefits=["better performance"],
            risk_assessment={"deployment_risk": "low"},
            validation_requirements=["integration_tests"],
            metadata={},
            timestamp=asyncio.get_event_loop().time(),
            status=ValidationStatus.VALIDATED
        )
        
        # Add to proposals
        csi.proposals[proposal.id] = proposal
        
        result = await csi.implement_improvement_adoption_mechanisms(
            proposal, "gradual"
        )
        
        assert result['proposal_id'] == "test_adoption"
        assert result['strategy'] == "gradual"
        assert 'phases' in result
        assert len(result['phases']) >= 1  # At least pilot phase should execute
        assert 'adoption_status' in result
        
        # Verify A2A broadcast
        mock_orchestrator.a2a_server.broadcast_evolution_event.assert_called()
        
        # Check final call for adoption completion
        calls = mock_orchestrator.a2a_server.broadcast_evolution_event.call_args_list
        final_call = calls[-1][0][0]
        assert final_call['event_type'] == 'improvement_adopted'
        assert final_call['proposal_id'] == "test_adoption"
    
    @pytest.mark.asyncio
    async def test_implement_improvement_adoption_canary(self, csi, mock_orchestrator):
        """Test A2A-enabled improvement adoption with canary strategy."""
        proposal = ImprovementProposal(
            id="test_canary",
            proposer_id="test_agent",
            improvement_type=ImprovementType.CAPABILITY_EXTENSION,
            title="Test Canary",
            description="Test canary description",
            target_component="test_component",
            proposed_changes={"extension": "test"},
            expected_benefits=["new functionality"],
            risk_assessment={"deployment_risk": "medium"},
            validation_requirements=["canary_tests"],
            metadata={},
            timestamp=asyncio.get_event_loop().time(),
            status=ValidationStatus.VALIDATED
        )
        
        result = await csi.implement_improvement_adoption_mechanisms(
            proposal, "canary"
        )
        
        assert result['proposal_id'] == "test_canary"
        assert result['strategy'] == "canary"
        assert 'canary_result' in result
        assert 'adoption_status' in result
    
    def test_improvement_proposal_creation(self):
        """Test improvement proposal data structure."""
        proposal = ImprovementProposal(
            id="test_id",
            proposer_id="agent1",
            improvement_type=ImprovementType.CODE_OPTIMIZATION,
            title="Test Proposal",
            description="Test description",
            target_component="test_component",
            proposed_changes={"change": "test"},
            expected_benefits=["benefit1"],
            risk_assessment={"risk": "low"},
            validation_requirements=["req1"],
            metadata={"meta": "data"},
            timestamp=1234567890.0
        )
        
        assert proposal.id == "test_id"
        assert proposal.improvement_type == ImprovementType.CODE_OPTIMIZATION
        assert proposal.status == ValidationStatus.PENDING
        assert proposal.votes == {}
        assert proposal.validation_results == {}
    
    def test_collaborative_task_creation(self):
        """Test collaborative task data structure."""
        task = CollaborativeTask(
            id="task_1",
            coordinator_id="agent1",
            participants=["agent2", "agent3"],
            problem_description="Test problem",
            decomposition=[{"subtask": "test", "complexity": 0.5}],
            task_assignments={"agent2": ["subtask1"]},
            progress={"subtask1": 0.5},
            results={"subtask1": "partial_result"},
            status="active",
            created_at=1234567890.0
        )
        
        assert task.id == "task_1"
        assert len(task.participants) == 2
        assert task.status == "active"
        assert task.deadline is None
    
    def test_code_generation_request_creation(self):
        """Test code generation request data structure."""
        request = CodeGenerationRequest(
            id="req_1",
            requester_id="agent1",
            specification="Test spec",
            requirements=["req1", "req2"],
            constraints=["const1"],
            target_language="python",
            quality_criteria=["quality1"],
            collaboration_mode="parallel",
            participants=["agent2", "agent3"]
        )
        
        assert request.id == "req_1"
        assert request.collaboration_mode == "parallel"
        assert len(request.participants) == 2
        assert request.deadline is None
    
    def test_testing_suite_creation(self):
        """Test testing suite data structure."""
        suite = TestingSuite(
            id="suite_1",
            name="Test Suite",
            test_cases=[{"test": "case1"}],
            coverage_requirements={"code": 0.8},
            performance_benchmarks={"speed": 100},
            security_checks=["check1"],
            compatibility_matrix={"lang": ["version1"]},
            execution_agents=["agent1"]
        )
        
        assert suite.id == "suite_1"
        assert suite.name == "Test Suite"
        assert len(suite.test_cases) == 1
        assert suite.results is None


class TestImprovementTypes:
    """Test improvement type enums."""
    
    def test_improvement_type_values(self):
        """Test improvement type enum values."""
        assert ImprovementType.CODE_OPTIMIZATION.value == "code_optimization"
        assert ImprovementType.ALGORITHM_ENHANCEMENT.value == "algorithm_enhancement"
        assert ImprovementType.ARCHITECTURE_REFINEMENT.value == "architecture_refinement"
        assert ImprovementType.PERFORMANCE_TUNING.value == "performance_tuning"
        assert ImprovementType.CAPABILITY_EXTENSION.value == "capability_extension"
        assert ImprovementType.KNOWLEDGE_INTEGRATION.value == "knowledge_integration"
    
    def test_validation_status_values(self):
        """Test validation status enum values."""
        assert ValidationStatus.PENDING.value == "pending"
        assert ValidationStatus.VALIDATING.value == "validating"
        assert ValidationStatus.VALIDATED.value == "validated"
        assert ValidationStatus.REJECTED.value == "rejected"
        assert ValidationStatus.DEPLOYED.value == "deployed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
