"""
Tests for DGM Engine integration
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.dgm.core.engine import DGMEngine
from src.dgm.models import (
    ImprovementCandidate, ImprovementType, ImprovementStatus,
    PerformanceMetric
)

class TestDGMEngine:
    """Test DGM Engine functionality"""
    
    @pytest.fixture
    def engine_config(self):
        """Test configuration for DGM engine"""
        return {
            "code_generation": {
                "model": "test_model",
                "temperature": 0.7
            },
            "validation": {
                "validation_timeout": 60,
                "test_iterations": 5
            },
            "archive_path": "./test_data/dgm/test_agent",
            "safety": {
                "min_safety_score": 0.8,
                "max_risk_level": 0.3
            },
            "max_concurrent_improvements": 2,
            "improvement_interval_minutes": 5,
            "safety_threshold": 0.8
        }
    
    @pytest.fixture
    def engine(self, engine_config):
        """Create test DGM engine"""
        return DGMEngine("test_agent", engine_config)
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initializes correctly"""
        assert engine.agent_id == "test_agent"
        assert engine.state.agent_id == "test_agent"
        assert engine.max_concurrent_improvements == 2
        assert not engine._running
    
    @pytest.mark.asyncio
    async def test_engine_start_stop(self, engine):
        """Test engine start and stop"""
        # Start engine
        await engine.start()
        assert engine._running
        assert engine._improvement_task is not None
        
        # Stop engine
        await engine.stop()
        assert not engine._running
    
    @pytest.mark.asyncio
    async def test_attempt_improvement(self, engine):
        """Test manual improvement attempt"""
        # Mock the code generator to return proper ImprovementCandidate
        test_candidate = ImprovementCandidate(
            id="test_candidate",
            agent_id="test_agent",
            improvement_type=ImprovementType.ALGORITHM_MODIFICATION,
            description="Test improvement",
            code_changes={"test.py": "print('improved')"},
            expected_improvement=0.1,
            risk_level=0.2
        )
        engine.code_generator.generate_improvement = AsyncMock(return_value=test_candidate)
        
        # Mock safety monitor
        engine.safety_monitor.evaluate_candidate_safety = AsyncMock(return_value={
            "safe": True,
            "safety_score": 0.9,
            "warnings": [],
            "blocking_issues": []
        })
          # Attempt improvement
        candidate_id = await engine.attempt_improvement({"test": "context"})
        
        assert candidate_id is not None
        assert len(engine.state.active_experiments) == 1
        assert candidate_id in engine.state.active_experiments
    
    @pytest.mark.asyncio
    async def test_improvement_rejection_due_to_safety(self, engine):
        """Test improvement rejection due to safety concerns"""
        # Mock the code generator to return dangerous candidate
        dangerous_candidate = ImprovementCandidate(
            id="dangerous_candidate",
            agent_id="test_agent",
            improvement_type=ImprovementType.ALGORITHM_MODIFICATION,
            description="Dangerous improvement",
            code_changes={"test.py": "os.system('rm -rf /')"},
            expected_improvement=0.1,
            risk_level=0.9
        )
        engine.code_generator.generate_improvement = AsyncMock(return_value=dangerous_candidate)
        
        # Mock safety monitor to reject
        engine.safety_monitor.evaluate_candidate_safety = AsyncMock(return_value={
            "safe": False,
            "safety_score": 0.1,
            "warnings": [],
            "blocking_issues": [{"type": "security_risk", "severity": "critical"}]
        })
        
        # Mock archive
        engine.archive.store_entry = AsyncMock(return_value=True)
        
        # Attempt improvement
        candidate_id = await engine.attempt_improvement()
        
        # Should be rejected
        candidate = await engine.get_improvement_status(candidate_id)
        assert candidate.status == ImprovementStatus.REJECTED
        assert len(engine.state.active_experiments) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_improvement_limit(self, engine):
        """Test concurrent improvement limit enforcement"""
        # Mock components
        concurrent_candidate = ImprovementCandidate(
            id="concurrent_test",
            agent_id="test_agent", 
            improvement_type=ImprovementType.ALGORITHM_MODIFICATION,
            description="Test improvement",
            code_changes={"test.py": "print('improved')"},
            expected_improvement=0.1,
            risk_level=0.2
        )
        engine.code_generator.generate_improvement = AsyncMock(return_value=concurrent_candidate)
        
        engine.safety_monitor.evaluate_candidate_safety = AsyncMock(return_value={
            "safe": True,
            "safety_score": 0.9,
            "warnings": [],
            "blocking_issues": []
        })
        
        # Fill up the concurrent slots
        for i in range(engine.max_concurrent_improvements):
            candidate_id = await engine.attempt_improvement()
            assert candidate_id is not None
        
        # Next attempt should fail
        with pytest.raises(ValueError, match="Maximum concurrent improvements reached"):
            await engine.attempt_improvement()
    
    @pytest.mark.asyncio
    async def test_get_current_state(self, engine):
        """Test getting current DGM state"""
        state = await engine.get_current_state()
        
        assert state.agent_id == "test_agent"
        assert state.generation == 0
        assert len(state.improvement_history) == 0
        assert len(state.active_experiments) == 0
    
    @pytest.mark.asyncio
    async def test_baseline_establishment(self, engine):
        """Test baseline performance establishment"""
        await engine._establish_baseline()
        
        assert len(engine.state.current_performance) > 0
        assert "response_time" in engine.state.current_performance
        assert "accuracy" in engine.state.current_performance

class TestDGMIntegration:
    """Integration tests for complete DGM workflow"""
    
    @pytest.mark.asyncio
    async def test_full_improvement_workflow(self):
        """Test complete improvement workflow"""
        config = {
            "code_generation": {"model": "test"},
            "validation": {"validation_timeout": 10},
            "archive_path": "./test_data/dgm/integration_test",
            "safety": {"min_safety_score": 0.5},
            "max_concurrent_improvements": 1
        }
        
        engine = DGMEngine("integration_test_agent", config)
          # Mock all components for successful workflow
        integration_candidate = ImprovementCandidate(
            id="integration_test",
            agent_id="integration_test_agent",
            improvement_type=ImprovementType.ALGORITHM_MODIFICATION,
            description="Performance optimization",
            code_changes={"optimizer.py": "# Optimized code"},
            expected_improvement=0.2,
            risk_level=0.1
        )
        engine.code_generator.generate_improvement = AsyncMock(return_value=integration_candidate)
        
        engine.safety_monitor.evaluate_candidate_safety = AsyncMock(return_value={
            "safe": True,
            "safety_score": 0.9,
            "warnings": [],
            "blocking_issues": []
        })
        
        # Mock validator for successful validation
        from src.dgm.models import ValidationResult, PerformanceMetric
        mock_validation_result = ValidationResult(
            candidate_id="test",
            success=True,
            performance_before=[
                PerformanceMetric(name="speed", value=1.0, unit="ops/sec")
            ],
            performance_after=[
                PerformanceMetric(name="speed", value=1.2, unit="ops/sec")
            ],
            improvement_score=0.2,
            safety_score=0.9,
            test_results={"tests_passed": 10},
            validation_time=5.0
        )
        
        engine.validator.validate_candidate = AsyncMock(return_value=mock_validation_result)
        
        engine.safety_monitor.evaluate_validation_result = AsyncMock(return_value={
            "safe": True,
            "concerns": []
        })
        
        # Mock archive
        engine.archive.store_entry = AsyncMock(return_value=True)
        
        # Start the improvement
        candidate_id = await engine.attempt_improvement()
        
        # Wait for processing to complete
        await asyncio.sleep(0.2)
        
        # Check results
        candidate = await engine.get_improvement_status(candidate_id)
        assert candidate is not None
        assert candidate.id == candidate_id
