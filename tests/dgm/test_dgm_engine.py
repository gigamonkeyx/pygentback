import pytest
import asyncio
from src.dgm.core.engine import DGMEngine
from src.dgm.models import ImprovementStatus

@pytest.mark.asyncio
async def test_dgm_engine_initialization():
    """Test DGM engine initialization"""
    config = {
        "max_concurrent_improvements": 2,
        "improvement_interval_minutes": 1,
        "safety_threshold": 0.8
    }
    
    engine = DGMEngine("test-agent", config)
    
    assert engine.agent_id == "test-agent"
    assert engine.max_concurrent_improvements == 2
    assert engine.safety_threshold == 0.8
    assert engine.state.agent_id == "test-agent"
    assert engine.state.generation == 0

@pytest.mark.asyncio
async def test_dgm_engine_start_stop():
    """Test DGM engine start and stop"""
    config = {
        "improvement_interval_minutes": 0.1  # Very short for testing
    }
    
    engine = DGMEngine("test-agent", config)
    
    # Start engine
    await engine.start()
    assert engine._running
    
    # Wait a moment
    await asyncio.sleep(0.2)
    
    # Stop engine
    await engine.stop()
    assert not engine._running

@pytest.mark.asyncio
async def test_dgm_improvement_attempt():
    """Test manual improvement attempt"""
    config = {
        "safety_threshold": 0.5  # Lower threshold for testing
    }
    
    engine = DGMEngine("test-agent", config)
    
    # Initialize baseline
    await engine._establish_baseline()
    
    # Attempt improvement
    candidate_id = await engine.attempt_improvement({"test_context": True})
    
    assert candidate_id is not None
    assert len(engine.state.active_experiments) == 1
    
    # Check improvement status
    status = await engine.get_improvement_status(candidate_id)
    assert status is not None
    assert status.agent_id == "test-agent"

@pytest.mark.asyncio 
async def test_dgm_state_retrieval():
    """Test DGM state retrieval"""
    config = {}
    engine = DGMEngine("test-agent", config)
    
    state = await engine.get_current_state()
    
    assert state.agent_id == "test-agent"
    assert state.generation == 0
    assert len(state.improvement_history) == 0
    assert len(state.active_experiments) == 0
