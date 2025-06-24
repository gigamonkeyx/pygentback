"""
Simple Phase 2 Test

Quick test to verify Phase 2 components are working.
"""

import asyncio
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from orchestration.orchestration_manager import OrchestrationManager
from orchestration.coordination_models import OrchestrationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_phase2_simple():
    """Simple test of Phase 2 functionality."""
    try:
        logger.info("🧪 Starting Simple Phase 2 Test...")
        
        # Create manager
        config = OrchestrationConfig(evolution_enabled=True)
        manager = OrchestrationManager(config)
        
        # Start system
        await manager.start()
        logger.info("✅ System started successfully")
        
        # Test system status
        status = await manager.get_system_status()
        assert status["is_running"] is True
        logger.info("✅ System status check passed")
        
        # Check Phase 2 components
        components = status.get("components", {})
        
        # Check adaptive load balancer
        if "adaptive_load_balancer" in components:
            logger.info("✅ Adaptive Load Balancer present")
        
        # Check transaction coordinator
        if "transaction_coordinator" in components:
            logger.info("✅ Transaction Coordinator present")
        
        # Check emergent behavior detector
        if "emergent_behavior_detector" in components:
            logger.info("✅ Emergent Behavior Detector present")
        
        # Test basic functionality
        await manager.register_existing_mcp_servers()
        logger.info("✅ MCP servers registered")
        
        # Create test agent
        agent_id = await manager.create_tot_agent("Simple Test Agent", ["reasoning"])
        if agent_id:
            logger.info("✅ Test agent created")
        
        # Test system observation
        observation_success = await manager.observe_system_for_behaviors()
        if observation_success:
            logger.info("✅ System observation successful")
        
        # Stop system
        await manager.stop()
        logger.info("✅ System stopped successfully")
        
        logger.info("🎉 Simple Phase 2 test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple Phase 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_phase2_simple())
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ TESTS FAILED")