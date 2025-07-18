#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Base Agent Test - Phase 1.2
Observer-approved validation for enhanced agent base class with augmentation hooks
"""

import asyncio
import logging
from typing import Dict, Any

# Fix import paths
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.agent.base import BaseAgent
from src.core.agent.config import AgentConfig
from src.core.agent.status import AgentStatus

logger = logging.getLogger(__name__)

class TestEnhancedAgent(BaseAgent):
    """Test implementation of enhanced base agent"""
    
    async def _agent_initialize(self) -> None:
        """Test agent initialization"""
        self.logger.info("Test agent initialized")
    
    async def _agent_shutdown(self) -> None:
        """Test agent shutdown"""
        self.logger.info("Test agent shutdown")
    
    async def _handle_request(self, message) -> Any:
        """Test request handler"""
        return f"Test response to: {message.content}"
    
    async def _handle_tool_call(self, message) -> Any:
        """Test tool call handler"""
        return f"Test tool call: {message.content}"
    
    async def _handle_capability_request(self, message) -> Any:
        """Test capability request handler"""
        return {"capabilities": ["test"]}
    
    async def _handle_notification(self, message) -> Any:
        """Test notification handler"""
        return "Notification received"


class TestEnhancedBaseAgent:
    """Test suite for enhanced base agent functionality"""
    
    async def test_basic_agent_creation(self):
        """Test basic enhanced agent creation"""
        config = AgentConfig(
            agent_id="test_agent_001",
            name="Test Enhanced Agent",
            agent_type="test",
            custom_config={
                "augmentation_enabled": True,
                "rag_enabled": True,
                "lora_enabled": False,
                "riper_omega_enabled": False,
                "cooperative_enabled": False
            }
        )
        
        agent = TestEnhancedAgent(config)
        
        # Validate augmentation configuration
        assert agent.augmentation_enabled == True, "Augmentation should be enabled"
        assert agent.rag_enabled == True, "RAG should be enabled"
        assert agent.lora_enabled == False, "LoRA should be disabled"
        assert agent.riper_omega_enabled == False, "RIPER-Ω should be disabled"
        assert agent.cooperative_enabled == False, "Cooperative should be disabled"
        
        # Validate augmentation components are initialized as None
        assert agent.rag_augmenter is None, "RAG augmenter should be None initially"
        assert agent.lora_adapter is None, "LoRA adapter should be None initially"
        assert agent.riper_omega_manager is None, "RIPER-Ω manager should be None initially"
        assert agent.cooperation_manager is None, "Cooperation manager should be None initially"
        
        logger.info("Basic enhanced agent creation test passed")
        return True
    
    async def test_augmentation_initialization(self):
        """Test augmentation initialization process"""
        config = AgentConfig(
            agent_id="test_agent_002",
            name="Test Augmentation Agent",
            agent_type="test",
            custom_config={
                "augmentation_enabled": True,
                "rag_enabled": True,
                "lora_enabled": True,
                "riper_omega_enabled": True,
                "cooperative_enabled": True
            }
        )
        
        agent = TestEnhancedAgent(config)
        
        # Initialize the agent (this will call _initialize_augmentations)
        await agent.initialize()
        
        # Validate agent is active
        assert agent.status == AgentStatus.ACTIVE, "Agent should be active after initialization"
        
        # Validate augmentation metrics are initialized
        metrics = agent.get_augmentation_metrics()
        assert metrics["total_requests"] == 0, "Total requests should be 0 initially"
        assert metrics["augmented_requests"] == 0, "Augmented requests should be 0 initially"
        assert metrics["augmentation_rate"] == 0.0, "Augmentation rate should be 0.0 initially"
        assert metrics["augmentation_enabled"] == True, "Augmentation should be enabled in metrics"
        
        await agent.shutdown()
        
        logger.info("✅ Augmentation initialization test passed")
        return True
    
    async def test_augmented_generation(self):
        """Test augmented generation functionality"""
        config = AgentConfig(
            agent_id="test_agent_003",
            name="Test Generation Agent",
            agent_type="test",
            custom_config={
                "augmentation_enabled": True,
                "rag_enabled": True,
                "lora_enabled": False,
                "riper_omega_enabled": False,
                "cooperative_enabled": False
            }
        )
        
        agent = TestEnhancedAgent(config)
        await agent.initialize()
        
        # Test augmented generation
        test_prompt = "write hello world in Python"
        result = await agent._augmented_generate(test_prompt)
        
        # Validate result
        assert result.startswith("[Augmented]"), "Result should be augmented"
        assert test_prompt in result, "Original prompt should be in result"
        
        # Check metrics
        metrics = agent.get_augmentation_metrics()
        assert metrics["total_requests"] == 1, "Total requests should be 1"
        assert metrics["rag_retrievals"] == 1, "RAG retrievals should be 1"
        
        await agent.shutdown()
        
        logger.info("✅ Augmented generation test passed")
        return True
    
    async def test_augmentation_metrics_tracking(self):
        """Test augmentation metrics tracking"""
        config = AgentConfig(
            agent_id="test_agent_004",
            name="Test Metrics Agent",
            agent_type="test",
            custom_config={
                "augmentation_enabled": True,
                "rag_enabled": True,
                "lora_enabled": True,
                "riper_omega_enabled": True,
                "cooperative_enabled": True
            }
        )
        
        agent = TestEnhancedAgent(config)
        await agent.initialize()
        
        # Perform multiple augmented generations
        for i in range(5):
            await agent._augmented_generate(f"test prompt {i}")
        
        # Check metrics
        metrics = agent.get_augmentation_metrics()
        assert metrics["total_requests"] == 5, "Total requests should be 5"
        assert metrics["augmented_requests"] == 5, "All requests should be augmented"
        assert metrics["augmentation_rate"] == 1.0, "Augmentation rate should be 100%"
        assert metrics["rag_retrievals"] == 5, "RAG retrievals should be 5"
        assert metrics["lora_adaptations"] == 5, "LoRA adaptations should be 5"
        assert metrics["riper_omega_chains"] == 5, "RIPER-Ω chains should be 5"
        assert metrics["cooperative_actions"] == 5, "Cooperative actions should be 5"
        
        await agent.shutdown()
        
        logger.info("✅ Augmentation metrics tracking test passed")
        return True
    
    async def test_augmentation_disabled(self):
        """Test behavior when augmentation is disabled"""
        config = AgentConfig(
            agent_id="test_agent_005",
            name="Test Disabled Agent",
            agent_type="test",
            custom_config={
                "augmentation_enabled": False
            }
        )
        
        agent = TestEnhancedAgent(config)
        await agent.initialize()
        
        # Test generation without augmentation
        test_prompt = "write hello world in Python"
        result = await agent._augmented_generate(test_prompt)
        
        # Should still work but without augmentation
        assert result == test_prompt, "Result should be unchanged when augmentation disabled"
        
        # Check metrics
        metrics = agent.get_augmentation_metrics()
        assert metrics["augmentation_enabled"] == False, "Augmentation should be disabled"
        assert metrics["augmentation_rate"] == 0.0, "Augmentation rate should be 0%"
        
        await agent.shutdown()
        
        logger.info("✅ Augmentation disabled test passed")
        return True


async def run_enhanced_base_agent_tests():
    """Run enhanced base agent tests"""
    print("\n🚀 PHASE 1.2 VALIDATION: Enhanced Agent Base Class Integration")
    print("=" * 70)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_instance = TestEnhancedBaseAgent()
    results = {}
    
    try:
        # Test 1: Basic agent creation
        print("\n1. Testing enhanced agent creation...")
        results['agent_creation'] = await test_instance.test_basic_agent_creation()
        
        # Test 2: Augmentation initialization
        print("\n2. Testing augmentation initialization...")
        results['augmentation_init'] = await test_instance.test_augmentation_initialization()
        
        # Test 3: Augmented generation
        print("\n3. Testing augmented generation...")
        results['augmented_generation'] = await test_instance.test_augmented_generation()
        
        # Test 4: Metrics tracking
        print("\n4. Testing augmentation metrics tracking...")
        results['metrics_tracking'] = await test_instance.test_augmentation_metrics_tracking()
        
        # Test 5: Disabled augmentation
        print("\n5. Testing disabled augmentation behavior...")
        results['disabled_augmentation'] = await test_instance.test_augmentation_disabled()
        
        # Summary
        print("\n" + "=" * 70)
        print("PHASE 1.2 ENHANCED BASE AGENT VALIDATION RESULTS:")
        print("=" * 70)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result)
        success_rate = passed_tests / total_tests
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
        
        if success_rate >= 1.0:  # 100% success required
            print("\n🎉 PHASE 1.2 VALIDATION: SUCCESS")
            print("✅ Enhanced base agent class with augmentation hooks ready")
            print("✅ Observer Checkpoint: Base class enhancement validated")
            print("🚀 Ready to proceed to Phase 2: RAG Augmentation Integration")
            return True
        else:
            print("\n⚠️ PHASE 1.2 VALIDATION: FAILED")
            print("Some tests failed. Please review and fix issues before proceeding.")
            return False
            
    except Exception as e:
        print(f"\n❌ PHASE 1.2 VALIDATION FAILED: {e}")
        logger.error(f"Phase 1.2 validation error: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(run_enhanced_base_agent_tests())
