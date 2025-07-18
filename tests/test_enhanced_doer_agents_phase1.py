#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Doer Agents Phase 1 Tests
Observer-approved validation for Ollama Llama3 8B base configuration
"""

import asyncio
import pytest
import logging
from typing import Dict, Any

# Test imports - Fix import paths
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.agent_factory import AgentFactory
from src.core.ollama_integration import OllamaManager
from src.config.settings import Settings
from src.agents.coding_agent import CodingAgent
from src.agents.reasoning_agent import ReasoningAgent

logger = logging.getLogger(__name__)

class TestEnhancedDoerAgentsPhase1:
    """Phase 1 validation tests for enhanced doer agents with Llama3 8B"""
    
    @pytest.fixture
    async def agent_factory(self):
        """Create agent factory for testing"""
        settings = Settings()
        factory = AgentFactory(settings=settings)
        await factory.initialize()
        yield factory
        await factory.shutdown()
    
    @pytest.fixture
    async def ollama_manager(self):
        """Create Ollama manager for testing"""
        manager = OllamaManager()
        await manager.initialize()
        return manager
    
    async def test_ollama_llama3_8b_configuration(self, ollama_manager):
        """Test Ollama Llama3 8B model configuration"""
        # Check if Llama3 8B is in available models
        available_models = ollama_manager.available_models
        assert "llama3:8b" in available_models, "Llama3 8B not configured in available models"
        
        # Validate model properties
        llama3_model = available_models["llama3:8b"]
        assert llama3_model.name == "llama3:8b"
        assert llama3_model.size_gb == 4.7
        assert "CODE_ANALYSIS" in [cap.name for cap in llama3_model.capabilities]
        assert "REASONING" in [cap.name for cap in llama3_model.capabilities]
        assert llama3_model.context_length == 8192
        
        logger.info("✅ Llama3 8B configuration validated")
    
    async def test_task_model_assignments(self, ollama_manager):
        """Test that task models are properly assigned to Llama3 8B"""
        task_models = ollama_manager.task_models
        
        # Check enhanced assignments
        assert task_models["CODE_ANALYSIS"] == "llama3:8b", "Code analysis not assigned to Llama3 8B"
        assert task_models["REASONING"] == "llama3:8b", "Reasoning not assigned to Llama3 8B"
        assert task_models["FAST_RESPONSE"] == "llama3:8b", "Fast response not assigned to Llama3 8B"
        
        logger.info("✅ Task model assignments validated")
    
    async def test_enhanced_agent_spawn_coding(self, agent_factory):
        """Test enhanced coding agent spawning with Llama3 8B"""
        try:
            # Spawn enhanced coding agent
            coding_agent = await agent_factory.enhanced_agent_spawn(
                agent_type="coding",
                name="test_coding_agent",
                use_llama3_8b=True,
                enable_augmentation=True
            )
            
            # Validate agent configuration
            assert coding_agent is not None, "Coding agent not created"
            assert coding_agent.config.agent_type == "coding"
            assert coding_agent.config.custom_config.get("model_name") == "llama3:8b"
            assert coding_agent.config.custom_config.get("augmentation_enabled") is True
            assert coding_agent.config.custom_config.get("rag_enabled") is True
            
            # Validate capabilities
            expected_capabilities = ["code_generation", "code_analysis", "debugging", "documentation"]
            for capability in expected_capabilities:
                assert capability in coding_agent.config.enabled_capabilities
            
            logger.info("✅ Enhanced coding agent spawn validated")
            
        except Exception as e:
            logger.error(f"Enhanced coding agent spawn failed: {e}")
            raise
    
    async def test_enhanced_agent_spawn_research(self, agent_factory):
        """Test enhanced research agent spawning with Llama3 8B"""
        try:
            # Spawn enhanced research agent
            research_agent = await agent_factory.enhanced_agent_spawn(
                agent_type="research",
                name="test_research_agent",
                use_llama3_8b=True,
                enable_augmentation=True
            )
            
            # Validate agent configuration
            assert research_agent is not None, "Research agent not created"
            assert research_agent.config.agent_type == "research"
            assert research_agent.config.custom_config.get("model_name") == "llama3:8b"
            assert research_agent.config.custom_config.get("augmentation_enabled") is True
            
            # Validate capabilities
            expected_capabilities = ["research", "analysis", "synthesis", "fact_checking"]
            for capability in expected_capabilities:
                assert capability in research_agent.config.enabled_capabilities
            
            logger.info("✅ Enhanced research agent spawn validated")
            
        except Exception as e:
            logger.error(f"Enhanced research agent spawn failed: {e}")
            raise
    
    async def test_basic_code_generation(self, agent_factory):
        """Test basic code generation with Llama3 8B"""
        try:
            # Create coding agent
            coding_agent = await agent_factory.enhanced_agent_spawn(
                agent_type="coding",
                name="test_code_gen",
                use_llama3_8b=True
            )
            
            # Test basic code generation
            test_prompt = "write hello world in Python"
            
            # This would normally call the agent's generate method
            # For now, just validate the agent is properly configured
            assert hasattr(coding_agent, 'ollama_backend'), "Coding agent missing Ollama backend"
            assert coding_agent.model_name == "llama3:8b", "Coding agent not using Llama3 8B"
            
            logger.info("✅ Basic code generation setup validated")
            
        except Exception as e:
            logger.error(f"Basic code generation test failed: {e}")
            raise
    
    async def test_model_health_check(self, ollama_manager):
        """Test model health check and fallback mechanisms"""
        try:
            # Test model availability check
            is_available = await ollama_manager.is_model_available("llama3:8b")
            
            if not is_available:
                logger.warning("Llama3 8B not available, testing fallback mechanism")
                
                # Test fallback to available models
                available_models = await ollama_manager.list_models()
                assert len(available_models) > 0, "No fallback models available"
                
                logger.info(f"Fallback models available: {available_models}")
            else:
                logger.info("✅ Llama3 8B model health check passed")
            
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            raise
    
    async def test_settings_configuration(self):
        """Test that settings are properly configured for Llama3 8B"""
        settings = Settings()
        
        # Check Ollama configuration
        assert settings.OLLAMA_MODEL == "llama3:8b", "Default Ollama model not set to Llama3 8B"
        assert settings.OLLAMA_EMBED_MODEL == "llama3:8b", "Ollama embed model not set to Llama3 8B"
        assert settings.OLLAMA_BASE_URL == "http://localhost:11434", "Ollama base URL incorrect"
        
        logger.info("✅ Settings configuration validated")


async def run_phase1_validation():
    """Run Phase 1 validation tests"""
    print("\n🚀 PHASE 1 VALIDATION: Ollama Llama3 8B Base Configuration")
    print("=" * 60)
    
    try:
        test_instance = TestEnhancedDoerAgentsPhase1()
        
        # Test 1: Settings configuration
        await test_instance.test_settings_configuration()
        
        # Test 2: Ollama manager setup
        ollama_manager = OllamaManager()
        await ollama_manager.initialize()
        await test_instance.test_ollama_llama3_8b_configuration(ollama_manager)
        await test_instance.test_task_model_assignments(ollama_manager)
        await test_instance.test_model_health_check(ollama_manager)
        
        # Test 3: Agent factory setup
        settings = Settings()
        agent_factory = AgentFactory(settings=settings)
        await agent_factory.initialize()
        
        await test_instance.test_enhanced_agent_spawn_coding(agent_factory)
        await test_instance.test_enhanced_agent_spawn_research(agent_factory)
        await test_instance.test_basic_code_generation(agent_factory)
        
        await agent_factory.shutdown()
        
        print("\n✅ PHASE 1 VALIDATION COMPLETE")
        print("Observer Checkpoint: Ollama Llama3 8B configuration validated")
        print("Ready for Phase 2: RAG Augmentation Integration")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 1 VALIDATION FAILED: {e}")
        logger.error(f"Phase 1 validation error: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(run_phase1_validation())
