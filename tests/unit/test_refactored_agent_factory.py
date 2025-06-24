#!/usr/bin/env python3
"""
Test Agent Factory Refactoring

This script tests the refactored agent factory with provider registry.
"""

import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_refactored_agent_factory():
    """Test the refactored agent factory."""
    try:        # Import the refactored agent factory
        from src.core.agent_factory import AgentFactory
        
        logger.info("=== Testing Refactored Agent Factory ===")
        
        # Create agent factory
        factory = AgentFactory()
        
        # Initialize with providers
        logger.info("Initializing providers...")
        await factory.initialize(
            enable_ollama=True,
            enable_openrouter=True
        )
        
        # Test system readiness
        logger.info("Checking system readiness...")
        readiness = await factory.get_system_readiness()
        logger.info(f"System ready: {readiness['providers']}")
        
        # Test available models
        logger.info("Getting available models...")
        models = await factory.get_available_models()
        for provider, model_list in models.items():
            logger.info(f"{provider}: {len(model_list)} models available")
            if model_list:
                logger.info(f"  First few: {model_list[:3]}")
        
        # Test model recommendations
        logger.info("Getting model recommendations...")
        recommendations = await factory.get_recommended_models(include_paid=False)
        logger.info(f"Free model recommendations: {recommendations}")
        
        # Test default config
        logger.info("Getting default configs...")
        for agent_type in ["reasoning", "coding", "general"]:
            config = factory.get_default_model_config(agent_type)
            logger.info(f"{agent_type}: {config}")
        
        # Test health check
        logger.info("Performing health check...")
        health = await factory.health_check()
        logger.info(f"Health status: {health}")
        
        # Test validation
        logger.info("Testing config validation...")
        test_config = {
            "provider": "openrouter",
            "model_name": "deepseek/deepseek-r1-0528-qwen3-8b:free"
        }
        validation = await factory.validate_agent_config("reasoning", test_config)
        logger.info(f"Validation result: {validation}")
        
        # Cleanup
        await factory.shutdown()
        
        logger.info("=== Refactoring Test Complete ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_refactored_agent_factory())
    if success:
        print("✅ Agent Factory refactoring test PASSED")
    else:
        print("❌ Agent Factory refactoring test FAILED")
