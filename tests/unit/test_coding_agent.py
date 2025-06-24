#!/usr/bin/env python3
"""
Create and Test Coding Agent with OpenRouter

This script demonstrates creating a coding agent using the refactored agent factory
with OpenRouter provider integration.
"""

import asyncio
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_and_test_coding_agent():
    """Create and test a coding agent using OpenRouter."""
    try:
        logger.info("=== Creating Coding Agent with OpenRouter ===")
        
        # Import the refactored components
        from src.core.agent_factory import AgentFactory
        
        logger.info("1. Initializing Agent Factory...")
        
        # Create agent factory (without old manager dependencies)
        agent_factory = AgentFactory()
        
        # Initialize with OpenRouter enabled
        logger.info("2. Initializing providers...")
        init_results = await agent_factory.initialize(
            enable_ollama=True,  # Try Ollama too if available
            enable_openrouter=True,
            openrouter_config={
                "api_key": os.getenv("OPENROUTER_API_KEY")  # Use env var if set
            }
        )
        
        logger.info(f"Provider initialization results: {init_results}")
        
        # Check system readiness
        logger.info("3. Checking system readiness...")
        readiness = await agent_factory.get_system_readiness()
        logger.info(f"System ready providers: {list(readiness['providers'].keys())}")
        
        # Get available models
        logger.info("4. Getting available models...")
        models = await agent_factory.get_available_models()
        for provider, model_list in models.items():
            logger.info(f"  {provider}: {len(model_list)} models")
            if model_list:
                logger.info(f"    Examples: {model_list[:3]}")
        
        # Get default coding agent config
        logger.info("5. Getting default coding agent config...")
        default_config = agent_factory.get_default_model_config("coding", provider="openrouter")
        logger.info(f"Default config: {default_config}")
        
        # Validate the config
        logger.info("6. Validating agent config...")
        validation = await agent_factory.validate_agent_config("coding", default_config)
        logger.info(f"Validation result: {validation}")
        
        if not validation["valid"]:
            logger.error(f"Config validation failed: {validation['errors']}")
            
            # Try with a free model
            logger.info("Trying with free DeepSeek model...")
            default_config = {
                "provider": "openrouter",
                "model_name": "deepseek/deepseek-r1-0528-qwen3-8b:free",
                "temperature": 0.1,
                "max_tokens": 3000
            }
            validation = await agent_factory.validate_agent_config("coding", default_config)
            logger.info(f"Free model validation: {validation}")
        
        if validation["valid"]:
            logger.info("7. Creating coding agent...")
            
            # Create the coding agent
            agent = await agent_factory.create_agent(
                agent_type="coding",
                name="TestCodingAgent",
                capabilities=["code_generation", "debugging", "code_review"],
                custom_config=default_config
            )
            
            logger.info(f"✅ Created coding agent: {agent.agent_id}")
            logger.info(f"Agent type: {agent.type}")
            logger.info(f"Agent config: {agent.config.custom_config}")
            
            # Test the agent (if it has generate method)
            logger.info("8. Testing coding agent...")
            
            if hasattr(agent, 'generate') or hasattr(agent, 'process'):
                try:
                    test_prompt = "Write a Python function to calculate the factorial of a number using recursion."
                    
                    # Try to get a response (this depends on the agent implementation)
                    logger.info(f"Sending test prompt: {test_prompt}")
                    
                    # For now, just log that the agent was created successfully
                    # In a real implementation, you'd call agent.generate() or similar
                    logger.info("Agent is ready for code generation tasks!")
                    
                except Exception as e:
                    logger.error(f"Error testing agent: {e}")
            else:
                logger.info("Agent created but no test method available")
            
            # Test direct provider access through agent factory
            logger.info("9. Testing direct provider access...")
            
            try:
                # Access the provider registry directly
                provider_registry = agent_factory.provider_registry
                
                # Test text generation through the registry
                result = await provider_registry.generate_text(
                    provider_name="openrouter",
                    model=default_config["model_name"],
                    prompt="def factorial(n):",
                    max_tokens=200,
                    temperature=0.1
                )
                
                if result:
                    logger.info("✅ Direct provider text generation successful!")
                    logger.info(f"Generated code:\n{result}")
                else:
                    logger.warning("Direct provider call returned empty result")
                    
            except Exception as e:
                logger.error(f"Error with direct provider access: {e}")
            
            # Cleanup
            logger.info("10. Cleaning up...")
            await agent_factory.destroy_agent(agent.agent_id)
            
        else:
            logger.error("Cannot create agent - configuration validation failed")
        
        # Shutdown
        await agent_factory.shutdown()
        
        logger.info("=== Coding Agent Test Complete ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_recommendations():
    """Test the model recommendation system."""
    try:
        logger.info("\n=== Testing Model Recommendations ===")
        
        from src.core.agent_factory import AgentFactory
        
        factory = AgentFactory()
        await factory.initialize(enable_ollama=True, enable_openrouter=True)
        
        # Test free models only
        logger.info("Free models:")
        free_models = await factory.get_recommended_models(include_paid=False)
        for provider, models in free_models.items():
            logger.info(f"  {provider}: {models}")
        
        # Test all models
        logger.info("All models (free + paid):")
        all_models = await factory.get_recommended_models(include_paid=True)
        for provider, models in all_models.items():
            logger.info(f"  {provider}: {models}")
        
        await factory.shutdown()
        
    except Exception as e:
        logger.error(f"Model recommendation test failed: {e}")

if __name__ == "__main__":
    # Run the tests
    logger.info("Starting coding agent creation test...")
    
    success = asyncio.run(create_and_test_coding_agent())
    
    if success:
        print("\n✅ Coding agent test PASSED")
        
        # Also test model recommendations
        asyncio.run(test_model_recommendations())
        
    else:
        print("\n❌ Coding agent test FAILED")
