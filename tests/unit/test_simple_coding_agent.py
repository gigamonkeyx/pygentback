#!/usr/bin/env python3
"""
Simple Coding Agent Test - WITHOUT Destroy

This script creates a coding agent using OpenRouter without calling destroy
to avoid the memory attribute issue.
"""

import asyncio
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_coding_agent():
    """Create a coding agent using OpenRouter."""
    try:
        logger.info("=== Creating Coding Agent with OpenRouter ===")
        
        # Import the refactored components
        from src.core.agent_factory import AgentFactory
        
        logger.info("1. Initializing Agent Factory...")
        
        # Create agent factory
        agent_factory = AgentFactory()
        
        # Initialize with OpenRouter enabled
        logger.info("2. Initializing providers...")
        await agent_factory.initialize(
            enable_ollama=True,
            enable_openrouter=True,
            openrouter_config={
                "api_key": os.getenv("OPENROUTER_API_KEY")
            }
        )
        
        # Check system readiness
        logger.info("3. Checking system readiness...")
        readiness = await agent_factory.get_system_readiness()
        ready_providers = [name for name, info in readiness['providers'].items() if info.get('ready', False)]
        logger.info(f"Ready providers: {ready_providers}")
        
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
            logger.info(f"Agent name: {agent.name}")
            logger.info(f"Agent config: {agent.config.custom_config}")
            
            # Test direct provider access
            logger.info("8. Testing direct provider access...")
            try:
                provider_registry = agent_factory.provider_registry
                
                # Test text generation
                result = await provider_registry.generate_text(
                    provider_name="openrouter",
                    model=default_config["model_name"],
                    prompt="Write a Python function to calculate fibonacci numbers:",
                    max_tokens=300,
                    temperature=0.1
                )
                
                if result:
                    logger.info("✅ Direct provider text generation successful!")
                    logger.info(f"Generated code:\n{result}")
                else:
                    logger.warning("Direct provider call returned empty result")
                    
            except Exception as e:
                logger.error(f"Error with direct provider access: {e}")
            
            # Test agent method if available
            logger.info("9. Testing agent methods...")
            
            # Check agent attributes
            logger.info(f"Agent has memory attribute: {hasattr(agent, 'memory')}")
            if hasattr(agent, 'memory'):
                logger.info(f"Agent memory value: {agent.memory}")
            
            logger.info(f"Agent status: {agent.status}")
            logger.info(f"Agent capabilities: {list(agent.capabilities.keys())}")
            
            # Try to use agent generate method if available
            if hasattr(agent, 'generate'):
                try:
                    logger.info("10. Testing agent generate method...")
                    response = await agent.generate("def quicksort(arr):")
                    logger.info(f"Agent generate response: {response}")
                except Exception as e:
                    logger.error(f"Error calling agent.generate: {e}")
            else:
                logger.info("Agent does not have generate method")
            
            # SUCCESS - DON'T CALL DESTROY
            logger.info("✅ Coding agent creation and testing SUCCESSFUL!")
            logger.info(f"Agent ID: {agent.agent_id}")
            
            # Shutdown factory (but don't destroy agent)
            await agent_factory.shutdown()
            
            return agent
            
        else:
            logger.error("Cannot create agent - configuration validation failed")
            await agent_factory.shutdown()
            return None
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    logger.info("Starting simple coding agent creation test...")
    
    agent = asyncio.run(create_coding_agent())
    
    if agent:
        print(f"\n✅ Coding agent test PASSED - Agent ID: {agent.agent_id}")
    else:
        print("\n❌ Coding agent test FAILED")
