#!/usr/bin/env python3
"""
Test Dual Coding Agents - Ollama vs OpenRouter

This script creates two coding agents - one using Ollama and one using OpenRouter,
then tests them with the same coding question to compare their responses.
"""

import asyncio
import logging
import os
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_dual_coding_agents():
    """Create and test two coding agents from different providers."""
    try:
        logger.info("=== Dual Coding Agent Test ===")
        
        # Import the refactored components
        from src.core.agent_factory import AgentFactory
        
        logger.info("1. Initializing Agent Factory...")
        
        # Create agent factory
        agent_factory = AgentFactory()
        
        # Initialize with both providers
        logger.info("2. Initializing providers...")
        init_results = await agent_factory.initialize(
            enable_ollama=True,
            enable_openrouter=True,
            openrouter_config={
                "api_key": os.getenv("OPENROUTER_API_KEY")
            }
        )
        
        logger.info(f"Provider initialization results: {init_results}")
        
        # Check system readiness
        logger.info("3. Checking system readiness...")
        readiness = await agent_factory.get_system_readiness()
        ready_providers = [name for name, info in readiness['providers'].items() if info.get('ready', False)]
        logger.info(f"Ready providers: {ready_providers}")
        
        # Get available models for each provider
        logger.info("4. Getting available models...")
        models = await agent_factory.get_available_models()
        
        ollama_models = models.get('ollama', [])
        openrouter_models = models.get('openrouter', [])
        
        logger.info(f"Ollama models: {ollama_models}")
        logger.info(f"OpenRouter models (first 10): {openrouter_models[:10]}")
        
        # Select appropriate models
        # For Ollama, look for a DeepSeek model
        ollama_model = None
        for model in ollama_models:
            if 'deepseek' in model.lower():
                ollama_model = model
                break
        
        if not ollama_model and ollama_models:
            ollama_model = ollama_models[0]  # Use first available
        
        # For OpenRouter, use the free DeepSeek model
        openrouter_model = "deepseek/deepseek-r1-0528-qwen3-8b:free"
        
        logger.info(f"Selected Ollama model: {ollama_model}")
        logger.info(f"Selected OpenRouter model: {openrouter_model}")
        
        # Verify models are available
        if not ollama_model:
            logger.error("No Ollama models available")
            return False
        
        if 'openrouter' not in ready_providers:
            logger.error("OpenRouter not ready")
            return False
        
        # Create coding question
        coding_question = """
        Create a Python class called 'BinarySearchTree' that implements a binary search tree with the following methods:
        1. insert(value) - Insert a value into the tree
        2. search(value) - Search for a value and return True/False
        3. delete(value) - Delete a value from the tree
        4. inorder_traversal() - Return a list of values in inorder traversal
        
        Include proper error handling and docstrings. Make it clean and efficient.
        """
        
        logger.info("5. Creating Ollama coding agent...")
        
        # Create Ollama agent configuration
        ollama_config = {
            "provider": "ollama",
            "model_name": ollama_model,
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        # Validate Ollama config
        ollama_validation = await agent_factory.validate_agent_config("coding", ollama_config)
        logger.info(f"Ollama validation: {ollama_validation}")
        
        ollama_agent = None
        if ollama_validation["valid"]:
            try:
                ollama_agent = await agent_factory.create_agent(
                    agent_type="coding",
                    name="OllamaCodingAgent",
                    capabilities=["code_generation", "debugging"],
                    custom_config=ollama_config
                )
                logger.info(f"✅ Created Ollama coding agent: {ollama_agent.agent_id}")
            except Exception as e:
                logger.error(f"Failed to create Ollama agent: {e}")
        
        logger.info("6. Creating OpenRouter coding agent...")
        
        # Create OpenRouter agent configuration
        openrouter_config = {
            "provider": "openrouter",
            "model_name": openrouter_model,
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        # Validate OpenRouter config
        openrouter_validation = await agent_factory.validate_agent_config("coding", openrouter_config)
        logger.info(f"OpenRouter validation: {openrouter_validation}")
        
        openrouter_agent = None
        if openrouter_validation["valid"]:
            try:
                openrouter_agent = await agent_factory.create_agent(
                    agent_type="coding",
                    name="OpenRouterCodingAgent", 
                    capabilities=["code_generation", "debugging"],
                    custom_config=openrouter_config
                )
                logger.info(f"✅ Created OpenRouter coding agent: {openrouter_agent.agent_id}")
            except Exception as e:
                logger.error(f"Failed to create OpenRouter agent: {e}")
        
        # Test both agents with the same question
        results = {}
        
        if ollama_agent or openrouter_agent:
            logger.info("7. Testing agents with coding question...")
            logger.info(f"Question: {coding_question}")
            
            # Test Ollama agent
            if ollama_agent:
                logger.info("Testing Ollama agent...")
                try:
                    ollama_result = await agent_factory.provider_registry.generate_text(
                        provider_name="ollama",
                        model=ollama_model,
                        prompt=coding_question,
                        temperature=0.1,
                        max_tokens=2000
                    )
                    results["ollama"] = {
                        "model": ollama_model,
                        "response": ollama_result,
                        "success": bool(ollama_result)
                    }
                    logger.info("✅ Ollama agent responded successfully")
                except Exception as e:
                    logger.error(f"Ollama agent failed: {e}")
                    results["ollama"] = {
                        "model": ollama_model,
                        "response": "",
                        "success": False,
                        "error": str(e)
                    }
            
            # Test OpenRouter agent
            if openrouter_agent:
                logger.info("Testing OpenRouter agent...")
                try:
                    openrouter_result = await agent_factory.provider_registry.generate_text(
                        provider_name="openrouter",
                        model=openrouter_model,
                        prompt=coding_question,
                        temperature=0.1,
                        max_tokens=2000
                    )
                    results["openrouter"] = {
                        "model": openrouter_model,
                        "response": openrouter_result,
                        "success": bool(openrouter_result)
                    }
                    logger.info("✅ OpenRouter agent responded successfully")
                except Exception as e:
                    logger.error(f"OpenRouter agent failed: {e}")
                    results["openrouter"] = {
                        "model": openrouter_model,
                        "response": "",
                        "success": False,
                        "error": str(e)
                    }
        
        # Display results
        logger.info("8. Comparing results...")
        
        print("\n" + "="*80)
        print("CODING AGENT COMPARISON RESULTS")
        print("="*80)
        
        print(f"\nQUESTION:\n{coding_question}")
        
        for provider, result in results.items():
            print(f"\n{'-'*40}")
            print(f"{provider.upper()} AGENT RESPONSE")
            print(f"Model: {result['model']}")
            print(f"Success: {result['success']}")
            print(f"{'-'*40}")
            
            if result['success']:
                print(f"Response:\n{result['response']}")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        print(f"\n{'='*80}")
        
        # Save results to file
        with open("dual_agent_results.json", "w") as f:
            json.dump({
                "question": coding_question,
                "results": results,
                "timestamp": str(asyncio.get_event_loop().time())
            }, f, indent=2)
        
        logger.info("Results saved to dual_agent_results.json")
        
        # Cleanup (skip for now due to memory bug)
        logger.info("9. Skipping cleanup due to known memory bug...")
        # if ollama_agent:
        #     await agent_factory.destroy_agent(ollama_agent.agent_id)
        # if openrouter_agent:
        #     await agent_factory.destroy_agent(openrouter_agent.agent_id)
        
        # Shutdown
        await agent_factory.shutdown()
        
        logger.info("=== Dual Coding Agent Test Complete ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting dual coding agent test...")
    
    success = asyncio.run(test_dual_coding_agents())
    
    if success:
        print("\n✅ Dual coding agent test COMPLETED")
    else:
        print("\n❌ Dual coding agent test FAILED")
