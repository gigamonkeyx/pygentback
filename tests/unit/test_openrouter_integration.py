#!/usr/bin/env python3
"""
Test OpenRouter Integration

This script tests the OpenRouter provider integration to ensure
it works correctly with the agent factory and reasoning agents.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.agent_factory import AgentFactory
from src.core.ollama_manager import get_ollama_manager
from src.ai.providers.openrouter_provider import get_openrouter_manager, OpenRouterBackend
from src.config.settings import get_settings


async def test_openrouter_backend():
    """Test OpenRouter backend directly"""
    print("🧪 Testing OpenRouter Backend...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return False
    
    try:
        backend = OpenRouterBackend(
            model_name="anthropic/claude-3-haiku",  # Use a fast, cheap model for testing
            api_key=api_key
        )
        
        response = await backend.generate("Say 'Hello from OpenRouter!' and nothing else.")
        
        if response and "Hello from OpenRouter" in response:
            print("✅ OpenRouter backend working")
            print(f"   Response: {response}")
            return True
        else:
            print("❌ OpenRouter backend failed")
            print(f"   Response: {response}")
            return False
    
    except Exception as e:
        print(f"❌ OpenRouter backend error: {e}")
        return False


async def test_openrouter_manager():
    """Test OpenRouter manager"""
    print("🧪 Testing OpenRouter Manager...")
    
    try:
        settings = get_settings()
        manager = get_openrouter_manager(settings)
        
        # Test connectivity
        if await manager.start():
            print("✅ OpenRouter manager connected")
            
            # Test model availability
            models = await manager.get_available_models()
            print(f"   Available models: {len(models)}")
            
            # Test popular models
            popular = manager.get_popular_models()
            print(f"   Popular models: {list(popular.keys())[:3]}")
            
            return True
        else:
            print("❌ OpenRouter manager failed to connect")
            return False
    
    except Exception as e:
        print(f"❌ OpenRouter manager error: {e}")
        return False


async def test_agent_creation_with_openrouter():
    """Test creating agents with OpenRouter"""
    print("🧪 Testing Agent Creation with OpenRouter...")
    
    try:
        # Initialize services
        settings = get_settings()
        ollama_manager = get_ollama_manager(settings)
        openrouter_manager = get_openrouter_manager(settings)
        
        agent_factory = AgentFactory(
            settings=settings,
            ollama_manager=ollama_manager,
            openrouter_manager=openrouter_manager
        )
        
        # Start services
        await ollama_manager.start()
        if not await openrouter_manager.start():
            print("❌ OpenRouter not available for agent testing")
            return False
        
        # Configuration for OpenRouter-based reasoning agent
        custom_config = {
            "provider": "openrouter",
            "model_name": "anthropic/claude-3-haiku",
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        # Validate configuration
        validation = await agent_factory.validate_agent_config_before_creation(
            "reasoning", 
            custom_config
        )
        
        if not validation["valid"]:
            print("❌ Agent configuration validation failed:")
            for error in validation["errors"]:
                print(f"   • {error}")
            return False
        
        # Create the agent
        agent = await agent_factory.create_agent(
            agent_type="reasoning",
            name="TestReasoningAgent-OpenRouter",
            custom_config=custom_config
        )
        
        print(f"✅ Created reasoning agent: {agent.agent_id}")
        print(f"   Model: {custom_config['model_name']}")
        print("   Provider: OpenRouter")
        
        # Test basic functionality
        from src.core.agent import AgentMessage, MessageType
        
        test_message = AgentMessage(
            type=MessageType.REQUEST,
            sender="test",
            recipient=agent.agent_id,
            content={"content": "What is 2+2? Answer briefly."}
        )
        
        response = await agent.process_message(test_message)
        
        if response and response.content:
            print("✅ Agent processed message successfully")
            print(f"   Response: {response.content}")
        else:
            print("❌ Agent failed to process message")
            return False
        
        # Cleanup
        await agent_factory.destroy_agent(agent.agent_id)
        print("✅ Agent cleaned up successfully")
        
        return True
    
    except Exception as e:
        print(f"❌ Agent creation test error: {e}")
        return False


async def test_system_readiness():
    """Test system readiness with OpenRouter"""
    print("🧪 Testing System Readiness...")
    
    try:
        # Initialize services
        settings = get_settings()
        ollama_manager = get_ollama_manager(settings)
        openrouter_manager = get_openrouter_manager(settings)
        
        agent_factory = AgentFactory(
            settings=settings,
            ollama_manager=ollama_manager,
            openrouter_manager=openrouter_manager
        )
        
        # Start services
        await ollama_manager.start()
        await openrouter_manager.start()
        
        # Get system readiness
        readiness = await agent_factory.get_system_readiness()
        
        print("✅ System readiness check completed")
        print(f"   Ollama: {'Ready' if readiness['providers']['ollama']['available'] else 'Not Ready'}")
        print(f"   OpenRouter: {'Ready' if readiness['providers']['openrouter']['available'] else 'Not Ready'}")
        
        # Test agent creation guide
        guide = await agent_factory.get_agent_creation_guide("reasoning")
        
        print(f"   Reasoning agents supported: {'Yes' if guide['system_ready'] else 'No'}")
        
        if guide["configuration_examples"]:
            print("   Available configurations:")
            for provider, config in guide["configuration_examples"].items():
                print(f"     {provider}: {config['model_name']}")
        
        return True
    
    except Exception as e:
        print(f"❌ System readiness test error: {e}")
        return False


async def run_all_tests():
    """Run all OpenRouter tests"""
    print("🚀 OpenRouter Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("OpenRouter Backend", test_openrouter_backend),
        ("OpenRouter Manager", test_openrouter_manager),
        ("System Readiness", test_system_readiness),
        ("Agent Creation", test_agent_creation_with_openrouter),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 Test Results")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Summary: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! OpenRouter integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check your OpenRouter configuration.")
        
        if not os.getenv("OPENROUTER_API_KEY"):
            print("\n💡 Missing OPENROUTER_API_KEY environment variable")
            print("   Set it with: export OPENROUTER_API_KEY=your_key_here")


async def main():
    """Main test runner"""
    print("OpenRouter Integration Test")
    print("This will test the OpenRouter provider integration.")
    print()
    
    # Check prerequisites
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  Warning: OPENROUTER_API_KEY not found in environment")
        print("Some tests may fail without an API key.")
        print()
    
    await run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
