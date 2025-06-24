#!/usr/bin/env python3
"""
PyGent Factory System Checker

Comprehensive system status checker to help users understand what's
available and what needs to be configured. This is crucial without a UI.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.agent_factory import AgentFactory
from src.ai.providers.openrouter_provider import get_openrouter_manager
from src.core.ollama_manager import get_ollama_manager


async def check_system_status():
    """Check comprehensive system status."""
    print("🔍 PyGent Factory System Status Check")
    print("=" * 60)
    
    # Initialize managers
    print("1. Initializing managers...")
    
    # OpenRouter with your API key
    openrouter_manager = get_openrouter_manager(
        api_key="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"
    )
    
    # Ollama
    ollama_manager = get_ollama_manager()
    
    print("\n2. Testing providers...")
    
    # Test OpenRouter
    print("   🌐 OpenRouter:")
    openrouter_success = await openrouter_manager.start()
    if openrouter_success:
        models = await openrouter_manager.get_available_models()
        print(f"      ✅ Ready - {len(models)} models available")
        popular = openrouter_manager.get_popular_models()
        print(f"      🔝 Popular: {', '.join(list(popular.keys())[:3])}")
    else:
        print("      ❌ Not available")
    
    # Test Ollama
    print("   🏠 Ollama:")
    ollama_success = await ollama_manager.start()
    if ollama_success:
        models = await ollama_manager.get_available_models()
        print(f"      ✅ Ready - {len(models)} models available")
        if models:
            print(f"      🔝 Models: {', '.join(models[:3])}")
        else:
            print("      ⚠️ No models downloaded")
    else:
        print("      ❌ Not available")
    
    print("\n3. Creating Agent Factory...")
    
    # Create agent factory with both managers
    factory = AgentFactory(
        ollama_manager=ollama_manager if ollama_success else None,
        openrouter_manager=openrouter_manager if openrouter_success else None
    )
    
    await factory.initialize()
    
    # Check system readiness
    print("\n4. System Readiness Assessment...")
    readiness = await factory.get_system_readiness()
    
    print("   📊 Providers:")
    for provider_name, info in readiness["providers"].items():
        status_icon = "✅" if info["available"] else "❌"
        print(f"      {status_icon} {provider_name.title()}: {info['status']} ({info['models_count']} models)")
    
    print("\n   🤖 Agent Types Supported:")
    for agent_type, support in readiness["agent_types_supported"].items():
        status_icon = "✅" if support["supported"] else "❌"
        print(f"      {status_icon} {agent_type.title()}: {'Ready' if support['supported'] else 'Not Ready'}")
        
        if not support["supported"]:
            reqs = support["requirements"]
            missing = [req for req, needed in reqs.items() if needed]
            print(f"         Needs: {', '.join(missing)}")
    
    print("\n   💡 Recommendations:")
    for rec in readiness["recommendations"]:
        print(f"      • {rec}")
    
    # Test agent creation
    print("\n5. Testing Agent Creation...")
    
    test_configs = []
    
    if openrouter_success:
        test_configs.append({
            "name": "OpenRouter Reasoning Agent",
            "agent_type": "reasoning", 
            "custom_config": {
                "provider": "openrouter",
                "model_name": "anthropic/claude-3.5-sonnet"
            }
        })
    
    if ollama_success and await ollama_manager.get_available_models():
        models = await ollama_manager.get_available_models()
        test_configs.append({
            "name": "Ollama Reasoning Agent",
            "agent_type": "reasoning",
            "custom_config": {
                "provider": "ollama", 
                "model_name": models[0]
            }
        })
    
    if not test_configs:
        print("   ⚠️ No providers available for testing agent creation")
    else:
        for config in test_configs:
            print(f"   🧪 Testing: {config['name']}")
            try:
                # Validate configuration before creation
                validation = await factory.validate_agent_config_before_creation(
                    config["agent_type"], 
                    config["custom_config"]
                )
                
                if validation["valid"]:
                    print(f"      ✅ Configuration valid")
                    
                    # Actually create the agent
                    agent = await factory.create_agent(**config)
                    print(f"      ✅ Agent created: {agent.agent_id}")
                    
                    # Clean up
                    await factory.destroy_agent(agent.agent_id)
                    print(f"      🧹 Agent cleaned up")
                
                else:
                    print(f"      ❌ Configuration invalid:")
                    for error in validation["errors"]:
                        print(f"         • {error}")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 SYSTEM STATUS SUMMARY")
    print("=" * 60)
    
    total_providers = 2
    ready_providers = sum([openrouter_success, ollama_success])
    
    total_agents = len(readiness["agent_types_supported"])
    ready_agents = sum(1 for support in readiness["agent_types_supported"].values() if support["supported"])
    
    print(f"📊 Providers: {ready_providers}/{total_providers} ready")
    print(f"🤖 Agent Types: {ready_agents}/{total_agents} supported")
    
    if ready_providers > 0:
        print("✅ System is operational!")
        print("\n📋 Quick Start Guide:")
        
        if openrouter_success:
            print("   🌐 For OpenRouter agents:")
            print("      config = {'provider': 'openrouter', 'model_name': 'anthropic/claude-3.5-sonnet'}")
            print("      agent = await factory.create_agent('reasoning', custom_config=config)")
        
        if ollama_success:
            models = await ollama_manager.get_available_models()
            if models:
                print("   🏠 For Ollama agents:")
                print(f"      config = {{'provider': 'ollama', 'model_name': '{models[0]}'}}")
                print("      agent = await factory.create_agent('reasoning', custom_config=config)")
    else:
        print("❌ System needs configuration!")
        print("\n🛠️ Setup Required:")
        if not openrouter_success:
            print("   • Configure OpenRouter API key")
        if not ollama_success:
            print("   • Install and start Ollama")
    
    return ready_providers > 0


async def main():
    """Run system check."""
    try:
        ready = await check_system_status()
        return ready
    except Exception as e:
        print(f"\n💥 System check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
