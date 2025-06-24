#!/usr/bin/env python3
"""
Simple System Health Check for PyGent Factory
Checks core functionality without complex imports
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

async def main():
    """Main execution function"""
    print("=" * 60)
    print("🏥 PYGENT FACTORY - SIMPLE SYSTEM HEALTH CHECK")
    print("=" * 60)
    
    results = []
    
    # Test 1: Check file structure
    print("\n🔍 Checking File Structure...")
    critical_files = [
        "src/core/agent_factory.py",
        "src/ai/providers/provider_registry.py", 
        "src/ai/providers/ollama_provider.py",
        "src/ai/providers/openrouter_provider.py",
        "src/ai/providers/base_provider.py"
    ]
    
    missing_files = []
    for file_path in critical_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        results.append(False)
    else:
        print("✅ All critical files present")
        results.append(True)
    
    # Test 2: Check core imports
    print("\n🔍 Testing Core Imports...")
    try:
        from ai.providers.provider_registry import ProviderRegistry
        print("✅ ProviderRegistry import successful")
        
        from core.agent_factory import AgentFactory
        print("✅ AgentFactory import successful")
        
        from ai.providers.ollama_provider import OllamaProvider
        from ai.providers.openrouter_provider import OpenRouterProvider
        print("✅ Provider imports successful")
        
        results.append(True)
    except Exception as e:
        print(f"❌ Import failed: {e}")
        results.append(False)
    
    # Test 3: Check provider system
    print("\n🔍 Testing Provider System...")
    try:
        from ai.providers.provider_registry import ProviderRegistry
        
        registry = ProviderRegistry()
        await registry.initialize()
        
        status = await registry.get_system_status()
        print(f"✅ Provider system status: {status.get('status', 'unknown')}")
        
        providers = await registry.get_available_providers()
        print(f"✅ Available providers: {len(providers)}")
        
        for provider in providers:
            print(f"   • {provider}")
        
        results.append(True)
    except Exception as e:
        print(f"❌ Provider system test failed: {e}")
        results.append(False)
    
    # Test 4: Check agent creation
    print("\n🔍 Testing Agent Creation...")
    try:
        from core.agent_factory import AgentFactory
        
        factory = AgentFactory()
        await factory.initialize()
        
        # Get available agent types
        available_types = []
        try:
            available_types = list(factory.registry.agent_types.keys())
            print(f"✅ Available agent types: {available_types}")
        except:
            print("⚠️ Could not get available agent types")
        
        # Try to create an agent with fallback types
        agent = None
        for agent_type in ["general", "basic", "coding"]:
            if agent_type in available_types:
                try:
                    agent = await factory.create_agent(
                        agent_type=agent_type,
                        custom_config={
                            "provider": "ollama",
                            "model_name": "qwen3:8b"
                        }
                    )
                    break
                except Exception as e:
                    print(f"⚠️ Failed to create {agent_type} agent: {e}")
                    continue
        
        if agent:
            print("✅ Agent creation successful")
            print(f"   • Agent ID: {agent.agent_id}")
            print(f"   • Agent Type: {agent.agent_type}")
            
            # Test cleanup
            await factory.destroy_agent(agent.agent_id)
            print("✅ Agent cleanup successful")
            
            results.append(True)
        else:
            print("❌ Could not create any agent")
            results.append(False)
            
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ SYSTEM IS HEALTHY!")
        print("\n💡 RECOMMENDATIONS:")
        print("   • System is ready for production")
        print("   • Consider integrating research orchestrator with main system")
        print("   • Add performance monitoring")
        print("   • Deploy to Cloudflare Pages")
    elif passed >= total * 0.75:
        print("⚠️ SYSTEM IS MOSTLY HEALTHY!")
        print("\n🔧 NEXT STEPS:")
        print("   • Address failing tests")
        print("   • Fix import dependencies")
        print("   • System can be used with caution")
    else:
        print("❌ SYSTEM HAS ISSUES!")
        print("\n🔧 NEXT STEPS:")
        print("   • Fix failing tests before deployment")
        print("   • Check import dependencies")
        print("   • Verify provider configurations")
    
    print("=" * 60)
    
    return passed >= total * 0.75

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
