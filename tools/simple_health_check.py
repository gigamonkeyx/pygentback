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

async def test_core_imports():
    """Test core imports"""
    print("🔍 Testing Core Imports...")
    
    try:
        # Test provider registry
        from ai.providers.provider_registry import ProviderRegistry
        print("✅ ProviderRegistry import successful")
        
        # Test agent factory
        from core.agent_factory import AgentFactory
        print("✅ AgentFactory import successful")
        
        # Test providers
        from ai.providers.ollama_provider import OllamaProvider
        from ai.providers.openrouter_provider import OpenRouterProvider
        print("✅ Provider imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

async def test_provider_system():
    """Test provider system functionality"""
    print("\n🔍 Testing Provider System...")
    
    try:
        from ai.providers.provider_registry import ProviderRegistry
        
        registry = ProviderRegistry()
        await registry.initialize()
        health = await registry.get_system_status()
        print(f"✅ Provider system health: {health.get('status', 'unknown')}")
        
        providers = await registry.get_available_providers()
        print(f"✅ Available providers: {len(providers)}")

        for provider in providers:
            print(f"   • {provider}")
        
        return True
        
    except Exception as e:
        print(f"❌ Provider system test failed: {e}")
        return False

async def test_agent_creation():
    """Test basic agent creation"""
    print("\n🔍 Testing Agent Creation...")
    
    try:
        from core.agent_factory import AgentFactory
        
        factory = AgentFactory()
        await factory.initialize()        # Try to create an agent (use general type since others may not be available)
        agent = await factory.create_agent(
            agent_type="general",
            custom_config={
                "provider": "ollama",
                "model_name": "qwen3:8b"
            }
        )
        
        if agent:
            print("✅ Agent creation successful")
            print(f"   • Agent ID: {agent.agent_id}")
            print(f"   • Agent Type: {agent.agent_type}")
            
            # Test cleanup
            await factory.destroy_agent(agent.agent_id)
            print("✅ Agent cleanup successful")
            
            return True
        else:
            print("❌ Agent creation returned None")
            return False
            
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        return False

async def check_file_structure():
    """Check critical file structure"""
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
        return False
    else:
        print("✅ All critical files present")
        return True

async def main():
    """Main execution function"""
    print("=" * 60)
    print("🏥 PYGENT FACTORY - SIMPLE SYSTEM HEALTH CHECK")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(await check_file_structure())
    results.append(await test_core_imports())
    results.append(await test_provider_system())
    results.append(await test_agent_creation())
    
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
    else:
        print("❌ SYSTEM HAS ISSUES!")
        print("\n🔧 NEXT STEPS:")
        print("   • Fix failing tests before deployment")
        print("   • Check import dependencies")
        print("   • Verify provider configurations")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
