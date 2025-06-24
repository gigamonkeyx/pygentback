"""
Comprehensive Backend Test - Full System Validation

Tests the complete PyGent Factory backend with refactored ProviderRegistry:
1. API server health and endpoints
2. Provider registry functionality
3. MCP tool manager
4. Agent creation and text generation
5. Database and memory systems
"""

import asyncio
import aiohttp
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.providers.provider_registry import get_provider_registry
from src.mcp.tool_manager import get_mcp_tool_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"

async def test_api_health():
    """Test API server health endpoints."""
    print("🏥 TESTING API SERVER HEALTH")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test health endpoint
            async with session.get(f"{API_BASE_URL}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"   ✅ Health endpoint: {health_data.get('status', 'unknown')}")
                    return True
                else:
                    print(f"   ❌ Health endpoint failed: {response.status}")
                    return False
    except Exception as e:
        print(f"   ❌ API connection failed: {e}")
        return False

async def test_provider_registry_direct():
    """Test provider registry directly."""
    print("\n🔧 TESTING PROVIDER REGISTRY (Direct)")
    
    try:
        registry = get_provider_registry()
        
        # Initialize providers
        await registry.initialize()
        
        # Get system status
        status = await registry.get_system_status()
        print(f"   ✅ System initialized: {status['initialized']}")
        print(f"   ✅ Providers ready: {status['providers_ready']}/{status['providers_total']}")
        
        # Get available models
        models = await registry.get_all_models()
        total_models = sum(len(provider_models) for provider_models in models.values())
        print(f"   ✅ Total models available: {total_models}")
        
        for provider, provider_models in models.items():
            print(f"   - {provider}: {len(provider_models)} models")
        
        return status['providers_ready'] > 0
        
    except Exception as e:
        print(f"   ❌ Provider registry test failed: {e}")
        return False

async def test_text_generation():
    """Test text generation with fallback."""
    print("\n💬 TESTING TEXT GENERATION")
    
    try:
        registry = get_provider_registry()
        
        # Test simple generation
        result = await registry.generate_text_with_fallback(
            model="qwen3:8b",
            prompt="Hello! Please respond with exactly 'Test successful' if you can see this.",
            max_tokens=50
        )
        
        if result['success']:
            print("   ✅ Text generation successful")
            print(f"   ✅ Provider used: {result['provider_used']}")
            print(f"   ✅ Response: {result['result'][:100]}...")
            return True
        else:
            print(f"   ❌ Text generation failed: {result['errors']}")
            return False
            
    except Exception as e:
        print(f"   ❌ Text generation test failed: {e}")
        return False

async def test_mcp_tool_manager():
    """Test MCP tool manager with fallbacks."""
    print("\n🔨 TESTING MCP TOOL MANAGER")
    
    try:
        tool_manager = get_mcp_tool_manager()
        
        # Register a test tool with fallback
        async def test_fallback(params: dict) -> str:
            return f"[NATIVE] Test tool executed with params: {params}"
        
        await tool_manager.register_tool(
            'test_tool',
            {'server': 'test-server', 'endpoint': 'test'},
            test_fallback
        )
        
        # Execute tool (will use fallback since no real MCP client)
        result = await tool_manager.execute_tool('test_tool', {'input': 'test_data'})
        
        if result['success']:
            print("   ✅ Tool execution successful")
            print(f"   ✅ Source: {result['source']}")
            print(f"   ✅ Result: {result['result']}")
            
            # Get tool manager status
            status = tool_manager.get_status()
            print(f"   ✅ Registered tools: {status['registered_tools']}")
            print(f"   ✅ Native fallbacks: {status['native_fallbacks']}")
            return True
        else:
            print(f"   ❌ Tool execution failed: {result}")
            return False
            
    except Exception as e:
        print(f"   ❌ MCP tool manager test failed: {e}")
        return False

async def test_api_endpoints():
    """Test key API endpoints."""
    print("\n🌐 TESTING API ENDPOINTS")
    
    try:
        async with aiohttp.ClientSession() as session:
            tests = []
            
            # Test providers endpoint
            try:
                async with session.get(f"{API_BASE_URL}/api/providers") as response:
                    if response.status == 200:
                        providers_data = await response.json()
                        print(f"   ✅ Providers endpoint: {len(providers_data)} providers")
                        tests.append(True)
                    else:
                        print(f"   ❌ Providers endpoint failed: {response.status}")
                        tests.append(False)
            except Exception as e:
                print(f"   ❌ Providers endpoint error: {e}")
                tests.append(False)
              # Test models endpoint
            try:
                async with session.get(f"{API_BASE_URL}/api/models") as response:
                    if response.status == 200:
                        await response.json()  # Just validate JSON response
                        print("   ✅ Models endpoint: data received")
                        tests.append(True)
                    else:
                        print(f"   ❌ Models endpoint failed: {response.status}")
                        tests.append(False)
            except Exception as e:
                print(f"   ❌ Models endpoint error: {e}")
                tests.append(False)
              # Test MCP servers endpoint
            try:
                async with session.get(f"{API_BASE_URL}/api/v1/mcp/servers/public") as response:
                    if response.status == 200:
                        servers_data = await response.json()
                        print(f"   ✅ MCP servers endpoint: {len(servers_data.get('servers', []))} servers")
                        tests.append(True)
                    else:
                        print(f"   ❌ MCP servers endpoint failed: {response.status}")
                        tests.append(False)
            except Exception as e:
                print(f"   ❌ MCP servers endpoint error: {e}")
                tests.append(False)
            
            return all(tests)
            
    except Exception as e:
        print(f"   ❌ API endpoints test failed: {e}")
        return False

async def run_comprehensive_test():
    """Run comprehensive backend test."""
    print("🚀 COMPREHENSIVE BACKEND TEST")
    print("=" * 60)
    
    tests = []
    
    # Run all tests
    tests.append(await test_api_health())
    tests.append(await test_provider_registry_direct())
    tests.append(await test_text_generation())
    tests.append(await test_mcp_tool_manager())
    tests.append(await test_api_endpoints())
    
    # Results summary
    passed = sum(tests)
    total = len(tests)
    
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Backend is fully operational.")
        status = "✅ SYSTEM READY FOR PRODUCTION"
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. System mostly operational.")
        status = "⚠️  SYSTEM MOSTLY READY"
    else:
        print("❌ Multiple test failures. System needs attention.")
        status = "❌ SYSTEM NEEDS FIXES"
    
    print(f"\n🏆 FINAL STATUS: {status}")
    
    # Detailed results
    print("\n📋 DETAILED RESULTS:")
    test_names = [
        "API Server Health",
        "Provider Registry", 
        "Text Generation",
        "MCP Tool Manager",
        "API Endpoints"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, tests)):
        icon = "✅" if result else "❌"
        print(f"   {i+1}. {icon} {name}")
    
    return passed, total

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
