"""
Demo: Separated Architecture - Provider Registry + MCP Tool Manager

Shows how the clean separation allows for:
1. LLM providers to focus on language models only
2. MCP tools to be managed separately with circuit breakers
3. Clean integration between both systems
"""

import asyncio
import logging
from src.ai.providers.provider_registry import ProviderRegistry
from src.mcp.tool_manager import get_mcp_tool_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example native fallback functions
async def file_read_fallback(params: dict) -> str:
    """Native fallback for file reading."""
    filename = params.get('filename', 'test.txt')
    try:
        with open(filename, 'r') as f:
            return f"[NATIVE] Read {len(f.read())} chars from {filename}"
    except FileNotFoundError:
        return f"[NATIVE] File {filename} not found (would create empty file)"

async def web_search_fallback(params: dict) -> dict:
    """Native fallback for web search."""
    query = params.get('query', 'test')
    return {
        'query': query,
        'results': ['[NATIVE] Simulated result 1', '[NATIVE] Simulated result 2'],
        'note': 'This is a native fallback - real MCP would give better results'
    }

async def calculator_fallback(params: dict) -> float:
    """Native fallback for calculator."""
    expression = params.get('expression', '1+1')
    try:
        # Safe eval for demo (real implementation would use ast.literal_eval or proper parser)
        result = eval(expression.replace(' ', ''))
        return f"[NATIVE] {expression} = {result}"
    except Exception:
        return "[NATIVE] Invalid expression"

async def demo_separated_architecture():
    """Demo the separated architecture."""
    print("🏗️  SEPARATED ARCHITECTURE DEMO")
    print("=" * 50)
    
    # 1. Initialize Provider Registry (LLM providers only)
    print("\n1️⃣  PROVIDER REGISTRY (LLM providers)")
    registry = ProviderRegistry()
      # Initialize providers first
    await registry.initialize()
    
    # Check provider health
    status = await registry.get_system_status()
    print(f"   Provider status: {status['providers']}")
    print(f"   Ready providers: {status['providers_ready']}/{status['providers_total']}")
      # List available models
    models = await registry.get_all_models()
    print(f"   Available models: {len(models)} providers found")
    for provider, provider_models in models.items():
        print(f"   - {provider}: {len(provider_models)} models")
    
    # 2. Initialize MCP Tool Manager (tools only)
    print("\n2️⃣  MCP TOOL MANAGER (Tools with fallbacks)")
    tool_manager = get_mcp_tool_manager()
    
    # Register tools with native fallbacks
    await tool_manager.register_tool(
        'file_read',
        {'server': 'file-server', 'endpoint': 'read'},
        file_read_fallback
    )
    
    await tool_manager.register_tool(
        'web_search',
        {'server': 'search-server', 'endpoint': 'search'},
        web_search_fallback
    )
    
    await tool_manager.register_tool(
        'calculator',
        {'server': 'calc-server', 'endpoint': 'calc'},
        calculator_fallback
    )
    
    # Register tool without fallback (to show error handling)
    await tool_manager.register_tool(
        'advanced_ai_tool',
        {'server': 'ai-server', 'endpoint': 'process'},
        None  # No fallback
    )
    
    # 3. Test tool execution with fallbacks
    print("\n3️⃣  TOOL EXECUTION (MCP failed → fallback)")
    
    # These will use fallbacks since no real MCP client is connected
    test_cases = [
        ('file_read', {'filename': 'demo.txt'}),
        ('web_search', {'query': 'python async programming'}),
        ('calculator', {'expression': '2 + 3 * 4'}),
        ('advanced_ai_tool', {'input': 'process this'})  # No fallback
    ]
    
    for tool_name, params in test_cases:
        print(f"\n   🔧 Testing tool: {tool_name}")
        result = await tool_manager.execute_tool(tool_name, params)
        
        if result['success']:
            print(f"   ✅ Success ({result['source']}): {result['result']}")
            if 'warning' in result:
                print(f"   ⚠️  {result['warning']}")
        else:
            print(f"   ❌ Failed: {result['error']}")
            if 'suggestions' in result:
                print(f"   💡 Suggestions: {result['suggestions'][:2]}")
    
    # 4. Status overview
    print("\n4️⃣  SYSTEM STATUS")    # Provider registry status
    provider_status = {
        'providers_loaded': len(registry.providers),
        'models_available': sum(len(models) for models in models.values()) if models else 0,
        'ready_providers': status['providers_ready'] if status else 0
    }
    print(f"   📊 Provider Registry: {provider_status}")
    
    # Tool manager status
    tool_status = tool_manager.get_status()
    print(f"   🔧 Tool Manager: {tool_status['registered_tools']} tools, {tool_status['native_fallbacks']} fallbacks")
    print(f"   🛡️  Circuit breakers: {tool_status['circuit_breakers']}")
    
    # 5. Integration benefits
    print("\n5️⃣  ARCHITECTURE BENEFITS")
    benefits = [
        "✅ Provider Registry focuses only on LLM providers",
        "✅ Tool Manager handles MCP tools with circuit breakers",
        "✅ Native fallbacks provide hyper-availability",
        "✅ Clean separation of concerns",
        "✅ Easy to test and maintain each component",
        "✅ Tools fail gracefully with helpful errors",
        "✅ Agents get best available functionality"
    ]
    for benefit in benefits:
        print(f"   {benefit}")
    
    print("\n🎉 Demo complete! Architecture is clean and robust.")

if __name__ == "__main__":
    asyncio.run(demo_separated_architecture())
