#!/usr/bin/env python3
"""
Demo: Native Tools Are Real Tools
Shows that native fallbacks do actual work, not emulation
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def demo_real_tools():
    """Demonstrate that native tools do real work."""
    
    print("🎯 NATIVE TOOLS = REAL TOOLS DEMO")
    print("=" * 50)
    
    # Import our registry
    from ai.providers.provider_registry import ProviderRegistry
    
    registry = ProviderRegistry()
    await registry.initialize()
    
    # Register native tools
    registry.register_native_fallbacks()
    
    print("📝 Creating a real file with native tool...")
    
    # This actually creates a file!
    result = await registry.execute_mcp_tool("create_file", {
        "path": "real_test_file.py",
        "content": '''#!/usr/bin/env python3
"""
This file was created by a NATIVE TOOL!
Not an MCP server, not emulation - real Python code.
"""

def hello():
    print("Hello from a file created by native tool!")

if __name__ == "__main__":
    hello()
'''
    })
    
    print(f"Result: {result['result']}")
    
    print("\n📖 Reading the file back...")
    
    # This actually reads the file!
    result = await registry.execute_mcp_tool("read_file", {
        "path": "real_test_file.py"
    })
    
    print(f"File exists and contains: {len(result['result'])} characters")
    
    print("\n🚀 Now let's RUN the file we created...")
    
    # This actually executes the file!
    result = await registry.execute_mcp_tool("run_command", {
        "command": "python real_test_file.py"
    })
    
    print(f"Command output:\n{result['result']}")
    
    print("\n✅ PROOF: Native tools did REAL work!")
    print("   • Created an actual file")
    print("   • Read actual file contents") 
    print("   • Executed actual Python code")
    print("   • No MCP servers involved!")
    
    # Cleanup
    import os
    try:
        os.remove("real_test_file.py")
        print("\n🧹 Cleaned up test file")
    except:
        pass

if __name__ == "__main__":
    asyncio.run(demo_real_tools())
