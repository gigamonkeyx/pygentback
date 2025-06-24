#!/usr/bin/env python3
"""
Quick MCP Server Test

Test the core MCP servers to ensure they're working.
"""

import asyncio
import sys

async def test_server(name, command):
    """Test a single MCP server"""
    try:
        print(f"Testing {name}...")
        
        # Create the test message
        test_msg = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}\n'
        
        # Run the command
        if isinstance(command, list):
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=test_msg.encode()), 
                timeout=10.0
            )
            
            if stdout and b'"result"' in stdout:
                print(f"✅ {name} - Working")
                return True
            else:
                print(f"❌ {name} - Failed: {stderr.decode()[:100]}")
                return False
                
        except asyncio.TimeoutError:
            print(f"❌ {name} - Timeout")
            proc.terminate()
            return False
            
    except Exception as e:
        print(f"❌ {name} - Error: {e}")
        return False

async def main():
    """Test core MCP servers"""
    print("Testing Core MCP Servers\n")
    
    # Core servers to test
    servers = [
        ("Filesystem", ["node", r"d:\mcp\pygent-factory\mcp-servers\src\filesystem\dist\index.js", r"d:\mcp\pygent-factory"]),
        ("Fetch", [sys.executable, "-m", "mcp_server_fetch"]),
        ("Git", [sys.executable, "-m", "mcp_server_git"]),
        ("Memory", ["node", r"d:\mcp\pygent-factory\mcp-servers\src\memory\dist\index.js"]),
        ("Sequential Thinking", ["node", r"d:\mcp\pygent-factory\mcp-servers\src\sequentialthinking\dist\index.js"]),
        ("Python Code", [sys.executable, r"d:\mcp\pygent-factory\mcp_server_python.py"]),
    ]
    
    results = []
    for name, command in servers:
        result = await test_server(name, command)
        results.append((name, result))
    
    print("\n=== Summary ===")
    working = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\nWorking: {working}/{total} servers")
    
    if working >= 4:  # Need at least core servers
        print("🎉 Sufficient servers are working!")
        return True
    else:
        print("⚠️  Not enough servers working")
        return False

if __name__ == "__main__":
    asyncio.run(main())
