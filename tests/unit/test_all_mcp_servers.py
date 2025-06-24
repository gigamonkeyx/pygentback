#!/usr/bin/env python3
"""
Complete MCP Server Validation Test

Test all configured MCP servers including npx-based ones.
"""

import asyncio
import json
import os

async def test_server(name, command, timeout=15):
    """Test a single MCP server with proper handling of different command types"""
    try:
        print(f"Testing {name}...")
        
        # Create the test message
        test_msg = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}\n'
        
        # Handle different command formats
        if isinstance(command, list):
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy()
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
                timeout=timeout
            )
            
            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""
            
            # Check for successful response
            if '"result"' in stdout_str and '"protocolVersion"' in stdout_str:
                print(f"✅ {name} - Working")
                return True
            elif '"error"' in stdout_str:
                print(f"❌ {name} - Error response: {stdout_str[:100]}")
                return False
            else:
                print(f"❌ {name} - No valid response. Stderr: {stderr_str[:100]}")
                return False
                
        except asyncio.TimeoutError:
            print(f"⏱️  {name} - Timeout after {timeout}s")
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                proc.kill()
            return False
            
    except Exception as e:
        print(f"❌ {name} - Error: {e}")
        return False

async def main():
    """Test all MCP servers from config"""
    print("Testing All Configured MCP Servers\n")
    
    # Load config
    if not os.path.exists("mcp_server_configs.json"):
        print("❌ Config file not found: mcp_server_configs.json")
        return False
    
    with open("mcp_server_configs.json", 'r') as f:
        config = json.load(f)
    
    servers = config.get("servers", [])
    if not servers:
        print("❌ No servers found in config")
        return False
    
    # Test each server
    results = []
    for server_config in servers:
        name = server_config.get("name", server_config.get("id", "unknown"))
        command = server_config.get("command", [])
        auto_start = server_config.get("auto_start", True)
        
        if not auto_start:
            print(f"⏭️  {name} - Skipped (auto_start=false)")
            continue
            
        if not command:
            print(f"❌ {name} - No command specified")
            results.append((name, False))
            continue
        
        result = await test_server(name, command)
        results.append((name, result))
    
    # Summary
    print("\n" + "="*50)
    print("MCP SERVER TEST RESULTS")
    print("="*50)
    
    working = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if result:
            working += 1
    
    print(f"\nSummary: {working}/{total} servers working")
    
    if working >= 5:  # Need at least 5 core servers
        print("🎉 Sufficient servers are operational!")
        return True
    else:
        print("⚠️  Not enough servers working for full functionality")
        return False

if __name__ == "__main__":
    asyncio.run(main())
