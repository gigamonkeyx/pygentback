#!/usr/bin/env python3
"""
Test Python MCP servers using the MCP SDK
"""

import asyncio
import subprocess
import sys
import json
from pathlib import Path

async def test_python_mcp_server(module_name: str, args: list = None):
    """Test a Python MCP server by starting it and checking if it responds"""
    if args is None:
        args = []
    
    cmd = [sys.executable, "-m", module_name] + args
    print(f"Testing: {' '.join(cmd)}")
    
    try:
        # Start the server process
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        # Send the request
        request_json = json.dumps(init_request) + "\n"
        stdout, stderr = process.communicate(input=request_json, timeout=10)
        
        # Check if we got a valid response
        if stdout.strip():
            try:
                response = json.loads(stdout.strip())
                if "result" in response:
                    print(f"✅ {module_name}: PASS - Server responded correctly")
                    return True
                else:
                    print(f"❌ {module_name}: FAIL - Invalid response: {response}")
                    return False
            except json.JSONDecodeError:
                print(f"❌ {module_name}: FAIL - Invalid JSON response: {stdout}")
                return False
        else:
            print(f"❌ {module_name}: FAIL - No response. Stderr: {stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        process.kill()
        print(f"❌ {module_name}: FAIL - Timeout")
        return False
    except Exception as e:
        print(f"❌ {module_name}: FAIL - Error: {e}")
        return False

async def main():
    """Test all Python MCP servers"""
    print("🧪 Testing Python MCP Servers with MCP SDK")
    print("=" * 50)
    
    servers_to_test = [
        ("mcp_server_fetch", []),
        ("mcp_server_time", ["--local-timezone", "UTC"]),
        ("mcp_server_git", ["--repository", str(Path.cwd())]),
    ]
    
    results = {}
    for module_name, args in servers_to_test:
        results[module_name] = await test_python_mcp_server(module_name, args)
    
    print("\n📊 Summary:")
    print("=" * 30)
    for module_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{module_name}: {status}")
    
    total_tested = len(results)
    total_passed = sum(results.values())
    print(f"\nResults: {total_passed}/{total_tested} servers working")
    
    return total_passed == total_tested

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
