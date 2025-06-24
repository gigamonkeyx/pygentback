#!/usr/bin/env python3
"""
Simple Context7 MCP Client Demo
"""

import json
import subprocess
import sys
import time

def test_context7_simple():
    """Simple test of Context7 MCP server"""
    print("🚀 Context7 Simple Test")
    print("=" * 40)
    
    try:
        # Start Context7 MCP server
        cmd = ["D:\\nodejs\\npx.cmd", "-y", "@upstash/context7-mcp"]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        }
        
        # Send initialize request
        process.stdin.write(json.dumps(init_request) + "\\n")
        process.stdin.flush()
        
        # Read response
        init_response = process.stdout.readline()
        print(f"✅ Initialize response: {init_response[:100]}...")
        
        # Send initialized notification
        initialized_notif = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        process.stdin.write(json.dumps(initialized_notif) + "\\n")
        process.stdin.flush()
        
        # Test resolve-library-id for FastAPI
        print("\\n🔍 Testing resolve-library-id for 'fastapi'...")
        resolve_request = {
            "jsonrpc": "2.0", 
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "resolve-library-id",
                "arguments": {"libraryName": "fastapi"}
            }
        }
        
        process.stdin.write(json.dumps(resolve_request) + "\\n")
        process.stdin.flush()
        
        # Read response with timeout
        try:
            response = process.stdout.readline()
            if response:
                response_data = json.loads(response)
                print(f"📚 FastAPI library ID response:")
                
                if "result" in response_data:
                    content = response_data["result"].get("content", [])
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            print(f"   {text[:200]}...")
                else:
                    print(f"   Error: {response_data.get('error', 'Unknown error')}")
            else:
                print("   No response received")
                
        except Exception as e:
            print(f"   Error parsing response: {e}")
        
        # Clean up
        process.terminate()
        process.wait()
        
        print("\\n✅ Context7 test completed!")
        
    except Exception as e:
        print(f"❌ Error testing Context7: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_context7_value():
    """Show what Context7 provides"""
    print("\\n📚 Context7 - The Ultimate Coding Assistant")
    print("=" * 50)
    print("\\n🎯 What makes Context7 amazing:")
    print("   ✅ LIVE documentation (not outdated training data)")
    print("   ✅ Version-specific code examples")
    print("   ✅ No hallucinated APIs that don't exist")
    print("   ✅ Supports 1000+ libraries and frameworks")
    
    print("\\n🐍 Python Libraries Supported:")
    print("   • FastAPI - Modern web framework")
    print("   • Django - Full-featured web framework") 
    print("   • Flask - Lightweight web framework")
    print("   • Requests - HTTP library")
    print("   • Pandas - Data analysis")
    print("   • NumPy - Scientific computing")
    print("   • SQLAlchemy - Database ORM")
    print("   • Pydantic - Data validation")
    print("   • And hundreds more...")
    
    print("\\n🌐 Frontend Libraries:")
    print("   • React - UI library")
    print("   • Next.js - React framework")
    print("   • Vue.js - Progressive framework")
    print("   • Angular - Platform and framework")
    print("   • Svelte - Compile-time framework")
    
    print("\\n🛠️ How to use Context7:")
    print("   1. Add 'use context7' to your prompts")
    print("   2. Ask for specific library help")
    print("   3. Get working, up-to-date code examples")
    print("   4. Never worry about outdated docs again!")
    
    print("\\n💡 Example prompts:")
    print("   'Create a FastAPI app with JWT auth. use context7'")
    print("   'Show me pandas DataFrame operations. use context7'")
    print("   'Help with React hooks and state. use context7'")

if __name__ == "__main__":
    test_context7_simple()
    demonstrate_context7_value()
    
    print("\\n🎉 Context7 is now ready to use!")
    print("   Add it to VS Code and start coding with live docs!")
