#!/usr/bin/env python3
"""
Demonstrate Context7 capabilities for Python SDK documentation
"""

import json
import subprocess
import sys
import time

def call_context7_mcp(method, params=None):
    """Call Context7 MCP server with a specific method"""
    try:
        # Prepare MCP request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {}
        }
        
        # Start Context7 MCP server
        cmd = ["D:\\nodejs\\npx.cmd", "-y", "@upstash/context7-mcp"]
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # First initialize the connection
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "python-test", "version": "1.0"}
            }
        }
        
        # Send initialize + actual request
        input_data = json.dumps(init_request) + "\n" + json.dumps(request) + "\n"
        
        stdout, stderr = process.communicate(input=input_data, timeout=10)
        
        # Parse responses
        lines = stdout.strip().split('\n')
        responses = []
        for line in lines:
            if line.strip():
                try:
                    responses.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return responses
        
    except Exception as e:
        print(f"Error calling Context7: {e}")
        return None

def demo_python_sdk_docs():
    """Demonstrate getting Python SDK documentation"""
    print("🐍 Context7 Python SDK Documentation Demo")
    print("=" * 50)
    
    # Test 1: Resolve Python library IDs
    print("\n📚 Step 1: Resolving Python library names...")
    
    libraries_to_test = ["fastapi", "django", "requests", "pandas"]
    
    for lib in libraries_to_test:
        print(f"\n🔍 Looking up '{lib}'...")
        
        responses = call_context7_mcp("tools/call", {
            "name": "resolve-library-id",
            "arguments": {"libraryName": lib}
        })
        
        if responses:
            for response in responses:
                if "result" in response and "content" in response["result"]:
                    for content in response["result"]["content"]:
                        if content["type"] == "text":
                            print(f"✅ Found: {content['text'][:100]}...")
        else:
            print(f"❌ Could not resolve '{lib}'")
    
    # Test 2: Get specific documentation
    print(f"\n📖 Step 2: Getting FastAPI documentation...")
    
    responses = call_context7_mcp("tools/call", {
        "name": "get-library-docs",
        "arguments": {
            "context7CompatibleLibraryID": "tiangolo/fastapi",
            "topic": "getting started",
            "tokens": 5000
        }
    })
    
    if responses:
        for response in responses:
            if "result" in response and "content" in response["result"]:
                for content in response["result"]["content"]:
                    if content["type"] == "text":
                        print(f"📚 FastAPI Docs Preview:")
                        print("-" * 30)
                        print(content['text'][:500] + "...")
                        print("-" * 30)

if __name__ == "__main__":
    print("🚀 Starting Context7 Demo...")
    
    # Check if Context7 is available
    try:
        result = subprocess.run(
            ["D:\\nodejs\\npx.cmd", "-y", "@upstash/context7-mcp", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        print("✅ Context7 is available!")
    except:
        print("❌ Context7 not available or npx issue")
        sys.exit(1)
    
    demo_python_sdk_docs()
    
    print("\n🎉 Context7 Demo Complete!")
    print("\n💡 How to use Context7 in practice:")
    print("1. Add 'use context7' to your prompts")
    print("2. Ask for specific library documentation")
    print("3. Get up-to-date code examples")
    print("4. Perfect for: FastAPI, Django, requests, pandas, numpy, etc.")
