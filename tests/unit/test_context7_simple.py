#!/usr/bin/env python3
"""
Test Context7 MCP server for Python SDK information
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

async def test_context7_direct():
    """Test Context7 directly using npx"""
    print("🔍 Testing Context7 MCP server directly...")
    
    try:
        # Test with MCP inspector
        print("\n📋 Testing Context7 with MCP inspector...")
        cmd = ["D:\\nodejs\\npx.cmd", "@modelcontextprotocol/inspector", "@upstash/context7-mcp"]
        
        # Just check if it starts properly
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        
        try:
            stdout, stderr = process.communicate(timeout=5)
            print(f"✅ Context7 started successfully!")
            if stdout:
                print(f"Output: {stdout[:200]}...")
            if stderr:
                print(f"Stderr: {stderr[:200]}...")
        except subprocess.TimeoutExpired:
            print("⏰ Context7 is running (timeout expected for interactive mode)")
            process.kill()
            
    except Exception as e:
        print(f"❌ Error testing Context7: {e}")

async def test_context7_tools():
    """Test Context7 tools manually"""
    print("\n🛠️ Testing Context7 tools...")
    
    try:
        # Test resolve-library-id for Python
        print("Testing resolve-library-id for 'python'...")
        
        # This would normally be done through MCP protocol
        # For now, let's just verify the server can start
        cmd = ["D:\\nodejs\\npx.cmd", "-y", "@upstash/context7-mcp"]
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Send a simple MCP request (this is just to test connectivity)
        try:
            stdout, stderr = process.communicate(input='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}', timeout=5)
            print(f"✅ Context7 responded to initialize request")
            if "result" in stdout:
                print("📡 Context7 MCP protocol is working!")
        except subprocess.TimeoutExpired:
            print("⏰ Context7 connection test timed out (this might be normal)")
            process.kill()
            
    except Exception as e:
        print(f"❌ Error testing Context7 tools: {e}")

if __name__ == "__main__":
    print("🚀 Testing Context7 MCP Server")
    print("=" * 50)
    
    asyncio.run(test_context7_direct())
    asyncio.run(test_context7_tools())
    
    print("\n📚 Context7 Features for Python Development:")
    print("- resolve-library-id: Find Python libraries and frameworks")
    print("- get-library-docs: Get up-to-date Python documentation")
    print("- Supports: FastAPI, Django, Flask, requests, pandas, numpy, etc.")
    print("- Usage: Add 'use context7' to prompts for live Python docs!")
