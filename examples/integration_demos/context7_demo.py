#!/usr/bin/env python3
"""
Simple MCP Client to demonstrate Context7 capabilities
"""

import asyncio
import json
import subprocess
import sys

class SimpleMCPClient:
    def __init__(self):
        self.process = None
        self.initialized = False
        
    async def start_server(self, command, args):
        """Start an MCP server"""
        try:
            print(f"Starting: {command} {' '.join(args)}")
            self.process = subprocess.Popen(
                [command] + args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Initialize the connection
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "python-client", "version": "1.0.0"}
                }
            }
            
            response = await self.send_request(init_request)
            if response and "result" in response:
                self.initialized = True
                print("✅ MCP server initialized")
                return True
            else:
                print(f"❌ Init failed: {response}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to start: {e}")
            return False
            
    async def send_request(self, request):
        """Send a request to the MCP server"""
        try:
            request_line = json.dumps(request) + '\n'
            self.process.stdin.write(request_line)
            self.process.stdin.flush()
            
            response_line = self.process.stdout.readline()
            if response_line:
                return json.loads(response_line.strip())
            return None
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    async def call_tool(self, tool_name, arguments):
        """Call a tool"""
        request = {
            "jsonrpc": "2.0", 
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments}
        }
        return await self.send_request(request)
    
    def cleanup(self):
        if self.process:
            self.process.terminate()


async def demo_context7():
    """Demo Context7"""
    print("🚀 Context7 Demo")
    print("=" * 30)
    
    client = SimpleMCPClient()
    
    try:
        # Start Context7
        success = await client.start_server("D:\\nodejs\\npx.cmd", ["-y", "@upstash/context7-mcp"])
        if not success:
            return
        
        # Test resolve library
        print("\n🔍 Resolving 'fastapi'...")
        response = await client.call_tool("resolve-library-id", {"libraryName": "fastapi"})
        print(f"Response: {response}")
        
        # Test getting docs
        print("\n📚 Getting FastAPI docs...")
        docs_response = await client.call_tool("get-library-docs", {
            "context7CompatibleLibraryID": "tiangolo/fastapi",
            "topic": "tutorial",
            "tokens": 1000
        })
        
        if docs_response and "result" in docs_response:
            content = docs_response["result"].get("content", [])
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    print(f"📖 Documentation preview:")
                    print(text[:300] + "..." if len(text) > 300 else text)
                    break
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.cleanup()


if __name__ == "__main__":
    asyncio.run(demo_context7())
