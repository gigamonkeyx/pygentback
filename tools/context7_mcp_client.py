#!/usr/bin/env python3
"""
MCP Client for Context7
A simple Model Context Protocol client to interact with Context7 MCP server
"""

import asyncio
import json
import subprocess
import sys
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPResponse:
    """MCP Response wrapper"""
    id: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class MCPClient:
    """Model Context Protocol Client"""
    
    def __init__(self, command: List[str]):
        self.command = command
        self.process = None
        self.initialized = False
        self.request_id = 0
        
    async def start(self):
        """Start the MCP server process"""        try:
            logger.info(f"Starting MCP server: {' '.join(self.command)}")
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=0
            )
            
            # Initialize the MCP connection
            await self._initialize()
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise
    
    async def _initialize(self):
        """Initialize the MCP connection"""
        try:
            init_request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": False},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "python-mcp-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            response = await self._send_request(init_request)
            if response.error:
                raise Exception(f"Failed to initialize: {response.error}")
                
            logger.info("✅ MCP server initialized successfully")
            self.initialized = True
            
            # Send initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            await self._send_notification(initialized_notification)
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP connection: {e}")
            raise
    
    def _next_id(self) -> int:
        """Get next request ID"""
        self.request_id += 1
        return self.request_id
    
    async def _send_request(self, request: Dict[str, Any]) -> MCPResponse:
        """Send a request and wait for response"""
        if not self.process:
            raise Exception("MCP server not started")
            
        try:
            # Send request
            request_line = json.dumps(request) + "\n"
            self.process.stdin.write(request_line)
            self.process.stdin.flush()
            
            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                raise Exception("No response from MCP server")
                
            response_data = json.loads(response_line.strip())
            
            return MCPResponse(
                id=response_data.get("id", 0),
                result=response_data.get("result"),
                error=response_data.get("error")
            )
            
        except Exception as e:
            logger.error(f"Failed to send request: {e}")
            raise
    
    async def _send_notification(self, notification: Dict[str, Any]):
        """Send a notification (no response expected)"""
        if not self.process:
            raise Exception("MCP server not started")
            
        try:
            notification_line = json.dumps(notification) + "\n"
            self.process.stdin.write(notification_line)
            self.process.stdin.flush()
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            raise
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        if not self.initialized:
            raise Exception("MCP client not initialized")
            
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list"
        }
        
        response = await self._send_request(request)
        if response.error:
            raise Exception(f"Failed to list tools: {response.error}")
            
        return response.result.get("tools", [])
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool"""
        if not self.initialized:
            raise Exception("MCP client not initialized")
            
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        
        response = await self._send_request(request)
        if response.error:
            raise Exception(f"Failed to call tool '{name}': {response.error}")
            
        return response.result
    
    async def close(self):
        """Close the MCP connection"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            logger.info("MCP server closed")

class Context7Client:
    """Context7 MCP Client wrapper"""
    
    def __init__(self):
        self.mcp_client = MCPClient(["D:\\nodejs\\npx.cmd", "-y", "@upstash/context7-mcp"])
    
    async def start(self):
        """Start the Context7 client"""
        await self.mcp_client.start()
        
        # List available tools
        tools = await self.mcp_client.list_tools()
        logger.info(f"📚 Available Context7 tools: {[tool['name'] for tool in tools]}")
        
        return tools
    
    async def resolve_library_id(self, library_name: str) -> str:
        """Resolve a library name to Context7 compatible ID"""
        logger.info(f"🔍 Resolving library ID for: {library_name}")
        
        result = await self.mcp_client.call_tool("resolve-library-id", {
            "libraryName": library_name
        })
        
        # Extract the library ID from the response
        content = result.get("content", [])
        for item in content:
            if item.get("type") == "text":
                text = item.get("text", "")
                # Parse the response to extract the library ID
                if "Context7-compatible library ID:" in text:
                    library_id = text.split("Context7-compatible library ID:")[-1].strip()
                    return library_id
                
        return ""
    
    async def get_library_docs(self, library_id: str, topic: str = "", tokens: int = 10000) -> str:
        """Get documentation for a library"""
        logger.info(f"📖 Getting docs for: {library_id}")
        if topic:
            logger.info(f"   Topic: {topic}")
            
        arguments = {
            "context7CompatibleLibraryID": library_id,
            "tokens": tokens
        }
        
        if topic:
            arguments["topic"] = topic
            
        result = await self.mcp_client.call_tool("get-library-docs", arguments)
        
        # Extract documentation from the response
        content = result.get("content", [])
        docs = ""
        for item in content:
            if item.get("type") == "text":
                docs += item.get("text", "")
                
        return docs
    
    async def close(self):
        """Close the Context7 client"""
        await self.mcp_client.close()

async def demo_context7():
    """Demonstrate Context7 capabilities"""
    print("🚀 Context7 MCP Client Demo")
    print("=" * 50)
    
    client = Context7Client()
    
    try:
        # Start the client
        tools = await client.start()
        print(f"✅ Connected to Context7 with {len(tools)} tools available")
        
        # Demo 1: Resolve Python library IDs
        print("\n📚 Demo 1: Resolving Library IDs")
        print("-" * 30)
        
        libraries = ["fastapi", "django", "requests", "pandas"]
        resolved_libs = {}
        
        for lib in libraries:
            try:
                lib_id = await client.resolve_library_id(lib)
                resolved_libs[lib] = lib_id
                print(f"✅ {lib} → {lib_id}")
            except Exception as e:
                print(f"❌ {lib} → Error: {e}")
        
        # Demo 2: Get FastAPI documentation
        if "fastapi" in resolved_libs and resolved_libs["fastapi"]:
            print("\n📖 Demo 2: Getting FastAPI Documentation")
            print("-" * 40)
            
            try:
                docs = await client.get_library_docs(
                    resolved_libs["fastapi"], 
                    topic="getting started",
                    tokens=3000
                )
                
                print("📚 FastAPI Getting Started Documentation:")
                print("=" * 50)
                print(docs[:1000] + "..." if len(docs) > 1000 else docs)
                print("=" * 50)
                
            except Exception as e:
                print(f"❌ Error getting FastAPI docs: {e}")
        
        # Demo 3: Get pandas documentation
        if "pandas" in resolved_libs and resolved_libs["pandas"]:
            print("\n📊 Demo 3: Getting Pandas Documentation")
            print("-" * 40)
            
            try:
                docs = await client.get_library_docs(
                    resolved_libs["pandas"],
                    topic="dataframe operations",
                    tokens=2000
                )
                
                print("📊 Pandas DataFrame Operations:")
                print("=" * 50)
                print(docs[:800] + "..." if len(docs) > 800 else docs)
                print("=" * 50)
                
            except Exception as e:
                print(f"❌ Error getting pandas docs: {e}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await client.close()
    
    print("\n🎉 Context7 Demo Complete!")
    print("\n💡 What Context7 provides:")
    print("- ✅ Up-to-date library documentation")
    print("- ✅ Version-specific code examples")
    print("- ✅ No hallucinated APIs")
    print("- ✅ Works with 1000+ libraries")

if __name__ == "__main__":
    asyncio.run(demo_context7())
