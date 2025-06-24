#!/usr/bin/env python3
"""
Fix MCP Tool Discovery - Core Implementation

This script implements the minimal fix needed to enable MCP tool discovery
according to the official MCP specification. It focuses on the core issue:
calling tools/list after server startup and storing the tool metadata.
"""

import asyncio
import json
import logging
import subprocess
import sys
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MCPToolDiscoveryFix:
    """
    Minimal implementation to fix MCP tool discovery
    
    This class implements the core MCP specification requirement:
    - Call tools/list after server startup
    - Store tool metadata for agent access
    """
    
    def __init__(self):
        self.discovered_tools = {}
        self.active_servers = {}
        
    async def discover_tools_for_server(self, server_name: str, command: List[str]) -> Dict[str, Any]:
        """
        Discover tools for a specific MCP server using the MCP Python SDK
        
        This implements the MCP spec requirement to call tools/list
        """
        logger.info(f"Discovering tools for server: {server_name}")
        
        try:            # Import MCP SDK
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            
            # Create server parameters
            server_params = StdioServerParameters(
                command=command[0],
                args=command[1:] if len(command) > 1 else []
            )
            
            # Create and initialize session
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()
                    
                    # Call tools/list (MCP spec requirement)
                    logger.info(f"Calling tools/list for {server_name}")
                    tools_result = await session.list_tools()
                    
                    # Process discovered tools
                    discovered_tools = []
                    for tool in tools_result.tools:
                        tool_info = {
                            "name": tool.name,
                            "description": tool.description or "",
                            "input_schema": tool.inputSchema or {},
                            "server": server_name
                        }
                        discovered_tools.append(tool_info)
                        logger.info(f"  Tool: {tool.name} - {tool.description}")
                    
                    # Store results
                    self.discovered_tools[server_name] = discovered_tools
                    self.active_servers[server_name] = {
                        "command": command,
                        "tool_count": len(discovered_tools),
                        "status": "active"
                    }
                    
                    logger.info(f"Successfully discovered {len(discovered_tools)} tools for {server_name}")
                    return {
                        "server": server_name,
                        "tools": discovered_tools,
                        "tool_count": len(discovered_tools)
                    }
                    
        except Exception as e:
            logger.error(f"Failed to discover tools for {server_name}: {e}")
            self.active_servers[server_name] = {
                "command": command,
                "tool_count": 0,
                "status": "error",
                "error": str(e)
            }
            return {
                "server": server_name,
                "tools": [],
                "tool_count": 0,
                "error": str(e)
            }
    
    async def discover_all_known_servers(self) -> Dict[str, Any]:
        """Discover tools for all known working MCP servers"""
        
        # Known working MCP servers
        servers = {
            "context7": ["npx", "-y", "@upstash/context7-mcp"],
            "filesystem": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "d:\\mcp\\pygent-factory"],
        }
        
        # Try Cloudflare servers if available
        try:
            result = subprocess.run(["npx", "list"], capture_output=True, text=True, timeout=10)
            if "@cloudflare/mcp-server-cloudflare" in result.stdout:
                servers["cloudflare"] = ["npx", "-y", "@cloudflare/mcp-server-cloudflare"]
        except:
            pass
        
        logger.info(f"Discovering tools for {len(servers)} servers...")
        
        results = {}
        for server_name, command in servers.items():
            try:
                result = await self.discover_tools_for_server(server_name, command)
                results[server_name] = result
            except Exception as e:
                logger.error(f"Failed to test server {server_name}: {e}")
                results[server_name] = {
                    "server": server_name,
                    "tools": [],
                    "tool_count": 0,
                    "error": str(e)
                }
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all discovered tools"""
        total_tools = sum(len(tools) for tools in self.discovered_tools.values())
        active_servers = sum(1 for s in self.active_servers.values() if s.get("status") == "active")
        
        return {
            "total_servers": len(self.active_servers),
            "active_servers": active_servers,
            "total_tools": total_tools,
            "servers": self.active_servers,
            "all_tools": self.discovered_tools
        }
    
    def save_results(self, filename: str = "mcp_tool_discovery_results.json"):
        """Save discovery results to JSON file"""
        results = self.get_summary()
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")


async def main():
    """Main function to test MCP tool discovery fix"""
    print("=" * 60)
    print("MCP Tool Discovery Fix - Core Implementation")
    print("=" * 60)
    
    # Check MCP SDK availability
    try:
        import mcp
        logger.info("✅ MCP SDK is available")
    except ImportError:
        logger.error("❌ MCP SDK not available - please install: pip install mcp")
        return False
    
    # Create discovery instance
    discovery = MCPToolDiscoveryFix()
    
    # Discover tools for all known servers
    results = await discovery.discover_all_known_servers()
    
    # Print results
    summary = discovery.get_summary()
    print(f"\n📊 Discovery Results:")
    print(f"  Total Servers: {summary['total_servers']}")
    print(f"  Active Servers: {summary['active_servers']}")
    print(f"  Total Tools Discovered: {summary['total_tools']}")
    
    print(f"\n🔧 Tools by Server:")
    for server_name, tools in discovery.discovered_tools.items():
        print(f"  {server_name}: {len(tools)} tools")
        for tool in tools[:3]:  # Show first 3 tools
            print(f"    - {tool['name']}: {tool['description'][:50]}...")
        if len(tools) > 3:
            print(f"    ... and {len(tools) - 3} more")
    
    # Save results
    discovery.save_results()
    
    if summary['total_tools'] > 0:
        print(f"\n✅ SUCCESS: Discovered {summary['total_tools']} tools from {summary['active_servers']} servers")
        print("🎯 This proves tool discovery works - now integrate into PyGent Factory!")
        return True
    else:
        print(f"\n❌ FAILURE: No tools discovered")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
