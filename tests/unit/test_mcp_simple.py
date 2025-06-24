#!/usr/bin/env python3
"""Simple MCP server test"""

from mcp.server.fastmcp import FastMCP

# Create simple MCP server
mcp = FastMCP("Test Server")

@mcp.tool()
def hello_world() -> str:
    """Simple hello world tool."""
    return "Hello from MCP!"

if __name__ == "__main__":
    print("Starting simple MCP server...")
    mcp.run()
