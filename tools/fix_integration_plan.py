#!/usr/bin/env python3
"""
Integration Plan: Fix MCP Tool Discovery in PyGent Factory

This script provides the exact code changes needed to integrate working tool discovery
into the existing PyGent Factory MCP server registry system.
"""

# File: src/mcp/server_registry.py
# Add this method to the MCPServerRegistry class

REGISTRY_ENHANCEMENT = """
# Add these imports at the top
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPServerRegistry:
    def __init__(self):
        # ...existing code...
        self.tool_metadata = {}  # Store discovered tools
        self._mcp_sessions = {}  # Active MCP sessions
    
    async def register_and_discover_server(self, config: MCPServerConfig):
        '''Enhanced registration with tool discovery per MCP spec'''
        try:
            # Step 1: Register server (existing logic)
            registration = await self.register_server(config)
            
            # Step 2: Start server (existing logic) 
            await self.start_server(config.id)
            
            # Step 3: NEW - Discover tools per MCP specification
            await self._discover_server_tools(config.id, config)
            
            # Step 4: Update status
            registration.status = MCPServerStatus.RUNNING
            return registration
            
        except Exception as e:
            logger.error(f"Failed to register and discover server {config.name}: {e}")
            raise
    
    async def _discover_server_tools(self, server_id: str, config: MCPServerConfig):
        '''Discover tools using MCP tools/list endpoint'''
        try:
            # Create MCP session
            if isinstance(config.command, str):
                command_parts = config.command.split()
            else:
                command_parts = config.command
            
            server_params = StdioServerParameters(
                command=command_parts[0],
                args=command_parts[1:] if len(command_parts) > 1 else []
            )
            
            # Use MCP client session
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Call tools/list (MCP spec requirement)
                    tools_result = await session.list_tools()
                    
                    # Store tool metadata
                    discovered_tools = []
                    for tool in tools_result.tools:
                        tool_metadata = {
                            "name": tool.name,
                            "description": tool.description or "",
                            "input_schema": tool.inputSchema or {},
                            "server_id": server_id,
                            "server_name": config.name
                        }
                        discovered_tools.append(tool_metadata)
                    
                    self.tool_metadata[server_id] = discovered_tools
                    logger.info(f"Discovered {len(discovered_tools)} tools for {config.name}")
                    
        except Exception as e:
            logger.error(f"Failed to discover tools for {config.name}: {e}")
            self.tool_metadata[server_id] = []
    
    async def get_all_tools(self):
        '''Get all discovered tools across all servers'''
        all_tools = []
        for server_id, tools in self.tool_metadata.items():
            if self.is_server_active(server_id):
                all_tools.extend(tools)
        return all_tools
    
    async def get_server_tools(self, server_id: str):
        '''Get tools for specific server'''
        if self.is_server_active(server_id):
            return self.tool_metadata.get(server_id, [])
        return []
    
    def get_tool_summary(self):
        '''Get summary of all tools for frontend display'''
        active_servers = sum(1 for sid in self.tool_metadata.keys() if self.is_server_active(sid))
        total_tools = sum(len(tools) for sid, tools in self.tool_metadata.items() if self.is_server_active(sid))
        
        return {
            "active_servers": active_servers,
            "total_tools": total_tools,
            "servers": {
                sid: {
                    "tool_count": len(tools),
                    "tools": [tool["name"] for tool in tools]
                }
                for sid, tools in self.tool_metadata.items() 
                if self.is_server_active(sid)
            }
        }
"""

# File: src/api/main.py  
# Add these new API endpoints

API_ENDPOINTS = """
@app.get("/api/mcp/tools/summary")
async def get_mcp_tools_summary():
    '''Get summary of available MCP tools for frontend'''
    try:
        summary = mcp_registry.get_tool_summary()
        return {
            "success": True,
            "data": summary
        }
    except Exception as e:
        logger.error(f"Failed to get MCP tools summary: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/mcp/tools") 
async def get_all_mcp_tools():
    '''Get all available MCP tools'''
    try:
        tools = await mcp_registry.get_all_tools()
        return {
            "success": True,
            "tools": tools,
            "count": len(tools)
        }
    except Exception as e:
        logger.error(f"Failed to get MCP tools: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/mcp/servers/{server_id}/tools")
async def get_server_tools(server_id: str):
    '''Get tools for specific MCP server'''
    try:
        tools = await mcp_registry.get_server_tools(server_id)
        return {
            "success": True,
            "server_id": server_id,
            "tools": tools,
            "count": len(tools)
        }
    except Exception as e:
        logger.error(f"Failed to get tools for server {server_id}: {e}")
        return {
            "success": False,
            "error": str(e)
        }
"""

# File: ui/src/components/MCPStatus.tsx
# Update frontend to show real tool counts

FRONTEND_UPDATE = """
// Update the MCP status component to call the new API
const MCPStatus = () => {
  const [toolSummary, setToolSummary] = useState(null);
  
  useEffect(() => {
    const fetchToolSummary = async () => {
      try {
        const response = await fetch('/api/mcp/tools/summary');
        const data = await response.json();
        if (data.success) {
          setToolSummary(data.data);
        }
      } catch (error) {
        console.error('Failed to fetch MCP tool summary:', error);
      }
    };
    
    fetchToolSummary();
    const interval = setInterval(fetchToolSummary, 10000); // Update every 10s
    return () => clearInterval(interval);
  }, []);
  
  return (
    <div className="mcp-status">
      <h3>MCP Server Status</h3>
      {toolSummary ? (
        <div>
          <p>Active Servers: {toolSummary.active_servers}</p>
          <p>Total Tools: {toolSummary.total_tools}</p>
          <details>
            <summary>Server Details</summary>
            {Object.entries(toolSummary.servers).map(([serverId, data]) => (
              <div key={serverId}>
                <strong>{serverId}</strong>: {data.tool_count} tools
                <ul>
                  {data.tools.map(tool => <li key={tool}>{tool}</li>)}
                </ul>
              </div>
            ))}
          </details>
        </div>
      ) : (
        <p>Loading MCP status...</p>
      )}
    </div>
  );
};
"""

print("=" * 60)
print("MCP Tool Discovery Integration Plan")
print("=" * 60)
print()
print("STEP 1: Enhance MCP Server Registry")
print("- Add tool discovery to server registration flow")
print("- Store discovered tool metadata")
print("- Provide tool query methods")
print()
print("STEP 2: Add API Endpoints")  
print("- /api/mcp/tools/summary - Tool counts for frontend")
print("- /api/mcp/tools - All available tools")
print("- /api/mcp/servers/{id}/tools - Server-specific tools")
print()
print("STEP 3: Update Frontend")
print("- Show real tool counts instead of '0 tools'")
print("- Display server details and available tools")
print("- Auto-refresh tool status")
print()
print("EXPECTED RESULT:")
print("✅ Frontend shows: '13 tools from 2 servers'")
print("✅ Agents can discover and use MCP tools")
print("✅ Evolution system factors in tool availability")
print("✅ Context7 docs and filesystem ops available")
print()
print("This fixes the core architectural flaw identified in the MCP specification analysis.")
