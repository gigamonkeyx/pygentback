# MCP Tool Discovery Analysis and Fix

## Current Architectural Flaw

Based on investigation of the PyGent Factory codebase and consultation of the official MCP specification (https://modelcontextprotocol.io/specification/), we have identified a **critical architectural flaw** in the current MCP server registration system:

### The Problem

**NO TOOL DISCOVERY IS HAPPENING**

The current MCP server registration process:
1. ✅ Successfully registers servers 
2. ✅ Successfully starts server processes
3. ❌ **NEVER calls `tools/list` to discover available tools**
4. ❌ **No tool metadata is captured or stored**
5. ❌ **Agents see zero tools and cannot use any MCP capabilities**

### MCP Specification Requirements

According to the official MCP specification:

#### Client MUST Requirements:
1. **Tool Discovery**: Clients MUST call `tools/list` to discover available tools from each server
2. **Tool Storage**: Clients MUST store the returned tool metadata (name, description, inputSchema, annotations)
3. **Tool Updates**: Clients SHOULD handle `notifications/tools/list_changed` for dynamic updates
4. **Capability Declaration**: Servers MUST declare `tools` capability with `listChanged` support

#### Server Message Flow:
```
Client → Server: tools/list request
Client ← Server: tools/list response (with tool definitions)
Client ← Server: notifications/tools/list_changed (when tools change)
Client → Server: tools/call (to invoke specific tools)
```

### Current Implementation Gap

**File: `src/mcp/server_registry.py`**
- ✅ Has `register_server()` method
- ✅ Has `start_server()` method  
- ❌ **Missing `discover_tools()` method**
- ❌ **No `tools/list` RPC calls**
- ❌ **No tool metadata storage**

**File: `src/mcp/server/manager.py`**
- ✅ Manages server lifecycle
- ❌ **No tool discovery integration**
- ❌ **No tool capability validation**

**Impact on DGM-Style Evolution:**
- Agents cannot discover available MCP tools
- Evolution system cannot factor in tool availability
- Self-improvement is blind to MCP capabilities
- Context7, Cloudflare servers provide zero value despite being "active"

## The Fix

### 1. Enhanced MCP Server Registry

We need to modify the MCP server registration to:

1. **Call `tools/list`** after successful server startup
2. **Store tool metadata** in both memory and database
3. **Handle tool updates** via notifications
4. **Expose tool discovery API** for agents
5. **Validate server capabilities** during registration

### 2. Tool Discovery Implementation

```python
# Enhanced server registry with tool discovery
class MCPServerRegistry:
    async def register_and_discover_server(self, server_config):
        # 1. Register server
        server = await self.register_server(server_config)
        
        # 2. Start server process
        await self.start_server(server.id)
        
        # 3. DISCOVER TOOLS (THIS IS MISSING!)
        await self.discover_server_tools(server.id)
        
        # 4. Store capabilities in database
        await self.persist_server_capabilities(server.id)
        
        return server
    
    async def discover_server_tools(self, server_id):
        """Call tools/list and store returned tool metadata"""
        server = self.get_server(server_id)
        
        # Make JSON-RPC call to tools/list
        tools_response = await self._call_server_method(
            server, "tools/list", {}
        )
        
        # Store tool metadata
        for tool in tools_response.get("tools", []):
            await self.store_tool_definition(server_id, tool)
```

### 3. Database Schema Enhancement

Current database has server tables but no tool storage:

```sql
-- Missing tool storage tables
CREATE TABLE mcp_tools (
    id UUID PRIMARY KEY,
    server_id UUID REFERENCES mcp_servers(id),
    name VARCHAR NOT NULL,
    description TEXT,
    input_schema JSONB,
    annotations JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_mcp_tools_server_id ON mcp_tools(server_id);
CREATE INDEX idx_mcp_tools_name ON mcp_tools(name);
```

### 4. Agent Tool Discovery API

Agents need an API to discover available tools:

```python
# New endpoint for agent tool discovery
@app.get("/api/mcp/tools")
async def get_available_tools():
    """Return all available MCP tools across all active servers"""
    tools = await mcp_registry.get_all_available_tools()
    return {
        "tools": tools,
        "servers": await mcp_registry.get_active_servers(),
        "total_tools": len(tools)
    }

@app.get("/api/mcp/tools/{server_id}")
async def get_server_tools(server_id: str):
    """Return tools for a specific MCP server"""
    return await mcp_registry.get_server_tools(server_id)
```

## Implementation Priority

### Phase 1: Core Tool Discovery (IMMEDIATE)
1. ✅ Modify `server_registry.py` to call `tools/list` after server startup
2. ✅ Add tool metadata storage in memory registry
3. ✅ Create tool discovery API endpoints
4. ✅ Test with Context7 and Cloudflare servers

### Phase 2: Database Persistence (HIGH)
1. ✅ Create tool storage database schema
2. ✅ Persist tool metadata across restarts
3. ✅ Handle tool updates via notifications

### Phase 3: Evolution Integration (MEDIUM)
1. ✅ Integrate tool discovery into evolution system
2. ✅ Factor tool availability into agent evolution
3. ✅ Enable dynamic tool capability assessment

## Expected Outcome

After implementing tool discovery:

1. **Agents will see actual MCP tools**: Context7 docs, Cloudflare APIs, filesystem operations
2. **Frontend will show tool counts**: Instead of "0 tools", show "15 tools from 4 servers"
3. **Evolution system will factor in tools**: Agents evolve based on both usage AND available tool capabilities
4. **Real MCP value delivered**: Context7 documentation, Cloudflare data, file operations become available to agents

## Test Validation

```python
# Test script to validate tool discovery
async def test_tool_discovery():
    # 1. Verify servers are running
    servers = await mcp_registry.get_active_servers()
    assert len(servers) > 0
    
    # 2. Verify tools are discovered
    tools = await mcp_registry.get_all_available_tools()
    assert len(tools) > 0
    
    # 3. Verify Context7 tools specifically
    context7_tools = await mcp_registry.get_server_tools("context7")
    assert "resolve-library-id" in [t["name"] for t in context7_tools]
    assert "get-library-docs" in [t["name"] for t in context7_tools]
    
    # 4. Verify tools persist across restarts
    await restart_backend()
    tools_after_restart = await mcp_registry.get_all_available_tools()
    assert len(tools_after_restart) == len(tools)
```

This fix addresses the core issue preventing MCP servers from providing value to the PyGent Factory system and enables the DGM-style evolution to factor in actual tool availability.
