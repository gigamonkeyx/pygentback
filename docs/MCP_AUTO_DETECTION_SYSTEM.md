# MCP Server Auto-Detection and Discovery System

## Overview

PyGent Factory implements a comprehensive MCP (Model Context Protocol) server auto-detection and discovery system that enables robust agent code evolution through dynamic tool and capability management. The system provides both static configuration-based server loading and dynamic marketplace-based server discovery and installation.

## Architecture Components

### 1. Core Components

#### A. Server Configuration (`mcp_server_configs.json`)
- **Location**: `d:\mcp\pygent-factory\mcp_server_configs.json`
- **Purpose**: Static configuration file defining the core MCP servers that should be available
- **Structure**: JSON array of server definitions with metadata, commands, capabilities, and startup options

#### B. Server Registry (`src/mcp/server_registry.py`)
- **Purpose**: Central registry for managing MCP server instances
- **Responsibilities**:
  - Server lifecycle management (start, stop, restart)
  - Health monitoring and status tracking
  - Server process management
  - Configuration validation

#### C. Real Server Loader (`src/mcp/real_server_loader.py`)
- **Purpose**: Loads and initializes MCP servers from configuration
- **Process**:
  1. Reads `mcp_server_configs.json`
  2. Validates server configurations
  3. Starts configured servers with auto_start=true
  4. Registers servers in the server registry
  5. Stores initialization results in app_state

#### D. Tool Discovery Engine (`src/mcp/tool_discovery.py`)
- **Purpose**: Discovers and catalogs tools/capabilities from running MCP servers
- **MCP Spec Compliance**: Implements the official MCP `tools/list` protocol
- **Process**:
  1. Connects to each running MCP server
  2. Calls `tools/list` to enumerate available tools
  3. Stores tool metadata and capabilities
  4. Caches discovery results for performance

### 2. API Layer (`src/api/routes/mcp.py`)

#### Discovery Endpoints
- `GET /api/v1/mcp/discovery/status` - Returns discovery system status
- `GET /api/v1/mcp/discovery/servers` - Lists discovered servers and their capabilities
- `POST /api/v1/mcp/discovery/refresh` - Triggers server re-discovery
- `GET /api/v1/mcp/servers` - Lists all registered servers

#### Management Endpoints
- `POST /api/v1/mcp/servers/{server_id}/install` - Installs a new MCP server
- `POST /api/v1/mcp/servers/{server_id}/start` - Starts a server
- `POST /api/v1/mcp/servers/{server_id}/stop` - Stops a server
- `GET /api/v1/mcp/servers/{server_id}/status` - Gets server status
- `GET /api/v1/mcp/tools` - Lists all available tools across servers

### 3. Frontend Marketplace (`ui/src/pages/MCPMarketplacePage.tsx`)

#### Discovery UI Features
- **Server Status Dashboard**: Real-time status of all registered servers
- **Capability Browser**: Explore tools and capabilities by category
- **One-Click Installation**: Install servers from community/official repositories
- **Health Monitoring**: Visual indicators for server health and availability

#### Discovery Flow
1. Page loads and fetches discovery status from `/api/v1/mcp/discovery/status`
2. Displays available servers from `/api/v1/mcp/discovery/servers`
3. Polls for server status updates every 5 seconds
4. Supports installation of new servers with progress tracking
5. Shows real-time tool/capability availability

### 4. Caching and Persistence

#### Discovery Cache
- **Location**: `./data/mcp_cache/discovered_servers.json`
- **Contents**: Cached server metadata, capabilities, and discovery results
- **Update Strategy**: Refreshed on server startup, configuration changes, or manual refresh

#### App State Management
- **Location**: FastAPI app_state object in memory
- **Contents**: 
  - MCP server registry instance
  - Discovery results and status
  - Active tool catalog
  - Server health metrics

## Server Auto-Detection Flow

### 1. System Startup
1. **FastAPI Initialization** (`src/api/main.py`):
   - Creates MCP server registry
   - Initializes discovery system
   - Stores components in app_state

2. **Server Loading** (`src/mcp/real_server_loader.py`):
   - Reads `mcp_server_configs.json`
   - Validates each server configuration
   - Starts servers with `auto_start: true`
   - Registers successful starts in server registry

3. **Tool Discovery** (`src/mcp/tool_discovery.py`):
   - Enumerates tools from each running server
   - Calls MCP `tools/list` on each server
   - Builds comprehensive tool catalog
   - Caches results for fast access

### 2. Runtime Discovery
1. **Continuous Monitoring**:
   - Health checks on registered servers
   - Automatic restart on failure (if configured)
   - Tool availability updates

2. **Dynamic Installation**:
   - API endpoints support runtime server installation
   - New servers are registered and started automatically
   - Tools are discovered and added to catalog

3. **UI Integration**:
   - Real-time status updates via polling
   - Installation progress tracking
   - Visual server and tool management

## Required MCP Servers Status

### ✅ Currently Configured and Available

1. **Filesystem Server** (`@modelcontextprotocol/server-filesystem`)
   - **Purpose**: File system operations (read, write, search, directory listing)
   - **Status**: ✅ Configured, Built, Auto-start enabled
   - **Location**: `mcp-servers/src/filesystem/`

2. **Fetch Server** (`@modelcontextprotocol/server-fetch`)
   - **Purpose**: HTTP requests and web content retrieval
   - **Status**: ✅ Configured, Built, Auto-start enabled
   - **Location**: `mcp-servers/src/fetch/`

3. **Time Server** (`@modelcontextprotocol/server-time`)
   - **Purpose**: Time/date operations and timezone management
   - **Status**: ✅ Configured, Built, Auto-start enabled
   - **Location**: `mcp-servers/src/time/`

4. **Sequential Thinking Server** (`@modelcontextprotocol/server-sequentialthinking`)
   - **Purpose**: Reasoning chains and logical progression
   - **Status**: ✅ Configured, Built, Auto-start enabled
   - **Location**: `mcp-servers/src/sequentialthinking/`

5. **Memory Server** (`@modelcontextprotocol/server-memory`)
   - **Purpose**: Context persistence and session memory
   - **Status**: ✅ Configured, Built, Auto-start enabled
   - **Location**: `mcp-servers/src/memory/`

6. **Git Server** (`@modelcontextprotocol/server-git`)
   - **Purpose**: Version control operations
   - **Status**: ✅ Configured, Built, Auto-start enabled
   - **Location**: `mcp-servers/src/git/`

7. **Cloudflare Server** (`@cloudflare/mcp-server-cloudflare`)
   - **Purpose**: Cloudflare infrastructure management
   - **Status**: ✅ Configured, Repository available
   - **Location**: `mcp-server-cloudflare/`

### ⚠️ Servers Requiring Attention

8. **Python Server** (`mcp_server_python`)
   - **Purpose**: Python code execution and analysis
   - **Status**: ⚠️ Configured but repository not found
   - **Action Required**: Find or implement Python MCP server

### ❌ Removed for Security

9. **~~A2A Server~~** (REMOVED)
   - **Reason**: Security concerns and repository unavailability
   - **Status**: ❌ Removed from configuration
   - **Alternative**: Agent communication handled through core PyGent Factory systems

## Security Considerations

### Server Validation
- All configured servers require verification of source repositories
- Only official ModelContextProtocol and verified community servers are used
- Commands use full paths to prevent PATH injection attacks

### Process Isolation
- Each MCP server runs in its own process
- Timeouts prevent runaway processes
- Restart limits prevent infinite restart loops

### Configuration Security
- Server commands are explicitly defined in configuration
- No dynamic command generation from user input
- Environment isolation for server processes

## Performance Optimizations

### Caching Strategy
- Discovery results cached to avoid repeated server queries
- Tool metadata cached for fast lookup
- Server status cached with TTL for performance

### Lazy Loading
- Servers with `auto_start: false` load on demand
- Tool discovery only runs when servers are active
- UI polling optimized to reduce server load

### Resource Management
- Server restart limits prevent resource exhaustion
- Timeout controls prevent hanging operations
- Process cleanup on server shutdown

## Troubleshooting and Monitoring

### Health Checks
- Server process monitoring
- Communication channel validation
- Tool availability verification

### Logging and Diagnostics
- Server startup/shutdown events logged
- Discovery failures tracked and reported
- Performance metrics collected

### Recovery Mechanisms
- Automatic server restart on failure
- Discovery cache rebuilding
- Graceful degradation when servers unavailable

## Configuration Examples

### Adding a New Server
```json
{
  "id": "new-server",
  "name": "New MCP Server",
  "command": ["npx", "-y", "@example/mcp-server"],
  "capabilities": ["example-capability"],
  "transport": "stdio",
  "config": {
    "category": "utilities",
    "author": "Example Author",
    "verified": true,
    "description": "Example server description",
    "priority": 3
  },
  "auto_start": true,
  "restart_on_failure": true,
  "max_restarts": 3,
  "timeout": 30
}
```

### Runtime Installation via API
```bash
curl -X POST "http://localhost:8000/api/v1/mcp/servers/new-server/install" \
  -H "Content-Type: application/json" \
  -d '{"source": "npm", "package": "@example/mcp-server"}'
```

## Future Enhancements

### Planned Features
1. **Auto-Discovery from GitHub**: Scan GitHub repositories for MCP servers
2. **Version Management**: Support multiple versions of the same server
3. **Dependency Resolution**: Automatic installation of server dependencies
4. **Performance Profiling**: Tool usage analytics and performance metrics
5. **Server Marketplace**: Community-driven server discovery and ratings

### Integration Roadmap
1. **Agent Evolution Engine**: Servers dynamically selected based on agent needs
2. **Capability Matching**: Automatic server recommendation for specific tasks
3. **Resource Optimization**: Smart server lifecycle management
4. **Security Scanning**: Automated security validation of community servers

## Conclusion

The MCP server auto-detection and discovery system in PyGent Factory provides a robust foundation for agent code evolution through:

- **Comprehensive Server Management**: Static configuration + dynamic discovery
- **Real-time Tool Availability**: Live tool catalog with health monitoring  
- **Secure Operation**: Validated servers with process isolation
- **Performance Optimization**: Caching and lazy loading strategies
- **User-Friendly Interface**: Marketplace UI for server management
- **Extensibility**: API-driven architecture supporting future enhancements

All required MCP servers (except Python) are successfully configured, built, and ready for agent use. The a2a server has been removed for security reasons. The system is production-ready for robust agent code evolution scenarios.
