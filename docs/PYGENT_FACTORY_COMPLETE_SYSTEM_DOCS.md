# PyGent Factory Complete System Documentation

## Overview
PyGent Factory is a comprehensive agent evolution system with integrated MCP (Model Context Protocol) server management, real-time monitoring, and automated discovery capabilities.

## System Architecture

### Core Components
1. **Agent Factory**: Creates and manages intelligent agents
2. **MCP Manager**: Handles Model Context Protocol server lifecycle
3. **Memory Manager**: Provides persistent memory and context management
4. **Retrieval System**: RAG-based knowledge retrieval
5. **Message Bus**: Inter-component communication
6. **Health Monitoring System**: Real-time system health tracking
7. **Discovery System**: Automated MCP server discovery and marketplace

## MCP Monitoring & Discovery System

### Health Monitoring Features
The system includes a comprehensive health monitoring infrastructure that was activated when Cloudflare MCP servers were added:

#### 1. MCP Health Monitor (`src/testing/mcp/health_monitor.py`)
- **Real-time Health Checks**: Continuous monitoring of all MCP servers
- **Performance Metrics**: Response time, CPU usage, memory consumption tracking
- **Alert System**: Configurable thresholds with callback-based alerting
- **Status Tracking**: HEALTHY, WARNING, CRITICAL, UNKNOWN, OFFLINE states
- **Historical Data**: Health history tracking for trend analysis

```python
class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"
    OFFLINE = "offline"
```

#### 2. Health API Endpoints (`src/api/routes/health.py`)
- `/api/v1/health` - Overall system health status
- `/api/v1/health/database` - Database connectivity and performance
- `/api/v1/health/agents` - Agent factory status and active agent count
- `/api/v1/health/mcp` - MCP server health and availability
- `/api/v1/health/memory` - Memory system performance
- `/api/v1/health/rag` - RAG system status
- `/api/v1/health/ollama` - Ollama integration health

#### 3. System Resource Monitoring
- **CPU Usage**: Per-server and system-wide CPU monitoring
- **Memory Usage**: Memory consumption tracking with alerts
- **Response Times**: Latency monitoring with configurable thresholds
- **Error Rates**: Failed request tracking and alerting
- **Uptime Tracking**: Server availability and uptime statistics

### MCP Discovery System

#### Real MCP Server Ecosystem (Updated 2025-06-08)
The system now features **11 working real MCP servers** with **100% real implementations** (all fake/mock servers eliminated):

**Local Python Servers:**
1. **Python Filesystem Server** - Real implementation from `punkpeye/mcp-filesystem-python`
   - Secure file operations with .gitignore support and path traversal protection
   - Command: `python mcp_servers/filesystem_server.py .`
   
2. **Fetch Server** - Official `mcp_server_fetch` module for HTTP requests
3. **Time Server** - Official `mcp_server_time` module for temporal operations  
4. **Git Server** - Official `mcp_server_git` module for version control

**Local Node.js Servers:**
5. **Sequential Thinking Server** - Custom reasoning chain management
6. **Memory Server** - Context persistence and session memory
7. **Python Code Server** - PyGent Factory custom development server

**JavaScript Servers (via npx):**
8. **Context7 Documentation** - Real `@upstash/context7-mcp` for library docs
9. **GitHub Repository Server** - Official `@modelcontextprotocol/server-github`

**Remote Cloudflare Servers (SSE):**
10. **Cloudflare Documentation** - Public documentation access ✅
11. **Cloudflare Radar** - Internet insights and security trends ✅

**Server Status: 11/13 Working (85% Success Rate)**
- 2 Failed servers are Cloudflare server-side HTTP 500 errors (not system issues)

#### 1. Server Discovery (`src/api/routes/mcp.py`)
The system provides automated MCP server discovery and marketplace functionality:

**Discovery Endpoints:**
- `/api/v1/mcp/discovery/servers` - List all discoverable MCP servers
- `/api/v1/mcp/discovery/install/{server_id}` - Install discovered servers
- `/api/v1/mcp/discovery/categories` - Browse servers by category
- `/api/v1/mcp/discovery/search` - Search available servers

**Marketplace Endpoints:**
- `/api/v1/mcp/marketplace/featured` - Featured MCP servers
- `/api/v1/mcp/marketplace/popular` - Popular servers by usage
- `/api/v1/mcp/marketplace/categories` - Server categories
- `/api/v1/mcp/marketplace/search` - Marketplace search

#### 2. Automated Server Registration
The system automatically registers and monitors MCP servers:

```python
# Server registration with health monitoring
await mcp_manager.register_server(
    server_id="cloudflare-browser",
    config=server_config,
    auto_restart=True,
    health_check_interval=30
)
```

#### 3. Dynamic Server Loading
- **Cache System**: `data/mcp_cache/discovered_servers.json` 
- **Configuration**: `mcp_server_configs.json`
- **Auto-detection**: Scans for available servers and capabilities
- **Install Commands**: Supports local, npm, and remote server installation

### Cloudflare MCP Integration

#### Registered Cloudflare Servers
1. **Cloudflare Documentation** (`npx mcp-remote https://docs.mcp.cloudflare.com/sse`)
   - Access to Cloudflare documentation and guides
   - Real-time documentation search and retrieval

2. **Cloudflare Workers Bindings** (`npx mcp-remote https://bindings.mcp.cloudflare.com/sse`)
   - Manage Workers KV, D1, R2, and other bindings
   - Environment variable and secret management

3. **Cloudflare Radar** (`npx mcp-remote https://radar.mcp.cloudflare.com/sse`)
   - Internet traffic and security analytics
   - Attack trends and threat intelligence

4. **Cloudflare Browser Rendering** (`npx mcp-remote https://browser.mcp.cloudflare.com/sse`)
   - Headless browser automation and screenshot capture
   - Web scraping and automated testing capabilities

#### Server Health Monitoring
Each Cloudflare server is monitored for:
- **Connection Status**: SSE connection health
- **Response Times**: API call latency
- **Error Rates**: Failed request tracking
- **Capability Status**: Available tools and resources

### Orchestration & Lifecycle Management

#### 1. MCP Server Lifecycle (`src/mcp/server/lifecycle.py`)
- **Automated Startup**: Servers start automatically with the system
- **Health Monitoring**: Continuous health checks with auto-restart
- **Graceful Shutdown**: Clean server termination
- **Error Recovery**: Automatic restart on failures
- **Process Management**: Resource usage monitoring and limits

#### 2. Orchestration Manager (`src/orchestration/orchestration_manager.py`)
- **System-wide Health**: Overall system health assessment
- **Performance Metrics**: System performance monitoring
- **Resource Optimization**: Dynamic resource allocation
- **Alert Management**: Centralized alerting and notification

### WebSocket Real-time Features

#### 1. Live Monitoring Dashboard
- **Real-time Health Updates**: Live health status via WebSocket
- **Performance Graphs**: Real-time metrics visualization
- **Alert Notifications**: Instant alert delivery
- **Server Status**: Live server availability updates

#### 2. Event Streaming
```python
# WebSocket events for monitoring
{
    "type": "health_update",
    "server_id": "cloudflare-browser", 
    "status": "healthy",
    "metrics": {...}
}
```

### Configuration & Deployment

#### 1. Server Configuration (`mcp_server_configs.json`)
```json
{
  "servers": [
    {
      "id": "cloudflare-browser",
      "name": "Cloudflare Browser Rendering",
      "command": ["npx", "mcp-remote", "https://browser.mcp.cloudflare.com/sse"],
      "category": "automation",
      "description": "Browser automation and rendering",
      "health_check_interval": 30,
      "auto_restart": true
    }
  ]
}
```

#### 2. Discovery Cache (`data/mcp_cache/discovered_servers.json`)
- **Server Metadata**: Descriptions, categories, capabilities
- **Install Commands**: How to install each server
- **Health Status**: Last known health state
- **Usage Statistics**: Installation and usage metrics

### API Integration

#### 1. FastAPI Backend
- **Async Operations**: Non-blocking health checks and monitoring
- **Dependency Injection**: Clean separation of concerns
- **Error Handling**: Comprehensive error handling and logging
- **Documentation**: Auto-generated API documentation

#### 2. Frontend Integration
- **React Components**: Health monitoring dashboards
- **Real-time Updates**: WebSocket-based live updates
- **Interactive Controls**: Server management and configuration
- **Visual Indicators**: Health status visualization

### Performance & Scalability

#### 1. Monitoring Efficiency
- **Configurable Intervals**: Adjustable health check frequency
- **Batch Operations**: Efficient bulk health checks
- **Resource Limits**: CPU and memory usage limits
- **Caching**: Intelligent caching of health data

#### 2. Alert System
- **Threshold Configuration**: Customizable alert thresholds
- **Multiple Channels**: Email, WebSocket, webhook alerts
- **Alert Aggregation**: Prevent alert flooding
- **Auto-resolution**: Automatic alert resolution detection

### Security Features

#### 1. Server Validation
- **Secure Connections**: HTTPS/WSS for remote servers
- **Authentication**: Token-based authentication where required
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: Request rate limiting and throttling

#### 2. Health Data Security
- **Access Control**: Role-based access to health data
- **Data Encryption**: Encrypted health data storage
- **Audit Logging**: Health access audit trails
- **Privacy Protection**: Sensitive data anonymization

## Usage Examples

### Starting the System
```bash
# Start backend with full monitoring
python -m src.api.main

# Validate all servers
python validate_mcp_servers.py

# Monitor health in real-time
curl http://localhost:8000/api/v1/health
```

### Health Monitoring
```python
# Check specific server health
response = await client.get("/api/v1/health/mcp")
print(f"MCP Health: {response.json()}")

# Subscribe to health updates
websocket = await websocket_connect("/ws")
await websocket.send_json({"type": "subscribe", "events": ["health_updates"]})
```

### Server Discovery
```python
# Discover available servers
servers = await client.get("/api/v1/mcp/discovery/servers")

# Install a discovered server  
result = await client.post("/api/v1/mcp/discovery/install/cloudflare-docs")
```

## Troubleshooting

### Common Issues
1. **Server Startup Failures**: Check logs in health endpoint
2. **Connection Issues**: Verify network connectivity and firewall
3. **Performance Degradation**: Monitor resource usage via health API
4. **Missing Servers**: Check discovery cache and configuration files

### Monitoring Commands
```bash
# Check system health
curl http://localhost:8000/api/v1/health | jq

# Monitor specific component
curl http://localhost:8000/api/v1/health/mcp | jq

# View discovered servers
curl http://localhost:8000/api/v1/mcp/discovery/servers | jq
```

## Future Enhancements

### Planned Features
1. **Advanced Analytics**: Machine learning-based anomaly detection
2. **Predictive Monitoring**: Proactive issue identification
3. **Auto-scaling**: Dynamic server scaling based on demand
4. **Custom Dashboards**: User-configurable monitoring dashboards
5. **Integration APIs**: Third-party monitoring system integration

### Performance Improvements
1. **Optimized Health Checks**: More efficient health checking algorithms
2. **Distributed Monitoring**: Multi-node monitoring capabilities
3. **Enhanced Caching**: Improved caching strategies
4. **Real-time Analytics**: Live performance analytics and insights

---

This documentation covers the comprehensive monitoring and discovery system that was unlocked by integrating Cloudflare MCP servers into the PyGent Factory system. The system provides enterprise-grade monitoring, health checking, and automated discovery capabilities for robust MCP server management.
