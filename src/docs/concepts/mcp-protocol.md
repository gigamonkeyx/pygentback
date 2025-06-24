# MCP Protocol Integration

PyGent Factory is built on the **Model Context Protocol (MCP)**, a standardized protocol for AI agent communication and tool integration. This page explains how MCP works within PyGent Factory and how to extend it for your needs.

## What is the Model Context Protocol?

<div class="alert info">
<strong>💡 MCP Definition</strong><br>
The Model Context Protocol (MCP) is an open standard for communication between AI models and external tools/services. It provides a structured way for AI agents to access external capabilities like file systems, databases, web services, and custom tools.
</div>

MCP solves several critical challenges in AI agent development:

1. **Standardization**: Common interface for all tool interactions
2. **Security**: Controlled access to external resources
3. **Extensibility**: Easy addition of new capabilities
4. **Interoperability**: Works across different AI systems

<ArchitectureDiagram
  title="MCP Protocol Overview"
  type="mermaid"
  content="graph LR
    A[AI Agent] --> B[MCP Client]
    B <--> C[MCP Server Registry]
    C --> D[Filesystem MCP]
    C --> E[GitHub MCP]
    C --> F[Database MCP]
    C --> G[Search MCP]
    C --> H[Custom MCP]
    
    style A fill:#e3f2fd
    style B fill:#e1f5fe
    style C fill:#f3e5f5
    style D fill:#e8f5e8
    style E fill:#e8f5e8
    style F fill:#e8f5e8
    style G fill:#e8f5e8
    style H fill:#e8f5e8"
  description="MCP provides a standardized way for AI agents to interact with external tools and services through a registry of MCP servers."
/>

## MCP in PyGent Factory

PyGent Factory implements MCP at its core, with several key components:

### MCPServerManager

The `MCPServerManager` is the central registry for all MCP servers:

```python
class MCPServerManager:
    """Central registry for MCP servers"""
    
    def __init__(self):
        self.servers = {}
        self.tools_registry = {}
    
    async def register_server(self, server_config: MCPServerConfig):
        """Register a new MCP server"""
        server = await self._create_server_instance(server_config)
        self.servers[server_config.name] = server
        
        # Register all tools from this server
        tools = await server.list_tools()
        for tool in tools:
            self.tools_registry[tool.name] = {
                "server": server_config.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
    
    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict
    ) -> ToolResult:
        """Execute a tool on an MCP server"""
        if server_name not in self.servers:
            raise ServerNotFoundError(f"Server {server_name} not found")
            
        server = self.servers[server_name]
        return await server.execute_tool(tool_name, arguments)
    
    async def list_available_tools(self) -> Dict[str, ToolInfo]:
        """List all available tools across all servers"""
        return self.tools_registry
```

### MCPServerRegistry

The `MCPServerRegistry` provides a global access point for MCP servers:

```python
class MCPServerRegistry:
    """Global registry for MCP servers"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'MCPServerRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.manager = MCPServerManager()
    
    async def initialize(self, config: Dict):
        """Initialize all configured MCP servers"""
        for server_config in config.get("servers", []):
            await self.manager.register_server(server_config)
    
    def get_manager(self) -> MCPServerManager:
        """Get the MCP server manager"""
        return self.manager
```

### FastMCP

`FastMCP` is a high-performance MCP client for agent-tool communication:

```python
class FastMCP:
    """High-performance MCP client"""
    
    def __init__(self, registry: MCPServerRegistry = None):
        self.registry = registry or MCPServerRegistry.get_instance()
        self.manager = self.registry.get_manager()
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict,
        server_name: Optional[str] = None
    ) -> Any:
        """Execute a tool by name"""
        if server_name is None:
            # Auto-discover server from tool name
            tool_info = self.manager.tools_registry.get(tool_name)
            if not tool_info:
                raise ToolNotFoundError(f"Tool {tool_name} not found")
            server_name = tool_info["server"]
        
        return await self.manager.call_tool(server_name, tool_name, arguments)
    
    async def list_tools(self) -> List[ToolInfo]:
        """List all available tools"""
        return list(self.manager.tools_registry.values())
```

## Built-in MCP Servers

PyGent Factory includes several built-in MCP servers:

### 1. Filesystem MCP

Provides file system operations with controlled access:

<CodeExample
  :tabs="[
    {
      name: 'Server Configuration',
      content: `# Configure Filesystem MCP
{
  "name": "filesystem",
  "type": "filesystem",
  "config": {
    "allowed_directories": [
      "/app/data",
      "/tmp/pygent"
    ],
    "read_only": false
  }
}`
    },
    {
      name: 'Usage Example',
      content: `# Using Filesystem MCP in an agent
from pygent_factory.mcp import FastMCP

async def read_file_example(mcp_client: FastMCP, path: str):
    result = await mcp_client.execute_tool(
        "read_file",
        {"path": path},
        server_name="filesystem"
    )
    return result.content

async def write_file_example(mcp_client: FastMCP, path: str, content: str):
    result = await mcp_client.execute_tool(
        "write_file",
        {"path": path, "content": content},
        server_name="filesystem"
    )
    return result.success`
    },
    {
      name: 'Available Tools',
      content: `# Filesystem MCP Tools
- read_file: Read file contents
- write_file: Write content to a file
- list_directory: List files in a directory
- create_directory: Create a new directory
- delete_file: Delete a file
- move_file: Move or rename a file
- copy_file: Copy a file
- get_file_info: Get file metadata
- search_files: Search for files matching a pattern`
    }
  ]"
  description="Filesystem MCP provides secure access to the file system with configurable permissions and a comprehensive set of file operations."
/>

### 2. GitHub MCP

Integrates with GitHub repositories for version control:

<CodeExample
  :tabs="[
    {
      name: 'Server Configuration',
      content: `# Configure GitHub MCP
{
  "name": "github",
  "type": "github",
  "config": {
    "auth_token": "${GITHUB_TOKEN}",
    "default_owner": "gigamonkeyx",
    "default_repo": "pygent",
    "allowed_repos": [
      "gigamonkeyx/pygent",
      "gigamonkeyx/pygent-factory"
    ]
  }
}`
    },
    {
      name: 'Usage Example',
      content: `# Using GitHub MCP in an agent
from pygent_factory.mcp import FastMCP

async def create_issue_example(mcp_client: FastMCP, title: str, body: str):
    result = await mcp_client.execute_tool(
        "create_issue",
        {
            "owner": "gigamonkeyx",
            "repo": "pygent",
            "title": title,
            "body": body
        },
        server_name="github"
    )
    return result.issue_number

async def get_file_contents_example(mcp_client: FastMCP, path: str):
    result = await mcp_client.execute_tool(
        "get_file_contents",
        {
            "owner": "gigamonkeyx",
            "repo": "pygent",
            "path": path,
            "branch": "main"
        },
        server_name="github"
    )
    return result.content`
    },
    {
      name: 'Available Tools',
      content: `# GitHub MCP Tools
- create_repository: Create a new repository
- fork_repository: Fork an existing repository
- create_issue: Create a new issue
- create_pull_request: Create a new pull request
- get_file_contents: Get file contents from a repository
- create_or_update_file: Create or update a file
- list_commits: List commits in a repository
- search_code: Search for code in repositories
- search_issues: Search for issues and pull requests
- get_pull_request: Get pull request details
- merge_pull_request: Merge a pull request`
    }
  ]"
  description="GitHub MCP provides comprehensive GitHub integration for version control, issue tracking, and repository management."
/>

### 3. PostgreSQL MCP

Provides database access with query capabilities:

<CodeExample
  :tabs="[
    {
      name: 'Server Configuration',
      content: `# Configure PostgreSQL MCP
{
  "name": "postgres",
  "type": "postgres",
  "config": {
    "connection_string": "postgresql://user:password@localhost:5432/pygent_factory",
    "read_only": true,
    "max_rows": 1000,
    "timeout_seconds": 30
  }
}`
    },
    {
      name: 'Usage Example',
      content: `# Using PostgreSQL MCP in an agent
from pygent_factory.mcp import FastMCP

async def query_database_example(mcp_client: FastMCP, sql: str):
    result = await mcp_client.execute_tool(
        "query",
        {"sql": sql},
        server_name="postgres"
    )
    return result.rows

# Example query
query = "SELECT agent_id, name, created_at FROM agents WHERE status = 'active' LIMIT 10"`
    },
    {
      name: 'Available Tools',
      content: `# PostgreSQL MCP Tools
- query: Execute a read-only SQL query
- describe_table: Get table schema information
- list_tables: List available tables
- get_row_count: Get the number of rows in a table
- export_query_results: Export query results to CSV`
    }
  ]"
  description="PostgreSQL MCP provides secure database access with read-only queries by default and configurable limits for resource protection."
/>

### 4. Search MCP

Enables web search and information retrieval:

<CodeExample
  :tabs="[
    {
      name: 'Server Configuration',
      content: `# Configure Search MCP
{
  "name": "search",
  "type": "search",
  "config": {
    "providers": [
      {
        "name": "google",
        "api_key": "${GOOGLE_API_KEY}",
        "cx": "${GOOGLE_CX_ID}"
      },
      {
        "name": "brave",
        "api_key": "${BRAVE_API_KEY}"
      },
      {
        "name": "semantic_scholar",
        "api_key": "${SEMANTIC_SCHOLAR_API_KEY}"
      }
    ],
    "default_provider": "brave",
    "max_results": 10,
    "cache_ttl_seconds": 3600
  }
}`
    },
    {
      name: 'Usage Example',
      content: `# Using Search MCP in an agent
from pygent_factory.mcp import FastMCP

async def web_search_example(mcp_client: FastMCP, query: str):
    result = await mcp_client.execute_tool(
        "web_search",
        {
            "query": query,
            "num_results": 5,
            "provider": "brave"
        },
        server_name="search"
    )
    return result.results

async def academic_search_example(mcp_client: FastMCP, query: str):
    result = await mcp_client.execute_tool(
        "academic_search",
        {
            "query": query,
            "num_results": 5,
            "provider": "semantic_scholar"
        },
        server_name="search"
    )
    return result.papers`
    },
    {
      name: 'Available Tools',
      content: `# Search MCP Tools
- web_search: Search the web for information
- academic_search: Search academic papers and research
- news_search: Search for news articles
- image_search: Search for images
- fetch_url: Fetch and parse content from a URL
- summarize_webpage: Fetch and summarize a webpage`
    }
  ]"
  description="Search MCP provides comprehensive search capabilities across web, academic, news, and image sources with multiple provider options."
/>

## Creating Custom MCP Servers

You can extend PyGent Factory with custom MCP servers:

### 1. Define Server Class

```python
from pygent_factory.mcp import MCPServer, Tool, ToolResult

class CustomAnalyticsServer(MCPServer):
    """Custom MCP server for data analytics"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.analytics.example.com")
        
    @Tool("analyze_data")
    async def analyze_data(self, data: str) -> ToolResult:
        """Analyze data and return insights"""
        # Implementation logic
        result = await self._call_analytics_api(data)
        return ToolResult(
            success=True,
            content=result,
            metadata={"source": "custom_analytics"}
        )
        
    @Tool("generate_report")
    async def generate_report(
        self,
        data: str,
        report_type: str = "summary"
    ) -> ToolResult:
        """Generate a report from data"""
        # Implementation logic
        report = await self._generate_report_from_data(data, report_type)
        return ToolResult(
            success=True,
            content=report,
            metadata={"report_type": report_type}
        )
        
    async def _call_analytics_api(self, data: str) -> Dict:
        """Call external analytics API"""
        # Implementation details
        pass
        
    async def _generate_report_from_data(
        self,
        data: str,
        report_type: str
    ) -> Dict:
        """Generate report from data"""
        # Implementation details
        pass
```

### 2. Register the Server

```python
# Register custom MCP server
from pygent_factory.mcp import MCPServerRegistry

async def register_custom_server():
    registry = MCPServerRegistry.get_instance()
    
    # Create server configuration
    server_config = {
        "name": "custom_analytics",
        "type": "custom",
        "class": "path.to.CustomAnalyticsServer",
        "config": {
            "api_key": "your-api-key",
            "base_url": "https://api.analytics.example.com"
        }
    }
    
    # Register the server
    await registry.manager.register_server(server_config)
```

### 3. Use in Agents

```python
# Use custom MCP server in an agent
from pygent_factory.mcp import FastMCP

async def use_custom_analytics(mcp_client: FastMCP, data: str):
    result = await mcp_client.execute_tool(
        "analyze_data",
        {"data": data},
        server_name="custom_analytics"
    )
    
    if result.success:
        insights = result.content
        print(f"Analysis insights: {insights}")
    else:
        print(f"Analysis failed: {result.error}")
```

## MCP Security Considerations

Security is a critical aspect of MCP implementation:

### Access Control

- **Server-level Permissions**: Configure which servers an agent can access
- **Tool-level Permissions**: Restrict access to specific tools
- **Parameter Validation**: Validate all tool parameters before execution

### Resource Limits

- **Rate Limiting**: Limit the number of tool calls per minute
- **Timeout Controls**: Set maximum execution time for tools
- **Result Size Limits**: Limit the size of tool results

### Audit Logging

- **Tool Call Logging**: Log all tool calls with parameters
- **Result Logging**: Log tool execution results
- **Error Tracking**: Track and alert on tool execution errors

## Best Practices

### MCP Server Design

1. **Single Responsibility**: Each MCP server should focus on a specific domain
2. **Clear Documentation**: Document all tools and parameters
3. **Error Handling**: Provide clear error messages and status codes
4. **Idempotency**: Design tools to be idempotent when possible
5. **Statelessness**: Minimize server-side state for better scalability

### Tool Implementation

1. **Parameter Validation**: Validate all input parameters
2. **Descriptive Names**: Use clear, descriptive tool names
3. **Consistent Return Format**: Maintain consistent return structures
4. **Appropriate Granularity**: Balance between too many small tools and too few large tools
5. **Performance Optimization**: Optimize for common use cases

## Debugging MCP

PyGent Factory includes tools for debugging MCP interactions:

<CodeExample
  code="# MCP debugging example
from pygent_factory.mcp import MCPDebugger

# Create debugger
debugger = MCPDebugger()

# Enable tracing
debugger.enable_tracing()

# Execute tool with tracing
result = await mcp_client.execute_tool(
    "read_file",
    {"path": "/app/data/example.txt"},
    server_name="filesystem"
)

# Get trace
trace = debugger.get_last_trace()
print(f"Tool call trace: {trace}")

# Disable tracing
debugger.disable_tracing()"
  language="python"
  description="MCP debugging tools help trace tool calls, parameters, and results for troubleshooting."
/>

## MCP Performance Optimization

Optimize MCP performance with these techniques:

### Caching

```python
# MCP caching example
from pygent_factory.mcp import MCPCache

# Create cache
cache = MCPCache(ttl_seconds=3600)

# Execute tool with caching
result = await mcp_client.execute_tool_cached(
    "web_search",
    {"query": "PyGent Factory documentation"},
    server_name="search",
    cache=cache
)
```

### Batching

```python
# MCP batching example
from pygent_factory.mcp import MCPBatcher

# Create batcher
batcher = MCPBatcher()

# Add tool calls to batch
batcher.add_call(
    "read_file",
    {"path": "/app/data/file1.txt"},
    server_name="filesystem"
)
batcher.add_call(
    "read_file",
    {"path": "/app/data/file2.txt"},
    server_name="filesystem"
)

# Execute batch
results = await batcher.execute(mcp_client)
```

### Parallel Execution

```python
# MCP parallel execution example
import asyncio
from pygent_factory.mcp import FastMCP

async def execute_parallel(mcp_client: FastMCP):
    # Create tasks
    tasks = [
        mcp_client.execute_tool(
            "read_file",
            {"path": f"/app/data/file{i}.txt"},
            server_name="filesystem"
        )
        for i in range(10)
    ]
    
    # Execute in parallel
    results = await asyncio.gather(*tasks)
    return results
```

## MCP Monitoring

Monitor MCP performance and health:

### Metrics Collection

```python
# MCP metrics example
from pygent_factory.mcp import MCPMetrics

# Create metrics collector
metrics = MCPMetrics()

# Enable metrics collection
metrics.enable()

# Execute tool with metrics
result = await mcp_client.execute_tool(
    "web_search",
    {"query": "PyGent Factory documentation"},
    server_name="search"
)

# Get metrics
tool_metrics = metrics.get_tool_metrics("web_search")
server_metrics = metrics.get_server_metrics("search")

print(f"Tool call count: {tool_metrics.call_count}")
print(f"Average response time: {tool_metrics.avg_response_time_ms}ms")
print(f"Error rate: {tool_metrics.error_rate}%")
```

### Health Checks

```python
# MCP health check example
from pygent_factory.mcp import MCPHealthChecker

# Create health checker
health_checker = MCPHealthChecker()

# Check all servers
health_status = await health_checker.check_all_servers()

for server_name, status in health_status.items():
    print(f"Server {server_name}: {'Healthy' if status.healthy else 'Unhealthy'}")
    if not status.healthy:
        print(f"  Error: {status.error}")
```

## Conclusion

The Model Context Protocol is a foundational element of PyGent Factory, providing a standardized, secure, and extensible way for agents to interact with external tools and services. By leveraging MCP, you can create powerful, flexible agent systems that can access a wide range of capabilities while maintaining security and performance.

**Next**: Learn about the [RAG System](/concepts/rag-system) that powers intelligent knowledge retrieval →