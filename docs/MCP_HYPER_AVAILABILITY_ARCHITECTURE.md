# MCP Hyper-Availability Architecture for PyGent Factory

## Understanding MCP Tool Integration & Agent Decision Making

### 🎯 **What Makes an Agent Call a Tool/MCP?**

In the PyGent Factory system, agents call MCP tools through several mechanisms:

#### 1. **Automatic Tool Discovery**
```python
# During agent initialization
async def _initialize_mcp_tools(self):
    """Discover and register available MCP tools"""
    self.available_tools = await self.mcp_manager.get_available_tools()
    self.tool_executor = MCPToolExecutor(self.mcp_manager)
```

#### 2. **LLM-Driven Tool Selection**
```python
# The LLM decides when to use tools based on:
# - Function descriptions in the prompt
# - Available tool schemas
# - Context requirements
system_prompt = f"""
You are an agent with access to these tools:
{self.get_tool_descriptions()}

When you need to perform actions, use the appropriate tool.
"""
```

#### 3. **Explicit Tool Invocation**
```python
# Agents can explicitly call tools
result = await self.execute_tool("search_arxiv", {
    "query": "machine learning",
    "max_results": 10
})
```

## 🏗️ **MCP Hyper-Availability Architecture**

### Current Architecture Issues:
1. **Single Point of Failure**: MCP servers can go down
2. **No Redundancy**: Tools unavailable if server fails
3. **Limited Discovery**: Static tool registration
4. **No Load Balancing**: All requests to same server instance

### Proposed Hyper-Available Solution:

#### 1. **Multi-Instance MCP Registry**
```python
class HyperAvailableMCPRegistry:
    def __init__(self):
        self.server_instances = {}  # Multiple instances per server type
        self.tool_routing_table = {}  # Route tools to best instance
        self.health_monitors = {}   # Monitor each instance
        self.failover_strategies = {}  # Failover rules per tool
```

#### 2. **Native MCP Agent Integration**
```python
class MCPNativeAgent(BaseAgent):
    """Agent with native MCP tool integration"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.native_tools = {}  # Built-in tool implementations
        self.mcp_tools = {}     # External MCP tools
        self.tool_preferences = {}  # Preferred tool sources
    
    async def execute_tool_with_fallback(self, tool_name: str, args: dict):
        """Execute tool with multiple fallback options"""
        # 1. Try native implementation first (fastest)
        if tool_name in self.native_tools:
            return await self.native_tools[tool_name].execute(args)
        
        # 2. Try primary MCP server
        primary_result = await self.try_mcp_server("primary", tool_name, args)
        if primary_result.success:
            return primary_result
        
        # 3. Try backup MCP servers
        for backup_server in self.get_backup_servers(tool_name):
            backup_result = await self.try_mcp_server(backup_server, tool_name, args)
            if backup_result.success:
                return backup_result
        
        # 4. Use cached/degraded mode
        return await self.execute_degraded_mode(tool_name, args)
```

#### 3. **Tool Circuit Breaker Pattern**
```python
class MCPToolCircuitBreaker:
    """Prevents cascading failures in MCP tool calls"""
    
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call_tool(self, tool_func, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise MCPUnavailableError("Circuit breaker is OPEN")
        
        try:
            result = await tool_func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

## 🚀 **Implementation Plan for MCP Hyper-Availability**

### Phase 1: Native Tool Integration
```python
class NativeMCPTools:
    """Native implementations of common MCP tools"""
    
    @staticmethod
    async def search_arxiv(query: str, max_results: int = 10):
        """Native ArXiv search without external MCP dependency"""
        import arxiv
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results)
        return [{"title": r.title, "summary": r.summary} for r in client.results(search)]
    
    @staticmethod
    async def web_search(query: str, num_results: int = 5):
        """Native web search implementation"""
        # Implementation using direct API calls
        pass
```

### Phase 2: Multi-Instance Server Management
```python
class MCPServerPool:
    """Manage multiple instances of the same MCP server"""
    
    def __init__(self, server_type: str):
        self.server_type = server_type
        self.instances = []
        self.load_balancer = RoundRobinBalancer()
        self.health_checker = HealthChecker()
    
    async def add_instance(self, server_config):
        """Add a new server instance to the pool"""
        instance = MCPServer(server_config)
        await instance.start()
        self.instances.append(instance)
        self.health_checker.monitor(instance)
    
    async def execute_tool(self, tool_name: str, args: dict):
        """Execute tool on best available instance"""
        healthy_instances = self.health_checker.get_healthy_instances()
        
        if not healthy_instances:
            raise AllInstancesDownError(f"No healthy instances for {self.server_type}")
        
        instance = self.load_balancer.select(healthy_instances)
        return await instance.execute_tool(tool_name, args)
```

### Phase 3: Intelligent Tool Routing
```python
class IntelligentToolRouter:
    """Route tool calls to optimal execution method"""
    
    def __init__(self):
        self.tool_performance_stats = {}
        self.tool_availability_map = {}
        self.routing_rules = {}
    
    async def route_tool_call(self, agent_id: str, tool_name: str, args: dict):
        """Intelligently route tool call to best execution method"""
        
        # Check tool urgency and agent preferences
        urgency = self.assess_urgency(tool_name, args)
        agent_prefs = self.get_agent_preferences(agent_id)
        
        # Route based on criteria
        if urgency == "HIGH" and tool_name in agent_prefs.native_tools:
            return await self.execute_native(tool_name, args)
        
        elif urgency == "MEDIUM":
            return await self.execute_with_fallback(tool_name, args)
        
        else:
            return await self.execute_best_available(tool_name, args)
```

## 💡 **Recommended Implementation**

### 1. **Immediate: Enhance ProviderRegistry for MCP Tools**
```python
# Add to provider_registry.py
class ProviderRegistry:
    def __init__(self):
        # ...existing code...
        self.mcp_tool_registry = MCPToolRegistry()
        self.native_tool_registry = NativeToolRegistry()
    
    async def execute_tool_with_hyper_availability(self, 
                                                  tool_name: str, 
                                                  args: dict,
                                                  agent_id: str = None):
        """Execute tool with maximum availability guarantees"""
        
        # 1. Try native implementation (fastest, most reliable)
        if self.native_tool_registry.has_tool(tool_name):
            try:
                return await self.native_tool_registry.execute(tool_name, args)
            except Exception as e:
                logger.warning(f"Native tool {tool_name} failed: {e}")
        
        # 2. Try primary MCP server
        try:
            return await self.mcp_tool_registry.execute_primary(tool_name, args)
        except Exception as e:
            logger.warning(f"Primary MCP server failed for {tool_name}: {e}")
        
        # 3. Try backup MCP servers
        for backup in self.mcp_tool_registry.get_backups(tool_name):
            try:
                return await backup.execute(tool_name, args)
            except Exception as e:
                continue
        
        # 4. Return cached result or graceful degradation
        return await self.get_cached_or_degraded_result(tool_name, args)
```

### 2. **Agent Integration**
```python
# Add to base agent
class BaseAgent:
    async def _execute_tool(self, tool_name: str, arguments: dict):
        """Execute tool with hyper-availability"""
        provider_registry = get_provider_registry()
        return await provider_registry.execute_tool_with_hyper_availability(
            tool_name=tool_name,
            args=arguments,
            agent_id=self.agent_id
        )
```

## 🎯 **Key Benefits of This Architecture**

1. **Zero Downtime**: Multiple fallback mechanisms ensure tools are always available
2. **Performance Optimization**: Native tools execute faster than external MCP calls
3. **Intelligent Routing**: Best execution method chosen based on context
4. **Graceful Degradation**: System continues working even with partial failures
5. **Scalability**: Easy to add more server instances or native implementations

## 📋 **Next Steps**

1. **Implement Native Tool Library**: Start with most critical tools (ArXiv, web search)
2. **Enhance MCP Registry**: Add multi-instance support and health monitoring
3. **Update Agent Base Class**: Integrate hyper-available tool execution
4. **Add Circuit Breakers**: Prevent cascading failures
5. **Performance Monitoring**: Track tool execution metrics and optimize routing

This architecture ensures that agents have **hyper-available access to MCP tools** through multiple redundancy layers, intelligent routing, and graceful degradation strategies.
