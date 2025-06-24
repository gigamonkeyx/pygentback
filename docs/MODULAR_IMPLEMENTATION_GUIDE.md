# PyGent Factory - Modular Implementation Guide

## 🚀 Getting Started with the Modular Architecture

This guide provides practical examples and best practices for leveraging PyGent Factory's new modular architecture. Whether you're building new features or migrating existing code, this guide will help you make the most of the modular design.

## 📋 Quick Start Checklist

### **For New Projects**
- [ ] Use modular interfaces from day one
- [ ] Implement comprehensive type hints
- [ ] Set up monitoring and health checks
- [ ] Configure appropriate backends for your use case
- [ ] Implement proper error handling and logging

### **For Existing Projects**
- [ ] Assess current usage patterns
- [ ] Plan gradual migration strategy
- [ ] Test compatibility with legacy wrappers
- [ ] Migrate component by component
- [ ] Validate performance improvements

## 🏗️ Architecture Patterns

### **1. Agent Development Pattern**

```python
# Modern modular approach
from src.agents.factory import AgentFactory, AgentConfig
from src.agents.specialized import CodeAnalysisAgent, DataProcessingAgent

# Create specialized agents with factory
factory = AgentFactory(settings)

# Code analysis agent
code_agent_config = AgentConfig(
    agent_type="code_analysis",
    capabilities=["code_review", "bug_detection", "optimization"],
    model_config={"model": "gpt-4", "temperature": 0.1},
    tools=["static_analyzer", "code_formatter"]
)
code_agent = await factory.create_agent(code_agent_config)

# Data processing agent
data_agent_config = AgentConfig(
    agent_type="data_processing",
    capabilities=["data_analysis", "visualization", "reporting"],
    model_config={"model": "gpt-4", "temperature": 0.3},
    tools=["pandas_analyzer", "chart_generator"]
)
data_agent = await factory.create_agent(data_agent_config)

# Agents automatically register and become discoverable
agents = await factory.list_agents(capability="code_review")
```

### **2. Storage Strategy Pattern**

```python
# Multi-backend storage strategy
from src.storage.vector import VectorStoreManager, VectorQuery
from src.storage.vector.postgresql import PostgreSQLVectorStore
from src.storage.vector.faiss import FAISSVectorStore

# Initialize manager with multiple backends
manager = VectorStoreManager(settings, db_manager)
await manager.initialize()

# Production: PostgreSQL for reliability
postgres_config = {
    "type": "postgresql",
    "host": "prod-db.example.com",
    "database": "vectors",
    "schema": "production"
}
await manager.add_store("production", postgres_config)

# Development: FAISS for speed
faiss_config = {
    "type": "faiss",
    "persist_directory": "./dev_vectors",
    "index_type": "IVFFlat"
}
await manager.add_store("development", faiss_config)

# Smart routing based on environment
store_name = "production" if settings.environment == "prod" else "development"
query = VectorQuery(
    query_vector=embedding,
    collection="documents",
    limit=10,
    similarity_threshold=0.8
)
results = await manager.search_similar(query, store_name=store_name)
```

### **3. RAG Implementation Pattern**

```python
# Advanced RAG with multiple strategies
from src.rag.retrieval import RetrievalManager, RetrievalQuery, RetrievalStrategy
from src.rag.retrieval.scorer import ScoringWeights

# Configure advanced scoring
weights = ScoringWeights(
    similarity=0.4,      # Semantic similarity
    temporal=0.2,        # Recency boost
    authority=0.2,       # Source authority
    quality=0.15,        # Content quality
    keyword=0.05         # Keyword matching
)

retrieval_manager = RetrievalManager(
    vector_store_manager, 
    embedding_service, 
    settings
)

# Multi-strategy retrieval for comprehensive results
async def advanced_rag_query(question: str, context: str = None):
    # Primary semantic search
    semantic_query = RetrievalQuery(
        text=question,
        strategy=RetrievalStrategy.SEMANTIC,
        max_results=15,
        similarity_threshold=0.75,
        context=context,
        rerank=True,
        diversify=True
    )
    
    # Get results from multiple strategies
    strategies = [RetrievalStrategy.SEMANTIC, RetrievalStrategy.HYBRID]
    ensemble_results = await retrieval_manager.ensemble_retrieve(
        semantic_query, 
        strategies
    )
    
    return ensemble_results[:10]  # Top 10 results

# Usage
results = await advanced_rag_query(
    "How to implement async functions in Python?",
    context="Python development tutorial"
)
```

### **4. MCP Integration Pattern**

```python
# Enterprise MCP server management
from src.mcp.server import MCPServerManager, MCPServerConfig, MCPServerType
from src.mcp.server.config import create_filesystem_server_config, create_postgres_server_config

manager = MCPServerManager(settings)
await manager.initialize()

# Filesystem server for document access
fs_config = create_filesystem_server_config(
    name="document_fs",
    root_path="/secure/documents"
)
fs_server_id = await manager.register_server(fs_config)

# Database server for data operations
db_config = create_postgres_server_config(
    name="analytics_db",
    connection_string="postgresql://user:pass@db.example.com/analytics"
)
db_server_id = await manager.register_server(db_config)

# Tool discovery and execution
async def execute_mcp_tool(tool_name: str, arguments: dict):
    # Find appropriate server
    server_id = await manager.find_tool_server(tool_name)
    if not server_id:
        raise ValueError(f"Tool {tool_name} not available")
    
    # Execute with error handling
    try:
        result = await manager.call_tool(tool_name, arguments)
        return result
    except Exception as e:
        # Automatic retry with different server if available
        servers = await manager.get_servers_by_tool(tool_name)
        for backup_server in servers[1:]:  # Try backup servers
            try:
                result = await manager.call_tool(tool_name, arguments)
                return result
            except Exception:
                continue
        raise e

# Usage
file_content = await execute_mcp_tool("read_file", {"path": "config.json"})
```

### **5. Communication Pattern**

```python
# Advanced communication with reliability
from src.communication.protocols import ProtocolManager, ProtocolMessage
from src.communication.protocols.base import MessagePriority, DeliveryMode

manager = ProtocolManager(settings)
await manager.initialize()

# Register multiple protocols
await manager.register_protocol(ProtocolType.INTERNAL, internal_protocol)
await manager.register_protocol(ProtocolType.HTTP, http_protocol, "external_api")
await manager.register_protocol(ProtocolType.WEBSOCKET, ws_protocol, "realtime")

# Reliable message sending
async def send_critical_message(recipient: str, payload: dict):
    message = ProtocolMessage(
        sender="system",
        recipient=recipient,
        message_type="critical_update",
        payload=payload,
        priority=MessagePriority.HIGH,
        delivery_mode=DeliveryMode.AT_LEAST_ONCE,
        ttl=300,  # 5 minutes
        max_retries=5,
        retry_delay=2.0
    )
    
    success = await manager.send_message(message)
    if not success:
        # Escalate to urgent and try broadcast
        message.priority = MessagePriority.URGENT
        results = await manager.broadcast_message(
            message,
            exclude_protocols=["failed_protocol"]
        )
        return any(results.values())
    
    return success

# Real-time communication
async def setup_realtime_communication():
    # Add message handlers
    async def handle_realtime_update(message: ProtocolMessage):
        if message.message_type == "agent_status":
            await update_agent_dashboard(message.payload)
        elif message.message_type == "system_alert":
            await send_notification(message.payload)
    
    manager.add_message_handler("agent_status", handle_realtime_update)
    manager.add_message_handler("system_alert", handle_realtime_update)
```

## 🔧 Configuration Best Practices

### **Environment-Specific Configuration**

```python
# config/environments.py
class ProductionConfig:
    # High-performance, reliable backends
    VECTOR_STORE_CONFIG = {
        "default_type": "postgresql",
        "connection_pool_size": 20,
        "max_connections": 100
    }
    
    RAG_CONFIG = {
        "similarity_weight": 0.4,
        "temporal_weight": 0.2,
        "authority_weight": 0.2,
        "quality_weight": 0.15,
        "keyword_weight": 0.05
    }
    
    MCP_CONFIG = {
        "auto_restart": True,
        "max_restarts": 5,
        "health_check_interval": 30
    }

class DevelopmentConfig:
    # Fast, local backends
    VECTOR_STORE_CONFIG = {
        "default_type": "faiss",
        "persist_directory": "./dev_vectors"
    }
    
    RAG_CONFIG = {
        "similarity_weight": 0.8,  # Simpler scoring for dev
        "temporal_weight": 0.1,
        "authority_weight": 0.05,
        "quality_weight": 0.05,
        "keyword_weight": 0.0
    }
    
    MCP_CONFIG = {
        "auto_restart": False,  # Manual control in dev
        "max_restarts": 1
    }
```

### **Monitoring and Health Checks**

```python
# monitoring/health.py
async def comprehensive_health_check():
    health_status = {
        "overall": "healthy",
        "components": {}
    }
    
    # Check all modular components
    components = [
        ("agents", agent_factory),
        ("storage", vector_store_manager),
        ("rag", retrieval_manager),
        ("mcp", mcp_server_manager),
        ("communication", protocol_manager)
    ]
    
    for name, component in components:
        try:
            component_health = await component.health_check()
            health_status["components"][name] = component_health
            
            if component_health.get("overall_status") != "healthy":
                health_status["overall"] = "degraded"
                
        except Exception as e:
            health_status["components"][name] = {"status": "error", "error": str(e)}
            health_status["overall"] = "unhealthy"
    
    return health_status

# Usage in API endpoint
@app.get("/health")
async def health_endpoint():
    return await comprehensive_health_check()
```

## 🧪 Testing Strategies

### **Unit Testing with Modular Components**

```python
# tests/test_modular_components.py
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.factory import AgentFactory
from src.storage.vector import VectorStoreManager
from src.rag.retrieval import RetrievalManager

@pytest.fixture
async def mock_vector_store():
    mock_store = AsyncMock()
    mock_store.search_similar.return_value = [
        # Mock search results
    ]
    return mock_store

@pytest.fixture
async def retrieval_manager(mock_vector_store):
    mock_embedding_service = AsyncMock()
    manager = RetrievalManager(mock_vector_store, mock_embedding_service)
    return manager

async def test_retrieval_with_multiple_strategies(retrieval_manager):
    query = RetrievalQuery(
        text="test query",
        strategy=RetrievalStrategy.SEMANTIC,
        max_results=5
    )
    
    results = await retrieval_manager.retrieve(query)
    assert len(results) <= 5
    assert all(isinstance(r, RetrievalResult) for r in results)

async def test_agent_factory_creation():
    factory = AgentFactory(mock_settings)
    
    config = AgentConfig(
        agent_type="test_agent",
        capabilities=["test_capability"]
    )
    
    agent = await factory.create_agent(config)
    assert agent.agent_type == "test_agent"
    assert "test_capability" in agent.capabilities
```

### **Integration Testing**

```python
# tests/test_integration.py
async def test_end_to_end_rag_pipeline():
    # Set up full pipeline
    vector_manager = VectorStoreManager(test_settings, test_db)
    await vector_manager.initialize()
    
    retrieval_manager = RetrievalManager(vector_manager, embedding_service)
    
    # Add test documents
    test_docs = [create_test_document() for _ in range(10)]
    await vector_manager.add_documents(test_docs)
    
    # Test retrieval
    query = RetrievalQuery(text="test query", max_results=3)
    results = await retrieval_manager.retrieve(query)
    
    assert len(results) == 3
    assert all(r.similarity_score > 0.5 for r in results)
```

## 📈 Performance Optimization

### **Caching Strategies**

```python
# performance/caching.py
from functools import lru_cache
import asyncio

class PerformanceOptimizedRAG:
    def __init__(self, retrieval_manager):
        self.retrieval_manager = retrieval_manager
        self.query_cache = {}
        self.embedding_cache = {}
    
    @lru_cache(maxsize=1000)
    def _cache_key(self, query_text: str, strategy: str) -> str:
        return f"{hash(query_text)}:{strategy}"
    
    async def cached_retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        cache_key = self._cache_key(query.text, query.strategy.value)
        
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        results = await self.retrieval_manager.retrieve(query)
        
        # Cache results for 5 minutes
        self.query_cache[cache_key] = results
        asyncio.create_task(self._expire_cache(cache_key, 300))
        
        return results
    
    async def _expire_cache(self, key: str, delay: int):
        await asyncio.sleep(delay)
        self.query_cache.pop(key, None)
```

## 🔒 Security Best Practices

### **Secure Configuration**

```python
# security/config.py
class SecureConfig:
    def __init__(self):
        # Use environment variables for sensitive data
        self.database_url = os.getenv("DATABASE_URL")
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY")
        }
        
        # Validate required secrets
        self._validate_secrets()
    
    def _validate_secrets(self):
        required_secrets = ["DATABASE_URL", "OPENAI_API_KEY"]
        missing = [s for s in required_secrets if not os.getenv(s)]
        
        if missing:
            raise ValueError(f"Missing required secrets: {missing}")

# Use secure protocols
async def setup_secure_communication():
    # Enable TLS for all external communications
    http_config = {
        "base_url": "https://api.example.com",
        "verify_ssl": True,
        "timeout": 30
    }
    
    ws_config = {
        "url": "wss://secure.example.com",
        "ssl_context": ssl.create_default_context()
    }
```

## 🎯 Migration Checklist

### **Step-by-Step Migration**

1. **Assessment Phase**
   - [ ] Inventory current usage patterns
   - [ ] Identify performance bottlenecks
   - [ ] Plan migration priorities

2. **Preparation Phase**
   - [ ] Set up development environment with modular components
   - [ ] Create comprehensive tests
   - [ ] Establish monitoring baselines

3. **Migration Phase**
   - [ ] Migrate storage layer first
   - [ ] Update RAG implementations
   - [ ] Migrate MCP integrations
   - [ ] Update communication protocols
   - [ ] Migrate API endpoints

4. **Validation Phase**
   - [ ] Performance testing
   - [ ] Functionality verification
   - [ ] Load testing
   - [ ] Security validation

5. **Deployment Phase**
   - [ ] Staged rollout
   - [ ] Monitor performance metrics
   - [ ] Validate backward compatibility
   - [ ] Full production deployment

## 🎉 Success Metrics

Track these metrics to measure modularization success:

- **Performance**: Response times, throughput, resource usage
- **Reliability**: Error rates, uptime, recovery times
- **Maintainability**: Code complexity, test coverage, deployment frequency
- **Developer Experience**: Development velocity, onboarding time, debugging efficiency

The modular architecture provides the foundation for continuous improvement and scaling. Use these patterns and practices to build robust, high-performance AI agent systems with PyGent Factory!
