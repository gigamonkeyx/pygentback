# PyGent Factory Architecture

## **SYSTEM OVERVIEW**

PyGent Factory is a complete rebuild of the Agent Factory in Python, designed with MCP-first architecture to eliminate the TypeScript complexity and messaging system issues of the original implementation.

## **CORE PRINCIPLES**

1. **MCP Specification Compliance** - Follow official Model Context Protocol for stability, foundation, and security
2. **Clean Python Architecture** - No legacy TypeScript complexity or messaging fragmentation
3. **Vector-First RAG** - Comprehensive knowledge management with embeddings and semantic search
4. **Production Ready** - Docker, monitoring, testing, and deployment automation
5. **Modular Design** - Clear separation of concerns with dependency injection

## **TECHNOLOGY STACK**

### **Backend Framework**
- **FastAPI** - Modern async Python web framework
- **SQLAlchemy** - ORM with async support
- **Alembic** - Database migrations
- **Pydantic** - Data validation and serialization

### **Database & Storage**
- **Supabase (PostgreSQL)** - Primary database with vector extensions
- **pgvector** - Vector similarity search
- **ChromaDB** - Document embeddings and vector storage
- **Redis** - Caching and session management (optional)

### **AI & ML**
- **sentence-transformers** - Text embeddings
- **transformers** - Hugging Face model integration
- **langchain** - LLM orchestration and chains
- **openai/anthropic** - LLM API clients

### **MCP Integration**
- **mcp (Python SDK)** - Official MCP client implementation
- **10 MCP Servers** - Comprehensive development capabilities
- **Custom MCP Tools** - Domain-specific functionality

## **DIRECTORY STRUCTURE**

```
pygent-factory/
├── src/
│   ├── core/                 # Core agent and factory logic
│   │   ├── agent.py         # Base agent interface
│   │   ├── agent_factory.py # Agent creation and management
│   │   ├── message_system.py # MCP-compliant messaging
│   │   └── capability_system.py # Agent capabilities
│   ├── database/            # Database layer
│   │   ├── models.py        # SQLAlchemy models
│   │   ├── connection.py    # Database connections
│   │   └── migrations/      # Alembic migrations
│   ├── storage/             # Storage and persistence
│   │   ├── vector_store.py  # Vector storage with pgvector
│   │   └── knowledge_base.py # Knowledge management
│   ├── memory/              # Agent memory systems
│   │   └── memory_manager.py # Memory persistence and retrieval
│   ├── mcp/                 # MCP integration
│   │   ├── client.py        # MCP client implementation
│   │   ├── server_registry.py # MCP server management
│   │   └── tools/           # MCP tool implementations
│   ├── communication/       # Communication layer
│   │   └── broker.py        # Message broker with MCP patterns
│   ├── rag/                 # RAG system
│   │   ├── embeddings.py    # Embedding generation
│   │   ├── retrieval.py     # Document retrieval
│   │   ├── indexing.py      # Document indexing
│   │   ├── query_engine.py  # RAG query processing
│   │   └── vector_search.py # Vector similarity search
│   ├── agents/              # Agent implementations
│   │   ├── base_agent.py    # Base agent implementation
│   │   ├── coding_agent.py  # Code generation and analysis
│   │   ├── research_agent.py # Research and information gathering
│   │   ├── documentation_agent.py # Documentation generation
│   │   ├── testing_agent.py # Test generation and execution
│   │   ├── debugging_agent.py # Debugging assistance
│   │   └── orchestrator.py  # Multi-agent orchestration
│   ├── api/                 # FastAPI application
│   │   ├── main.py          # Application entry point
│   │   └── routes/          # API route definitions
│   ├── security/            # Security and authentication
│   │   └── auth.py          # Authentication and authorization
│   ├── evaluation/          # Agent evaluation
│   │   └── framework.py     # Evaluation metrics and testing
│   ├── config/              # Configuration management
│   │   └── settings.py      # Application settings
│   └── utils/               # Utilities
│       ├── logging.py       # Logging configuration
│       └── helpers.py       # Common utility functions
├── tests/                   # Test suite
├── docs/                    # Documentation
├── data/                    # Data files and knowledge base
├── docker/                  # Docker configurations
├── scripts/                 # Deployment and utility scripts
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Multi-service deployment
├── .env.example            # Environment variables template
└── README.md               # Project documentation
```

## **AGENT ARCHITECTURE**

### **Base Agent Interface**
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from mcp.types import Tool, Resource

class BaseAgent(ABC):
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.mcp_tools = {}
        self.memory = None
    
    @abstractmethod
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def execute_capability(self, capability: str, params: Dict[str, Any]) -> Any:
        pass
    
    async def register_mcp_tool(self, tool_name: str, server_id: str):
        self.mcp_tools[tool_name] = server_id
```

### **Agent Factory Pattern**
```python
class AgentFactory:
    def __init__(self, mcp_manager: MCPServerManager, memory_manager: MemoryManager):
        self.mcp_manager = mcp_manager
        self.memory_manager = memory_manager
        self.agent_registry = {}
    
    async def create_agent(self, agent_type: str, config: Dict[str, Any]) -> BaseAgent:
        agent_class = self.get_agent_class(agent_type)
        agent = agent_class(config)
        
        # Configure MCP tools
        await self.configure_mcp_tools(agent, config.get("mcp_tools", []))
        
        # Initialize memory
        agent.memory = await self.memory_manager.create_memory_space(agent.agent_id)
        
        self.agent_registry[agent.agent_id] = agent
        return agent
```

## **MCP INTEGRATION PATTERNS**

### **Tool Execution Framework**
```python
class MCPToolExecutor:
    async def execute_with_context(self, tool_name: str, params: Dict[str, Any], context: Dict[str, Any]):
        # Add context to tool execution
        enhanced_params = {**params, "context": context}
        return await self.mcp_manager.call_tool(tool_name, enhanced_params)
```

### **Resource Management**
```python
class MCPResourceManager:
    async def get_resource(self, resource_uri: str) -> Resource:
        # Retrieve resources through MCP
        return await self.mcp_client.get_resource(resource_uri)
    
    async def list_resources(self, resource_type: str = None) -> List[Resource]:
        # List available resources
        return await self.mcp_client.list_resources(resource_type)
```

## **RAG SYSTEM ARCHITECTURE**

### **Vector Storage**
```python
class VectorStore:
    def __init__(self, connection_string: str):
        self.db = create_async_engine(connection_string)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def store_document(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        embedding = self.embedding_model.encode(content)
        # Store in pgvector
        
    async def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query)
        # Perform vector similarity search
```

This architecture provides a solid foundation for the MCP-first Python rebuild of Agent Factory.
