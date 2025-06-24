# System Architecture

PyGent Factory is built with a **modular, microservices-inspired architecture** that prioritizes scalability, maintainability, and extensibility. This page provides a comprehensive overview of how all components work together.

## High-Level Architecture

<ArchitectureDiagram
  title="PyGent Factory Complete System Architecture"
  type="mermaid"
  :content="`graph TB
    subgraph 'Client Layer'
        WEB[Web Browser]
        API_CLIENT[API Clients]
        CLI[CLI Tools]
    end
    
    subgraph 'Frontend Layer'
        UI[React UI]
        WS_CLIENT[WebSocket Client]
    end
    
    subgraph 'API Gateway'
        NGINX[Nginx/Load Balancer]
        AUTH[Authentication Service]
        RATE[Rate Limiting]
    end
    
    subgraph 'Application Layer'
        API[FastAPI Server]
        WSS[WebSocket Server]
        SCHEDULER[Task Scheduler]
    end
    
    subgraph 'Core Services'
        AF[Agent Factory]
        MM[Memory Manager]
        MCP_MGR[MCP Manager]
        ORCH[Orchestrator]
    end
    
    subgraph 'AI Components'
        TOT[Tree of Thought Engine]
        RAG[RAG System]
        EMBED[Embedding Service]
        LLM[Language Models]
    end
    
    subgraph 'Agent Types'
        REASON[Reasoning Agent]
        RESEARCH[Research Agent]
        CODE[Coding Agent]
        SEARCH[Search Agent]
        GENERAL[General Agent]
        EVOLUTION[Evolution Agent]
    end
    
    subgraph 'MCP Servers'
        FS[Filesystem MCP]
        GH[GitHub MCP]
        SEARCH_MCP[Search MCP]
        DB_MCP[Database MCP]
        CUSTOM[Custom MCP Servers]
    end
    
    subgraph 'Storage Layer'
        POSTGRES[(PostgreSQL)]
        VECTOR[(Vector Database)]
        REDIS[(Redis Cache)]
        FILES[(File Storage)]
    end
    
    subgraph 'External Services'
        OLLAMA[Ollama/Local LLMs]
        APIS[External APIs]
        ACADEMIC[Academic Databases]
    end
    
    WEB --> UI
    API_CLIENT --> API
    CLI --> API
    
    UI --> NGINX
    WS_CLIENT --> NGINX
    
    NGINX --> AUTH
    AUTH --> API
    AUTH --> WSS
    
    API --> AF
    WSS --> MM
    API --> SCHEDULER
    
    AF --> ORCH
    MM --> VECTOR
    MCP_MGR --> FS
    MCP_MGR --> GH
    MCP_MGR --> SEARCH_MCP
    MCP_MGR --> DB_MCP
    
    ORCH --> REASON
    ORCH --> RESEARCH
    ORCH --> CODE
    ORCH --> SEARCH
    ORCH --> GENERAL
    ORCH --> EVOLUTION
    
    REASON --> TOT
    RESEARCH --> RAG
    CODE --> LLM
    SEARCH --> EMBED
    
    TOT --> LLM
    RAG --> EMBED
    EMBED --> VECTOR
    
    LLM --> OLLAMA
    RESEARCH --> ACADEMIC
    MCP_MGR --> APIS
    
    MM --> POSTGRES
    AF --> POSTGRES
    SCHEDULER --> REDIS
    
    style WEB fill:#e3f2fd
    style UI fill:#e1f5fe
    style API fill:#f3e5f5
    style AF fill:#e8f5e8
    style TOT fill:#fff3e0
    style RAG fill:#fff3e0
    style POSTGRES fill:#f1f8e9
    style VECTOR fill:#f1f8e9`"
  description="Complete system architecture showing all layers from client interfaces to storage, with clear separation of concerns and data flow."
/>

## Core Design Principles

### 🏗️ **Modular Architecture**
Each component has a single responsibility and well-defined interfaces, making the system easy to understand, test, and extend.

### 🔌 **MCP-First Design**
Built around the Model Context Protocol for standardized, secure, and extensible agent communication.

### 🚀 **Scalability**
Designed to scale horizontally with load balancing, caching, and distributed processing capabilities.

### 🔒 **Security**
Security is built into every layer with authentication, authorization, input validation, and secure communication.

### 🧪 **Testability**
Comprehensive testing strategy with unit, integration, and end-to-end tests for all components.

## Layer-by-Layer Breakdown

### 1. Client Layer

The client layer provides multiple ways to interact with PyGent Factory:

<CodeExample
  :tabs="[
    {
      name: 'Web Browser',
      content: `// Modern React-based web interface
// Features:
// - Real-time agent communication
// - Visual agent management
// - Interactive documentation
// - System monitoring dashboard

// Access via: http://localhost:8000`
    },
    {
      name: 'API Clients',
      content: `# Python SDK
from pygent_factory import AgentFactory, Settings

factory = AgentFactory(Settings())
agent = await factory.create_agent('reasoning')

# JavaScript/TypeScript
import { PyGentClient } from 'pygent-factory-js'

const client = new PyGentClient('http://localhost:8000')
const response = await client.chat('reasoning', 'Hello!')`
    },
    {
      name: 'CLI Tools',
      content: `# Command-line interface
pygent create-agent --type reasoning --name my-agent
pygent chat --agent my-agent --message "Hello!"
pygent list-agents
pygent deploy --environment production`
    }
  ]"
  description="Multiple client interfaces ensure PyGent Factory can be integrated into any workflow or application."
/>

### 2. Frontend Layer

The frontend layer provides the user interface and real-time communication:

#### React UI Components
- **Agent Dashboard**: Visual management of all agents
- **Chat Interface**: Real-time conversation with agents
- **System Monitor**: Live system metrics and health
- **Configuration Panel**: Agent and system settings

#### WebSocket Client
- **Real-time Updates**: Instant agent responses
- **Live Monitoring**: System status and metrics
- **Event Streaming**: Agent lifecycle events

### 3. API Gateway

The API gateway handles all incoming requests and provides cross-cutting concerns:

<CodeExample
  code="# Nginx configuration for PyGent Factory
upstream pygent_backend {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;  # Load balancing
}

server {
    listen 80;
    server_name api.pygent.ai;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    location / {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://pygent_backend;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
    }
}"
  language="nginx"
  description="API gateway configuration with load balancing, rate limiting, and WebSocket support."
/>

### 4. Application Layer

The application layer handles HTTP requests and WebSocket connections:

#### FastAPI Server
- **REST API**: Standard HTTP endpoints for agent management
- **OpenAPI Documentation**: Auto-generated API documentation
- **Request Validation**: Pydantic models for type safety
- **Error Handling**: Comprehensive error responses

#### WebSocket Server
- **Real-time Communication**: Bidirectional agent communication
- **Connection Management**: Handle multiple concurrent connections
- **Event Broadcasting**: System-wide event notifications

#### Task Scheduler
- **Background Jobs**: Long-running agent tasks
- **Periodic Tasks**: System maintenance and cleanup
- **Queue Management**: Task prioritization and retry logic

### 5. Core Services

The core services implement the main business logic:

<ArchitectureDiagram
  title="Core Services Interaction"
  type="mermaid"
  content="graph LR
    AF[Agent Factory] --> ORCH[Orchestrator]
    ORCH --> AGENT[Agent Instance]
    AGENT --> MM[Memory Manager]
    AGENT --> MCP[MCP Manager]
    MM --> VECTOR[(Vector DB)]
    MCP --> TOOLS[MCP Tools]
    
    style AF fill:#e8f5e8
    style ORCH fill:#fff3e0
    style AGENT fill:#f3e5f5
    style MM fill:#e1f5fe
    style MCP fill:#fce4ec"
  description="Core services work together to create, manage, and orchestrate agent interactions."
/>

#### Agent Factory
```python
class AgentFactory:
    """Central factory for creating and managing agents"""
    
    async def create_agent(
        self,
        agent_type: AgentType,
        name: str,
        custom_config: Optional[Dict] = None
    ) -> Agent:
        """Create a new agent instance"""
        
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Retrieve an existing agent"""
        
    async def list_agents(self) -> List[Agent]:
        """List all active agents"""
```

#### Memory Manager
```python
class MemoryManager:
    """Manages agent memory and context"""
    
    async def store_memory(
        self,
        agent_id: str,
        memory: Memory
    ) -> str:
        """Store a memory with vector embedding"""
        
    async def retrieve_memories(
        self,
        agent_id: str,
        query: str,
        limit: int = 10
    ) -> List[Memory]:
        """Retrieve relevant memories using vector search"""
```

#### MCP Manager
```python
class MCPManager:
    """Manages MCP server connections and tools"""
    
    async def register_server(self, server_config: MCPServerConfig):
        """Register a new MCP server"""
        
    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict
    ) -> ToolResult:
        """Execute a tool on an MCP server"""
```

### 6. AI Components

The AI components provide the intelligence for agent operations:

#### Tree of Thought Engine
- **Multi-path Reasoning**: Explore multiple solution paths
- **Decision Trees**: Structured problem-solving approach
- **Confidence Scoring**: Evaluate solution quality

#### RAG System
- **s3 RAG Framework**: Advanced retrieval-augmented generation
- **Vector Search**: Semantic similarity matching
- **Knowledge Integration**: Combine multiple information sources

#### Embedding Service
- **Text Embeddings**: Convert text to vector representations
- **Semantic Search**: Find semantically similar content
- **Clustering**: Group related information

#### Language Models
- **Local Models**: Ollama integration for privacy
- **Cloud Models**: API integration for scale
- **Model Selection**: Choose optimal model for each task

### 7. Agent Types

PyGent Factory includes specialized agent implementations:

<CodeExample
  :tabs="[
    {
      name: 'Reasoning Agent',
      content: `class ReasoningAgent(BaseAgent):
    \"\"\"Agent specialized for complex reasoning tasks\"\"\"
    
    def __init__(self):
        super().__init__()
        self.tot_engine = TreeOfThoughtEngine()
        self.reasoning_model = "deepseek2:latest"
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        # Use Tree of Thought for complex reasoning
        thoughts = await self.tot_engine.generate_thoughts(message.content)
        solution = await self.tot_engine.select_best_solution(thoughts)
        return self.create_response(solution)`
    },
    {
      name: 'Research Agent',
      content: `class ResearchAgent(BaseAgent):
    \"\"\"Agent specialized for academic research\"\"\"
    
    def __init__(self):
        super().__init__()
        self.research_sources = [
            ArxivSource(),
            SemanticScholarSource(),
            GoogleScholarSource()
        ]
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        # Multi-source research with citation
        results = await self.search_all_sources(message.content)
        summary = await self.synthesize_results(results)
        return self.create_response(summary)`
    },
    {
      name: 'Coding Agent',
      content: `class CodingAgent(BaseAgent):
    \"\"\"Agent specialized for software development\"\"\"
    
    def __init__(self):
        super().__init__()
        self.supported_languages = [
            'python', 'javascript', 'typescript', 'rust',
            'go', 'java', 'cpp', 'csharp', # ... 22 total
        ]
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        # Code generation with context awareness
        code = await self.generate_code(message.content)
        tests = await self.generate_tests(code)
        return self.create_response({'code': code, 'tests': tests})`
    }
  ]"
  description="Each agent type is optimized for specific tasks while sharing a common base interface."
/>

### 8. MCP Servers

MCP servers provide tools and capabilities to agents:

#### Built-in MCP Servers
- **Filesystem**: File operations and management
- **GitHub**: Repository integration and version control
- **Search**: Web search and information retrieval
- **Database**: Data storage and querying

#### Custom MCP Servers
```python
# Example custom MCP server
from mcp import Server, Tool

class CustomAnalyticsServer(Server):
    @Tool("analyze_data")
    async def analyze_data(self, data: str) -> Dict:
        """Analyze data and return insights"""
        # Custom analytics logic
        return {"insights": "...", "metrics": "..."}

# Register with PyGent Factory
mcp_manager.register_server(CustomAnalyticsServer())
```

### 9. Storage Layer

The storage layer provides persistence for all system data:

#### PostgreSQL
- **Relational Data**: Agent configurations, user data, system settings
- **ACID Compliance**: Reliable data consistency
- **Scalability**: Read replicas and partitioning

#### Vector Database
- **Embeddings**: High-dimensional vector storage
- **Similarity Search**: Fast semantic search capabilities
- **Indexing**: Optimized for vector operations

#### Redis Cache
- **Session Storage**: User sessions and temporary data
- **Task Queue**: Background job management
- **Caching**: Frequently accessed data

#### File Storage
- **Documents**: User uploads and generated files
- **Models**: AI model storage and versioning
- **Backups**: System backup and recovery

## Data Flow

Understanding how data flows through the system is crucial for debugging and optimization:

<ArchitectureDiagram
  title="Request Processing Flow"
  type="mermaid"
  content="sequenceDiagram
    participant U as User
    participant UI as React UI
    participant WS as WebSocket
    participant API as FastAPI
    participant AF as Agent Factory
    participant A as Agent
    participant MCP as MCP Server
    participant DB as Database
    
    U->>UI: Send Message
    UI->>WS: WebSocket Message
    WS->>API: Route Request
    API->>AF: Create/Get Agent
    AF->>A: Process Message
    A->>MCP: Use Tool
    MCP-->>A: Tool Result
    A->>DB: Store Memory
    A-->>AF: Response
    AF-->>API: Agent Response
    API-->>WS: Send Response
    WS-->>UI: Update UI
    UI-->>U: Display Result"
  description="Complete request flow from user input to agent response, showing all system interactions."
/>

## Performance Considerations

### Scalability Patterns

1. **Horizontal Scaling**: Multiple API server instances behind load balancer
2. **Database Sharding**: Distribute data across multiple database instances
3. **Caching Strategy**: Multi-level caching with Redis and application-level cache
4. **Async Processing**: Non-blocking I/O for all operations

### Optimization Strategies

1. **Connection Pooling**: Reuse database and external service connections
2. **Batch Processing**: Group similar operations for efficiency
3. **Lazy Loading**: Load data only when needed
4. **Compression**: Compress large responses and stored data

## Security Architecture

Security is implemented at every layer:

### Authentication & Authorization
- **JWT Tokens**: Stateless authentication
- **Role-Based Access**: Fine-grained permissions
- **API Keys**: Service-to-service authentication

### Data Protection
- **Encryption at Rest**: Database and file encryption
- **Encryption in Transit**: TLS for all communications
- **Input Validation**: Comprehensive input sanitization

### Network Security
- **Rate Limiting**: Prevent abuse and DoS attacks
- **CORS Configuration**: Secure cross-origin requests
- **Firewall Rules**: Network-level protection

## Monitoring & Observability

Comprehensive monitoring ensures system reliability:

### Metrics Collection
- **Application Metrics**: Response times, error rates, throughput
- **System Metrics**: CPU, memory, disk, network usage
- **Business Metrics**: Agent usage, user engagement

### Logging Strategy
- **Structured Logging**: JSON format for easy parsing
- **Log Levels**: Appropriate logging levels for different environments
- **Centralized Logging**: Aggregate logs from all services

### Health Checks
- **Endpoint Health**: Monitor API endpoint availability
- **Database Health**: Check database connectivity and performance
- **External Service Health**: Monitor MCP server and external API status

---

This architecture provides a solid foundation for building scalable, maintainable AI agent systems. The modular design allows for easy extension and customization while maintaining system reliability and performance.

**Next**: Learn about the different [agent types](/concepts/agents) and their specialized capabilities →