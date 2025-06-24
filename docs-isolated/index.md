---
layout: home

hero:
  name: "PyGent Factory"
  text: "MCP-Compliant Agent Factory"
  tagline: "Build, deploy, and orchestrate AI agents with enterprise-grade reliability and modern architecture"
  image:
    src: /logo.svg
    alt: PyGent Factory Logo
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started/introduction
    - theme: alt
      text: View on GitHub
      link: https://github.com/gigamonkeyx/pygent

features:
  - icon: 🤖
    title: Multi-Agent System
    details: Create specialized agents for reasoning, research, coding, and general tasks with seamless orchestration
    link: /concepts/agents
  
  - icon: 🔌
    title: MCP Protocol Integration
    details: Built on the Model Context Protocol for standardized, secure, and extensible agent communication
    link: /concepts/mcp-protocol
  
  - icon: 🧠
    title: Advanced RAG System
    details: s3 RAG framework with GPU-accelerated vector search and intelligent knowledge management
    link: /concepts/rag-system
  
  - icon: ⚡
    title: Real-time Communication
    details: WebSocket-based messaging for instant agent responses and live system monitoring
    link: /concepts/communication
  
  - icon: 💾
    title: Persistent Memory
    details: Sophisticated memory management with vector embeddings and contextual recall
    link: /concepts/memory-system
  
  - icon: 🚀
    title: Production Ready
    details: Docker deployment, monitoring, testing, and scalable architecture for enterprise use
    link: /guides/deployment/
---

<div class="feature-grid">
  <FeatureCard
    title="Tree of Thought Reasoning"
    description="Advanced reasoning capabilities using Tree of Thought methodology for complex problem solving and decision making."
    icon="reasoning"
    :badges="[{ text: 'Advanced', type: 'new' }]"
    link="/examples/reasoning-agent/"
  />
  
  <FeatureCard
    title="Research Orchestration"
    description="Zero-cost academic research with multiple source integration including ArXiv, Semantic Scholar, and Google Scholar."
    icon="search"
    :badges="[{ text: 'Academic', type: 'stable' }]"
    link="/examples/research-agent/"
  />
  
  <FeatureCard
    title="Code Generation"
    description="Intelligent coding assistance with 22+ programming language support and context-aware suggestions."
    icon="api"
    :badges="[{ text: 'Multi-lang', type: 'stable' }]"
    link="/examples/coding-assistant/"
  />
</div>

## Quick Start

Get PyGent Factory running in under 5 minutes:

<CodeExample
  :tabs="[
    {
      name: 'Installation',
      content: `# Clone the repository
git clone https://github.com/gigamonkeyx/pygent.git
cd pygent-factory

# Install dependencies
pip install -r requirements.txt

# Start the system
python main.py server --host 0.0.0.0 --port 8000`
    },
    {
      name: 'Docker',
      content: `# Using Docker Compose
docker-compose up -d

# Or with Docker
docker run -p 8000:8000 pygent-factory:latest`
    },
    {
      name: 'Python API',
      content: `from pygent_factory import AgentFactory, Settings

# Initialize the factory
settings = Settings()
factory = AgentFactory(settings)

# Create a reasoning agent
agent = await factory.create_agent(
    agent_type="reasoning",
    name="my_reasoning_agent"
)

# Send a message
response = await agent.process_message({
    "content": "Solve this complex problem..."
})

print(response.content)`
    }
  ]"
/>

## System Architecture

<ArchitectureDiagram
  title="PyGent Factory System Overview"
  type="mermaid"
  content="graph TB
    subgraph 'Frontend Layer'
        UI[React UI]
        WS[WebSocket Client]
    end
    
    subgraph 'API Layer'
        API[FastAPI Server]
        WSS[WebSocket Server]
        AUTH[Authentication]
    end
    
    subgraph 'Core System'
        AF[Agent Factory]
        MM[Memory Manager]
        MCP[MCP Manager]
    end
    
    subgraph 'AI Components'
        TOT[Tree of Thought]
        RAG[RAG System]
        EMBED[Embedding Service]
    end
    
    subgraph 'Storage Layer'
        DB[(PostgreSQL)]
        VECTOR[(Vector Store)]
        FILES[(File Storage)]
    end
    
    UI --> API
    WS --> WSS
    API --> AF
    WSS --> MM
    AF --> TOT
    AF --> RAG
    RAG --> EMBED
    MM --> VECTOR
    API --> DB
    
    style UI fill:#e1f5fe
    style API fill:#f3e5f5
    style AF fill:#e8f5e8
    style TOT fill:#fff3e0
    style RAG fill:#fff3e0"
  description="PyGent Factory uses a modular architecture with clear separation between frontend, API, core systems, and storage layers."
/>

## Interactive Demo

<InteractiveDemo
  title="Try PyGent Factory API"
  description="Test the API endpoints directly from the documentation"
  type="api"
  endpoint="/api/v1/agents/chat"
/>

## What Makes PyGent Factory Special?

### 🏗️ **Modern Architecture**
Built from the ground up with Python and the Model Context Protocol, eliminating the complexity and fragmentation of legacy TypeScript implementations.

### 🧠 **Advanced AI Integration**
- **Tree of Thought Reasoning**: Multi-path reasoning for complex problem solving
- **s3 RAG Framework**: Superior performance with 90% less training data
- **GPU Vector Search**: Lightning-fast semantic search and retrieval

### 🔧 **Developer Experience**
- **Type-safe APIs**: Full TypeScript support for frontend integration
- **Comprehensive Testing**: Unit, integration, and end-to-end test coverage
- **Hot Reload**: Instant development feedback with live reloading

### 🚀 **Production Ready**
- **Docker Deployment**: Containerized for easy deployment and scaling
- **Monitoring**: Built-in metrics, logging, and health checks
- **Security**: Authentication, authorization, and secure communication

## Community & Support

<div class="feature-grid">
  <FeatureCard
    title="Documentation"
    description="Comprehensive guides, tutorials, and API reference for all skill levels."
    icon="📚"
    link="/getting-started/introduction"
  />
  
  <FeatureCard
    title="GitHub Repository"
    description="Source code, issues, discussions, and contribution guidelines."
    icon="🐙"
    link="https://github.com/gigamonkeyx/pygent"
  />
  
  <FeatureCard
    title="Examples"
    description="Real-world examples and tutorials for common use cases."
    icon="💡"
    link="/examples/"
  />
</div>

---

<div style="text-align: center; margin: 2rem 0;">
  <p><strong>Ready to build intelligent agents?</strong></p>
  <a href="/getting-started/introduction" class="vp-button vp-button-brand">Get Started Now</a>
</div>