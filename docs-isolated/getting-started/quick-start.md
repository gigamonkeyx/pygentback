# Quick Start Guide

Get PyGent Factory up and running in under 5 minutes! This guide will walk you through the fastest way to start creating and interacting with AI agents.

## Prerequisites

Before you begin, make sure you have:

- **Python 3.9+** installed on your system
- **Git** for cloning the repository
- **8GB+ RAM** recommended for optimal performance
- **NVIDIA GPU** (optional, but recommended for vector search acceleration)

<div class="alert info">
<strong>💡 Don't have these?</strong> Check our <a href="/getting-started/installation">detailed installation guide</a> for help setting up your environment.
</div>

## Step 1: Clone and Setup

<CodeExample
  :tabs="[
    {
      name: 'Git Clone',
      content: \`# Clone the repository
git clone https://github.com/gigamonkeyx/pygent.git
cd pygent-factory

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\\\Scripts\\\\activate
# On macOS/Linux:
source venv/bin/activate\`
    },
    {
      name: 'Install Dependencies',
      content: \`# Install core dependencies
pip install -r requirements.txt

# Install optional GPU support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
\`
    }
  ]"
  description="This will download PyGent Factory and install all necessary dependencies. GPU support is optional but recommended for better performance."
/>

## Step 2: Start the System

<CodeExample
  code="# Start PyGent Factory server
python main.py server --host 0.0.0.0 --port 8000

# You should see output like:
# 🚀 PyGent Factory - Advanced AI System
# ✅ Server starting at http://0.0.0.0:8000"
  language="bash"
  description="This starts the PyGent Factory backend server with all AI components initialized."
/>

<div class="alert success">
<strong>🎉 Success!</strong> If you see the startup messages without errors, PyGent Factory is running! The system will automatically:
<ul>
<li>Initialize the agent factory</li>
<li>Start the MCP server manager</li>
<li>Load available AI models</li>
<li>Set up the vector database</li>
<li>Enable WebSocket communication</li>
</ul>
</div>

## Step 3: Access the Web Interface

1. Open your web browser
2. Navigate to `http://localhost:8000`
3. You should see the PyGent Factory dashboard

<div class="alert warning">
<strong>⚠️ Can't access the interface?</strong> Make sure:
<ul>
<li>The server is running (check the terminal for errors)</li>
<li>Port 8000 isn't blocked by your firewall</li>
<li>You're using the correct URL: <code>http://localhost:8000</code></li>
</ul>
</div>

## Step 4: Create Your First Agent

Let's create a simple reasoning agent using the Python API:

<CodeExample
  :tabs="[
    {
      name: 'Python API',
      content: `import asyncio
from pygent_factory import AgentFactory, Settings

async def create_first_agent():
    # Initialize PyGent Factory
    settings = Settings()
    factory = AgentFactory(settings)
    await factory.initialize()
    
    # Create a reasoning agent
    agent = await factory.create_agent(
        agent_type="reasoning",
        name="my_first_agent",
        custom_config={
            "model_name": "deepseek2:latest",
            "temperature": 0.7,
            "max_tokens": 500
        }
    )
    
    print(f"✅ Created agent: {agent.agent_id}")
    return agent

# Run the example
agent = asyncio.run(create_first_agent())`
    },
    {
      name: 'Web Interface',
      content: `// Using the web interface:

1. Click "Create New Agent" in the dashboard
2. Select "Reasoning Agent" from the dropdown
3. Enter a name: "my_first_agent"
4. Configure settings:
   - Model: deepseek2:latest
   - Temperature: 0.7
   - Max Tokens: 500
5. Click "Create Agent"

// The agent will appear in your agent list`
    },
    {
      name: 'REST API',
      content: `# Using curl to create an agent
curl -X POST http://localhost:8000/api/v1/agents \\
  -H "Content-Type: application/json" \\
  -d '{
    "agent_type": "reasoning",
    "name": "my_first_agent",
    "custom_config": {
      "model_name": "deepseek2:latest",
      "temperature": 0.7,
      "max_tokens": 500
    }
  }'

# Response:
# {
#   "agent_id": "reasoning_agent_123",
#   "status": "created",
#   "message": "Agent created successfully"
# }`
    }
  ]"
  description="Choose your preferred method to create your first agent. The Python API gives you the most control, while the web interface is more user-friendly."
/>

## Step 5: Interact with Your Agent

Now let's send a message to your agent:

<CodeExample
  :tabs="[
    {
      name: 'Python API',
      content: `async def chat_with_agent(agent):
    from pygent_factory.core.agent import AgentMessage, MessageType
    
    # Create a message
    message = AgentMessage(
        type=MessageType.REQUEST,
        sender="user",
        recipient=agent.agent_id,
        content={
            "content": "Explain the benefits of using AI agents in software development"
        }
    )
    
    # Send message and get response
    response = await agent.process_message(message)
    
    print("🤖 Agent Response:")
    print(response.content.get("solution", "No response"))
    
    return response

# Chat with your agent
response = asyncio.run(chat_with_agent(agent))`
    },
    {
      name: 'WebSocket',
      content: `// Connect via WebSocket (JavaScript)
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function() {
    console.log('Connected to PyGent Factory');
    
    // Send a chat message
    ws.send(JSON.stringify({
        type: 'chat_message',
        data: {
            message: {
                agentId: 'reasoning',
                content: 'Explain the benefits of using AI agents in software development'
            }
        }
    }));
};

ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    console.log('Agent response:', response);
};`
    },
    {
      name: 'REST API',
      content: `# Send a message via REST API
curl -X POST http://localhost:8000/api/v1/agents/reasoning_agent_123/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Explain the benefits of using AI agents in software development",
    "stream": false
  }'

# Response:
# {
#   "response": "AI agents offer several key benefits...",
#   "agent_id": "reasoning_agent_123",
#   "timestamp": "2024-01-01T12:00:00Z",
#   "metadata": {
#     "processing_time": 2.5,
#     "confidence": 0.95
#   }
# }`
    }
  ]"
  description="Try different ways to interact with your agent. WebSocket provides real-time communication, while REST API is great for simple integrations."
/>

## Step 6: Explore Different Agent Types

PyGent Factory includes several specialized agents. Let's try a research agent:

<CodeExample
  code="# Create a research agent
research_agent = await factory.create_agent(
    agent_type='research',
    name='research_assistant'
)

# Ask it to research a topic
research_message = AgentMessage(
    type=MessageType.REQUEST,
    sender='user',
    recipient=research_agent.agent_id,
    content={
        'content': 'Research the latest developments in quantum computing and provide a summary'
    }
)

research_response = await research_agent.process_message(research_message)
print('📚 Research Results:')
print(research_response.content.get('solution', 'No research results'))"
  language="python"
  description="Research agents can search academic databases, web sources, and provide comprehensive summaries with citations."
/>

## Interactive Demo

Try the PyGent Factory API directly from this documentation:

<InteractiveDemo
  title="Test PyGent Factory API"
  description="Send a message to a PyGent Factory agent and see the response"
  type="api"
  endpoint="/api/v1/agents/chat"
/>

## What's Next?

Congratulations! You now have PyGent Factory running and have created your first agent. Here's what to explore next:

<div class="feature-grid">
  <FeatureCard
    title="Learn Core Concepts"
    description="Understand the architecture, agent types, and how everything works together."
    icon="📚"
    link="/concepts/architecture"
  />
  
  <FeatureCard
    title="Detailed Installation"
    description="Set up PyGent Factory for production use with Docker, databases, and monitoring."
    icon="⚙️"
    link="/getting-started/installation"
  />
  
  <FeatureCard
    title="Agent Creation Guide"
    description="Learn how to create custom agents and configure them for specific use cases."
    icon="🤖"
    link="/guides/agent-creation/"
  />
  
  <FeatureCard
    title="API Reference"
    description="Explore the complete API documentation with examples and best practices."
    icon="📖"
    link="/api/rest-endpoints"
  />
</div>

## Common Next Steps

### For Developers
- **[System Architecture](/concepts/architecture)** - Understand how PyGent Factory works
- **[API Reference](/api/rest-endpoints)** - Integrate PyGent Factory into your applications
- **[Custom Agents](/advanced/custom-agents)** - Build specialized agents for your use case

### For Researchers
- **[Research Agent Guide](/examples/research-agent/)** - Set up academic research workflows
- **[RAG System](/concepts/rag-system)** - Understand the knowledge management system
- **[Memory System](/concepts/memory-system)** - Learn about agent memory and context

### For DevOps
- **[Production Deployment](/guides/deployment/)** - Deploy PyGent Factory at scale
- **[Monitoring](/guides/monitoring/)** - Set up monitoring and observability
- **[Security](/advanced/security)** - Secure your PyGent Factory installation

## Troubleshooting

Having issues? Check our [troubleshooting guide](/getting-started/troubleshooting) for common problems and solutions.

<div class="alert info">
<strong>💬 Need Help?</strong> Join our community on GitHub for support, discussions, and to share your PyGent Factory projects!
</div>

---

**Next**: Set up PyGent Factory for production use with our [detailed installation guide](/getting-started/installation) →