# A2A + DGM Master Implementation Plan - Part 2: A2A Integration

## Phase 1 Continued: A2A Integration with Existing Systems (Weeks 2-3)

### 1.3 Integration with FastAPI Main Application

**Step 8:** Update main FastAPI application to include A2A router
File: `src/api/main.py` (add to existing imports)
```python
# Add to existing imports
from ..protocols.a2a.router import router as a2a_router
```

**Step 9:** Include A2A router in main app
File: `src/api/main.py` (add after existing router includes)
```python
# Add after existing router includes (around line 200)
app.include_router(a2a_router)
```

**Step 10:** Update A2A handler with dependency injection
File: `src/protocols/a2a/handler.py` (add to __init__ method)
```python
# Modify __init__ method to accept dependencies
def __init__(self, agent_factory=None, memory_manager=None, db_manager=None):
    self.tasks: Dict[str, Task] = {}
    self.active_streams: Dict[str, asyncio.Queue] = {}
    self.agent_cards: Dict[str, AgentCard] = {}
    
    # Dependency injection
    self.agent_factory = agent_factory
    self.memory_manager = memory_manager
    self.db_manager = db_manager
```

**Step 11:** Create A2A dependency provider
File: `src/protocols/a2a/dependencies.py`
```python
from fastapi import Depends
from typing import Optional

from ...core.agent_factory import AgentFactory
from ...memory.memory_manager import MemoryManager
from ...database.connection import DatabaseManager
from .handler import A2AProtocolHandler

# Global handler instance
_a2a_handler: Optional[A2AProtocolHandler] = None

def get_a2a_handler(
    agent_factory: AgentFactory = Depends(),
    memory_manager: MemoryManager = Depends(),
    db_manager: DatabaseManager = Depends()
) -> A2AProtocolHandler:
    """Get A2A handler with dependencies"""
    global _a2a_handler
    if _a2a_handler is None:
        _a2a_handler = A2AProtocolHandler(
            agent_factory=agent_factory,
            memory_manager=memory_manager,
            db_manager=db_manager
        )
    return _a2a_handler

def set_a2a_handler(handler: A2AProtocolHandler):
    """Set global A2A handler instance"""
    global _a2a_handler
    _a2a_handler = handler
```

**Step 12:** Update router to use new dependencies
File: `src/protocols/a2a/router.py` (replace get_a2a_handler import and function)
```python
# Replace existing import
from .dependencies import get_a2a_handler

# Remove old get_a2a_handler function and global handler instance
```

### 1.4 Agent Card Generation System

**Step 13:** Create agent card generator
File: `src/protocols/a2a/agent_card_generator.py`
```python
from typing import Dict, Any, List
from datetime import datetime

from .models import AgentCard
from ...core.agent_factory import AgentFactory

class AgentCardGenerator:
    """Generate A2A agent cards from PyGent Factory agents"""
    
    def __init__(self, agent_factory: AgentFactory):
        self.agent_factory = agent_factory
    
    def generate_card(self, agent_id: str) -> AgentCard:
        """Generate agent card for specific agent"""
        # Get agent info from factory
        agent_info = self.agent_factory.get_agent_info(agent_id)
        
        if not agent_info:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Extract capabilities from agent
        capabilities = self._extract_capabilities(agent_info)
        
        # Generate card
        card = AgentCard(
            metadata={
                "name": agent_id,
                "displayName": agent_info.get("display_name", agent_id),
                "description": agent_info.get("description", "PyGent Factory Agent"),
                "version": "1.0.0",
                "tags": agent_info.get("tags", []),
                "created": datetime.utcnow().isoformat()
            },
            spec={
                "endpoints": {
                    "a2a": "/a2a/v1"
                },
                "capabilities": capabilities,
                "authentication": {
                    "type": "bearer",
                    "description": "Bearer token authentication"
                },
                "maxConcurrentTasks": agent_info.get("max_concurrent_tasks", 5),
                "supportedContentTypes": [
                    "text/plain",
                    "application/json",
                    "image/png",
                    "image/jpeg"
                ]
            }
        )
        
        return card
    
    def _extract_capabilities(self, agent_info: Dict[str, Any]) -> List[str]:
        """Extract capabilities from agent info"""
        capabilities = []
        
        # Add basic capabilities
        capabilities.append("reasoning")
        capabilities.append("task_execution")
        
        # Add specific capabilities based on agent type
        agent_type = agent_info.get("type", "")
        
        if "evolution" in agent_type.lower():
            capabilities.append("evolution")
            capabilities.append("optimization")
        
        if "integration" in agent_type.lower():
            capabilities.append("integration")
            capabilities.append("data_flow")
        
        if "reasoning" in agent_type.lower():
            capabilities.append("reasoning")
            capabilities.append("problem_solving")
        
        # Add MCP tool capabilities
        mcp_tools = agent_info.get("mcp_tools", [])
        for tool in mcp_tools:
            capabilities.append(f"mcp_{tool}")
        
        return list(set(capabilities))  # Remove duplicates
    
    def generate_factory_card(self) -> AgentCard:
        """Generate card for the entire PyGent Factory"""
        return AgentCard(
            metadata={
                "name": "pygent-factory",
                "displayName": "PyGent Factory",
                "description": "Advanced AI Agent Factory with Genetic Algorithms and Multi-Agent Collaboration",
                "version": "1.0.0",
                "tags": ["ai", "agents", "evolution", "collaboration"],
                "created": datetime.utcnow().isoformat()
            },
            spec={
                "endpoints": {
                    "a2a": "/a2a/v1",
                    "api": "/api/v1"
                },
                "capabilities": [
                    "agent_creation",
                    "multi_agent_coordination", 
                    "evolutionary_optimization",
                    "reasoning",
                    "memory_management",
                    "mcp_integration",
                    "real_time_communication"
                ],
                "authentication": {
                    "type": "bearer",
                    "description": "Bearer token authentication"
                },
                "maxConcurrentTasks": 50,
                "supportedContentTypes": [
                    "text/plain",
                    "application/json",
                    "image/png",
                    "image/jpeg",
                    "audio/wav",
                    "video/mp4"
                ]
            }
        )
```

**Step 14:** Update A2A handler to use agent card generator
File: `src/protocols/a2a/handler.py` (add to imports and __init__)
```python
# Add to imports
from .agent_card_generator import AgentCardGenerator

# Add to __init__ method after existing code
def __init__(self, agent_factory=None, memory_manager=None, db_manager=None):
    # ...existing code...
    
    # Initialize agent card generator
    if agent_factory:
        self.card_generator = AgentCardGenerator(agent_factory)
        # Generate and cache factory card
        self.agent_cards["factory"] = self.card_generator.generate_factory_card()

# Add new method to handler
async def get_agent_card(self, agent_id: str = "factory") -> AgentCard:
    """Get agent card by ID"""
    if agent_id in self.agent_cards:
        return self.agent_cards[agent_id]
    
    if self.card_generator and agent_id != "factory":
        # Generate card for specific agent
        try:
            card = self.card_generator.generate_card(agent_id)
            self.agent_cards[agent_id] = card
            return card
        except ValueError:
            pass
    
    # Return factory card as fallback
    return self.agent_cards.get("factory", self.card_generator.generate_factory_card())
```

**Step 15:** Update router to serve dynamic agent cards
File: `src/protocols/a2a/router.py` (replace the /.well-known/agent.json endpoint)
```python
@router.get("/.well-known/agent.json")
async def get_agent_card(
    agent_id: str = "factory",
    handler: A2AProtocolHandler = Depends(get_a2a_handler)
):
    """Get public agent card"""
    card = await handler.get_agent_card(agent_id)
    return card.dict()

@router.get("/agents/{agent_id}/card")
async def get_specific_agent_card(
    agent_id: str,
    handler: A2AProtocolHandler = Depends(get_a2a_handler)
):
    """Get agent card for specific agent"""
    card = await handler.get_agent_card(agent_id)
    return card.dict()
```

## Phase 1 Continued: Integration Testing (Week 3)

### 1.5 Basic Integration Testing

**Step 16:** Create A2A integration tests
File: `tests/protocols/a2a/test_integration.py`
```python
import pytest
import asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.api.main import app
from src.protocols.a2a.models import MessageSendParams, Message, TextPart

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

def test_agent_card_endpoint(client):
    """Test agent card endpoint"""
    response = client.get("/a2a/v1/.well-known/agent.json")
    assert response.status_code == 200
    
    data = response.json()
    assert data["kind"] == "agent"
    assert data["apiVersion"] == "a2a/v1"
    assert "metadata" in data
    assert "spec" in data

@pytest.mark.asyncio
async def test_message_send(async_client):
    """Test message send endpoint"""
    message_data = {
        "contextId": "test-context",
        "message": {
            "role": "user",
            "parts": [
                {
                    "kind": "text",
                    "text": "Hello, agent!"
                }
            ]
        }
    }
    
    response = await async_client.post("/a2a/v1/message/send", json=message_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["kind"] == "task"
    assert "id" in data
    assert data["contextId"] == "test-context"

@pytest.mark.asyncio
async def test_task_retrieval(async_client):
    """Test task retrieval"""
    # First send a message to create a task
    message_data = {
        "contextId": "test-context",
        "message": {
            "role": "user", 
            "parts": [{"kind": "text", "text": "Test message"}]
        }
    }
    
    send_response = await async_client.post("/a2a/v1/message/send", json=message_data)
    task_data = send_response.json()
    task_id = task_data["id"]
    
    # Wait a moment for processing
    await asyncio.sleep(0.2)
    
    # Retrieve the task
    get_response = await async_client.get(f"/a2a/v1/tasks/{task_id}")
    assert get_response.status_code == 200
    
    retrieved_task = get_response.json()
    assert retrieved_task["id"] == task_id
    assert retrieved_task["status"]["state"] in ["working", "completed"]

def test_specific_agent_card(client):
    """Test specific agent card endpoint"""
    response = client.get("/a2a/v1/agents/factory/card")
    assert response.status_code == 200
    
    data = response.json()
    assert data["metadata"]["name"] == "pygent-factory"
    assert "capabilities" in data["spec"]
```

## Next Steps

This completes Part 2 of the master implementation plan. The remaining parts will cover:

- **Part 3**: DGM Core Engine Implementation (Steps 17-26)
- **Part 4**: DGM Integration & Validation (Steps 27-36)  
- **Part 5**: Advanced Features & Testing (Steps 37-46)
- **Part 6**: Deployment & Production (Steps 47-52)

Each step remains explicit and numbered for precise execution.
