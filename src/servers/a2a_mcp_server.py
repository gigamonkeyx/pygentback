#!/usr/bin/env python3
"""
A2A (Agent-to-Agent) MCP Server

This server implements the Google A2A Protocol as an MCP server,
enabling PyGent Factory agents to communicate with other A2A-compliant agents.

Features:
- Agent discovery and registration
- A2A message sending and receiving
- Task management and coordination
- Multi-modal communication (text, files, data)
- Streaming and push notifications
- Integration with PyGent Factory orchestration

Usage:
    python src/servers/a2a_mcp_server.py [host] [port]
"""

import asyncio
import logging
import sys
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# FastAPI and MCP imports
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# A2A Standard imports
from a2a_standard import (
    AgentCard, AgentProvider, AgentCapabilities, AgentSkill,
    Task, TaskState, TaskStatus, Message, Part, TextPart, FilePart, DataPart,
    Artifact, A2AError
)

# HTTP client for A2A communication
import aiohttp

# PyGent Factory imports
sys.path.append(str(Path(__file__).parent.parent))

# Simple health response type
from dataclasses import dataclass
from typing import Optional

@dataclass
class HealthResponse:
    status: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def dict(self):
        result = {
            "status": self.status,
            "timestamp": self.timestamp.isoformat()
        }
        if self.details:
            result["details"] = self.details
        if self.error:
            result["error"] = self.error
        return result

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('a2a_mcp_server.log')
    ]
)

# Configure logging
logger = logging.getLogger(__name__)


class A2AMCPServer:
    """A2A MCP Server for PyGent Factory"""

    def __init__(self, port: int = 8006):
        self.port = port
        self.app = FastAPI(
            title="A2A MCP Server",
            description="Agent-to-Agent Protocol MCP Server for PyGent Factory",
            version="1.0.0"
        )
        
        # A2A components
        self.agent_card: Optional[AgentCard] = None
        self.registered_agents: Dict[str, AgentCard] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.client_sessions: Dict[str, aiohttp.ClientSession] = {}
        
        # Server state
        self.is_running = False
        self.start_time = datetime.now(timezone.utc)
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'tasks_created': 0,
            'tasks_completed': 0,
            'agents_discovered': 0,
            'connections_established': 0
        }
        
        # Setup CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()

        # Initialize A2A agent card
        self._initialize_agent_card()
    
    def _initialize_agent_card(self):
        """Initialize the A2A agent card with PyGent Factory capabilities"""
        try:
            # Create agent card for PyGent Factory
            self.agent_card = AgentCard(
                name="PyGent Factory A2A Agent",
                description="Multi-agent orchestration and coordination system with A2A protocol support",
                version="1.0.0",
                url="http://127.0.0.1:8006",  # A2A server endpoint
                defaultInputModes=["text", "application/json"],
                defaultOutputModes=["text", "application/json"],
                provider=AgentProvider(
                    name="PyGent Factory",
                    organization="PyGent Factory",
                    description="Advanced AI agent orchestration platform",
                    url="https://github.com/gigamonkeyx/pygentback"
                ),
                capabilities=AgentCapabilities(
                    streaming=True,
                    push_notifications=True,
                    multi_turn=True,
                    file_upload=True,
                    file_download=True,
                    structured_data=True
                ),
                skills=[
                    AgentSkill(
                        id="agent_orchestration",
                        name="agent_orchestration",
                        description="Coordinate and manage multiple AI agents",
                        input_modalities=["text", "application/json"],
                        output_modalities=["text", "application/json"],
                        tags=["orchestration", "coordination", "management"],
                        examples=[
                            "Create a new agent with specific capabilities",
                            "Coordinate task execution across multiple agents",
                            "Monitor agent performance and health"
                        ]
                    ),
                    AgentSkill(
                        id="document_processing",
                        name="document_processing",
                        description="Process and analyze documents using AI",
                        input_modalities=["text", "application/pdf", "application/json"],
                        output_modalities=["text", "application/json"],
                        tags=["documents", "processing", "analysis"],
                        examples=[
                            "Extract information from PDF documents",
                            "Summarize large text documents",
                            "Convert documents between formats"
                        ]
                    ),
                    AgentSkill(
                        id="vector_search",
                        name="vector_search",
                        description="Semantic search and similarity matching",
                        input_modalities=["text", "application/json"],
                        output_modalities=["application/json"],
                        tags=["search", "similarity", "vectors"],
                        examples=[
                            "Find similar documents in a knowledge base",
                            "Semantic search across document collections",
                            "Generate embeddings for text content"
                        ]
                    ),
                    AgentSkill(
                        id="embedding_generation",
                        name="embedding_generation",
                        description="Generate high-quality embeddings for text and documents",
                        input_modalities=["text", "application/json"],
                        output_modalities=["application/json"],
                        tags=["embeddings", "vectors", "generation"],
                        examples=[
                            "Generate embeddings for text content",
                            "Create vector representations of documents",
                            "Support for multiple embedding models"
                        ]
                    )
                ]
            )
            
            # Register self as an A2A agent
            self_agent_id = f"pygent_factory_{uuid.uuid4().hex[:8]}"
            self.registered_agents[self_agent_id] = self.agent_card
            self.stats['agents_discovered'] += 1

            logger.info("A2A agent card initialized with PyGent Factory capabilities")
            logger.info(f"Registered self as A2A agent: {self_agent_id}")

        except Exception as e:
            logger.error(f"Failed to initialize A2A agent card: {str(e)}")
            raise
    
    def _setup_routes(self):
        """Setup FastAPI routes for A2A and MCP endpoints"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with server information"""
            return {
                "name": "A2A MCP Server",
                "description": "Agent-to-Agent Protocol MCP Server for PyGent Factory",
                "version": "1.0.0",
                "protocol": "A2A",
                "capabilities": [
                    "agent_discovery",
                    "message_sending",
                    "task_management", 
                    "multi_modal_communication",
                    "streaming_support",
                    "push_notifications",
                    "mcp_integration"
                ],
                "uptime": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
                "stats": self.stats
            }
        
        @self.app.get("/health")
        async def get_health():
            """Health check endpoint"""
            try:
                # Check A2A server status
                a2a_healthy = self.agent_card is not None

                # Check active connections
                active_connections = len(self.client_sessions)

                # Check task processing
                active_tasks = len([t for t in self.active_tasks.values()
                                  if hasattr(t.status, 'state') and
                                  t.status.state in [TaskState.submitted, TaskState.working]])

                status = "healthy" if a2a_healthy else "unhealthy"
                
                return HealthResponse(
                    status=status,
                    timestamp=datetime.now(timezone.utc),
                    details={
                        "a2a_server_initialized": a2a_healthy,
                        "active_connections": active_connections,
                        "active_tasks": active_tasks,
                        "registered_agents": len(self.registered_agents),
                        "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds()
                    }
                ).dict()
                
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                return HealthResponse(
                    status="unhealthy",
                    timestamp=datetime.now(timezone.utc),
                    error=str(e)
                ).dict()
        
        @self.app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Serve the agent card for A2A discovery"""
            if not self.agent_card:
                raise HTTPException(status_code=503, detail="Agent card not initialized")
            
            return self.agent_card.dict()
        
        # A2A Protocol endpoints
        @self.app.post("/a2a/message/send")
        async def send_message(request: dict):
            """Send a message via A2A protocol"""
            try:
                logger.info(f"Received A2A message: {request}")

                # Process the message and create/update task
                task = await self._process_a2a_message(request)

                self.stats['messages_received'] += 1

                # Return task response with proper serialization
                status_dict = {
                    "state": task.status.state.value if hasattr(task.status.state, 'value') else str(task.status.state)
                }
                if hasattr(task.status, 'error') and task.status.error:
                    status_dict["error"] = task.status.error

                response = {
                    "id": task.id,
                    "contextId": task.context_id,
                    "status": status_dict,
                    "artifacts": task.artifacts,
                    "history": task.history,
                    "kind": "task",
                    "metadata": task.metadata
                }

                logger.info(f"Returning A2A response: {response}")
                return response

            except Exception as e:
                logger.error(f"Failed to send message: {str(e)}", exc_info=True)
                return {"error": str(e), "status": "failed"}
        
        @self.app.get("/a2a/tasks/{task_id}")
        async def get_task(task_id: str):
            """Get task status and results"""
            try:
                task = self.active_tasks.get(task_id)
                if not task:
                    raise HTTPException(status_code=404, detail="Task not found")
                
                return task.dict()
                
            except Exception as e:
                logger.error(f"Failed to get task: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # MCP-specific endpoints for A2A management
        @self.app.post("/mcp/a2a/discover_agent")
        async def discover_agent(request: dict):
            """Discover and register an A2A agent"""
            try:
                agent_url = request.get("agent_url")
                if not agent_url:
                    raise HTTPException(status_code=400, detail="agent_url is required")

                # Fetch agent card via HTTP
                async with aiohttp.ClientSession() as session:
                    agent_card_url = f"{agent_url.rstrip('/')}/.well-known/agent.json"
                    async with session.get(agent_card_url) as response:
                        if response.status != 200:
                            raise HTTPException(status_code=400, detail=f"Failed to fetch agent card from {agent_card_url}")

                        agent_card_data = await response.json()
                        agent_card = AgentCard(**agent_card_data)

                # Register the agent
                agent_id = f"{agent_card.name}_{uuid.uuid4().hex[:8]}"
                self.registered_agents[agent_id] = agent_card

                # Create HTTP session for this agent
                self.client_sessions[agent_id] = aiohttp.ClientSession()

                self.stats['agents_discovered'] += 1
                self.stats['connections_established'] += 1

                logger.info(f"Discovered and registered agent: {agent_card.name}")

                return {
                    "agent_id": agent_id,
                    "agent_card": agent_card.dict() if hasattr(agent_card, 'dict') else agent_card_data,
                    "status": "registered"
                }

            except Exception as e:
                logger.error(f"Failed to discover agent: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/mcp/a2a/agents")
        async def list_agents():
            """List all registered A2A agents"""
            return {
                "agents": {
                    agent_id: {
                        "name": card.name,
                        "description": card.description,
                        "version": card.version,
                        "url": card.url,
                        "capabilities": card.capabilities.dict() if hasattr(card.capabilities, 'dict') else card.capabilities,
                        "skills": [skill.dict() if hasattr(skill, 'dict') else skill for skill in (card.skills or [])]
                    }
                    for agent_id, card in self.registered_agents.items()
                },
                "total_agents": len(self.registered_agents)
            }
        
        @self.app.post("/mcp/a2a/send_to_agent")
        async def send_to_agent(request: dict):
            """Send a message to a specific registered agent"""
            try:
                agent_id = request.get("agent_id")
                message = request.get("message")
                context_id = request.get("context_id")

                if not agent_id or not message:
                    raise HTTPException(status_code=400, detail="agent_id and message are required")

                if agent_id not in self.registered_agents:
                    raise HTTPException(status_code=404, detail="Agent not found")

                agent_card = self.registered_agents[agent_id]

                # Check if this is a self-message (to the A2A MCP Server itself)
                if agent_card.url == f"http://127.0.0.1:{self.port}" or agent_id.startswith("pygent_factory_"):
                    # Handle self-messaging by processing directly
                    return await self._handle_self_message(message, context_id)

                # For external agents, use HTTP session
                if agent_id not in self.client_sessions:
                    self.client_sessions[agent_id] = aiohttp.ClientSession()

                session = self.client_sessions[agent_id]

                # Create A2A message payload
                message_payload = {
                    "jsonrpc": "2.0",
                    "id": str(uuid.uuid4()),
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"kind": "text", "text": message}],
                            "messageId": str(uuid.uuid4()),
                            "contextId": context_id or str(uuid.uuid4())
                        }
                    }
                }

                # Send message to agent
                agent_url = f"{agent_card.url.rstrip('/')}/a2a/message/send"
                async with session.post(agent_url, json=message_payload) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=400, detail=f"Agent returned status {response.status}")

                    response_data = await response.json()

                self.stats['messages_sent'] += 1

                return {
                    "agent_id": agent_id,
                    "message_sent": message,
                    "response": response_data,
                    "status": "sent"
                }

            except Exception as e:
                logger.error(f"Failed to send message to agent: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _process_a2a_message(self, request: dict) -> Task:
        """Process an incoming A2A message and create/update task"""
        try:
            # Extract message from request
            message_data = request.get("params", {}).get("message", {})

            # Create or get existing task
            task_id = message_data.get("taskId") or str(uuid.uuid4())
            context_id = message_data.get("contextId") or str(uuid.uuid4())

            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
            else:
                # Create new task (simplified structure)
                task_status = type('TaskStatus', (), {
                    'state': TaskState.submitted,
                    'error': None
                })()

                task = type('Task', (), {
                    'id': task_id,
                    'context_id': context_id,
                    'status': task_status,
                    'history': [],
                    'artifacts': [],
                    'metadata': {}
                })()
                self.active_tasks[task_id] = task
                self.stats['tasks_created'] += 1

            # Add message to task history
            task.history.append(message_data)

            # Process the message based on content
            await self._handle_message_content(task, message_data)

            return task

        except Exception as e:
            logger.error(f"Failed to process A2A message: {str(e)}", exc_info=True)
            # Create a failed task to return
            failed_task = type('Task', (), {
                'id': str(uuid.uuid4()),
                'context_id': str(uuid.uuid4()),
                'status': type('TaskStatus', (), {
                    'state': TaskState.failed,
                    'error': str(e)
                })(),
                'history': [],
                'artifacts': [],
                'metadata': {}
            })()
            return failed_task
    
    async def _handle_message_content(self, task, message_data: dict):
        """Handle the content of an A2A message"""
        try:
            # Update task status to working
            task.status.state = TaskState.working

            # Extract text content
            text_content = ""
            parts = message_data.get("parts", [])
            for part in parts:
                if part.get("kind") == "text":
                    text_content += part.get("text", "") + "\n"

            # Simple echo response for now (can be enhanced with actual processing)
            response_text = f"Processed message: {text_content.strip()}"

            # Create response artifact (simplified structure)
            artifact = {
                "artifactId": str(uuid.uuid4()),
                "name": "response",
                "parts": [{"kind": "text", "text": response_text}]
            }

            task.artifacts.append(artifact)

            # Mark task as completed
            task.status.state = TaskState.completed
            self.stats['tasks_completed'] += 1

            logger.info(f"Processed A2A message for task {task.id}")

        except Exception as e:
            # Mark task as failed
            task.status.state = TaskState.failed
            if hasattr(task.status, 'error'):
                task.status.error = str(e)
            logger.error(f"Failed to handle message content: {str(e)}", exc_info=True)

    async def _handle_self_message(self, message: str, context_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle messages sent to the A2A MCP Server itself"""
        try:
            # Create a self-message request
            self_request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": message}],
                        "messageId": str(uuid.uuid4()),
                        "contextId": context_id or str(uuid.uuid4())
                    }
                }
            }

            # Process the message internally
            task = await self._process_a2a_message(self_request)

            # Return the response
            status_dict = {
                "state": task.status.state.value if hasattr(task.status.state, 'value') else str(task.status.state)
            }
            if hasattr(task.status, 'error') and task.status.error:
                status_dict["error"] = task.status.error

            return {
                "agent_id": "self",
                "message_sent": message,
                "response": {
                    "id": task.id,
                    "contextId": task.context_id,
                    "status": status_dict,
                    "artifacts": task.artifacts,
                    "history": task.history,
                    "kind": "task",
                    "metadata": task.metadata
                },
                "status": "sent"
            }

        except Exception as e:
            logger.error(f"Failed to handle self-message: {str(e)}")
            return {
                "agent_id": "self",
                "message_sent": message,
                "error": str(e),
                "status": "failed"
            }
    
    async def start_server(self, host: str = "127.0.0.1", port: int = 8006):
        """Start the A2A MCP server"""
        try:
            self.is_running = True
            logger.info(f"Starting A2A MCP Server on {host}:{port}")
            
            config = uvicorn.Config(
                app=self.app,
                host=host,
                port=port,
                log_level="info"
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            logger.error(f"Failed to start A2A MCP server: {str(e)}")
            raise
        finally:
            self.is_running = False


async def main():
    """Main entry point"""
    # Parse command line arguments
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8006
    
    # Create and start server
    server = A2AMCPServer(port=port)
    await server.start_server(host, port)


if __name__ == "__main__":
    asyncio.run(main())
