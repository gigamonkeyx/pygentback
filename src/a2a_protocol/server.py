#!/usr/bin/env python3
"""
A2A Server Implementation

Provides A2A protocol JSON-RPC server for PyGent Factory agents.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .protocol import a2a_protocol, Message, TextPart, TaskState
from .agent_integration import a2a_agent_registry

logger = logging.getLogger(__name__)


class A2AServer:
    """A2A Protocol Server"""
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="PyGent Factory A2A Server",
            description="Agent-to-Agent Protocol Server for PyGent Factory",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup A2A server routes"""
        
        @self.app.get("/.well-known/agent.json")
        async def get_agent_discovery():
            """A2A agent discovery endpoint"""
            agents = await a2a_protocol.discover_agents()
            if agents:
                # Return the first agent's card for discovery
                agent = agents[0]
                return {
                    "name": agent.name,
                    "description": agent.description,
                    "url": agent.url,
                    "version": agent.version,
                    "provider": {
                        "organization": agent.provider.organization if agent.provider else "PyGent Factory",
                        "url": agent.provider.url if agent.provider else "https://github.com/gigamonkeyx/pygentback"
                    },
                    "authentication": {
                        "schemes": agent.authentication.schemes
                    },
                    "defaultInputModes": agent.defaultInputModes,
                    "defaultOutputModes": agent.defaultOutputModes,
                    "capabilities": {
                        "streaming": agent.capabilities.streaming,
                        "pushNotifications": agent.capabilities.pushNotifications,
                        "stateTransitionHistory": agent.capabilities.stateTransitionHistory
                    },
                    "skills": [
                        {
                            "id": skill.id,
                            "name": skill.name,
                            "description": skill.description,
                            "tags": skill.tags,
                            "examples": skill.examples,
                            "inputModes": skill.inputModes,
                            "outputModes": skill.outputModes
                        }
                        for skill in agent.skills
                    ]
                }
            else:
                return {
                    "name": "PyGent Factory A2A Server",
                    "description": "Multi-agent system with A2A protocol support",
                    "url": f"http://{self.host}:{self.port}",
                    "version": "1.0.0",
                    "capabilities": {
                        "streaming": True,
                        "pushNotifications": False
                    },
                    "skills": []
                }
        
        @self.app.post("/")
        async def handle_jsonrpc(request: Request):
            """Handle A2A JSON-RPC requests"""
            try:
                body = await request.json()
                
                # Handle single request
                if isinstance(body, dict):
                    response = await self._process_single_request(body)
                    return JSONResponse(content=response)
                
                # Handle batch requests
                elif isinstance(body, list):
                    responses = []
                    for req in body:
                        response = await self._process_single_request(req)
                        responses.append(response)
                    return JSONResponse(content=responses)
                
                else:
                    return JSONResponse(
                        content={
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32600,
                                "message": "Invalid Request"
                            },
                            "id": None
                        },
                        status_code=400
                    )
                    
            except Exception as e:
                logger.error(f"A2A server error: {e}")
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        },
                        "id": None
                    },
                    status_code=500
                )
        
        @self.app.get("/agents")
        async def list_agents():
            """List all registered A2A agents"""
            agents = await a2a_agent_registry.list_agents()
            return [
                {
                    "agent_id": wrapper.agent.agent_id,
                    "name": wrapper.agent.name,
                    "type": str(wrapper.agent.agent_type),
                    "url": wrapper.agent_url,
                    "status": str(wrapper.agent.status)
                }
                for wrapper in agents
            ]
        
        @self.app.get("/tasks/{task_id}")
        async def get_task(task_id: str):
            """Get A2A task details"""
            if task_id not in a2a_protocol.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = a2a_protocol.tasks[task_id]
            return {
                "id": task.id,
                "sessionId": task.sessionId,
                "status": {
                    "state": task.status.state.value,
                    "timestamp": task.status.timestamp
                },
                "artifacts": [
                    {
                        "name": artifact.name,
                        "description": artifact.description,
                        "parts": [
                            {
                                "type": part.type,
                                "text": getattr(part, "text", ""),
                                "metadata": part.metadata
                            }
                            for part in artifact.parts
                        ],
                        "metadata": artifact.metadata
                    }
                    for artifact in task.artifacts
                ],
                "metadata": task.metadata
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            agents = await a2a_agent_registry.list_agents()
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "agents_registered": len(agents),
                "tasks_active": len(a2a_protocol.tasks)
            }
    
    async def _process_single_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single JSON-RPC request"""
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            # Handle A2A-specific methods
            if method == "tasks/send":
                return await self._handle_task_send(params, request_id)
            elif method == "tasks/get":
                return await self._handle_task_get(params, request_id)
            elif method == "tasks/cancel":
                return await self._handle_task_cancel(params, request_id)
            else:
                # Delegate to protocol handler
                return await a2a_protocol.process_jsonrpc_request(request)
                
        except Exception as e:
            logger.error(f"Error processing A2A request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def _handle_task_send(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """Handle tasks/send with agent execution"""
        try:
            # Create or continue task
            task_id = params.get("id")
            session_id = params.get("sessionId")
            message_data = params.get("message", {})
            
            # Convert message data to Message object
            message = Message(
                role=message_data.get("role", "user"),
                parts=[
                    TextPart(
                        text=part.get("text", ""),
                        metadata=part.get("metadata", {})
                    )
                    for part in message_data.get("parts", [])
                ],
                metadata=message_data.get("metadata", {})
            )
            
            # Create or update task
            if task_id and task_id in a2a_protocol.tasks:
                task = await a2a_protocol.send_message(task_id, message)
            else:
                # Create new task
                agent_url = params.get("agent_url", "default")
                task = await a2a_protocol.create_task(agent_url, message, session_id)
            
            # Execute task with available agents
            await self._execute_task_with_agents(task)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "id": task.id,
                    "sessionId": task.sessionId,
                    "status": {
                        "state": task.status.state.value,
                        "timestamp": task.status.timestamp
                    },
                    "artifacts": [
                        {
                            "name": artifact.name,
                            "parts": [
                                {"type": part.type, "text": getattr(part, "text", "")}
                                for part in artifact.parts
                            ],
                            "metadata": artifact.metadata
                        }
                        for artifact in task.artifacts
                    ],
                    "metadata": task.metadata
                }
            }
            
        except Exception as e:
            logger.error(f"Error in tasks/send: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Task execution failed: {str(e)}"
                }
            }
    
    async def _handle_task_get(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """Handle tasks/get request"""
        task_id = params.get("id")
        
        if task_id not in a2a_protocol.tasks:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": f"Task not found: {task_id}"
                }
            }
        
        task = a2a_protocol.tasks[task_id]
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "id": task.id,
                "sessionId": task.sessionId,
                "status": {
                    "state": task.status.state.value,
                    "timestamp": task.status.timestamp
                },
                "artifacts": [
                    {
                        "name": artifact.name,
                        "parts": [
                            {"type": part.type, "text": getattr(part, "text", "")}
                            for part in artifact.parts
                        ],
                        "metadata": artifact.metadata
                    }
                    for artifact in task.artifacts
                ],
                "history": [
                    {
                        "role": msg.role,
                        "parts": [
                            {"type": part.type, "text": getattr(part, "text", "")}
                            for part in msg.parts
                        ]
                    }
                    for msg in task.history
                ],
                "metadata": task.metadata
            }
        }
    
    async def _handle_task_cancel(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """Handle tasks/cancel request"""
        task_id = params.get("id")
        
        if task_id not in a2a_protocol.tasks:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": f"Task not found: {task_id}"
                }
            }
        
        task = a2a_protocol.tasks[task_id]
        task.status.state = TaskState.CANCELED
        task.status.timestamp = datetime.utcnow().isoformat()
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "id": task.id,
                "sessionId": task.sessionId,
                "status": {
                    "state": task.status.state.value,
                    "timestamp": task.status.timestamp
                },
                "metadata": task.metadata
            }
        }
    
    async def _execute_task_with_agents(self, task):
        """Execute task using available agents"""
        try:
            # Get available agents
            agents = await a2a_agent_registry.list_agents()
            
            if not agents:
                logger.warning("No agents available to execute task")
                task.status.state = TaskState.FAILED
                task.status.timestamp = datetime.utcnow().isoformat()
                return
            
            # For now, use the first available agent
            # In a more sophisticated implementation, we would route based on task content
            agent_wrapper = agents[0]
            
            # Execute task
            await agent_wrapper.handle_a2a_task(task)
            
        except Exception as e:
            logger.error(f"Error executing task with agents: {e}")
            task.status.state = TaskState.FAILED
            task.status.timestamp = datetime.utcnow().isoformat()
    
    async def start(self):
        """Start the A2A server"""
        import uvicorn
        
        logger.info(f"Starting A2A server on {self.host}:{self.port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()


# Global A2A server instance
a2a_server = A2AServer()
