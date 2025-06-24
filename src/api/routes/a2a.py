#!/usr/bin/env python3
"""
A2A Protocol API Routes

FastAPI routes for A2A (Agent-to-Agent) protocol endpoints.
Implements Google A2A specification for agent communication.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# A2A Protocol imports
try:
    import sys
    import os
    from pathlib import Path

    # Add src to path for imports
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from a2a_protocol.transport import A2ATransportLayer
    from a2a_protocol.task_manager import A2ATaskManager
    from a2a_protocol.security import A2ASecurityManager
    from a2a_protocol.discovery import A2AAgentDiscovery
    from a2a_protocol.agent_card_generator import A2AAgentCardGenerator
    from a2a_protocol.error_handling import A2AError, A2ATransportError
    from a2a_standard import AgentCard, Task, TaskState, Message, TextPart
    from core.agent_factory import AgentFactory
    A2A_AVAILABLE = True
except ImportError as e:
    A2A_AVAILABLE = False
    print(f"A2A imports failed: {e}")

    # Fallback imports for when A2A is not available
    class AgentFactory:
        def __init__(self, *args, **kwargs):
            self.a2a_enabled = False

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/a2a/v1", tags=["A2A Protocol"])

# A2A Protocol Models
class A2AMessageRequest(BaseModel):
    """A2A message request model"""
    method: str = Field(..., description="A2A method name")
    params: Dict[str, Any] = Field(..., description="Method parameters")
    id: Optional[str] = Field(None, description="Request ID")

class A2AMessageResponse(BaseModel):
    """A2A message response model"""
    result: Optional[Dict[str, Any]] = Field(None, description="Method result")
    error: Optional[Dict[str, Any]] = Field(None, description="Error information")
    id: Optional[str] = Field(None, description="Request ID")

class AgentDiscoveryResponse(BaseModel):
    """Agent discovery response model"""
    agents: List[Dict[str, Any]] = Field(..., description="Discovered agents")
    total: int = Field(..., description="Total number of agents")
    timestamp: str = Field(..., description="Discovery timestamp")

# Dependency injection
async def get_agent_factory() -> AgentFactory:
    """Get agent factory instance"""
    # This should be injected from the main application
    # For now, create a new instance
    return AgentFactory()

async def get_a2a_components():
    """Get A2A protocol components"""
    if not A2A_AVAILABLE:
        raise HTTPException(status_code=503, detail="A2A protocol not available")
    
    return {
        "transport": A2ATransportLayer(),
        "task_manager": A2ATaskManager(),
        "security": A2ASecurityManager(),
        "discovery": A2AAgentDiscovery(),
        "card_generator": A2AAgentCardGenerator()
    }

# Well-known endpoints
@router.get("/.well-known/agent.json")
async def get_agent_card(
    agent_factory: AgentFactory = Depends(get_agent_factory)
):
    """Get agent card for A2A discovery"""
    try:
        if not agent_factory.a2a_enabled:
            raise HTTPException(status_code=503, detail="A2A protocol not enabled")
        
        # Generate agent card for the PyGent Factory system
        agent_card = await agent_factory.a2a_card_generator.generate_system_agent_card()
        
        return JSONResponse(content=agent_card)
        
    except Exception as e:
        logger.error(f"Failed to get agent card: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Agent discovery endpoints
@router.get("/agents/discover", response_model=AgentDiscoveryResponse)
async def discover_agents(
    agent_factory: AgentFactory = Depends(get_agent_factory)
):
    """Discover available A2A agents"""
    try:
        if not agent_factory.a2a_enabled:
            raise HTTPException(status_code=503, detail="A2A protocol not enabled")
        
        # Get all registered agents
        agents = []
        for agent_id, agent in agent_factory.agents.items():
            try:
                # Generate agent card for each agent
                agent_card = await agent_factory.a2a_card_generator.generate_agent_card(
                    agent=agent,
                    agent_type=agent.type,
                    enable_authentication=True
                )
                agents.append(agent_card)
            except Exception as e:
                logger.warning(f"Failed to generate card for agent {agent_id}: {e}")
        
        return AgentDiscoveryResponse(
            agents=agents,
            total=len(agents),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to discover agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_id}/card")
async def get_agent_card_by_id(
    agent_id: str,
    agent_factory: AgentFactory = Depends(get_agent_factory)
):
    """Get agent card for specific agent"""
    try:
        if not agent_factory.a2a_enabled:
            raise HTTPException(status_code=503, detail="A2A protocol not enabled")
        
        # Get agent
        agent = await agent_factory.get_agent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Generate agent card
        agent_card = await agent_factory.a2a_card_generator.generate_agent_card(
            agent=agent,
            agent_type=agent.type,
            enable_authentication=True
        )
        
        return JSONResponse(content=agent_card)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent card for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Message handling endpoints
@router.post("/message/send", response_model=A2AMessageResponse)
async def send_message(
    request: A2AMessageRequest,
    agent_factory: AgentFactory = Depends(get_agent_factory),
    a2a_components: Dict = Depends(get_a2a_components)
):
    """Send A2A message to agent"""
    try:
        transport = a2a_components["transport"]
        
        # Process A2A message through transport layer
        response = await transport.handle_jsonrpc_request({
            "jsonrpc": "2.0",
            "method": request.method,
            "params": request.params,
            "id": request.id
        })
        
        return A2AMessageResponse(
            result=response.get("result"),
            error=response.get("error"),
            id=response.get("id")
        )
        
    except Exception as e:
        logger.error(f"Failed to send A2A message: {e}")
        return A2AMessageResponse(
            error={
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            },
            id=request.id
        )

@router.post("/message/stream")
async def stream_message(
    request: A2AMessageRequest,
    agent_factory: AgentFactory = Depends(get_agent_factory),
    a2a_components: Dict = Depends(get_a2a_components)
):
    """Stream A2A message to agent"""
    try:
        transport = a2a_components["transport"]
        
        # Handle streaming message
        async def generate_stream():
            try:
                # Process streaming request
                response = await transport.handle_streaming_request({
                    "method": request.method,
                    "params": request.params,
                    "id": request.id
                })
                
                # Yield response chunks
                if response:
                    yield f"data: {response}\n\n"
                else:
                    yield "data: {\"status\": \"completed\"}\n\n"
                    
            except Exception as e:
                error_response = {
                    "error": {
                        "code": -32603,
                        "message": "Streaming error",
                        "data": str(e)
                    }
                }
                yield f"data: {error_response}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to stream A2A message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Task management endpoints
@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    a2a_components: Dict = Depends(get_a2a_components)
):
    """Get A2A task by ID"""
    try:
        task_manager = a2a_components["task_manager"]
        
        # Get task
        task = task_manager.get_task_sync(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Convert task to dict
        task_dict = {
            "id": task.id,
            "contextId": task.contextId,
            "status": {
                "state": task.status.state.value if hasattr(task.status.state, 'value') else str(task.status.state),
                "timestamp": task.status.timestamp,
                "message": task.status.message
            },
            "created_at": task.created_at,
            "updated_at": task.updated_at
        }
        
        return JSONResponse(content=task_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    a2a_components: Dict = Depends(get_a2a_components)
):
    """Cancel A2A task"""
    try:
        task_manager = a2a_components["task_manager"]
        
        # Cancel task
        success = task_manager.update_task_status_sync(
            task_id=task_id,
            state=TaskState.CANCELLED,
            message="Task cancelled by user request"
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
        
        return {"status": "cancelled", "task_id": task_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def a2a_health_check(
    agent_factory: AgentFactory = Depends(get_agent_factory)
):
    """A2A protocol health check"""
    try:
        health_status = {
            "status": "healthy",
            "a2a_enabled": agent_factory.a2a_enabled if agent_factory else False,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "transport": A2A_AVAILABLE,
                "task_manager": A2A_AVAILABLE,
                "security": A2A_AVAILABLE,
                "discovery": A2A_AVAILABLE,
                "card_generator": A2A_AVAILABLE
            }
        }
        
        if agent_factory and agent_factory.a2a_enabled:
            health_status["registered_agents"] = len(agent_factory.agents)
        
        return health_status
        
    except Exception as e:
        logger.error(f"A2A health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
