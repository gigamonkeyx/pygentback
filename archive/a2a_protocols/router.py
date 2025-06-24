from fastapi import APIRouter, Depends

from .dependencies import get_a2a_handler
from .handler import A2AProtocolHandler
from .models import MessageSendParams, TaskQueryParams, Task

router = APIRouter(prefix="/a2a/v1", tags=["A2A Protocol"])

@router.post("/message/send", response_model=Task)
async def message_send(
    params: MessageSendParams,
    handler: A2AProtocolHandler = Depends(get_a2a_handler)
):
    """Send message to agent"""
    return await handler.handle_message_send(params)

@router.post("/message/stream")
async def message_stream(
    params: MessageSendParams,
    handler: A2AProtocolHandler = Depends(get_a2a_handler)
):
    """Stream message to agent"""
    return await handler.handle_message_stream(params)

@router.get("/tasks/{task_id}", response_model=Task)
async def get_task(
    task_id: str,
    handler: A2AProtocolHandler = Depends(get_a2a_handler)
):
    """Get task by ID"""
    query_params = TaskQueryParams(task_id=task_id)
    return await handler.handle_tasks_get(query_params)

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

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "protocol": "a2a/v1"}
