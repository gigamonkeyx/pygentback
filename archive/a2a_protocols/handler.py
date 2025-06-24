import uuid
import asyncio
import logging
from typing import Dict
from datetime import datetime
from fastapi import HTTPException
from sse_starlette import EventSourceResponse

from .models import (
    Task, TaskState, TaskStatus, MessageSendParams, 
    TaskQueryParams, AgentCard
)
from .agent_card_generator import AgentCardGenerator

logger = logging.getLogger(__name__)

class A2AProtocolHandler:
    """A2A Protocol v0.2.1 implementation"""
    
    def __init__(self, agent_factory=None, memory_manager=None, db_manager=None):
        self.tasks: Dict[str, Task] = {}
        self.active_streams: Dict[str, asyncio.Queue] = {}
        self.agent_cards: Dict[str, AgentCard] = {}
        
        # Dependency injection
        self.agent_factory = agent_factory
        self.memory_manager = memory_manager
        self.db_manager = db_manager
        
        # Initialize agent card generator
        if agent_factory:
            self.card_generator = AgentCardGenerator(agent_factory)
            # Generate and cache factory card
            self.agent_cards["factory"] = self.card_generator.generate_factory_card()
        else:
            self.card_generator = None
    
    async def handle_message_send(self, params: MessageSendParams) -> Task:
        """Implement message/send method"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            contextId=params.contextId,
            status=TaskStatus(state=TaskState.SUBMITTED),
            history=[params.message] if params.message else None
        )
        
        self.tasks[task_id] = task
        
        # Process message asynchronously
        asyncio.create_task(self._process_message_async(task, params))
        
        return task
    
    async def handle_message_stream(self, params: MessageSendParams) -> EventSourceResponse:
        """Implement streaming with Server-Sent Events"""
        task_id = str(uuid.uuid4())
        stream_queue = asyncio.Queue()
        self.active_streams[task_id] = stream_queue
        
        # Start processing in background
        asyncio.create_task(self._process_stream_async(task_id, params, stream_queue))
        
        return EventSourceResponse(self._stream_generator(task_id, stream_queue))
    
    async def handle_tasks_get(self, params: TaskQueryParams) -> Task:
        """Implement task retrieval"""
        task = self.tasks.get(params.task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {params.task_id} not found")
        return task
    
    async def _process_message_async(self, task: Task, params: MessageSendParams):
        """Process message asynchronously"""
        try:
            # Update task status
            task.status.state = TaskState.WORKING
            task.updated_at = datetime.utcnow()
            
            # Simulate processing
            await asyncio.sleep(0.1)
            
            # Mark as completed
            task.status.state = TaskState.COMPLETED
            task.status.message = "Message processed successfully"
            task.updated_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error processing task {task.id}: {e}")
            task.status.state = TaskState.FAILED
            task.status.message = str(e)
            task.updated_at = datetime.utcnow()
    
    async def _process_stream_async(self, task_id: str, params: MessageSendParams, stream_queue: asyncio.Queue):
        """Process streaming message"""
        try:
            # Send initial status
            await stream_queue.put({
                "type": "task_update",
                "task_id": task_id,
                "status": "working"
            })
            
            # Simulate processing with progress updates
            for progress in [0.25, 0.5, 0.75, 1.0]:
                await asyncio.sleep(0.1)
                await stream_queue.put({
                    "type": "progress",
                    "task_id": task_id,
                    "progress": progress
                })
            
            # Send completion
            await stream_queue.put({
                "type": "completed",
                "task_id": task_id,
                "message": "Stream processing completed"
            })
            
        except Exception as e:
            await stream_queue.put({
                "type": "error",
                "task_id": task_id,
                "error": str(e)
            })
        finally:
            await stream_queue.put(None)  # End of stream marker
    
    async def _stream_generator(self, task_id: str, stream_queue: asyncio.Queue):
        """Generator for Server-Sent Events"""
        try:
            while True:
                data = await stream_queue.get()
                if data is None:  # End of stream
                    break
                yield {
                    "event": data.get("type", "message"),
                    "data": data
                }
        except Exception as e:
            logger.error(f"Error in stream generator for task {task_id}: {e}")
            yield {
                "event": "error",
                "data": {"error": str(e)}
            }
        finally:            # Clean up
            if task_id in self.active_streams:
                del self.active_streams[task_id]
    
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
        if "factory" in self.agent_cards:
            return self.agent_cards["factory"]
        elif self.card_generator:
            return self.card_generator.generate_factory_card()
        else:
            # Fallback static card
            from .models import AgentCard
            return AgentCard(
                metadata={
                    "name": "pygent-factory",
                    "displayName": "PyGent Factory",
                    "description": "Advanced AI Agent Factory",
                    "version": "1.0.0"
                },
                spec={
                    "endpoints": {"a2a": "/a2a/v1"},
                    "capabilities": ["reasoning", "evolution", "multi-agent"]
                }
            )
    
    def register_agent_card(self, agent_id: str, card: AgentCard):
        """Register an agent card"""
        self.agent_cards[agent_id] = card
        logger.info(f"Registered agent card for {agent_id}")
    
    def get_all_tasks(self) -> Dict[str, Task]:
        """Get all tasks for debugging/monitoring"""
        return self.tasks.copy()
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks"""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        to_remove = []
        
        for task_id, task in self.tasks.items():
            if (task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED] and 
                task.updated_at.timestamp() < cutoff_time):
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.tasks[task_id]
            logger.info(f"Cleaned up old task {task_id}")
        
        return len(to_remove)
