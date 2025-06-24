#!/usr/bin/env python3
"""
A2A Protocol Streaming Support

Implements Server-Sent Events (SSE) streaming according to Google A2A specification.
Handles real-time task updates, artifact streaming, and status notifications.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)


@dataclass
class StreamingEvent:
    """Server-Sent Event structure"""
    event: Optional[str] = None
    data: Optional[str] = None
    id: Optional[str] = None
    retry: Optional[int] = None


@dataclass
class TaskStatusUpdateEvent:
    """A2A Task Status Update Event"""
    taskId: str
    contextId: str
    status: Dict[str, Any]
    final: bool = False
    kind: str = "status-update"


@dataclass
class TaskArtifactUpdateEvent:
    """A2A Task Artifact Update Event"""
    taskId: str
    contextId: str
    artifact: Dict[str, Any]
    append: bool = False
    lastChunk: bool = False
    kind: str = "artifact-update"


@dataclass
class SendStreamingMessageResponse:
    """A2A Streaming Message Response"""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class A2AStreamingManager:
    """Manages A2A streaming connections and events"""
    
    def __init__(self):
        self.active_streams: Dict[str, asyncio.Queue] = {}
        self.task_streams: Dict[str, List[str]] = {}  # task_id -> list of stream_ids
        self._lock = asyncio.Lock()
    
    async def create_stream(self, task_id: str, stream_id: Optional[str] = None) -> str:
        """Create a new streaming connection for a task"""
        if not stream_id:
            stream_id = str(uuid.uuid4())
        
        async with self._lock:
            # Create event queue for this stream
            self.active_streams[stream_id] = asyncio.Queue(maxsize=100)
            
            # Track stream for task
            if task_id not in self.task_streams:
                self.task_streams[task_id] = []
            self.task_streams[task_id].append(stream_id)
        
        logger.info(f"Created stream {stream_id} for task {task_id}")
        return stream_id
    
    async def close_stream(self, stream_id: str):
        """Close a streaming connection"""
        async with self._lock:
            if stream_id in self.active_streams:
                # Signal end of stream
                try:
                    await self.active_streams[stream_id].put(None)
                except asyncio.QueueFull:
                    pass
                
                # Remove from active streams
                del self.active_streams[stream_id]
                
                # Remove from task streams
                for task_id, streams in self.task_streams.items():
                    if stream_id in streams:
                        streams.remove(stream_id)
                        if not streams:
                            del self.task_streams[task_id]
                        break
        
        logger.info(f"Closed stream {stream_id}")
    
    async def send_task_status_update(self, task_id: str, status: Dict[str, Any], final: bool = False):
        """Send task status update to all streams for this task"""
        if task_id not in self.task_streams:
            return
        
        event = TaskStatusUpdateEvent(
            taskId=task_id,
            contextId=status.get("contextId", str(uuid.uuid4())),
            status=status,
            final=final
        )
        
        response = SendStreamingMessageResponse(
            id=str(uuid.uuid4()),
            result=asdict(event)
        )
        
        await self._broadcast_to_task_streams(task_id, response)
        
        if final:
            # Close all streams for this task
            for stream_id in self.task_streams.get(task_id, []):
                await self.close_stream(stream_id)
    
    async def send_artifact_update(self, task_id: str, artifact: Dict[str, Any], 
                                 append: bool = False, last_chunk: bool = False):
        """Send artifact update to all streams for this task"""
        if task_id not in self.task_streams:
            return
        
        event = TaskArtifactUpdateEvent(
            taskId=task_id,
            contextId=artifact.get("contextId", str(uuid.uuid4())),
            artifact=artifact,
            append=append,
            lastChunk=last_chunk
        )
        
        response = SendStreamingMessageResponse(
            id=str(uuid.uuid4()),
            result=asdict(event)
        )
        
        await self._broadcast_to_task_streams(task_id, response)
    
    async def _broadcast_to_task_streams(self, task_id: str, response: SendStreamingMessageResponse):
        """Broadcast response to all streams for a task"""
        stream_ids = self.task_streams.get(task_id, [])
        
        for stream_id in stream_ids:
            if stream_id in self.active_streams:
                try:
                    await self.active_streams[stream_id].put(response)
                except asyncio.QueueFull:
                    logger.warning(f"Stream {stream_id} queue full, dropping event")
    
    async def get_stream_events(self, stream_id: str) -> AsyncGenerator[str, None]:
        """Get events for a specific stream"""
        if stream_id not in self.active_streams:
            return
        
        queue = self.active_streams[stream_id]
        
        try:
            while True:
                # Wait for next event
                event = await queue.get()
                
                # None signals end of stream
                if event is None:
                    break
                
                # Convert to SSE format
                sse_data = self._format_sse_event(event)
                yield sse_data
                
        except asyncio.CancelledError:
            logger.info(f"Stream {stream_id} cancelled")
        except Exception as e:
            logger.error(f"Error in stream {stream_id}: {e}")
        finally:
            await self.close_stream(stream_id)
    
    def _format_sse_event(self, response: SendStreamingMessageResponse) -> str:
        """Format response as Server-Sent Event"""
        data = json.dumps(asdict(response))
        return f"data: {data}\n\n"


class A2AStreamingHandler:
    """Handles A2A streaming endpoints and responses"""
    
    def __init__(self, streaming_manager: Optional[A2AStreamingManager] = None):
        self.streaming_manager = streaming_manager or A2AStreamingManager()
    
    async def handle_message_stream(self, params: Dict[str, Any], 
                                  auth: Optional[HTTPAuthorizationCredentials] = None) -> StreamingResponse:
        """Handle message/stream request with SSE response"""
        try:
            # Validate parameters
            if "message" not in params:
                raise HTTPException(status_code=400, detail="Invalid params: 'message' required")
            
            message = params["message"]
            task_id = params.get("taskId") or str(uuid.uuid4())
            context_id = params.get("contextId") or str(uuid.uuid4())
            
            # Create streaming connection
            stream_id = await self.streaming_manager.create_stream(task_id)
            
            # Send initial task status
            initial_status = {
                "state": "submitted",
                "timestamp": datetime.utcnow().isoformat(),
                "contextId": context_id
            }
            
            # Create initial response
            initial_response = SendStreamingMessageResponse(
                id=str(uuid.uuid4()),
                result={
                    "id": task_id,
                    "contextId": context_id,
                    "status": initial_status,
                    "history": [message],
                    "artifacts": [],
                    "kind": "task",
                    "metadata": {}
                }
            )
            
            # Add initial response to stream
            await self.streaming_manager.active_streams[stream_id].put(initial_response)
            
            # Start background task processing
            asyncio.create_task(self._process_streaming_task(task_id, context_id, message))
            
            # Return streaming response
            return StreamingResponse(
                self.streaming_manager.get_stream_events(stream_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control"
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling message/stream: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def handle_tasks_resubscribe(self, params: Dict[str, Any], 
                                     auth: Optional[HTTPAuthorizationCredentials] = None) -> StreamingResponse:
        """Handle tasks/resubscribe request with SSE response"""
        try:
            # Validate parameters
            if "id" not in params:
                raise HTTPException(status_code=400, detail="Invalid params: 'id' required")
            
            task_id = params["id"]
            
            # Create new streaming connection for existing task
            stream_id = await self.streaming_manager.create_stream(task_id)
            
            # Send current task status (placeholder)
            current_status = {
                "state": "working",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.streaming_manager.send_task_status_update(task_id, current_status)
            
            # Return streaming response
            return StreamingResponse(
                self.streaming_manager.get_stream_events(stream_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control"
                }
            )
            
        except Exception as e:
            logger.error(f"Error handling tasks/resubscribe: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _process_streaming_task(self, task_id: str, context_id: str, message: Dict[str, Any]):
        """Process task with streaming updates - REAL IMPLEMENTATION"""
        try:
            # Import real task processing components
            from core.agent_factory import AgentFactory
            from ai.multi_agent.models import Task, TaskStatus

            # Update to working status immediately
            working_status = {
                "state": "working",
                "timestamp": datetime.utcnow().isoformat(),
                "contextId": context_id
            }
            await self.streaming_manager.send_task_status_update(task_id, working_status)

            # Create real task and process it
            agent_factory = AgentFactory()
            task_content = message.get('parts', [{}])[0].get('text', 'No text')

            # Create and execute real task
            real_task = Task(
                task_id=task_id,
                description=f"A2A streaming task: {task_content}",
                status=TaskStatus.RUNNING
            )

            # Process with real agent (get first available agent)
            available_agents = await agent_factory.get_available_agents()
            if available_agents:
                agent = available_agents[0]
                result = await agent.execute_task(real_task)
                response_text = result.get('result', f"Processed: {task_content}")
            else:
                # No agents available - create a general agent
                agent = await agent_factory.create_agent("general", name=f"streaming_agent_{task_id[:8]}")
                result = await agent.execute_task(real_task)
                response_text = result.get('result', f"Processed: {task_content}")

            # Generate real artifact from task result
            artifact = {
                "artifactId": str(uuid.uuid4()),
                "name": "response",
                "parts": [
                    {
                        "kind": "text",
                        "text": response_text
                    }
                ]
            }
            
            await self.streaming_manager.send_artifact_update(
                task_id, artifact, append=False, last_chunk=True
            )

            # Complete task immediately (no simulation delay)
            completed_status = {
                "state": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "contextId": context_id
            }
            await self.streaming_manager.send_task_status_update(task_id, completed_status, final=True)
            
            logger.info(f"Completed streaming task {task_id}")
            
        except Exception as e:
            logger.error(f"Error processing streaming task {task_id}: {e}")
            
            # Send error status
            error_status = {
                "state": "failed",
                "timestamp": datetime.utcnow().isoformat(),
                "contextId": context_id,
                "error": str(e)
            }
            await self.streaming_manager.send_task_status_update(task_id, error_status, final=True)
    
    def setup_fastapi_routes(self, app: FastAPI):
        """Setup FastAPI routes for A2A streaming"""
        
        @app.post("/a2a/v1/stream")
        async def a2a_stream_endpoint(request: Request):
            """A2A streaming endpoint for message/stream and tasks/resubscribe"""
            try:
                # Parse request body
                body = await request.body()
                request_data = json.loads(body)
                
                method = request_data.get("method")
                params = request_data.get("params", {})
                
                if method == "message/stream":
                    return await self.handle_message_stream(params)
                elif method == "tasks/resubscribe":
                    return await self.handle_tasks_resubscribe(params)
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported streaming method: {method}")
                    
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON")
            except Exception as e:
                logger.error(f"A2A streaming endpoint error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        logger.info("A2A streaming routes registered with FastAPI")


# Global streaming instances
streaming_manager = A2AStreamingManager()
streaming_handler = A2AStreamingHandler(streaming_manager)
