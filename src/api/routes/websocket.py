"""
WebSocket API Routes for Real-time Communication

This module provides WebSocket endpoints for real-time communication
between the frontend and AI components.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import re

from fastapi import WebSocket, WebSocketDisconnect, Depends
from fastapi.routing import APIRouter

from ...mcp.server_registry import MCPServerManager
# Removed circular import - will initialize dependencies locally

# Import ToT reasoning components
from ...ai.reasoning.tot.models import ToTConfig, GenerationStrategy, EvaluationMethod, SearchMethod
from ...ai.reasoning.tot.tot_engine import ToTEngine
from ...ai.reasoning.tot.thought_generator import OllamaBackend
from ...agents.reasoning_agent import ReasoningAgent
from ...core.agent.config import AgentConfig

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, str] = {}  # user_id -> connection_id
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.user_connections[user_id] = connection_id
        logger.info(f"WebSocket connection established: {connection_id} for user {user_id}")
    
    def disconnect(self, connection_id: str, user_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if user_id in self.user_connections:
            del self.user_connections[user_id]
        logger.info(f"WebSocket connection closed: {connection_id} for user {user_id}")
    
    async def send_personal_message(self, message: dict, user_id: str):
        """Send a message to a specific user"""
        connection_id = self.user_connections.get(user_id)
        if connection_id and connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}: {e}")
                # Remove broken connection
                self.disconnect(connection_id, user_id)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients"""
        if not self.active_connections:
            return
        
        message_text = json.dumps(message)
        disconnected = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message_text)
            except Exception as e:
                logger.error(f"Failed to broadcast to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected clients
        for connection_id in disconnected:
            # Find user_id for this connection
            user_id = None
            for uid, cid in self.user_connections.items():
                if cid == connection_id:
                    user_id = uid
                    break
            if user_id:
                self.disconnect(connection_id, user_id)


# Global connection manager
manager = ConnectionManager()


# Test endpoint to verify router is working
@router.get("/ws-test")
async def websocket_test():
    """Test endpoint to verify WebSocket router is registered"""
    return {"message": "WebSocket router is working", "endpoint": "/ws"}


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time communication"""
    
    # Generate unique connection ID and user ID
    connection_id = f"conn_{datetime.now().timestamp()}_{id(websocket)}"
    user_id = f"user_{datetime.now().timestamp()}_{id(websocket)}"  # Generate unique user ID per connection
    
    await manager.connect(websocket, connection_id, user_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Route message based on type
            await handle_websocket_message(message, user_id)
            
    except WebSocketDisconnect:
        manager.disconnect(connection_id, user_id)
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
        manager.disconnect(connection_id, user_id)


async def handle_websocket_message(message: dict, user_id: str):
    """Handle incoming WebSocket messages and route to appropriate handlers"""

    message_type = message.get('type')
    data = message.get('data', {})

    logger.info(f"Received WebSocket message: type={message_type}, user={user_id}, data_keys={list(data.keys()) if isinstance(data, dict) else 'not_dict'}")

    try:
        if message_type == 'chat_message':
            logger.info(f"Processing chat message: {data}")
            await handle_chat_message(data, user_id)

        elif message_type == 'ping':
            logger.info(f"Received ping from {user_id}")
            await manager.send_personal_message({
                'type': 'pong',
                'data': {'timestamp': datetime.now().isoformat()}
            }, user_id)

        elif message_type == 'request_system_metrics':
            await handle_system_metrics_request(data, user_id)

        elif message_type == 'start_reasoning':
            logger.info(f"Starting reasoning session for {user_id}")
            await handle_start_reasoning(data, user_id)

        elif message_type == 'stop_reasoning':
            logger.info(f"Stopping reasoning session for {user_id}")
            await handle_stop_reasoning(data, user_id)

        else:
            logger.warning(f"Unknown message type: {message_type}")
            await manager.send_personal_message({
                'type': 'error',
                'data': {'message': f'Unknown message type: {message_type}'}
            }, user_id)

    except Exception as e:
        logger.error(f"Error handling message {message_type}: {e}")
        await manager.send_personal_message({
            'type': 'error',
            'data': {'message': f'Error processing {message_type}: {str(e)}'}
        }, user_id)


async def handle_chat_message(data: dict, user_id: str):
    """Handle chat messages and generate AI responses"""

    # Write debug logs to file
    debug_log = f"\n[{datetime.now()}] === NEW CHAT MESSAGE ===\n"
    debug_log += f"WEBSOCKET: handle_chat_message called with data: {data}\n"

    message = data.get('message', {})
    agent_type = message.get('agentId', 'general')
    content = message.get('content', '')

    debug_log += f"WEBSOCKET: Extracted: agent_type={agent_type}, content_length={len(content)}\n"
    debug_log += f"WEBSOCKET: RAW USER CONTENT: '{content}'\n"
    
    # Write to debug file
    try:
        with open("debug_pipeline.log", "a", encoding="utf-8") as f:
            f.write(debug_log)
    except Exception as e:
        print(f"Failed to write debug log: {e}")
    
    logger.info(f"WEBSOCKET: handle_chat_message called with data: {data}")
    
    logger.info(f"WEBSOCKET: Extracted: agent_type={agent_type}, content_length={len(content)}")
    logger.info(f"WEBSOCKET: RAW USER CONTENT: '{content}'")

    # Send typing indicator
    await manager.send_personal_message({
        'type': 'typing_indicator',
        'data': {'user_id': 'ai_agent', 'typing': True}
    }, user_id)
    
    try:
        # Emit agent thinking start
        await emit_agent_thinking(user_id, agent_type, f"Processing your request: {content[:100]}...")
        await emit_agent_status(user_id, agent_type, "initializing", {"request_type": "chat_message"})
        
        # Get real AI response from appropriate agent
        logger.info(f"Getting AI response from agent: {agent_type}")
        await emit_agent_status(user_id, agent_type, "processing", {"model_loading": True})
        
        ai_response = await get_agent_response(agent_type, content, user_id)
        logger.info(f"AI response received: {ai_response}")

        # Emit thinking completion
        await emit_agent_thinking(
            user_id, 
            agent_type, 
            f"Generated response with {ai_response.get('confidence', 0.8):.1%} confidence",
            confidence=ai_response.get('confidence', 0.8),
            step="response_generation"
        )

        # Extract and emit reasoning content for the reasoning page
        raw_response = ai_response.get('response', 'Sorry, I encountered an issue processing your request.')
        cleaned_response = await extract_and_emit_reasoning(raw_response, user_id, agent_type)
        
        # Create response message
        response_message = {
            'id': f"msg_{datetime.now().timestamp()}",
            'type': 'agent',
            'content': cleaned_response,
            'timestamp': datetime.now().isoformat(),
            'agentId': agent_type,
            'metadata': {
                'agent_type': ai_response.get('agent', agent_type),
                'processing_time': ai_response.get('metrics', {}).get('total_time', 0),
                'confidence': ai_response.get('confidence', 0.8)
            }
        }

        logger.info(f"Sending response message: {response_message}")

        # Check if we should create a task based on the response
        task_id = await create_task_from_response(ai_response, agent_type, content, user_id)
        
        if task_id:
            # Add task information to the response
            response_message['metadata']['task_created'] = task_id
            
            # Send task creation notification
            await manager.send_personal_message({
                'type': 'task_created',
                'data': {
                    'task_id': task_id,
                    'task_type': 'coding',
                    'description': f"Task created from your request: {content[:100]}..."
                }
            }, user_id)

        # Send response
        await manager.send_personal_message({
            'type': 'chat_response',
            'data': {'message': response_message}
        }, user_id)

        logger.info(f"Response sent successfully to user: {user_id}")
        
        # Emit response completion
        await emit_agent_activity(
            user_id=user_id,
            activity_type='response',
            agent_name=agent_type,
            content=f"Response delivered ({len(ai_response.get('response', ''))} chars)",
            metadata={
                'response_length': len(ai_response.get('response', '')),
                'confidence': ai_response.get('confidence', 0.8),
                'processing_time': ai_response.get('metrics', {}).get('total_time', 0)
            }
        )

        # Create task if AI response suggests actionable items
        task_id = await create_task_from_response(ai_response, agent_type, content, user_id)
        if task_id:
            logger.info(f"Task {task_id} created successfully from AI response")
            # Emit task creation activity
            await emit_agent_activity(
                user_id=user_id,
                activity_type='task_created',
                agent_name='system',
                content=f"Created task {task_id} from {agent_type} response",
                metadata={
                    'task_id': task_id,
                    'source_agent': agent_type,
                    'original_request': content[:200]
                }
            )

    except Exception as e:
        logger.error(f"Error handling chat message: {e}")
        
        # Emit error activity
        await emit_agent_activity(
            user_id=user_id,
            activity_type='error',
            agent_name=agent_type,
            content=f"Error processing request: {str(e)}",
            metadata={'error_type': type(e).__name__, 'error_message': str(e)}
        )
        
        # Send error response
        error_message = {
            'id': f"msg_{datetime.now().timestamp()}",
            'type': 'agent',
            'content': "I apologize, but I encountered an error processing your request. Please try again.",
            'timestamp': datetime.now().isoformat(),
            'agentId': agent_type,
            'error': True
        }

        await manager.send_personal_message({
            'type': 'chat_response',
            'data': {'message': error_message}
        }, user_id)
    
    finally:
        # Stop typing indicator
        await manager.send_personal_message({
            'type': 'typing_indicator',
            'data': {'user_id': 'ai_agent', 'typing': False}
        }, user_id)


async def handle_system_metrics_request(data: dict, user_id: str):
    """Handle system metrics requests"""
    
    try:
        # Import here to avoid circular imports
        from ...monitoring.system_monitor import get_system_metrics
        
        metrics = await get_system_metrics()
        
        await manager.send_personal_message({
            'type': 'system_metrics',
            'data': {'metrics': metrics}
        }, user_id)
    
    except Exception as e:
        await manager.send_personal_message({
            'type': 'system_metrics_error',
            'data': {'error': str(e)}
        }, user_id)


async def get_agent_response(agent_type: str, content: str, user_id: str) -> Dict[str, Any]:
    """Get AI response from the appropriate agent"""

    # Write debug logs to file
    debug_log = f"\n[{datetime.now()}] === GET AGENT RESPONSE ===\n"
    debug_log += f"BACKEND: Getting agent response: type={agent_type}, user={user_id}, content_length={len(content)}\n"
    debug_log += f"BACKEND: USER QUERY CONTENT: '{content}'\n"
    
    try:
        with open("debug_pipeline.log", "a", encoding="utf-8") as f:
            f.write(debug_log)
    except Exception as e:
        print(f"Failed to write debug log: {e}")

    logger.info(f"BACKEND: Getting agent response: type={agent_type}, user={user_id}, content_length={len(content)}")
    logger.info(f"BACKEND: USER QUERY CONTENT: '{content}'")

    try:
        # Emit initialization step
        await emit_agent_thinking(user_id, agent_type, "Initializing agent components...", step="initialization")
        
        # Import dependencies
        from ...core.agent_factory import AgentFactory
        from ...core.agent import AgentMessage, MessageType
        from ...core.agent.config import AgentConfig
        from ...api.main import app_state

        # Get initialized components from app state
        settings = app_state.get("settings")
        memory_manager = app_state.get("memory_manager")
        mcp_manager = app_state.get("mcp_manager")

        if not all([settings, memory_manager, mcp_manager]):
            raise Exception("Required components not initialized in app state")

        # Emit model selection step
        await emit_agent_thinking(user_id, agent_type, "Selecting optimal model for task...", step="model_selection")

        # Create agent factory with Ollama manager
        ollama_manager = app_state.get("ollama_manager")
        agent_factory = AgentFactory(mcp_manager, memory_manager, settings, ollama_manager)

        # Get available models from Ollama manager
        if not ollama_manager:
            raise ValueError("Ollama manager not available")
        available_models = await ollama_manager.get_available_models()

        # Dynamic model selection based on capabilities and task type
        default_model = None
        reasoning_models = []
        general_models = []
        
        if available_models:
            # Categorize models by capabilities
            for model in available_models:
                model_lower = model.lower()
                if any(keyword in model_lower for keyword in ["deepseek-r1", "qwen3", "deepseek-coder"]):
                    reasoning_models.append(model)
                else:
                    general_models.append(model)
            
            # Select model based on agent type and task complexity
            if agent_type == "reasoning" or "math" in content.lower() or any(op in content for op in ["*", "+", "-", "/", "^", "="]):
                # For reasoning tasks, prefer reasoning-capable models
                if reasoning_models:
                    # Prefer deepseek-r1 for math/reasoning, qwen3 for general reasoning
                    if any("deepseek-r1" in model for model in reasoning_models):
                        default_model = next(model for model in reasoning_models if "deepseek-r1" in model)
                    elif any("qwen3" in model for model in reasoning_models):
                        default_model = next(model for model in reasoning_models if "qwen3" in model)
                    else:
                        default_model = reasoning_models[0]
                else:
                    default_model = general_models[0] if general_models else available_models[0]
            elif agent_type == "coding":
                # For coding tasks, prefer coding-capable models
                coding_preferences = ["deepseek-coder", "qwen3", "deepseek-r1"]
                for pref in coding_preferences:
                    matching_models = [m for m in available_models if pref in m.lower()]
                    if matching_models:
                        default_model = matching_models[0]
                        break
                if not default_model:
                    default_model = available_models[0]
            else:
                # For general tasks, use any available model
                if reasoning_models:
                    default_model = reasoning_models[0]
                elif general_models:
                    default_model = general_models[0]
                else:
                    default_model = available_models[0]

        if not default_model:
            raise ValueError("No Ollama models available")

        logger.info(f"Using model {default_model} for {agent_type} agent")
        
        # Emit model confirmation
        await emit_agent_thinking(
            user_id, 
            agent_type, 
            f"Selected {default_model} model for {agent_type} tasks", 
            confidence=0.9,
            step="model_configured"
        )

        # Determine if we should use Tree of Thought reasoning
        use_tot = (
            agent_type == "reasoning" or 
            any(keyword in content.lower() for keyword in ["solve", "calculate", "reason", "think", "analyze"]) or
            any(op in content for op in ["*", "+", "-", "/", "^", "=", "?"])
        )
        
        if use_tot and any(model in default_model for model in ["deepseek-r1", "qwen3"]):
            logger.info(f"Activating Tree of Thought reasoning for query: {content[:50]}...")
            
            # Emit ToT activation
            await emit_agent_thinking(
                user_id, 
                agent_type, 
                "Activating Tree of Thought multi-path reasoning...", 
                confidence=0.95,
                step="tot_activation"
            )
            
            # Use ToT enhanced response
            try:
                response = await get_tot_enhanced_response(
                    content, default_model, user_id, agent_type
                )
                if response and response.get("content"):
                    return response
                else:
                    logger.warning("ToT reasoning failed, falling back to standard approach")
            except Exception as e:
                logger.error(f"ToT reasoning error: {e}")
                await emit_agent_thinking(
                    user_id, 
                    agent_type, 
                    "ToT reasoning encountered an issue, using standard approach...", 
                    confidence=0.7,
                    step="tot_fallback"
                )

        # Create agent config
        agent_config = AgentConfig(
            agent_type=agent_type,
            name=f"{agent_type}_agent_{user_id}",
            custom_config={
                "model_name": default_model,
                "ollama_url": "http://localhost:11434",
                "max_tokens": 500,
                "temperature": 0.7
            }
        )

        logger.info(f"Creating agent: {agent_config.name}")
        
        # Emit agent creation step
        await emit_agent_thinking(user_id, agent_type, f"Creating {agent_type} agent instance...", step="agent_creation")

        # Create or get agent
        agent = await agent_factory.create_agent(
            agent_type=agent_type,
            name=agent_config.name,
            custom_config=agent_config.custom_config
        )

        logger.info(f"Agent created successfully: {agent.agent_id}")
        
        # Emit processing start
        await emit_agent_thinking(user_id, agent_type, "Processing your request with AI model...", step="llm_processing")

        # Create message for agent
        agent_message = AgentMessage(
            type=MessageType.REQUEST,
            sender=user_id,
            recipient=agent.agent_id,
            content={"content": content}
        )

        # DEBUG: Log what we're sending to the agent
        logger.info(f"BACKEND: Creating AgentMessage with content: {agent_message.content}")
        logger.info(f"BACKEND: Agent ID: {agent.agent_id}, Agent Type: {agent_type}")

        # Process message with agent
        logger.info(f"BACKEND: Processing message with agent: {agent.agent_id}")
        response = await agent.process_message(agent_message)
        
        # DEBUG: Log what comes back from the agent
        logger.info(f"BACKEND: Agent response received: type={type(response)}, content_type={type(response.content)}")
        if hasattr(response, 'content'):
            logger.info(f"BACKEND: Response content preview: {str(response.content)[:200]}...")
        
        logger.info(f"Agent response received: type={type(response)}, content_type={type(response.content)}")
        
        # Emit processing completion
        await emit_agent_thinking(
            user_id, 
            agent_type, 
            "AI processing complete, formatting response...", 
            confidence=0.95,
            step="response_formatting"
        )

        if hasattr(response, 'content') and response.content:
            logger.info(f"Response content keys: {list(response.content.keys()) if isinstance(response.content, dict) else 'not_dict'}")
            if isinstance(response.content, dict) and 'solution' in response.content:
                logger.info(f"Solution found: {response.content['solution'][:100]}...")

        # Extract response content
        if isinstance(response.content, dict):
            # Handle different agent response formats
            if "solution" in response.content:
                # Reasoning agent format
                return {
                    "response": response.content.get("solution", "No solution found"),
                    "agent": agent_type,
                    "confidence": response.content.get("confidence", 0.8),
                    "metrics": response.content.get("metrics", {"total_time": 0.5}),
                    "reasoning_path": response.content.get("reasoning_path", []),
                    "alternatives": response.content.get("alternatives", [])
                }
            elif "response" in response.content:
                # Standard response format
                return response.content
            elif "message" in response.content:
                # General agent format
                return {
                    "response": response.content.get("message", "No response"),
                    "agent": agent_type,
                    "confidence": response.content.get("confidence", 0.8),
                    "metrics": response.content.get("metrics", {"total_time": 0.5})
                }
            else:
                # Fallback for unknown format
                return {
                    "response": str(response.content),
                    "agent": agent_type,
                    "confidence": 0.8,
                    "metrics": {"total_time": 0.5}
                }
        else:
            return {
                "response": str(response.content),
                "agent": agent_type,
                "confidence": 0.8,
                "metrics": {"total_time": 0.5}
            }

    except Exception as e:
        logger.error(f"Error getting agent response: {e}")
        return {
            "response": f"I apologize, but I'm currently experiencing technical difficulties. Error: {str(e)}",
            "agent": agent_type,
            "confidence": 0.1,
            "metrics": {"total_time": 0.0},
            "error": True
        }


async def create_task_from_response(ai_response: Dict[str, Any], agent_type: str, original_content: str, user_id: str) -> Optional[str]:
    """Create a task if the AI response suggests actionable work"""
    
    try:
        from ...orchestration.coordination_models import TaskRequest, TaskPriority
        from ...api.main import app_state
        
        response_text = ai_response.get('response', '').lower()
        
        # Keywords that suggest actionable tasks
        task_indicators = [
            'write', 'create', 'build', 'implement', 'develop', 'code', 'program',
            'design', 'generate', 'make', 'construct', 'setup', 'configure'
        ]
        
        # Check if response suggests creating something
        suggests_task = any(indicator in response_text for indicator in task_indicators)
        
        if suggests_task and agent_type == 'coding':
            # Get task dispatcher from app state
            task_dispatcher = app_state.get("task_dispatcher")
            
            if task_dispatcher:
                # Create task request
                task = TaskRequest(
                    task_type="coding",
                    description=f"User request: {original_content}",
                    priority=TaskPriority.NORMAL,
                    input_data={
                        "user_request": original_content,
                        "ai_suggestion": ai_response.get('response', ''),
                        "agent_type": agent_type,
                        "user_id": user_id
                    },
                    required_capabilities={"code_generation", "file_creation"},
                    metadata={
                        "source": "chat_interaction",
                        "created_by": user_id,
                        "ai_agent": agent_type
                    }
                )
                
                # Submit task
                success = await task_dispatcher.submit_task(task)
                
                if success:
                    logger.info(f"Created task {task.task_id} from chat response for user {user_id}")
                    return task.task_id
                else:
                    logger.error(f"Failed to submit task for user {user_id}")
            else:
                logger.warning("Task dispatcher not available in app state")
        
        return None
        
    except Exception as e:
        logger.error(f"Error creating task from response: {e}")
        return None


async def emit_agent_activity(user_id: str, activity_type: str, agent_name: str, content: str, metadata: dict = None):
    """Emit agent activity events for real-time logging"""
    try:
        activity_event = {
            'type': 'agent_activity',
            'data': {
                'id': f"activity_{datetime.now().timestamp()}_{activity_type}",
                'timestamp': datetime.now().isoformat(),
                'activity_type': activity_type,  # 'thinking', 'processing', 'response', 'task_created', 'error'
                'agent': agent_name,
                'content': content,
                'metadata': metadata or {}
            }
        }
        
        await manager.send_personal_message(activity_event, user_id)
        logger.info(f"Agent activity emitted: {activity_type} from {agent_name} to {user_id}")
        
    except Exception as e:
        logger.error(f"Failed to emit agent activity: {e}")


async def emit_agent_thinking(user_id: str, agent_name: str, thought: str, confidence: float = 0.8, step: str = None):
    """Emit agent thinking/reasoning events"""
    await emit_agent_activity(
        user_id=user_id,
        activity_type='thinking',
        agent_name=agent_name,
        content=thought,
        metadata={
            'confidence': confidence,
            'reasoning_step': step,
            'timestamp': datetime.now().isoformat()
        }
    )


async def emit_agent_status(user_id: str, agent_name: str, status: str, details: dict = None):
    """Emit agent status updates"""
    await emit_agent_activity(
        user_id=user_id,
        activity_type='status',
        agent_name=agent_name,
        content=f"Agent {status}",
        metadata={
            'status': status,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
    )


# Global reasoning sessions storage
reasoning_sessions: Dict[str, Dict[str, Any]] = {}


async def handle_start_reasoning(data: dict, user_id: str):
    """Handle start reasoning WebSocket message - connect UI to real ToT engine"""
    
    problem = data.get('problem', '')
    mode = data.get('mode', 'adaptive')
    
    logger.info(f"Starting ToT reasoning for user {user_id}: problem='{problem[:100]}...', mode={mode}")
    
    # Create session ID
    session_id = f"reasoning_{user_id}_{datetime.now().timestamp()}"
    
    try:
        # Emit reasoning start status
        await emit_agent_status(user_id, "reasoning_engine", "initializing", {
            "session_id": session_id,
            "problem": problem[:200],
            "mode": mode
        })
        
        # Create ToT configuration based on mode
        tot_config = ToTConfig(
            generation_strategy=GenerationStrategy.PROPOSE if mode != 'creative' else GenerationStrategy.SAMPLE,
            evaluation_method=EvaluationMethod.VALUE,
            search_method=SearchMethod.BFS if mode == 'thorough' else SearchMethod.DFS,
            n_generate_sample=4 if mode == 'thorough' else 3,
            n_evaluate_sample=3 if mode == 'thorough' else 2,
            n_select_sample=2,
            max_depth=6 if mode == 'thorough' else 4,
            temperature=0.9 if mode == 'creative' else 0.7,
            model_name="llama3.2:3b"  # Default model
        )
        
        # Create reasoning agent
        agent_config = AgentConfig(
            name="reasoning_agent",
            type="reasoning",
            custom_config={
                "model_name": "llama3.2:3b",
                "ollama_url": "http://localhost:11434"
            }
        )
        
        reasoning_agent = ReasoningAgent(agent_config)
        
        # Store session
        reasoning_sessions[user_id] = {
            "session_id": session_id,
            "agent": reasoning_agent,
            "config": tot_config,
            "problem": problem,
            "mode": mode,
            "status": "running",
            "start_time": datetime.now().isoformat()
        }
        
        # Send reasoning started response
        await manager.send_personal_message({
            'type': 'reasoning_started',
            'data': {
                'session_id': session_id,
                'problem': problem,
                'mode': mode,
                'config': {
                    'generation_strategy': tot_config.generation_strategy.value,
                    'search_method': tot_config.search_method.value,
                    'max_depth': tot_config.max_depth,
                    'n_generate': tot_config.n_generate_sample
                }
            }
        }, user_id)
        
        # Start reasoning in background
        asyncio.create_task(run_reasoning_session(user_id, session_id, problem))
        
        logger.info(f"ToT reasoning session {session_id} started successfully")
        
    except Exception as e:
        logger.error(f"Error starting reasoning session: {e}")
        await manager.send_personal_message({
            'type': 'reasoning_error',
            'data': {
                'error': str(e),
                'message': 'Failed to start reasoning session'
            }
        }, user_id)


async def handle_stop_reasoning(data: dict, user_id: str):
    """Handle stop reasoning WebSocket message"""
    
    logger.info(f"Stopping reasoning session for user {user_id}")
    
    try:
        if user_id in reasoning_sessions:
            session = reasoning_sessions[user_id]
            session['status'] = 'stopped'
            
            await emit_agent_status(user_id, "reasoning_engine", "stopped", {
                "session_id": session['session_id'],
                "reason": "user_requested"
            })
            
            # Send stopped confirmation
            await manager.send_personal_message({
                'type': 'reasoning_stopped',
                'data': {
                    'session_id': session['session_id'],
                    'reason': 'user_requested'
                }
            }, user_id)
            
            # Clean up session
            del reasoning_sessions[user_id]
            
            logger.info(f"Reasoning session stopped for user {user_id}")
        else:
            await manager.send_personal_message({
                'type': 'reasoning_error',
                'data': {
                    'error': 'No active reasoning session',
                    'message': 'No reasoning session to stop'
                }
            }, user_id)
            
    except Exception as e:
        logger.error(f"Error stopping reasoning session: {e}")
        await manager.send_personal_message({
            'type': 'reasoning_error',
            'data': {
                'error': str(e),
                'message': 'Failed to stop reasoning session'
            }
        }, user_id)


async def run_reasoning_session(user_id: str, session_id: str, problem: str):
    """Run the actual ToT reasoning session in background"""
    
    logger.info(f"Running ToT reasoning session {session_id} for problem: {problem[:100]}...")
    
    try:
        session = reasoning_sessions.get(user_id)
        if not session or session['status'] != 'running':
            logger.info(f"Session {session_id} was cancelled or not found")
            return
            
        reasoning_agent = session['agent']
        
        # Emit thinking start
        await emit_agent_thinking(
            user_id, 
            "reasoning_engine", 
            "Starting Tree of Thought reasoning process...",
            confidence=1.0,
            step="initialization"
        )
        
        # Call the reasoning agent
        await emit_agent_status(user_id, "reasoning_engine", "processing", {
            "step": "generating_thoughts",
            "session_id": session_id
        })
        
        # Process the problem with ToT reasoning
        response = await reasoning_agent.process_message({
            'content': problem,
            'type': 'reasoning_request',
            'metadata': {
                'session_id': session_id,
                'user_id': user_id
            }
        })
        
        # Check if session was stopped
        if user_id not in reasoning_sessions or reasoning_sessions[user_id]['status'] != 'running':
            logger.info(f"Session {session_id} was stopped during processing")
            return
        
        # Emit final thinking
        await emit_agent_thinking(
            user_id,
            "reasoning_engine",
            "Tree of Thought reasoning completed",
            confidence=response.get('confidence', 0.8),
            step="completion"
        )
        
        # Send final result
        await manager.send_personal_message({
            'type': 'reasoning_result',
            'data': {
                'session_id': session_id,
                'result': response.get('response', 'No result generated'),
                'confidence': response.get('confidence', 0.8),
                'metadata': response.get('metadata', {}),
                'reasoning_tree': response.get('reasoning_tree', [])  # ToT tree structure
            }
        }, user_id)
        
        # Update session status
        if user_id in reasoning_sessions:
            reasoning_sessions[user_id]['status'] = 'completed'
        
        logger.info(f"ToT reasoning session {session_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in reasoning session {session_id}: {e}")
        
        # Emit error
        await emit_agent_activity(
            user_id=user_id,
            activity_type='error',
            agent_name='reasoning_engine',
            content=f"Reasoning session failed: {str(e)}",
            metadata={
                'session_id': session_id,
                'error_type': type(e).__name__
            }
        )
        
        # Send error to client
        await manager.send_personal_message({
            'type': 'reasoning_error',
            'data': {
                'session_id': session_id,
                'error': str(e),
                'message': 'Reasoning session encountered an error'
            }
        }, user_id)
        
        # Clean up session
        if user_id in reasoning_sessions:
            reasoning_sessions[user_id]['status'] = 'error'


async def get_tot_enhanced_response(content: str, model_name: str, user_id: str, agent_type: str) -> Dict[str, Any]:
    """
    Generate response using Tree of Thought reasoning for complex queries
    """
    try:
        from ...ai.reasoning.tot.tot_engine import ToTEngine
        from ...ai.reasoning.tot.models import ToTConfig, GenerationStrategy, EvaluationMethod, SearchMethod
        from ...ai.reasoning.tot.thought_generator import OllamaBackend
        
        # Configure ToT for the specific task
        tot_config = ToTConfig(
            model_name=model_name,
            generation_strategy=GenerationStrategy.PROPOSE,
            evaluation_method=EvaluationMethod.VALUE,
            search_method=SearchMethod.BFS,
            n_generate_sample=3,
            n_evaluate_sample=2,
            n_select_sample=2,
            max_depth=4,
            temperature=0.7,
            max_tokens=800,
            task_description=f"Solve or analyze: {content}",
            success_criteria="Provide accurate, well-reasoned solution"
        )
        
        # Create Ollama backend
        ollama_backend = OllamaBackend(model_name=model_name)
        
        # Create ToT engine
        tot_engine = ToTEngine(tot_config, ollama_backend, enable_vector_search=False)
        
        # Emit thinking step
        await emit_agent_thinking(
            user_id, 
            agent_type, 
            "Generating multiple reasoning paths...", 
            confidence=0.8,
            step="tot_generation"
        )
        
        # Run ToT reasoning
        search_results = await tot_engine.solve(content, max_iterations=10)
        
        if search_results and search_results.solutions:
            best_solution = search_results.solutions[0]
            
            # Emit final thinking step
            await emit_agent_thinking(
                user_id, 
                agent_type, 
                f"Found solution with confidence {best_solution.confidence:.2f}", 
                confidence=best_solution.confidence,
                step="tot_solution"
            )
            
            return {
                "content": best_solution.content,
                "confidence": best_solution.confidence,
                "reasoning_paths": len([s for s in search_results.solutions if s.confidence > 0.5]),
                "metadata": {
                    "reasoning_method": "tree_of_thought",
                    "model_used": model_name,
                    "search_depth": best_solution.depth,
                    "total_thoughts": len(search_results.visited_states)
                }
            }
        else:
            logger.warning("ToT reasoning produced no valid solutions")
            return None
            
    except Exception as e:
        logger.error(f"ToT enhanced response error: {e}")
        return None


async def extract_and_emit_reasoning(content: str, user_id: str, agent_type: str):
    """Extract <think> content and emit reasoning updates for the reasoning page"""
    
    # Extract content between <think> and </think> tags
    think_pattern = r'<think>(.*?)</think>'
    matches = re.findall(think_pattern, content, re.DOTALL)
    
    if not matches:
        return content  # No reasoning content found, return original
    
    # Process the first (and usually only) thinking block
    reasoning_content = matches[0].strip()
    
    # Split reasoning into logical steps (by paragraphs or sentences)
    reasoning_steps = []
    
    # Split by double newlines (paragraphs) first
    paragraphs = reasoning_content.split('\n\n')
    
    for i, paragraph in enumerate(paragraphs):
        if paragraph.strip():
            # Further split long paragraphs by sentences
            sentences = paragraph.split('. ')
            if len(sentences) > 3:  # If paragraph is too long, split by sentences
                for j, sentence in enumerate(sentences):
                    if sentence.strip():
                        reasoning_steps.append({
                            'content': sentence.strip() + ('.' if not sentence.endswith('.') else ''),
                            'step_type': 'analysis',
                            'confidence': 0.7 + (j * 0.05),  # Gradually increase confidence
                            'timestamp': datetime.now().isoformat()
                        })
            else:
                reasoning_steps.append({
                    'content': paragraph.strip(),
                    'step_type': 'reasoning',
                    'confidence': 0.8 + (i * 0.02),
                    'timestamp': datetime.now().isoformat()
                })
    
    # Emit reasoning updates for each step
    for i, step in enumerate(reasoning_steps):
        await manager.send_personal_message({
            'type': 'reasoning_update',
            'data': {
                'step': i + 1,
                'total_steps': len(reasoning_steps),
                'content': step['content'],
                'confidence': step['confidence'],
                'step_type': step['step_type'],
                'timestamp': step['timestamp'],
                'agent': agent_type
            }
        }, user_id)
        
        # Small delay between steps for visual effect
        await asyncio.sleep(0.1)
    
    # Send reasoning complete event
    await manager.send_personal_message({
        'type': 'reasoning_complete',
        'data': {
            'total_steps': len(reasoning_steps),
            'processing_time': len(reasoning_steps) * 0.1,
            'paths_explored': len(reasoning_steps),
            'agent': agent_type,
            'timestamp': datetime.now().isoformat()
        }
    }, user_id)
    
    # Return content with <think> tags removed
    cleaned_content = re.sub(think_pattern, '', content, flags=re.DOTALL).strip()
    return cleaned_content
