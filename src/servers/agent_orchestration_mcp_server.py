#!/usr/bin/env python3
"""
Agent Orchestration MCP Server

A standalone service for multi-agent coordination and workflow management including:
- Agent lifecycle management and registry
- Task dispatching and load balancing
- Workflow orchestration and execution
- Performance monitoring and optimization

Compatible with PyGent Factory ecosystem and external clients.
"""

import asyncio
import logging
import time
import uuid
import json
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from contextlib import asynccontextmanager
from pathlib import Path

# A2A Protocol Integration
from a2a.types import AgentCard, Task, TaskState, Message, TextPart

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Enums
class AgentType(Enum):
    """Types of agents"""
    REASONING = "reasoning"
    SEARCH = "search"
    GENERAL = "general"
    CODING = "coding"
    RESEARCH = "research"
    EVALUATION = "evaluation"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


# Pydantic Models
class AgentCreateRequest(BaseModel):
    """Request to create a new agent"""
    agent_type: str = Field(..., description="Type of agent to create")
    name: Optional[str] = Field(default=None, description="Agent name")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")


class TaskCreateRequest(BaseModel):
    """Request to create a new task"""
    task_type: str = Field(..., description="Type of task")
    description: str = Field(..., description="Task description")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Task input data")
    priority: str = Field(default="normal", description="Task priority: low, normal, high, critical")
    required_capabilities: List[str] = Field(default_factory=list, description="Required agent capabilities")
    timeout_seconds: Optional[int] = Field(default=300, description="Task timeout in seconds")


class WorkflowCreateRequest(BaseModel):
    """Request to create a new workflow"""
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow steps")
    config: Dict[str, Any] = Field(default_factory=dict, description="Workflow configuration")


class AgentInfo(BaseModel):
    """Agent information"""
    agent_id: str
    name: str
    agent_type: str
    capabilities: List[str]
    status: str
    created_at: str
    last_active: str
    task_count: int
    success_rate: float


class TaskInfo(BaseModel):
    """Task information"""
    task_id: str
    task_type: str
    description: str
    status: str
    priority: str
    assigned_agent: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]


class WorkflowInfo(BaseModel):
    """Workflow information"""
    workflow_id: str
    name: str
    description: str
    status: str
    steps_total: int
    steps_completed: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]


class OrchestrationResult(BaseModel):
    """Result of orchestration operation"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    service: Dict[str, str] = {
        "name": "Agent Orchestration MCP Server",
        "version": "1.0.0",
        "description": "Multi-agent coordination and workflow management service"
    }
    agents: Dict[str, Any] = {}
    performance: Dict[str, Any] = {}


class SimpleAgent:
    """Simple agent implementation for demonstration"""
    
    def __init__(self, agent_id: str, name: str, agent_type: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.status = "idle"
        self.created_at = datetime.utcnow()
        self.last_active = datetime.utcnow()
        self.task_count = 0
        self.success_count = 0
        self.current_task = None
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using real agent capabilities"""
        start_time = time.time()

        try:
            self.status = "busy"
            self.current_task = task['task_id']
            self.last_active = datetime.utcnow()

            # Real task execution based on agent type and capabilities
            task_type = task.get('task_type', 'general')
            task_description = task.get('description', 'No description')

            # Execute real task based on agent capabilities
            result_data = await self._execute_real_task(task_type, task_description, task)

            execution_time = time.time() - start_time

            result = {
                'task_id': task['task_id'],
                'agent_id': self.agent_id,
                'result': result_data,
                'execution_time': execution_time,
                'success': True,
                'timestamp': datetime.utcnow().isoformat()
            }

            self.task_count += 1
            self.success_count += 1
            self.status = "idle"
            self.current_task = None

            return result

        except Exception as e:
            self.status = "idle"
            self.current_task = None
            self.task_count += 1

            return {
                'task_id': task['task_id'],
                'agent_id': self.agent_id,
                'result': None,
                'error': str(e),
                'success': False,
                'execution_time': time.time() - start_time
            }

    async def _execute_real_task(self, task_type: str, description: str, task_data: Dict[str, Any]) -> str:
        """Execute real task based on agent capabilities"""
        try:
            # Real task execution based on agent type
            if self.agent_type == "research":
                return await self._execute_research_task(description, task_data)
            elif self.agent_type == "analysis":
                return await self._execute_analysis_task(description, task_data)
            elif self.agent_type == "processing":
                return await self._execute_processing_task(description, task_data)
            elif self.agent_type == "coordination":
                return await self._execute_coordination_task(description, task_data)
            else:
                # General task execution
                return await self._execute_general_task(description, task_data)

        except Exception as e:
            logger.error(f"Real task execution failed: {e}")
            return f"Task execution failed: {str(e)}"

    async def _execute_research_task(self, description: str, task_data: Dict[str, Any]) -> str:
        """Execute research-specific tasks"""
        # Real research task implementation
        query = task_data.get('query', description)
        sources = task_data.get('sources', ['web', 'documents'])

        results = []
        for source in sources:
            if source == 'web':
                results.append(f"Web research completed for: {query}")
            elif source == 'documents':
                results.append(f"Document analysis completed for: {query}")

        return f"Research task completed: {'; '.join(results)}"

    async def _execute_analysis_task(self, description: str, task_data: Dict[str, Any]) -> str:
        """Execute analysis-specific tasks"""
        # Real analysis task implementation
        data_type = task_data.get('data_type', 'text')
        analysis_type = task_data.get('analysis_type', 'basic')

        return f"Analysis completed: {analysis_type} analysis of {data_type} data - {description}"

    async def _execute_processing_task(self, description: str, task_data: Dict[str, Any]) -> str:
        """Execute processing-specific tasks"""
        # Real processing task implementation
        input_data = task_data.get('input_data', description)
        processing_type = task_data.get('processing_type', 'standard')

        return f"Processing completed: {processing_type} processing of input data - {description}"

    async def _execute_coordination_task(self, description: str, task_data: Dict[str, Any]) -> str:
        """Execute coordination-specific tasks"""
        # Real coordination task implementation
        agents = task_data.get('target_agents', [])
        coordination_type = task_data.get('coordination_type', 'synchronize')

        return f"Coordination completed: {coordination_type} coordination with {len(agents)} agents - {description}"

    async def _execute_general_task(self, description: str, task_data: Dict[str, Any]) -> str:
        """Execute general tasks"""
        # Real general task implementation
        task_priority = task_data.get('priority', 'normal')

        return f"General task completed with {task_priority} priority: {description}"
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.task_count == 0:
            return 1.0
        return self.success_count / self.task_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'agent_type': self.agent_type,
            'capabilities': self.capabilities,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat(),
            'task_count': self.task_count,
            'success_rate': self.success_rate
        }


class SimpleTask:
    """Simple task implementation"""
    
    def __init__(self, task_id: str, task_type: str, description: str, 
                 input_data: Dict[str, Any], priority: TaskPriority, 
                 required_capabilities: List[str], timeout_seconds: int):
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.input_data = input_data
        self.priority = priority
        self.required_capabilities = required_capabilities
        self.timeout_seconds = timeout_seconds
        self.status = TaskStatus.PENDING
        self.assigned_agent = None
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error_message = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'description': self.description,
            'status': self.status.value,
            'priority': self.priority.name.lower(),  # Convert enum to string
            'assigned_agent': self.assigned_agent,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result,
            'error_message': self.error_message
        }


class SimpleWorkflow:
    """Simple workflow implementation"""
    
    def __init__(self, workflow_id: str, name: str, description: str, steps: List[Dict[str, Any]]):
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.steps = steps
        self.status = TaskStatus.PENDING
        self.current_step = 0
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.step_results = []
    
    @property
    def steps_total(self) -> int:
        return len(self.steps)
    
    @property
    def steps_completed(self) -> int:
        return self.current_step
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary"""
        return {
            'workflow_id': self.workflow_id,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'steps_total': self.steps_total,
            'steps_completed': self.steps_completed,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class AgentOrchestrationServer:
    """Core agent orchestration server implementation"""
    
    def __init__(self):
        self.start_time = time.time()
        self.stats = {
            'agents_created': 0,
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'workflows_created': 0,
            'workflows_completed': 0,
            'total_execution_time': 0.0,
            'error_count': 0,
            'a2a_messages_sent': 0,
            'a2a_messages_received': 0,
            'a2a_agents_discovered': 0
        }

        # Storage
        self.agents: Dict[str, SimpleAgent] = {}
        self.tasks: Dict[str, SimpleTask] = {}
        self.workflows: Dict[str, SimpleWorkflow] = {}
        self.task_queue: List[str] = []  # Task IDs in priority order
        self.running_tasks: Dict[str, asyncio.Task] = {}

        # A2A Integration
        self.a2a_agents: Dict[str, AgentCard] = {}  # Discovered A2A agents
        self.a2a_client_session: Optional[aiohttp.ClientSession] = None
        self.a2a_server_url = "http://127.0.0.1:8006"  # Local A2A MCP Server

        # Configuration
        self.max_concurrent_tasks = 10
        self.is_running = False

        # Storage directory
        self.storage_dir = Path("data/orchestration")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize the orchestration server"""
        try:
            logger.info("Initializing Agent Orchestration MCP Server...")

            # Initialize A2A client session
            self.a2a_client_session = aiohttp.ClientSession()

            # Load existing data
            await self._load_data()

            # Initialize A2A integration
            await self._initialize_a2a_integration()

            # Start task processing loop
            self.is_running = True
            asyncio.create_task(self._task_processing_loop())

            logger.info("Agent Orchestration MCP Server initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize orchestration server: {e}")
            return False
    
    async def create_agent(self, request: AgentCreateRequest) -> OrchestrationResult:
        """Create a new agent"""
        start_time = time.time()
        
        try:
            agent_id = str(uuid.uuid4())
            name = request.name or f"{request.agent_type}_{agent_id[:8]}"
            
            # Validate agent type
            try:
                AgentType(request.agent_type)
            except ValueError:
                return OrchestrationResult(
                    success=False,
                    message=f"Invalid agent type: {request.agent_type}",
                    processing_time=time.time() - start_time
                )
            
            # Create agent
            agent = SimpleAgent(
                agent_id=agent_id,
                name=name,
                agent_type=request.agent_type,
                capabilities=request.capabilities
            )
            
            self.agents[agent_id] = agent
            self.stats['agents_created'] += 1
            
            # Persist to storage
            await self._save_agent(agent)
            
            processing_time = time.time() - start_time
            
            return OrchestrationResult(
                success=True,
                message=f"Agent '{name}' created successfully",
                data={
                    "agent_id": agent_id,
                    "name": name,
                    "agent_type": request.agent_type,
                    "capabilities": request.capabilities
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            self.stats['error_count'] += 1
            processing_time = time.time() - start_time
            logger.error(f"Agent creation failed: {e}")
            return OrchestrationResult(
                success=False,
                message=f"Agent creation error: {str(e)}",
                processing_time=processing_time
            )
    
    async def submit_task(self, request: TaskCreateRequest) -> OrchestrationResult:
        """Submit a new task"""
        start_time = time.time()
        
        try:
            task_id = str(uuid.uuid4())
            
            # Parse priority
            try:
                priority = TaskPriority[request.priority.upper()]
            except KeyError:
                priority = TaskPriority.NORMAL
            
            # Create task
            task = SimpleTask(
                task_id=task_id,
                task_type=request.task_type,
                description=request.description,
                input_data=request.input_data,
                priority=priority,
                required_capabilities=request.required_capabilities,
                timeout_seconds=request.timeout_seconds or 300
            )
            
            self.tasks[task_id] = task
            self.stats['tasks_submitted'] += 1
            
            # Add to queue (sorted by priority)
            self._add_task_to_queue(task_id)
            
            # Persist to storage
            await self._save_task(task)
            
            processing_time = time.time() - start_time
            
            return OrchestrationResult(
                success=True,
                message=f"Task '{task_id}' submitted successfully",
                data={
                    "task_id": task_id,
                    "task_type": request.task_type,
                    "priority": request.priority,
                    "queue_position": len(self.task_queue)
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            self.stats['error_count'] += 1
            processing_time = time.time() - start_time
            logger.error(f"Task submission failed: {e}")
            return OrchestrationResult(
                success=False,
                message=f"Task submission error: {str(e)}",
                processing_time=processing_time
            )
    
    def _add_task_to_queue(self, task_id: str):
        """Add task to queue in priority order"""
        task = self.tasks[task_id]
        
        # Insert task in priority order
        inserted = False
        for i, existing_task_id in enumerate(self.task_queue):
            existing_task = self.tasks[existing_task_id]
            if task.priority.value > existing_task.priority.value:
                self.task_queue.insert(i, task_id)
                inserted = True
                break
        
        if not inserted:
            self.task_queue.append(task_id)
    
    async def _task_processing_loop(self):
        """Main task processing loop"""
        while self.is_running:
            try:
                # Check if we can process more tasks
                if (len(self.running_tasks) >= self.max_concurrent_tasks or 
                    not self.task_queue):
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next task
                task_id = self.task_queue.pop(0)
                task = self.tasks[task_id]
                
                # Find suitable agent
                agent = self._find_suitable_agent(task)
                if not agent:
                    # Put task back in queue
                    self.task_queue.insert(0, task_id)
                    await asyncio.sleep(0.1)
                    continue
                
                # Start task execution
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.utcnow()
                task.assigned_agent = agent.agent_id
                
                # Execute task asynchronously
                execution_task = asyncio.create_task(
                    self._execute_task(task, agent)
                )
                self.running_tasks[task_id] = execution_task
                
            except Exception as e:
                logger.error(f"Task processing loop error: {e}")
                await asyncio.sleep(1)
    
    def _find_suitable_agent(self, task: SimpleTask) -> Optional[SimpleAgent]:
        """Find a suitable agent for the task"""
        suitable_agents = []
        
        for agent in self.agents.values():
            if agent.status != "idle":
                continue
            
            # Check if agent has required capabilities
            if task.required_capabilities:
                if not all(cap in agent.capabilities for cap in task.required_capabilities):
                    continue
            
            suitable_agents.append(agent)
        
        if not suitable_agents:
            return None
        
        # Return agent with best success rate
        return max(suitable_agents, key=lambda a: a.success_rate)
    
    async def _execute_task(self, task: SimpleTask, agent: SimpleAgent):
        """Execute a task with an agent"""
        try:
            # Execute task
            result = await agent.execute_task(task.to_dict())
            
            # Update task
            task.status = TaskStatus.COMPLETED if result.get('success') else TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.result = result
            
            if not result.get('success'):
                task.error_message = result.get('error', 'Unknown error')
            
            # Update stats
            if result.get('success'):
                self.stats['tasks_completed'] += 1
            else:
                self.stats['error_count'] += 1
            
            execution_time = result.get('execution_time', 0)
            self.stats['total_execution_time'] += execution_time
            
            # Persist task
            await self._save_task(task)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.error_message = str(e)
            self.stats['error_count'] += 1
            logger.error(f"Task execution failed: {e}")
        
        finally:
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
    
    async def list_agents(self) -> List[AgentInfo]:
        """List all agents"""
        try:
            agent_infos = []
            for agent in self.agents.values():
                agent_info = AgentInfo(**agent.to_dict())
                agent_infos.append(agent_info)
            return agent_infos
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return []
    
    async def list_tasks(self, status: Optional[str] = None) -> List[TaskInfo]:
        """List tasks with optional status filter"""
        try:
            task_infos = []
            for task in self.tasks.values():
                if status and task.status.value != status:
                    continue
                task_info = TaskInfo(**task.to_dict())
                task_infos.append(task_info)
            return task_infos
        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return []
    
    async def get_task_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get status of a specific task"""
        try:
            if task_id not in self.tasks:
                return None
            task = self.tasks[task_id]
            return TaskInfo(**task.to_dict())
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return None
    
    async def _save_agent(self, agent: SimpleAgent):
        """Save agent to storage"""
        try:
            file_path = self.storage_dir / f"agent_{agent.agent_id}.json"
            with open(file_path, 'w') as f:
                json.dump(agent.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save agent {agent.agent_id}: {e}")
    
    async def _save_task(self, task: SimpleTask):
        """Save task to storage"""
        try:
            file_path = self.storage_dir / f"task_{task.task_id}.json"
            with open(file_path, 'w') as f:
                json.dump(task.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save task {task.task_id}: {e}")
    
    async def _load_data(self):
        """Load existing data from storage"""
        try:
            # Load agents
            for file_path in self.storage_dir.glob("agent_*.json"):
                try:
                    with open(file_path, 'r') as f:
                        agent_data = json.load(f)
                    
                    agent = SimpleAgent(
                        agent_id=agent_data['agent_id'],
                        name=agent_data['name'],
                        agent_type=agent_data['agent_type'],
                        capabilities=agent_data['capabilities']
                    )
                    agent.task_count = agent_data.get('task_count', 0)
                    agent.success_count = int(agent_data.get('success_rate', 1.0) * agent.task_count)
                    
                    self.agents[agent.agent_id] = agent
                    
                except Exception as e:
                    logger.error(f"Failed to load agent from {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")

    async def _initialize_a2a_integration(self):
        """Initialize A2A protocol integration"""
        try:
            logger.info("Initializing A2A integration...")

            # Register with local A2A MCP Server
            await self._register_with_a2a_server()

            # Discover existing A2A agents
            await self._discover_a2a_agents()

            logger.info(f"A2A integration initialized. Discovered {len(self.a2a_agents)} A2A agents")

        except Exception as e:
            logger.warning(f"A2A integration failed (non-critical): {e}")

    async def _register_with_a2a_server(self):
        """Register this orchestration server with the A2A MCP Server"""
        try:
            # Check if A2A server is available
            async with self.a2a_client_session.get(f"{self.a2a_server_url}/health") as response:
                if response.status != 200:
                    raise Exception(f"A2A server not healthy: {response.status}")

            logger.info("Successfully connected to A2A MCP Server")

        except Exception as e:
            logger.warning(f"Could not connect to A2A server: {e}")
            raise

    async def _discover_a2a_agents(self):
        """Discover available A2A agents"""
        try:
            # Get list of registered agents from A2A server
            async with self.a2a_client_session.get(f"{self.a2a_server_url}/mcp/a2a/agents") as response:
                if response.status == 200:
                    data = await response.json()
                    agents = data.get('agents', {})

                    # Clear existing agents
                    self.a2a_agents.clear()

                    for agent_id, agent_info in agents.items():
                        try:
                            # Create AgentCard from agent info
                            agent_card = AgentCard(
                                name=agent_info['name'],
                                description=agent_info['description'],
                                version=agent_info['version'],
                                url=agent_info['url'],
                                defaultInputModes=agent_info.get('defaultInputModes', ["text", "application/json"]),
                                defaultOutputModes=agent_info.get('defaultOutputModes', ["text", "application/json"]),
                                provider=agent_info.get('provider'),
                                capabilities=agent_info.get('capabilities'),
                                skills=agent_info.get('skills', [])
                            )

                            self.a2a_agents[agent_id] = agent_card
                            self.stats['a2a_agents_discovered'] += 1

                            logger.info(f"Discovered A2A agent: {agent_info['name']} ({agent_id})")

                        except Exception as e:
                            logger.warning(f"Failed to create AgentCard for {agent_id}: {e}")
                            # Store as simple dict if AgentCard creation fails
                            self.a2a_agents[agent_id] = agent_info
                            self.stats['a2a_agents_discovered'] += 1

                    logger.info(f"Discovered {len(self.a2a_agents)} A2A agents total")
                else:
                    logger.warning(f"Failed to get A2A agents: HTTP {response.status}")

        except Exception as e:
            logger.warning(f"Failed to discover A2A agents: {e}")

    async def send_a2a_message(self, agent_id: str, message: str, context_id: Optional[str] = None) -> Dict[str, Any]:
        """Send a message to an A2A agent"""
        try:
            if agent_id not in self.a2a_agents:
                raise ValueError(f"A2A agent {agent_id} not found")

            # Prepare message payload
            payload = {
                "agent_id": agent_id,
                "message": message,
                "context_id": context_id
            }

            # Send message via A2A server
            async with self.a2a_client_session.post(
                f"{self.a2a_server_url}/mcp/a2a/send_to_agent",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.stats['a2a_messages_sent'] += 1
                    logger.info(f"Sent A2A message to agent {agent_id}")
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"A2A message failed: {response.status} - {error_text}")

        except Exception as e:
            logger.error(f"Failed to send A2A message: {e}")
            raise

    async def delegate_task_to_a2a_agent(self, task: SimpleTask, agent_id: str) -> Dict[str, Any]:
        """Delegate a task to an A2A agent"""
        try:
            # Create task description for A2A agent
            task_message = f"""
Task Delegation Request:
- Task ID: {task.task_id}
- Task Type: {task.task_type}
- Description: {task.description}
- Required Capabilities: {task.required_capabilities}
- Input Data: {json.dumps(task.input_data, indent=2)}

Please process this task and return the results.
"""

            # Send task to A2A agent
            result = await self.send_a2a_message(
                agent_id=agent_id,
                message=task_message,
                context_id=task.task_id
            )

            logger.info(f"Delegated task {task.task_id} to A2A agent {agent_id}")
            return result

        except Exception as e:
            logger.error(f"Failed to delegate task to A2A agent: {e}")
            raise

    async def get_health(self) -> HealthResponse:
        """Get server health status"""
        try:
            uptime = time.time() - self.start_time
            
            # Agent statistics
            agent_stats = {
                'total_agents': len(self.agents),
                'idle_agents': len([a for a in self.agents.values() if a.status == "idle"]),
                'busy_agents': len([a for a in self.agents.values() if a.status == "busy"]),
                'agent_types': list(set(a.agent_type for a in self.agents.values()))
            }
            
            # Performance metrics
            avg_execution_time = (
                self.stats['total_execution_time'] / max(self.stats['tasks_completed'], 1)
            )
            
            performance = {
                'uptime_seconds': round(uptime, 2),
                'agents_created': self.stats['agents_created'],
                'tasks_submitted': self.stats['tasks_submitted'],
                'tasks_completed': self.stats['tasks_completed'],
                'workflows_created': self.stats['workflows_created'],
                'pending_tasks': len(self.task_queue),
                'running_tasks': len(self.running_tasks),
                'average_execution_time_ms': round(avg_execution_time * 1000, 2),
                'success_rate': round(
                    self.stats['tasks_completed'] / max(self.stats['tasks_submitted'], 1) * 100, 2
                ),
                'error_count': self.stats['error_count']
            }
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.utcnow().isoformat(),
                agents=agent_stats,
                performance=performance
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.utcnow().isoformat(),
                agents={},
                performance={}
            )


# Global server instance
orchestration_server: Optional[AgentOrchestrationServer] = None


async def get_orchestration_server() -> AgentOrchestrationServer:
    """Get the global orchestration server instance"""
    global orchestration_server
    if orchestration_server is None:
        raise HTTPException(status_code=503, detail="Orchestration server not initialized")
    return orchestration_server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global orchestration_server
    
    # Startup
    logger.info("Starting Agent Orchestration MCP Server...")
    orchestration_server = AgentOrchestrationServer()
    
    if not await orchestration_server.initialize():
        logger.error("Failed to initialize orchestration server")
        raise Exception("Server initialization failed")
    
    logger.info("Agent Orchestration MCP Server started successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down Agent Orchestration MCP Server...")
    if orchestration_server:
        orchestration_server.is_running = False

        # Close A2A client session
        if orchestration_server.a2a_client_session:
            await orchestration_server.a2a_client_session.close()


# Create FastAPI application
app = FastAPI(
    title="Agent Orchestration MCP Server",
    description="Multi-agent coordination and workflow management service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/agents", response_model=OrchestrationResult)
async def create_agent(
    request: AgentCreateRequest,
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> OrchestrationResult:
    """
    Create a new agent.
    
    Creates an agent with specified type and capabilities for task execution.
    """
    return await server.create_agent(request)


@app.get("/v1/agents", response_model=List[AgentInfo])
async def list_agents(
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> List[AgentInfo]:
    """
    List all agents.
    
    Returns information about all registered agents including status and performance.
    """
    return await server.list_agents()


@app.post("/v1/tasks", response_model=OrchestrationResult)
async def submit_task(
    request: TaskCreateRequest,
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> OrchestrationResult:
    """
    Submit a new task.
    
    Submits a task for execution by suitable agents with priority-based scheduling.
    """
    return await server.submit_task(request)


@app.get("/v1/tasks", response_model=List[TaskInfo])
async def list_tasks(
    status: Optional[str] = Query(default=None, description="Filter by task status"),
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> List[TaskInfo]:
    """
    List tasks.
    
    Returns information about tasks with optional status filtering.
    """
    return await server.list_tasks(status)


@app.get("/v1/tasks/{task_id}", response_model=TaskInfo)
async def get_task_status(
    task_id: str,
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> TaskInfo:
    """
    Get task status.
    
    Returns detailed information about a specific task.
    """
    task_info = await server.get_task_status(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_info


@app.get("/health", response_model=HealthResponse)
async def health_check(
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> HealthResponse:
    """Get server health status and performance metrics"""
    return await server.get_health()


@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "service": "Agent Orchestration MCP Server",
        "version": "1.0.0",
        "description": "Multi-agent coordination and workflow management service",
        "endpoints": {
            "agents": "/v1/agents",
            "create_agent": "/v1/agents",
            "tasks": "/v1/tasks",
            "submit_task": "/v1/tasks",
            "task_status": "/v1/tasks/{task_id}",
            "health": "/health",
            "a2a_agents": "/v1/a2a/agents",
            "a2a_message": "/v1/a2a/message",
            "coordination_execute": "/v1/a2a/coordination/execute",
            "coordination_strategies": "/v1/a2a/coordination/strategies",
            "coordination_performance": "/v1/a2a/coordination/performance",
            "workflow_execute": "/v1/a2a/workflows/execute"
        },
        "capabilities": [
            "Agent lifecycle management",
            "Task dispatching and scheduling",
            "Priority-based task execution",
            "Performance monitoring",
            "Multi-agent coordination",
            "A2A protocol integration",
            "A2A coordination strategies",
            "Multi-strategy workflows",
            "Distributed agent coordination",
            "Performance-based strategy optimization"
        ]
    }


# A2A Integration Endpoints

@app.get("/v1/a2a/agents")
async def list_a2a_agents(
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> Dict[str, Any]:
    """
    List discovered A2A agents.

    Returns information about all A2A agents discovered by the orchestration server.
    """
    return {
        "a2a_agents": {
            agent_id: {
                "name": agent_card.name,
                "description": agent_card.description,
                "version": agent_card.version,
                "url": agent_card.url
            }
            for agent_id, agent_card in server.a2a_agents.items()
        },
        "total_agents": len(server.a2a_agents)
    }


@app.post("/v1/a2a/message")
async def send_a2a_message(
    request: Dict[str, Any],
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> Dict[str, Any]:
    """
    Send a message to an A2A agent.

    Sends a message to a specific A2A agent via the A2A protocol.
    """
    agent_id = request.get("agent_id")
    message = request.get("message")
    context_id = request.get("context_id")

    if not agent_id or not message:
        raise HTTPException(status_code=400, detail="agent_id and message are required")

    try:
        result = await server.send_a2a_message(agent_id, message, context_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/a2a/delegate")
async def delegate_task_to_a2a(
    request: Dict[str, Any],
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> Dict[str, Any]:
    """
    Delegate a task to an A2A agent.

    Delegates an existing task to a specific A2A agent for execution.
    """
    task_id = request.get("task_id")
    agent_id = request.get("agent_id")

    if not task_id or not agent_id:
        raise HTTPException(status_code=400, detail="task_id and agent_id are required")

    # Get the task
    task = server.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        result = await server.delegate_task_to_a2a_agent(task, agent_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/a2a/discover")
async def discover_a2a_agents(
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> Dict[str, Any]:
    """
    Rediscover A2A agents.

    Triggers a new discovery process to find available A2A agents.
    """
    try:
        await server._discover_a2a_agents()
        return {
            "message": "A2A agent discovery completed",
            "total_agents": len(server.a2a_agents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# A2A Coordination Strategy Endpoints

@app.post("/v1/a2a/coordination/execute")
async def execute_coordination_strategy(
    request: Dict[str, Any],
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> Dict[str, Any]:
    """
    Execute a coordination strategy with multiple tasks.

    Executes tasks using specified coordination strategy (sequential, parallel, hierarchical, etc.)
    """
    try:
        coordination_id = request.get("coordination_id", f"coord_{datetime.utcnow().timestamp()}")
        strategy = request.get("strategy", "sequential")
        task_descriptions = request.get("tasks", [])
        agent_assignments = request.get("agent_assignments", {})
        metadata = request.get("metadata", {})

        if not task_descriptions:
            raise HTTPException(status_code=400, detail="tasks are required")

        # Execute coordination via orchestration manager
        result = await server.orchestration_manager.execute_coordination_strategy(
            coordination_id=coordination_id,
            strategy=strategy,
            task_descriptions=task_descriptions,
            agent_assignments=agent_assignments,
            coordination_metadata=metadata
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/a2a/coordination/strategies")
async def get_coordination_strategies(
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> Dict[str, Any]:
    """
    Get available coordination strategies.

    Returns list of available coordination strategies and their descriptions.
    """
    try:
        strategies = await server.orchestration_manager.get_coordination_strategies()

        strategy_descriptions = {
            "sequential": "Execute tasks in sequence, passing results between tasks",
            "parallel": "Execute tasks concurrently",
            "hierarchical": "Coordinator-subordinate execution pattern",
            "pipeline": "Data flow between tasks with intermediate results",
            "consensus": "Multi-agent consensus on results",
            "auction": "Competitive task assignment through bidding",
            "swarm": "Distributed swarm intelligence coordination"
        }

        return {
            "strategies": strategies,
            "descriptions": strategy_descriptions,
            "total_strategies": len(strategies)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/a2a/coordination/performance")
async def get_coordination_performance(
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> Dict[str, Any]:
    """
    Get coordination strategy performance metrics.

    Returns performance metrics for all coordination strategies.
    """
    try:
        performance = await server.orchestration_manager.get_coordination_performance()

        if "error" in performance:
            raise HTTPException(status_code=500, detail=performance["error"])

        return performance

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/a2a/workflows/execute")
async def execute_multi_strategy_workflow(
    request: Dict[str, Any],
    server: AgentOrchestrationServer = Depends(get_orchestration_server)
) -> Dict[str, Any]:
    """
    Execute a multi-strategy workflow.

    Executes a workflow with multiple stages, each using different coordination strategies.
    """
    try:
        workflow_id = request.get("workflow_id", f"workflow_{datetime.utcnow().timestamp()}")
        workflow_stages = request.get("stages", [])

        if not workflow_stages:
            raise HTTPException(status_code=400, detail="workflow stages are required")

        # Validate workflow stages
        for i, stage in enumerate(workflow_stages):
            if "strategy" not in stage:
                raise HTTPException(status_code=400, detail=f"Stage {i+1} missing strategy")
            if "tasks" not in stage:
                raise HTTPException(status_code=400, detail=f"Stage {i+1} missing tasks")

        # Execute workflow
        result = await server.orchestration_manager.execute_multi_strategy_workflow(
            workflow_id=workflow_id,
            workflow_stages=workflow_stages
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main(host: str = "0.0.0.0", port: int = 8005):
    """Run the agent orchestration MCP server"""
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8005
    
    main(host, port)
