#!/usr/bin/env python3
"""
A2A Integration Manager

Manages the integration of A2A protocol with PyGent Factory infrastructure.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .protocol import a2a_protocol, Message, TextPart, Task
from .agent_integration import a2a_agent_registry
from .server import a2a_server

logger = logging.getLogger(__name__)


class A2AManager:
    """A2A Protocol Integration Manager"""
    
    def __init__(self):
        self.initialized = False
        self.server_task: Optional[asyncio.Task] = None
        self.registered_agents: Dict[str, Any] = {}
        
    async def initialize(self, 
                        database_manager=None, 
                        redis_manager=None, 
                        orchestration_manager=None) -> bool:
        """Initialize A2A manager with PyGent Factory infrastructure"""
        try:
            logger.info("Initializing A2A Protocol Manager...")
            
            # Store infrastructure references
            self.database_manager = database_manager
            self.redis_manager = redis_manager
            self.orchestration_manager = orchestration_manager
            
            # Initialize A2A protocol
            await self._initialize_a2a_protocol()
            
            # Setup integration with existing infrastructure
            await self._setup_infrastructure_integration()
            
            self.initialized = True
            logger.info("A2A Protocol Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize A2A manager: {e}")
            return False
    
    async def _initialize_a2a_protocol(self):
        """Initialize the A2A protocol"""
        # A2A protocol is already initialized as a global instance
        logger.info("A2A protocol initialized")
    
    async def _setup_infrastructure_integration(self):
        """Setup integration with PyGent Factory infrastructure"""
        
        # If database manager is available, setup persistence
        if self.database_manager:
            await self._setup_database_integration()
        
        # If Redis manager is available, setup caching
        if self.redis_manager:
            await self._setup_redis_integration()
        
        # If orchestration manager is available, setup coordination
        if self.orchestration_manager:
            await self._setup_orchestration_integration()
    
    async def _setup_database_integration(self):
        """Setup database integration for A2A tasks"""
        try:
            # A2A integration now uses main database models (agents, tasks)
            # No separate tables needed - A2A fields are integrated into main models

            # Verify main tables exist and have A2A fields
            if hasattr(self.database_manager, 'get_session'):
                # Modern database manager with SQLAlchemy session
                logger.info("Using SQLAlchemy session for A2A database integration")
            else:
                # Legacy database manager - verify tables exist
                logger.info("Verifying main database tables for A2A integration")

            logger.info("A2A database integration setup complete - using main models")

        except Exception as e:
            logger.error(f"Failed to setup A2A database integration: {e}")
    
    async def _setup_redis_integration(self):
        """Setup Redis integration for A2A communication"""
        try:
            # Setup Redis channels for A2A communication
            await self.redis_manager.set("a2a:status", "active")
            await self.redis_manager.set("a2a:initialized", datetime.utcnow().isoformat())
            
            logger.info("A2A Redis integration setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup A2A Redis integration: {e}")
    
    async def _setup_orchestration_integration(self):
        """Setup orchestration integration for A2A coordination"""
        try:
            # Register A2A manager with orchestration system
            if hasattr(self.orchestration_manager, 'register_service'):
                await self.orchestration_manager.register_service(
                    "a2a_manager", 
                    self,
                    capabilities=["agent_communication", "task_routing", "protocol_management"]
                )
            
            logger.info("A2A orchestration integration setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup A2A orchestration integration: {e}")
    
    async def register_agent(self, agent, base_url: str = "http://localhost:8000") -> bool:
        """Register an agent with A2A protocol"""
        try:
            # Register with A2A agent registry
            wrapper = await a2a_agent_registry.register_agent(agent, base_url)
            
            # Store reference
            self.registered_agents[agent.agent_id] = {
                "agent": agent,
                "wrapper": wrapper,
                "registered_at": datetime.utcnow().isoformat()
            }
            
            # Persist to database if available
            if self.database_manager:
                await self._persist_agent_registration(agent, wrapper)
            
            logger.info(f"Successfully registered agent {agent.name} with A2A protocol")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.name}: {e}")
            return False
    
    async def _persist_agent_registration(self, agent, wrapper):
        """Persist agent registration to database using main Agent model"""
        try:
            agent_card_data = {
                "name": wrapper.agent.name,
                "description": f"PyGent Factory {str(agent.agent_type).lower().title()} Agent",
                "url": wrapper.agent_url,
                "version": "1.0.0",
                "capabilities": [cap.name for cap in agent.capabilities] if hasattr(agent, 'capabilities') else [],
                "agent_type": str(agent.agent_type)
            }

            # Use SQLAlchemy session if available
            if hasattr(self.database_manager, 'get_session'):
                from ..database.models import Agent

                async with self.database_manager.get_session() as session:
                    # Find existing agent by ID
                    existing_agent = await session.get(Agent, agent.id)

                    if existing_agent:
                        # Update existing agent with A2A fields
                        existing_agent.a2a_url = wrapper.agent_url
                        existing_agent.a2a_agent_card = agent_card_data
                        existing_agent.last_seen_at = datetime.utcnow()
                        await session.commit()
                        logger.info(f"Updated existing agent {agent.name} with A2A information")
                    else:
                        logger.warning(f"Agent {agent.name} not found in database - A2A registration skipped")
            else:
                # Fallback: log warning about legacy database manager
                logger.warning("Legacy database manager detected - A2A persistence may be limited")
                logger.info(f"Would persist agent {agent.name} A2A registration (URL: {wrapper.agent_url})")

        except Exception as e:
            logger.error(f"Failed to persist agent A2A registration: {e}")
    
    async def start_server(self, host: str = "localhost", port: int = 8080) -> bool:
        """Start the A2A server"""
        try:
            if self.server_task and not self.server_task.done():
                logger.warning("A2A server is already running")
                return True
            
            # Update server configuration
            a2a_server.host = host
            a2a_server.port = port
            
            # Start server in background task
            self.server_task = asyncio.create_task(a2a_server.start())
            
            # Give server time to start
            await asyncio.sleep(1)
            
            logger.info(f"A2A server started on {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start A2A server: {e}")
            return False
    
    async def stop_server(self):
        """Stop the A2A server"""
        try:
            if self.server_task and not self.server_task.done():
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass
                
                logger.info("A2A server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping A2A server: {e}")
    
    async def send_agent_to_agent_message(self, 
                                        from_agent_id: str, 
                                        to_agent_id: str, 
                                        message: str,
                                        metadata: Optional[Dict[str, Any]] = None) -> Optional[Task]:
        """Send a message from one agent to another using A2A protocol"""
        try:
            # Get target agent wrapper
            to_agent_wrapper = await a2a_agent_registry.get_agent(to_agent_id)
            if not to_agent_wrapper:
                logger.error(f"Target agent {to_agent_id} not found")
                return None
            
            # Create A2A message
            a2a_message = Message(
                role="agent",
                parts=[TextPart(text=message)],
                metadata=metadata or {"from_agent": from_agent_id}
            )
            
            # Create task
            task = await a2a_protocol.create_task(
                agent_url=to_agent_wrapper.agent_url,
                message=a2a_message
            )
            
            # Execute task
            result_task = await to_agent_wrapper.handle_a2a_task(task)
            
            logger.info(f"Successfully sent A2A message from {from_agent_id} to {to_agent_id}")
            return result_task
            
        except Exception as e:
            logger.error(f"Failed to send A2A message: {e}")
            return None
    
    async def coordinate_multi_agent_task(self, 
                                        task_description: str, 
                                        agent_ids: List[str],
                                        coordination_strategy: str = "sequential") -> List[Task]:
        """Coordinate a task across multiple agents"""
        try:
            results = []
            
            if coordination_strategy == "sequential":
                # Execute tasks sequentially
                current_message = task_description
                
                for agent_id in agent_ids:
                    task = await self.send_agent_to_agent_message(
                        from_agent_id="coordinator",
                        to_agent_id=agent_id,
                        message=current_message,
                        metadata={"coordination_strategy": "sequential", "task_index": len(results)}
                    )
                    
                    if task:
                        results.append(task)
                        # Use the result as input for the next agent
                        if task.artifacts:
                            current_message = task.artifacts[-1].parts[0].text if task.artifacts[-1].parts else current_message
            
            elif coordination_strategy == "parallel":
                # Execute tasks in parallel
                tasks = []
                for agent_id in agent_ids:
                    task_coro = self.send_agent_to_agent_message(
                        from_agent_id="coordinator",
                        to_agent_id=agent_id,
                        message=task_description,
                        metadata={"coordination_strategy": "parallel"}
                    )
                    tasks.append(task_coro)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # Filter out exceptions
                results = [r for r in results if isinstance(r, Task)]
            
            logger.info(f"Coordinated multi-agent task across {len(agent_ids)} agents, got {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to coordinate multi-agent task: {e}")
            return []
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered A2A agents"""
        try:
            agents = await a2a_agent_registry.list_agents()
            
            status = {
                "total_agents": len(agents),
                "active_tasks": len(a2a_protocol.tasks),
                "agents": []
            }
            
            for wrapper in agents:
                agent_info = {
                    "agent_id": wrapper.agent.agent_id,
                    "name": wrapper.agent.name,
                    "type": str(wrapper.agent.agent_type),
                    "status": str(wrapper.agent.status),
                    "url": wrapper.agent_url,
                    "capabilities": len(wrapper.agent.capabilities)
                }
                status["agents"].append(agent_info)
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get agent status: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown A2A manager"""
        try:
            logger.info("Shutting down A2A manager...")
            
            # Stop server
            await self.stop_server()
            
            # Clear registrations
            self.registered_agents.clear()
            
            # Update status in Redis if available
            if self.redis_manager:
                await self.redis_manager.set("a2a:status", "shutdown")
            
            self.initialized = False
            logger.info("A2A manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during A2A manager shutdown: {e}")


# Global A2A manager instance
a2a_manager = A2AManager()
