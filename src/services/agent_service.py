"""
Agent Service - Database-backed agent management

This module provides database-backed agent management services, ensuring
all agents are associated with users and persisted to the database.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

from ..database.models import Agent as AgentModel, User
from ..database.connection import get_database_session
from ..core.agent_factory import AgentFactory
from ..core.agent import AgentStatus


logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing agents with database persistence and user association."""
    
    def __init__(self, agent_factory: AgentFactory):
        self.agent_factory = agent_factory
    
    async def create_agent(
        self,
        user_id: str,
        agent_type: str,
        name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        mcp_tools: Optional[List[str]] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> AgentModel:
        """
        Create a new agent associated with a user.
        
        Args:
            user_id: ID of the user creating the agent
            agent_type: Type of agent to create
            name: Optional name for the agent
            capabilities: List of capabilities to enable
            mcp_tools: List of MCP tools to register
            custom_config: Custom configuration parameters
            
        Returns:
            AgentModel: The created agent database record
            
        Raises:
            ValueError: If user not found or agent creation fails
            SQLAlchemyError: If database operation fails
        """
        session = next(get_database_session())
        try:
            # Verify user exists
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError(f"User not found: {user_id}")
            
            # Create agent instance using factory
            agent_instance = await self.agent_factory.create_agent(
                agent_type=agent_type,
                name=name,
                capabilities=capabilities,
                mcp_tools=mcp_tools,
                custom_config=custom_config
            )
            
            # Create database record
            agent_record = AgentModel(
                id=agent_instance.agent_id,
                user_id=user_id,
                name=agent_instance.name,
                type=agent_instance.type,
                capabilities=capabilities or [],
                config={
                    'mcp_tools': mcp_tools or [],
                    'custom_config': custom_config or {}
                },
                status=agent_instance.status.value
            )
            
            session.add(agent_record)
            session.commit()
            session.refresh(agent_record)
            
            logger.info(f"Created agent {agent_record.id} for user {user_id}")
            return agent_record
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error creating agent: {str(e)}")
            raise
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating agent: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_user_agents(
        self,
        user_id: str,
        agent_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[AgentModel]:
        """
        Get all agents for a specific user with optional filtering.
        
        Args:
            user_id: ID of the user
            agent_type: Optional agent type filter
            status: Optional status filter
            
        Returns:
            List[AgentModel]: List of user's agents
        """
        session = next(get_database_session())
        try:
            query = session.query(AgentModel).filter(AgentModel.user_id == user_id)
            
            if agent_type:
                query = query.filter(AgentModel.type == agent_type)
            
            if status:
                query = query.filter(AgentModel.status == status)
            
            agents = query.order_by(AgentModel.created_at.desc()).all()
            return agents
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting user agents: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_agent_by_id(self, agent_id: str, user_id: str) -> Optional[AgentModel]:
        """
        Get an agent by ID, ensuring it belongs to the specified user.
        
        Args:
            agent_id: ID of the agent
            user_id: ID of the user (for authorization)
            
        Returns:
            AgentModel or None: The agent if found and owned by user
        """
        session = next(get_database_session())
        try:
            agent = session.query(AgentModel).filter(
                AgentModel.id == agent_id,
                AgentModel.user_id == user_id
            ).first()
            return agent
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting agent: {str(e)}")
            raise
        finally:
            session.close()
    
    async def update_agent_status(
        self,
        agent_id: str,
        user_id: str,
        status: str
    ) -> Optional[AgentModel]:
        """
        Update an agent's status.
        
        Args:
            agent_id: ID of the agent
            user_id: ID of the user (for authorization)
            status: New status
            
        Returns:
            AgentModel or None: Updated agent if found and owned by user
        """
        session = next(get_database_session())
        try:
            agent = session.query(AgentModel).filter(
                AgentModel.id == agent_id,
                AgentModel.user_id == user_id
            ).first()
            
            if not agent:
                return None
            
            agent.status = status
            agent.updated_at = datetime.utcnow()
            
            session.commit()
            session.refresh(agent)
            
            # Also update the runtime agent if it exists
            runtime_agent = await self.agent_factory.get_agent(agent_id)
            if runtime_agent:
                runtime_agent.status = AgentStatus(status)
            
            logger.info(f"Updated agent {agent_id} status to {status}")
            return agent
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error updating agent status: {str(e)}")
            raise
        finally:
            session.close()
    
    async def delete_agent(self, agent_id: str, user_id: str) -> bool:
        """
        Delete an agent.
        
        Args:
            agent_id: ID of the agent
            user_id: ID of the user (for authorization)
            
        Returns:
            bool: True if agent was deleted, False if not found
        """
        session = next(get_database_session())
        try:
            agent = session.query(AgentModel).filter(
                AgentModel.id == agent_id,
                AgentModel.user_id == user_id
            ).first()
            
            if not agent:
                return False
            
            # Shut down runtime agent if it exists
            await self.agent_factory.registry.unregister_agent(agent_id)
            
            # Delete database record
            session.delete(agent)
            session.commit()
            
            logger.info(f"Deleted agent {agent_id} for user {user_id}")
            return True
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error deleting agent: {str(e)}")
            raise
        finally:
            session.close()
    
    def get_agent_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics about a user's agents.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dict with agent statistics
        """
        session = next(get_database_session())
        try:
            agents = session.query(AgentModel).filter(AgentModel.user_id == user_id).all()
            
            stats = {
                'total_agents': len(agents),
                'agents_by_type': {},
                'agents_by_status': {},
                'most_recent_agent': None
            }
            
            for agent in agents:
                # Count by type
                stats['agents_by_type'][agent.type] = stats['agents_by_type'].get(agent.type, 0) + 1
                
                # Count by status
                stats['agents_by_status'][agent.status] = stats['agents_by_status'].get(agent.status, 0) + 1
            
            # Get most recent agent
            if agents:
                most_recent = max(agents, key=lambda a: a.created_at)
                stats['most_recent_agent'] = {
                    'id': most_recent.id,
                    'name': most_recent.name,
                    'type': most_recent.type,
                    'created_at': most_recent.created_at.isoformat()
                }
            
            return stats
            
        except SQLAlchemyError as e:
            logger.error(f"Database error getting agent statistics: {str(e)}")
            raise
        finally:
            session.close()


# Global agent service instance
_agent_service: Optional[AgentService] = None


def set_agent_service(service: AgentService):
    """Set the global agent service instance"""
    global _agent_service
    _agent_service = service


def get_agent_service() -> AgentService:
    """Get the agent service dependency"""
    if _agent_service is None:
        raise RuntimeError("Agent service not initialized")
    return _agent_service
