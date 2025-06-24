"""
Database Manager for PyGent Factory Startup Service
Async SQLAlchemy with PostgreSQL and SQLite support for service orchestration.
"""

import os
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, Text, JSON, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from ..utils.logging_config import database_logger
from ..models.database_models import Base, ServiceConfiguration, StartupSequence, SystemState, ConfigurationProfile


class DatabaseManager:
    """Async database manager for startup service operations."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.async_session_maker = None
        self.logger = database_logger
        self._initialized = False
    
    def _get_database_url(self) -> str:
        """Get database URL from environment or use default SQLite."""
        # Check for PostgreSQL first (production)
        if os.getenv("DATABASE_URL"):
            url = os.getenv("DATABASE_URL")
            # Convert postgres:// to postgresql+asyncpg://
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+asyncpg://", 1)
            elif url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url
        
        # Check for individual PostgreSQL components
        pg_host = os.getenv("POSTGRES_HOST", "localhost")
        pg_port = os.getenv("POSTGRES_PORT", "5432")
        pg_db = os.getenv("POSTGRES_DB", "pygent_factory")
        pg_user = os.getenv("POSTGRES_USER", "postgres")
        pg_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        
        if pg_password != "postgres" or os.getenv("USE_POSTGRESQL", "").lower() == "true":
            return f"postgresql+asyncpg://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}"
        
        # Default to SQLite for development
        db_path = os.getenv("SQLITE_DB_PATH", "data/startup_service.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return f"sqlite+aiosqlite:///{db_path}"
    
    async def initialize(self) -> bool:
        """Initialize database connection and create tables."""
        try:
            self.logger.info(f"Initializing database connection: {self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url}")
            
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                echo=os.getenv("SQL_ECHO", "false").lower() == "true",
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={"check_same_thread": False} if "sqlite" in self.database_url else {}
            )
            
            # Create session maker
            self.async_session_maker = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.logger.info("Database tables created successfully")
            
            # Test connection
            await self.health_check()
            
            self._initialized = True
            self.logger.info("Database manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check database connectivity and health."""
        try:
            if not self.engine:
                return False
            
            async with self.async_session_maker() as session:
                result = await session.execute(func.now() if "postgresql" in self.database_url else func.datetime('now'))
                timestamp = result.scalar()
                
                self.logger.debug(f"Database health check successful: {timestamp}")
                return True
                
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return False
    
    @asynccontextmanager
    async def get_session(self):
        """Get async database session with automatic cleanup."""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        
        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def create_service_configuration(self, config_data: Dict[str, Any]) -> str:
        """Create a new service configuration."""
        try:
            async with self.get_session() as session:
                config = ServiceConfiguration(
                    id=str(uuid.uuid4()),
                    service_name=config_data["service_name"],
                    service_type=config_data["service_type"],
                    configuration=config_data["configuration"],
                    environment=config_data.get("environment", "development"),
                    is_active=config_data.get("is_active", True),
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                session.add(config)
                await session.flush()
                
                self.logger.info(f"Service configuration created: {config.service_name}")
                return config.id
                
        except Exception as e:
            self.logger.error(f"Failed to create service configuration: {e}")
            raise
    
    async def get_service_configuration(self, service_name: str, environment: str = "development") -> Optional[Dict[str, Any]]:
        """Get service configuration by name and environment."""
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    session.query(ServiceConfiguration).filter(
                        ServiceConfiguration.service_name == service_name,
                        ServiceConfiguration.environment == environment,
                        ServiceConfiguration.is_active == True
                    )
                )
                config = result.scalar_one_or_none()
                
                if config:
                    return {
                        "id": config.id,
                        "service_name": config.service_name,
                        "service_type": config.service_type,
                        "configuration": config.configuration,
                        "environment": config.environment,
                        "created_at": config.created_at,
                        "updated_at": config.updated_at
                    }
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get service configuration: {e}")
            raise
    
    async def create_startup_sequence(self, sequence_data: Dict[str, Any]) -> str:
        """Create a new startup sequence record."""
        try:
            async with self.get_session() as session:
                sequence = StartupSequence(
                    id=str(uuid.uuid4()),
                    sequence_name=sequence_data["sequence_name"],
                    services=sequence_data["services"],
                    dependencies=sequence_data.get("dependencies", {}),
                    environment=sequence_data.get("environment", "development"),
                    status=sequence_data.get("status", "pending"),
                    started_at=datetime.utcnow() if sequence_data.get("status") == "running" else None,
                    completed_at=None,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                session.add(sequence)
                await session.flush()
                
                self.logger.info(f"Startup sequence created: {sequence.sequence_name}")
                return sequence.id
                
        except Exception as e:
            self.logger.error(f"Failed to create startup sequence: {e}")
            raise
    
    async def update_startup_sequence_status(self, sequence_id: str, status: str, details: Dict[str, Any] = None) -> bool:
        """Update startup sequence status."""
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    session.query(StartupSequence).filter(StartupSequence.id == sequence_id)
                )
                sequence = result.scalar_one_or_none()
                
                if not sequence:
                    self.logger.warning(f"Startup sequence not found: {sequence_id}")
                    return False
                
                sequence.status = status
                sequence.updated_at = datetime.utcnow()
                
                if status == "running" and not sequence.started_at:
                    sequence.started_at = datetime.utcnow()
                elif status in ["completed", "failed"]:
                    sequence.completed_at = datetime.utcnow()
                
                if details:
                    sequence.execution_details = details
                
                self.logger.info(f"Startup sequence status updated: {sequence_id} -> {status}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update startup sequence status: {e}")
            raise
    
    async def get_system_state(self) -> Dict[str, Any]:
        """Get current system state."""
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    session.query(SystemState).order_by(SystemState.updated_at.desc()).limit(1)
                )
                state = result.scalar_one_or_none()
                
                if state:
                    return {
                        "id": state.id,
                        "overall_status": state.overall_status,
                        "services_status": state.services_status,
                        "last_startup_sequence": state.last_startup_sequence,
                        "system_metrics": state.system_metrics,
                        "updated_at": state.updated_at
                    }
                
                # Return default state if none exists
                return {
                    "overall_status": "unknown",
                    "services_status": {},
                    "last_startup_sequence": None,
                    "system_metrics": {},
                    "updated_at": datetime.utcnow()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get system state: {e}")
            raise
    
    async def update_system_state(self, state_data: Dict[str, Any]) -> bool:
        """Update system state."""
        try:
            async with self.get_session() as session:
                # Get existing state or create new one
                result = await session.execute(
                    session.query(SystemState).order_by(SystemState.updated_at.desc()).limit(1)
                )
                state = result.scalar_one_or_none()
                
                if state:
                    # Update existing state
                    state.overall_status = state_data.get("overall_status", state.overall_status)
                    state.services_status = state_data.get("services_status", state.services_status)
                    state.last_startup_sequence = state_data.get("last_startup_sequence", state.last_startup_sequence)
                    state.system_metrics = state_data.get("system_metrics", state.system_metrics)
                    state.updated_at = datetime.utcnow()
                else:
                    # Create new state
                    state = SystemState(
                        id=str(uuid.uuid4()),
                        overall_status=state_data.get("overall_status", "unknown"),
                        services_status=state_data.get("services_status", {}),
                        last_startup_sequence=state_data.get("last_startup_sequence"),
                        system_metrics=state_data.get("system_metrics", {}),
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    session.add(state)
                
                self.logger.debug("System state updated successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update system state: {e}")
            raise
    
    async def close(self):
        """Close database connections."""
        try:
            if self.engine:
                await self.engine.dispose()
                self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")
    
    async def execute_raw_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute raw SQL query (use with caution)."""
        try:
            async with self.get_session() as session:
                result = await session.execute(query, params or {})
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Failed to execute raw query: {e}")
            raise
