#!/usr/bin/env python3
"""
Production Database Manager

Enterprise-grade PostgreSQL database manager with connection pooling,
migrations, and comprehensive error handling.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import text, event
from sqlalchemy.engine import Engine

from .models import Base, PRODUCTION_MODEL_REGISTRY

logger = logging.getLogger(__name__)


class ProductionDatabaseManager:
    """Production-ready PostgreSQL database manager"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.session_factory = None
        self.connection_pool = None
        self.is_initialized = False
        
        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "30"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        
        # Performance monitoring
        self.connection_count = 0
        self.query_count = 0
        self.error_count = 0
        self.start_time = datetime.utcnow()
    
    def _get_database_url(self) -> str:
        """Get database URL from environment"""
        # Production PostgreSQL URL
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            return db_url
        
        # Construct from components
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        database = os.getenv("DB_NAME", "pygent_factory")
        username = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "postgres")
        
        return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    
    async def initialize(self) -> bool:
        """Initialize database engine and connection pool"""
        try:
            logger.info("Initializing production database manager...")
            
            # Create async engine with optimized settings
            self.engine = create_async_engine(
                self.database_url,
                poolclass=NullPool,  # Use external connection pooling
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                future=True,
                pool_pre_ping=True,
                pool_recycle=self.pool_recycle,
                connect_args={
                    "server_settings": {
                        "application_name": "pygent_factory_production",
                        "timezone": "UTC"
                    }
                }
            )
            
            # Setup event listeners for monitoring
            self._setup_event_listeners()
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Initialize connection pool
            await self._initialize_connection_pool()
            
            # Test connection
            await self.health_check()
            
            # Create tables if needed
            await self.create_tables()
            
            self.is_initialized = True
            logger.info("Production database manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            return False
    
    async def _initialize_connection_pool(self):
        """Initialize asyncpg connection pool for high-performance operations"""
        try:
            # Extract connection parameters from URL
            url_parts = self.database_url.replace("postgresql+asyncpg://", "postgresql://")
            
            self.connection_pool = await asyncpg.create_pool(
                url_parts,
                min_size=5,
                max_size=self.pool_size,
                command_timeout=60,
                server_settings={
                    'application_name': 'pygent_factory_pool',
                    'timezone': 'UTC'
                }
            )
            
            logger.info(f"Connection pool initialized with {self.pool_size} max connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def _setup_event_listeners(self):
        """Setup SQLAlchemy event listeners for monitoring"""
        
        @event.listens_for(self.engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            self.connection_count += 1
            logger.debug(f"Database connection established (total: {self.connection_count})")
        
        @event.listens_for(self.engine.sync_engine, "before_cursor_execute")
        def on_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            self.query_count += 1
            context._query_start_time = datetime.utcnow()
        
        @event.listens_for(self.engine.sync_engine, "after_cursor_execute")
        def on_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            if hasattr(context, '_query_start_time'):
                duration = (datetime.utcnow() - context._query_start_time).total_seconds()
                if duration > 1.0:  # Log slow queries
                    logger.warning(f"Slow query detected: {duration:.2f}s - {statement[:100]}...")
    
    async def create_tables(self):
        """Create all database tables"""
        try:
            async with self.engine.begin() as conn:
                # Enable pgvector extension if available
                try:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    logger.info("pgvector extension enabled")
                except Exception as e:
                    logger.warning(f"Could not enable pgvector extension: {e}")
                
                # Create all tables
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created successfully")
                
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def drop_tables(self):
        """Drop all database tables (use with caution)"""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                logger.warning("All database tables dropped")
                
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup"""
        if not self.is_initialized:
            raise RuntimeError("Database manager not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self.error_count += 1
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get raw asyncpg connection for high-performance operations"""
        if not self.connection_pool:
            raise RuntimeError("Connection pool not initialized")
        
        async with self.connection_pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                self.error_count += 1
                logger.error(f"Database connection error: {e}")
                raise
    
    async def execute_query(self, query: str, *args) -> Any:
        """Execute raw SQL query"""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def execute_command(self, command: str, *args) -> str:
        """Execute SQL command (INSERT, UPDATE, DELETE)"""
        async with self.get_connection() as conn:
            return await conn.execute(command, *args)

    async def fetch_all(self, query: str, *args) -> list:
        """Fetch all rows from query"""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)

    async def fetch_one(self, query: str, *args) -> dict:
        """Fetch one row from query"""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)

    async def fetch_val(self, query: str, *args) -> Any:
        """Fetch single value from query"""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive database health check"""
        try:
            start_time = datetime.utcnow()
            
            # Test SQLAlchemy connection
            async with self.get_session() as session:
                result = await session.execute(text("SELECT version()"))
                pg_version = result.scalar()
            
            # Test connection pool
            async with self.get_connection() as conn:
                pool_size = await conn.fetchval("SELECT count(*) FROM pg_stat_activity WHERE application_name LIKE 'pygent_factory%'")
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            health_data = {
                "status": "healthy",
                "postgresql_version": pg_version,
                "response_time_ms": round(response_time * 1000, 2),
                "uptime_seconds": round(uptime, 2),
                "connection_count": self.connection_count,
                "query_count": self.query_count,
                "error_count": self.error_count,
                "pool_size": pool_size,
                "max_pool_size": self.pool_size,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return health_data
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database performance statistics"""
        try:
            async with self.get_connection() as conn:
                # Get table sizes
                table_stats = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                    FROM pg_tables 
                    WHERE schemaname = 'public'
                    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                """)
                
                # Get connection stats
                connection_stats = await conn.fetch("""
                    SELECT 
                        state,
                        count(*) as count
                    FROM pg_stat_activity 
                    WHERE application_name LIKE 'pygent_factory%'
                    GROUP BY state
                """)
                
                return {
                    "table_statistics": [dict(row) for row in table_stats],
                    "connection_statistics": [dict(row) for row in connection_stats],
                    "performance_metrics": {
                        "total_connections": self.connection_count,
                        "total_queries": self.query_count,
                        "total_errors": self.error_count,
                        "error_rate": (self.error_count / max(self.query_count, 1)) * 100
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup database connections and resources"""
        try:
            if self.connection_pool:
                await self.connection_pool.close()
                logger.info("Connection pool closed")
            
            if self.engine:
                await self.engine.dispose()
                logger.info("Database engine disposed")
            
            self.is_initialized = False
            logger.info("Database manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")


# Global database manager instance
db_manager = ProductionDatabaseManager()

async def initialize_database():
    """Initialize the global database manager"""
    if not db_manager.is_initialized:
        success = await db_manager.initialize()
        if not success:
            logger.error("Failed to initialize database manager")
            return False
    return True

async def ensure_database_initialized():
    """Ensure database is initialized, initialize if not"""
    if not db_manager.is_initialized:
        return await initialize_database()
    return True
