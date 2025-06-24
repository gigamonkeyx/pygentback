"""
Database Connection Management

This module provides database connection management for PyGent Factory,
including async SQLAlchemy setup, connection pooling, and health monitoring.
"""

import logging
from typing import Optional, Dict, Any, AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncSession, 
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool
import time

from .models import Base
from ..config.settings import Settings


logger = logging.getLogger(__name__)


# Global synchronous database components for legacy compatibility
sync_engine: Optional[Any] = None
sync_session_factory: Optional[sessionmaker] = None


def get_sync_database_url(settings: Optional[Settings] = None) -> str:
    """Get synchronous database URL from async URL."""
    if settings is None:
        settings = Settings()
    
    # Convert async postgres URL to sync
    async_url = settings.DATABASE_URL
    if async_url.startswith('postgresql+asyncpg://'):
        return async_url.replace('postgresql+asyncpg://', 'postgresql://')
    elif async_url.startswith('postgresql+psycopg://'):
        return async_url.replace('postgresql+psycopg://', 'postgresql://')
    elif async_url.startswith('sqlite+aiosqlite://'):
        return async_url.replace('sqlite+aiosqlite://', 'sqlite://')
    else:
        return async_url


def init_sync_database(settings: Optional[Settings] = None):
    """Initialize synchronous database components."""
    global sync_engine, sync_session_factory
    
    if settings is None:
        settings = Settings()
    
    try:
        # Create synchronous engine
        sync_url = get_sync_database_url(settings)
        sync_engine = create_engine(
            sync_url,
            poolclass=NullPool,
            echo=getattr(settings, 'database_echo', False),
            pool_pre_ping=True
        )
        
        # Create session factory
        sync_session_factory = sessionmaker(
            bind=sync_engine,
            autocommit=False,
            autoflush=False
        )
        
        logger.info("Synchronous database components initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize synchronous database: {e}")
        raise


def get_database_session() -> Generator[Session, None, None]:
    """
    Get synchronous database session for legacy compatibility.
    
    Yields:
        Session: Synchronous database session
    """
    global sync_session_factory
    
    if sync_session_factory is None:
        init_sync_database()
    
    session = sync_session_factory()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_sync_db_session() -> Generator[Session, None, None]:
    """
    Context manager for synchronous database sessions.
    
    Yields:
        Session: Synchronous database session
    """
    session = next(get_database_session())
    try:
        yield session
    finally:
        session.close()


class DatabaseManager:
    """
    Database connection and session management.
    
    Provides async database connections, session management,
    connection pooling, and health monitoring.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize database manager.
        
        Args:
            settings: Application settings containing database configuration
        """
        self.settings = settings
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._connection_count = 0
        self._query_count = 0
        self._error_count = 0
        self._last_health_check = None
        self._health_status = False

    def _is_postgresql(self) -> bool:
        """Check if the database is PostgreSQL."""
        return "postgresql" in self.settings.ASYNC_DATABASE_URL.lower()

    def _is_sqlite(self) -> bool:
        """Check if the database is SQLite."""
        return "sqlite" in self.settings.ASYNC_DATABASE_URL.lower()

    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        try:
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                self.settings.ASYNC_DATABASE_URL,
                poolclass=NullPool,  # Use NullPool for async engines
                echo=self.settings.DEBUG,  # Log SQL queries in debug mode
                future=True
            )
            
            # Set up event listeners for monitoring
            self._setup_event_listeners()
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Test connection
            await self.health_check()
            
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {str(e)}")
            raise
    
    def _setup_event_listeners(self) -> None:
        """Set up SQLAlchemy event listeners for monitoring."""
        
        @event.listens_for(self.engine.sync_engine, "connect")
        def on_connect(dbapi_connection, connection_record):
            """Handle new database connections."""
            self._connection_count += 1
            logger.debug(f"New database connection established (total: {self._connection_count})")
        
        @event.listens_for(self.engine.sync_engine, "before_cursor_execute")
        def on_before_execute(conn, cursor, statement, parameters, context, executemany):
            """Handle query execution start."""
            context._query_start_time = time.time()
            self._query_count += 1
        
        @event.listens_for(self.engine.sync_engine, "after_cursor_execute")
        def on_after_execute(conn, cursor, statement, parameters, context, executemany):
            """Handle query execution completion."""
            if hasattr(context, '_query_start_time'):
                duration = time.time() - context._query_start_time
                if duration > 1.0:  # Log slow queries
                    logger.warning(f"Slow query detected: {duration:.2f}s - {statement[:100]}...")
        
        @event.listens_for(self.engine.sync_engine, "handle_error")
        def on_error(exception_context):
            """Handle database errors."""
            self._error_count += 1
            logger.error(f"Database error: {exception_context.original_exception}")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an async database session.
        
        Yields:
            AsyncSession: Database session
            
        Example:
            async with db_manager.get_session() as session:
                result = await session.execute(select(Agent))
        """
        if not self.session_factory:
            raise RuntimeError("Database manager not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {str(e)}")
                raise
            finally:
                await session.close()
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a raw SQL query.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            
        Returns:
            Query result
        """
        async with self.get_session() as session:
            result = await session.execute(text(query), parameters or {})
            return result
    
    async def health_check(self) -> bool:
        """
        Perform database health check.
        
        Returns:
            bool: True if database is healthy
        """
        try:
            async with self.get_session() as session:
                # Simple query to test connection
                result = await session.execute(text("SELECT 1"))
                result.fetchone()
                
                # Check if pgvector extension is available (PostgreSQL only)
                if self._is_postgresql():
                    try:
                        result = await session.execute(
                            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                        )
                        vector_available = result.fetchone() is not None

                        if not vector_available:
                            logger.warning("pgvector extension not found")
                    except Exception:
                        logger.warning("Could not check for pgvector extension")
                
                self._health_status = True
                self._last_health_check = time.time()
                
                logger.debug("Database health check passed")
                return True
                
        except Exception as e:
            self._health_status = False
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    async def create_schemas(self) -> None:
        """Create database schemas if they don't exist."""
        # SQLite doesn't support schemas, only PostgreSQL does
        if not self._is_postgresql():
            logger.info("Skipping schema creation for SQLite database")
            return

        schemas = ["agents", "knowledge", "mcp", "evaluation"]

        async with self.get_session() as session:
            for schema in schemas:
                try:
                    await session.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
                    logger.info(f"Created schema: {schema}")
                except Exception as e:
                    logger.error(f"Failed to create schema {schema}: {str(e)}")
                    raise
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise
    
    async def drop_tables(self) -> None:
        """Drop all database tables."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {str(e)}")
            raise
    
    async def setup_vector_extension(self) -> None:
        """Set up pgvector extension and functions."""
        # Vector extensions are only available for PostgreSQL
        if not self._is_postgresql():
            logger.info("Skipping vector extension setup for SQLite database")
            return

        try:
            async with self.get_session() as session:
                # Enable pgvector extension
                await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

                # Create vector similarity search functions
                similarity_function = """
                CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
                RETURNS float AS $$
                BEGIN
                    RETURN 1 - (a <=> b);
                END;
                $$ LANGUAGE plpgsql;
                """

                await session.execute(text(similarity_function))

                logger.info("Vector extension and functions set up successfully")

        except Exception as e:
            logger.error(f"Failed to set up vector extension: {str(e)}")
            raise
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information."""
        if not self.engine:
            return {"status": "not_initialized"}

        pool = self.engine.pool

        # Handle different pool types (SQLite uses NullPool)
        pool_info = {}
        try:
            if hasattr(pool, 'size'):
                pool_info.update({
                    "pool_size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                })
            else:
                # SQLite NullPool doesn't have these methods
                pool_info.update({
                    "pool_type": "NullPool (SQLite)",
                    "pool_size": "N/A",
                    "checked_in": "N/A",
                    "checked_out": "N/A",
                    "overflow": "N/A",
                })
        except Exception as e:
            pool_info.update({
                "pool_error": str(e),
                "pool_size": "Unknown",
            })

        return {
            "status": "connected" if self._health_status else "disconnected",
            "database_type": "SQLite" if "sqlite" in str(self.engine.url).lower() else "PostgreSQL",
            "database_url": str(self.engine.url),
            "connection_count": self._connection_count,
            "query_count": self._query_count,
            "error_count": self._error_count,
            "last_health_check": self._last_health_check,
            **pool_info
        }
    
    async def close(self) -> None:
        """Close database connections and cleanup."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


async def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global db_manager
    if db_manager is None:
        raise RuntimeError("Database manager not initialized")
    return db_manager


async def initialize_database(settings: Settings) -> DatabaseManager:
    """
    Initialize the global database manager.
    
    Args:
        settings: Application settings
        
    Returns:
        DatabaseManager: Initialized database manager
    """
    global db_manager
    
    if db_manager is not None:
        logger.warning("Database manager already initialized")
        return db_manager
    
    db_manager = DatabaseManager(settings)
    await db_manager.initialize()
    
    # Set up database schema and tables
    await db_manager.create_schemas()
    await db_manager.setup_vector_extension()
    await db_manager.create_tables()
    
    return db_manager


async def close_database() -> None:
    """Close the global database manager."""
    global db_manager
    if db_manager:
        await db_manager.close()
        db_manager = None


# Dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting database sessions.
    
    Yields:
        AsyncSession: Database session
    """
    db = await get_database_manager()
    async with db.get_session() as session:
        yield session
