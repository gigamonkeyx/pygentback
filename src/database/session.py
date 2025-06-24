"""
Database Session Dependencies

This module provides FastAPI dependency injection for database sessions
following Context7 MCP best practices with proper yield-based session management.
"""

from typing import Annotated, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from .connection import get_db_session


# Type annotation for database session dependency injection
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Alternative session dependency function.
    
    This is an alias for get_db_session for consistency with Context7 patterns.
    
    Yields:
        AsyncSession: Database session with automatic commit/rollback
    """
    async for session in get_db_session():
        yield session


# Alternative type annotation
SessionDependency = Annotated[AsyncSession, Depends(get_session)]


__all__ = [
    "SessionDep",
    "SessionDependency", 
    "get_session",
    "get_db_session"
]
