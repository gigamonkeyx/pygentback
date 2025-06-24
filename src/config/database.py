"""
Database Configuration Module

Provides database connection and configuration utilities.
"""

import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker
from typing import Optional

def get_database_url() -> str:
    """Get database URL from environment or use default"""
    return os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:password@localhost:5432/pygent_factory"
    )

def create_engine(database_url: Optional[str] = None) -> Engine:
    """Create SQLAlchemy engine"""
    from sqlalchemy import create_engine as sa_create_engine
    
    url = database_url or get_database_url()
    return sa_create_engine(url)

def get_session_maker(engine: Optional[Engine] = None):
    """Get SQLAlchemy session maker"""
    if engine is None:
        engine = create_engine()
    return sessionmaker(bind=engine)
