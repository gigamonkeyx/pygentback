#!/usr/bin/env python3
"""
Create MCP database tables for tool storage

This script creates all the necessary database tables for storing MCP tool metadata
according to the MCP specification requirements.
"""

import asyncio
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sqlalchemy import create_engine, text
from src.mcp.database.models import Base, CREATE_INDEXES_SQL


async def create_mcp_tables():
    """Create MCP database tables and indexes"""
    try:
        # Use SQLite database (same as the main application)
        database_url = "sqlite:///./pygent_factory.db"
        
        print(f"Creating MCP tables in database: {database_url}")
          # Create engine
        engine = create_engine(database_url, echo=True)
        
        # Create all tables
        print("\n=== Creating tables ===")
        Base.metadata.create_all(bind=engine)
        
        # Create indexes for performance
        print("\n=== Creating indexes ===")
        with engine.connect() as conn:
            # Split the SQL and execute each statement
            for statement in CREATE_INDEXES_SQL.strip().split(';'):
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    print(f"Executing: {statement[:50]}...")
                    conn.execute(text(statement))
            conn.commit()
        
        # Verify tables were created
        print("\n=== Verifying tables ===")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            
            expected_tables = ['mcp_servers', 'mcp_tools', 'mcp_resources', 'mcp_prompts', 'mcp_tool_calls']
            
            print(f"Tables found: {tables}")
            
            for table in expected_tables:
                if table in tables:
                    print(f"✅ {table} - Created successfully")
                else:
                    print(f"❌ {table} - Missing!")
                    
            # Check indexes
            print("\n=== Verifying indexes ===")
            for table in expected_tables:
                if table in tables:
                    result = conn.execute(text(f"PRAGMA index_list({table})"))
                    indexes = [row[1] for row in result]
                    print(f"{table} indexes: {indexes}")
        
        print("\n✅ MCP database setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating MCP tables: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(create_mcp_tables())
    sys.exit(0 if success else 1)
