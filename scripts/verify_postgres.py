#!/usr/bin/env python3
"""
PostgreSQL Connection Verification Script
Verifies connection to the running PostgreSQL container.
"""

import asyncio
import asyncpg
import sys
import os

async def verify_postgres_connection():
    """Verify PostgreSQL connection and create database if needed."""
    
    # Connection parameters
    host = "localhost"
    port = 54321
    user = "postgres"
    password = "postgres"
    database = "pygent_factory"
    
    try:
        # First, connect to default postgres database
        print(f"Connecting to PostgreSQL at {host}:{port}...")
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres"
        )
        
        print("✅ Connected to PostgreSQL successfully")
        
        # Check if pygent_factory database exists
        result = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", 
            database
        )
        
        if not result:
            print(f"Creating database: {database}")
            await conn.execute(f"CREATE DATABASE {database}")
            print(f"✅ Database {database} created")
        else:
            print(f"✅ Database {database} already exists")
        
        await conn.close()
        
        # Now connect to the target database
        conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        # Test pgvector extension
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("✅ pgvector extension available")
        except Exception as e:
            print(f"⚠️ pgvector extension not available: {e}")
        
        # Test basic query
        version = await conn.fetchval("SELECT version()")
        print(f"✅ PostgreSQL version: {version[:50]}...")
        
        await conn.close()
        
        # Set environment variable for the session
        connection_string = f"postgresql://postgres:postgres@localhost:54321/pygent_factory"
        print(f"✅ Connection string: {connection_string}")
        
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(verify_postgres_connection())
    sys.exit(0 if success else 1)
