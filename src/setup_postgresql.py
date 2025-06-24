"""
PostgreSQL Setup Helper

Helps set up PostgreSQL database for PyGent Factory.
"""

import asyncio
import logging
import subprocess
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_postgresql_installation():
    """Check if PostgreSQL is installed."""
    try:
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ PostgreSQL found: {result.stdout.strip()}")
            return True
        else:
            logger.info("❌ PostgreSQL not found in PATH")
            return False
    except FileNotFoundError:
        logger.info("❌ PostgreSQL not installed or not in PATH")
        return False


async def test_postgresql_connection():
    """Test PostgreSQL connection."""
    try:
        import asyncpg
        
        # Try to connect
        conn = await asyncpg.connect("postgresql://postgres:postgres@localhost:5432/postgres")
        result = await conn.fetchval("SELECT 'PostgreSQL Connected' as status")
        await conn.close()
        
        if result == "PostgreSQL Connected":
            logger.info("✅ PostgreSQL connection successful")
            return True
        else:
            logger.info("❌ PostgreSQL connection failed")
            return False
            
    except Exception as e:
        logger.info(f"❌ PostgreSQL connection failed: {e}")
        return False


async def create_pygent_database():
    """Create PyGent Factory database."""
    try:
        import asyncpg
        
        # Connect to default postgres database
        conn = await asyncpg.connect("postgresql://postgres:postgres@localhost:5432/postgres")
        
        # Check if database exists
        exists = await conn.fetchval("""
            SELECT 1 FROM pg_database WHERE datname = 'pygent_factory'
        """)
        
        if exists:
            logger.info("✅ pygent_factory database already exists")
        else:
            # Create database
            await conn.execute("CREATE DATABASE pygent_factory")
            logger.info("✅ pygent_factory database created")
        
        await conn.close()
        
        # Connect to new database and create schema
        conn = await asyncpg.connect("postgresql://postgres:postgres@localhost:5432/pygent_factory")
        
        # Create tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS orchestration_events (
                id SERIAL PRIMARY KEY,
                event_type VARCHAR(100) NOT NULL,
                event_data JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS task_results (
                task_id VARCHAR(255) PRIMARY KEY,
                result_data JSONB NOT NULL,
                completed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DOUBLE PRECISION NOT NULL,
                recorded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
            )
        """)
        
        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_orchestration_events_type 
            ON orchestration_events(event_type)
        """)
        
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_orchestration_events_created 
            ON orchestration_events(created_at)
        """)
        
        # Test insert
        await conn.execute("""
            INSERT INTO orchestration_events (event_type, event_data)
            VALUES ('setup_test', '{"message": "Database setup successful"}')
        """)
        
        # Test query
        result = await conn.fetchval("""
            SELECT COUNT(*) FROM orchestration_events WHERE event_type = 'setup_test'
        """)
        
        if result > 0:
            logger.info("✅ Database schema created and tested successfully")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False


async def main():
    """Main setup function."""
    logger.info("🗄️ PostgreSQL Setup for PyGent Factory")
    logger.info("="*50)
    
    # Check if PostgreSQL is installed
    pg_installed = await check_postgresql_installation()
    
    if not pg_installed:
        logger.info("\n📋 PostgreSQL Installation Instructions:")
        logger.info("1. Download PostgreSQL from: https://www.postgresql.org/download/")
        logger.info("2. Install with default settings")
        logger.info("3. Set password for 'postgres' user to 'postgres'")
        logger.info("4. Ensure PostgreSQL service is running")
        logger.info("5. Add PostgreSQL bin directory to PATH")
        return False
    
    # Test connection
    logger.info("\n🔌 Testing PostgreSQL Connection...")
    connection_ok = await test_postgresql_connection()
    
    if not connection_ok:
        logger.info("\n📋 Connection Troubleshooting:")
        logger.info("1. Ensure PostgreSQL service is running")
        logger.info("2. Check if password for 'postgres' user is 'postgres'")
        logger.info("3. Verify PostgreSQL is listening on localhost:5432")
        logger.info("4. Check firewall settings")
        return False
    
    # Create database and schema
    logger.info("\n🏗️ Setting up PyGent Factory Database...")
    db_setup_ok = await create_pygent_database()
    
    if db_setup_ok:
        logger.info("\n🎉 PostgreSQL Setup Complete!")
        logger.info("✅ Database: pygent_factory")
        logger.info("✅ Connection: postgresql://postgres:postgres@localhost:5432/pygent_factory")
        logger.info("✅ Schema: Created and tested")
        logger.info("🚀 Ready for zero mock code integration!")
        return True
    else:
        logger.error("\n❌ PostgreSQL setup failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)