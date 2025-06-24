#!/usr/bin/env python3
"""
Database Migration System

Production-ready database migration system for PyGent Factory.
Handles schema changes, data migrations, and version control.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from .production_manager import db_manager

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Database migration definition"""
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str
    dependencies: List[str] = None


class MigrationManager:
    """Manages database migrations and schema versioning"""
    
    def __init__(self):
        self.migrations: List[Migration] = []
        self.applied_migrations: List[str] = []
    
    async def initialize(self):
        """Initialize migration tracking table"""
        try:
            async with db_manager.get_session() as session:
                # Create migrations table if it doesn't exist
                await session.execute(text("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        execution_time_ms INTEGER,
                        checksum VARCHAR(64)
                    )
                """))
                await session.commit()
                
                # Load applied migrations
                result = await session.execute(text("SELECT version FROM schema_migrations ORDER BY applied_at"))
                self.applied_migrations = [row[0] for row in result.fetchall()]
                
                logger.info(f"Migration manager initialized. {len(self.applied_migrations)} migrations applied.")
                
        except Exception as e:
            logger.error(f"Failed to initialize migration manager: {e}")
            raise
    
    def add_migration(self, migration: Migration):
        """Add a migration to the registry"""
        self.migrations.append(migration)
        # Sort by version
        self.migrations.sort(key=lambda m: m.version)
    
    async def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations"""
        return [m for m in self.migrations if m.version not in self.applied_migrations]
    
    async def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration"""
        try:
            start_time = datetime.utcnow()
            
            async with db_manager.get_session() as session:
                # Execute migration SQL
                await session.execute(text(migration.up_sql))
                
                # Record migration
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                await session.execute(text("""
                    INSERT INTO schema_migrations (version, name, description, execution_time_ms)
                    VALUES (:version, :name, :description, :execution_time)
                """), {
                    "version": migration.version,
                    "name": migration.name,
                    "description": migration.description,
                    "execution_time": execution_time
                })
                
                await session.commit()
                
                self.applied_migrations.append(migration.version)
                logger.info(f"Applied migration {migration.version}: {migration.name} ({execution_time}ms)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply migration {migration.version}: {e}")
            return False
    
    async def rollback_migration(self, migration: Migration) -> bool:
        """Rollback a single migration"""
        try:
            start_time = datetime.utcnow()
            
            async with db_manager.get_session() as session:
                # Execute rollback SQL
                await session.execute(text(migration.down_sql))
                
                # Remove migration record
                await session.execute(text("""
                    DELETE FROM schema_migrations WHERE version = :version
                """), {"version": migration.version})
                
                await session.commit()
                
                if migration.version in self.applied_migrations:
                    self.applied_migrations.remove(migration.version)
                
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                logger.info(f"Rolled back migration {migration.version}: {migration.name} ({execution_time}ms)")
                return True
                
        except Exception as e:
            logger.error(f"Failed to rollback migration {migration.version}: {e}")
            return False
    
    async def migrate_up(self, target_version: Optional[str] = None) -> bool:
        """Apply all pending migrations up to target version"""
        try:
            pending = await self.get_pending_migrations()
            
            if target_version:
                pending = [m for m in pending if m.version <= target_version]
            
            if not pending:
                logger.info("No pending migrations to apply")
                return True
            
            logger.info(f"Applying {len(pending)} migrations...")
            
            for migration in pending:
                success = await self.apply_migration(migration)
                if not success:
                    logger.error(f"Migration failed at {migration.version}")
                    return False
            
            logger.info("All migrations applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            return False
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        try:
            pending = await self.get_pending_migrations()
            
            async with db_manager.get_session() as session:
                result = await session.execute(text("""
                    SELECT version, name, applied_at, execution_time_ms
                    FROM schema_migrations 
                    ORDER BY applied_at DESC 
                    LIMIT 10
                """))
                recent_migrations = [dict(row._mapping) for row in result.fetchall()]
            
            return {
                "total_migrations": len(self.migrations),
                "applied_migrations": len(self.applied_migrations),
                "pending_migrations": len(pending),
                "recent_migrations": recent_migrations,
                "pending_list": [{"version": m.version, "name": m.name} for m in pending]
            }
            
        except Exception as e:
            logger.error(f"Failed to get migration status: {e}")
            return {"error": str(e)}


# Initialize migration manager
migration_manager = MigrationManager()

# Define production migrations
PRODUCTION_MIGRATIONS = [
    Migration(
        version="001",
        name="create_production_schema",
        description="Create production database schema with PostgreSQL optimizations",
        up_sql="""
        -- Enable required extensions
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        CREATE EXTENSION IF NOT EXISTS "pgcrypto";
        
        -- Create indexes for performance
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_type_status_concurrent 
        ON agents(agent_type, status) WHERE is_active = true;
        
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_state_priority_concurrent 
        ON tasks(state, priority) WHERE state IN ('pending', 'running');
        
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_processing_status_concurrent 
        ON documents(processing_status) WHERE processing_status != 'completed';
        
        -- Add performance monitoring
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        -- Create triggers for updated_at
        DO $$
        DECLARE
            t text;
        BEGIN
            FOR t IN
                SELECT table_name 
                FROM information_schema.columns 
                WHERE column_name = 'updated_at' 
                AND table_schema = 'public'
            LOOP
                EXECUTE format('CREATE TRIGGER update_%I_updated_at 
                    BEFORE UPDATE ON %I 
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()', t, t);
            END LOOP;
        END;
        $$;
        """,
        down_sql="""
        -- Remove triggers
        DO $$
        DECLARE
            t text;
        BEGIN
            FOR t IN
                SELECT table_name 
                FROM information_schema.columns 
                WHERE column_name = 'updated_at' 
                AND table_schema = 'public'
            LOOP
                EXECUTE format('DROP TRIGGER IF EXISTS update_%I_updated_at ON %I', t, t);
            END LOOP;
        END;
        $$;
        
        DROP FUNCTION IF EXISTS update_updated_at_column();
        """
    ),
    
    Migration(
        version="002",
        name="add_vector_support",
        description="Add pgvector support for embeddings",
        up_sql="""
        -- Enable pgvector extension
        CREATE EXTENSION IF NOT EXISTS vector;
        
        -- Add vector columns to existing tables
        ALTER TABLE document_embeddings 
        ADD COLUMN IF NOT EXISTS embedding_vector_v2 vector(1536);
        
        ALTER TABLE code_embeddings 
        ADD COLUMN IF NOT EXISTS embedding_vector_v2 vector(1536);
        
        ALTER TABLE agent_memory 
        ADD COLUMN IF NOT EXISTS embedding_v2 vector(1536);
        
        -- Create vector indexes for similarity search
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_embeddings_vector 
        ON document_embeddings USING ivfflat (embedding_vector_v2 vector_cosine_ops) 
        WITH (lists = 100);
        
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_code_embeddings_vector 
        ON code_embeddings USING ivfflat (embedding_vector_v2 vector_cosine_ops) 
        WITH (lists = 100);
        """,
        down_sql="""
        DROP INDEX IF EXISTS idx_document_embeddings_vector;
        DROP INDEX IF EXISTS idx_code_embeddings_vector;
        
        ALTER TABLE document_embeddings DROP COLUMN IF EXISTS embedding_vector_v2;
        ALTER TABLE code_embeddings DROP COLUMN IF EXISTS embedding_vector_v2;
        ALTER TABLE agent_memory DROP COLUMN IF EXISTS embedding_v2;
        """
    ),
    
    Migration(
        version="003",
        name="add_performance_monitoring",
        description="Add performance monitoring and analytics tables",
        up_sql="""
        -- Create performance monitoring table
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            metric_name VARCHAR(255) NOT NULL,
            metric_value FLOAT NOT NULL,
            metric_type VARCHAR(50) NOT NULL,
            tags JSONB DEFAULT '{}',
            recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX idx_performance_metrics_name_time 
        ON performance_metrics(metric_name, recorded_at);
        
        -- Create audit log table
        CREATE TABLE IF NOT EXISTS audit_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            table_name VARCHAR(255) NOT NULL,
            operation VARCHAR(20) NOT NULL,
            old_values JSONB,
            new_values JSONB,
            user_id UUID,
            performed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        
        CREATE INDEX idx_audit_log_table_time 
        ON audit_log(table_name, performed_at);
        """,
        down_sql="""
        DROP TABLE IF EXISTS performance_metrics;
        DROP TABLE IF EXISTS audit_log;
        """
    )
]

# Register all migrations
for migration in PRODUCTION_MIGRATIONS:
    migration_manager.add_migration(migration)


async def initialize_production_database():
    """Initialize production database with all migrations"""
    try:
        logger.info("Initializing production database...")
        
        # Initialize database manager
        await db_manager.initialize()
        
        # Initialize migration manager
        await migration_manager.initialize()
        
        # Apply all migrations
        await migration_manager.migrate_up()
        
        logger.info("Production database initialization completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize production database: {e}")
        return False


if __name__ == "__main__":
    # Run database initialization
    asyncio.run(initialize_production_database())
