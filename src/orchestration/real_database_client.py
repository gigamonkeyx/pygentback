"""
Real Database Client

Production-grade PostgreSQL integration for enterprise database operations.
Provides real SQL execution, connection pooling, and transaction management.
"""

import asyncio
import logging
import asyncpg
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class RealDatabaseClient:
    """
    Production PostgreSQL client with real database operations.
    
    Features:
    - Real SQL execution with parameter binding
    - Connection pooling for performance
    - Transaction management
    - Proper error handling and logging
    - Schema management and migrations
    """
    
    def __init__(self, connection_string: str, pool_size: int = 10):
        self.connection_string = connection_string
        self.pool_size = pool_size
        self.pool: Optional[asyncpg.Pool] = None
        self.is_connected = False
        
    async def connect(self) -> bool:
        """Establish real database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=self.pool_size,
                command_timeout=30,
                server_settings={
                    'application_name': 'pygent_factory_orchestration',
                    'timezone': 'UTC'
                }
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                result = await conn.fetchval('SELECT version()')
                logger.info(f"Connected to PostgreSQL: {result}")
            
            # Initialize schema
            await self._initialize_schema()
            
            self.is_connected = True
            return True
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self.is_connected = False
            logger.info("Database connection pool closed")
    
    async def execute_query(self, sql: str, params: List[Any] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query and return results."""
        if not self.is_connected:
            raise ConnectionError("Database not connected")
        
        try:
            async with self.pool.acquire() as conn:
                if params:
                    rows = await conn.fetch(sql, *params)
                else:
                    rows = await conn.fetch(sql)
                
                # Convert to list of dictionaries
                results = []
                for row in rows:
                    results.append(dict(row))
                
                logger.debug(f"Query executed: {sql[:100]}... | Rows: {len(results)}")
                return results
                
        except Exception as e:
            logger.error(f"Query execution failed: {sql[:100]}... | Error: {e}")
            raise
    
    async def execute_command(self, sql: str, params: List[Any] = None) -> int:
        """Execute INSERT/UPDATE/DELETE command and return affected rows."""
        if not self.is_connected:
            raise ConnectionError("Database not connected")
        
        try:
            async with self.pool.acquire() as conn:
                if params:
                    result = await conn.execute(sql, *params)
                else:
                    result = await conn.execute(sql)
                
                # Extract affected row count
                affected_rows = int(result.split()[-1]) if result else 0
                
                logger.debug(f"Command executed: {sql[:100]}... | Affected: {affected_rows}")
                return affected_rows
                
        except Exception as e:
            logger.error(f"Command execution failed: {sql[:100]}... | Error: {e}")
            raise
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions."""
        if not self.is_connected:
            raise ConnectionError("Database not connected")
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    async def log_orchestration_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Log orchestration events to database."""
        try:
            sql = """
                INSERT INTO orchestration_events (event_type, event_data, created_at)
                VALUES ($1, $2, $3)
            """
            
            await self.execute_command(sql, [
                event_type,
                json.dumps(event_data),
                datetime.utcnow()
            ])
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log orchestration event: {e}")
            return False
    
    async def store_task_result(self, task_id: str, result_data: Dict[str, Any]) -> bool:
        """Store task execution results."""
        try:
            sql = """
                INSERT INTO task_results (task_id, result_data, completed_at)
                VALUES ($1, $2, $3)
                ON CONFLICT (task_id) DO UPDATE SET
                    result_data = EXCLUDED.result_data,
                    completed_at = EXCLUDED.completed_at
            """
            
            await self.execute_command(sql, [
                task_id,
                json.dumps(result_data),
                datetime.utcnow()
            ])
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store task result: {e}")
            return False
    
    async def get_task_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve task execution history."""
        try:
            sql = """
                SELECT task_id, result_data, completed_at
                FROM task_results
                ORDER BY completed_at DESC
                LIMIT $1
            """
            
            return await self.execute_query(sql, [limit])
            
        except Exception as e:
            logger.error(f"Failed to retrieve task history: {e}")
            return []
    
    async def store_performance_metrics(self, metrics: Dict[str, float]) -> bool:
        """Store system performance metrics."""
        try:
            sql = """
                INSERT INTO performance_metrics (metric_name, metric_value, recorded_at)
                VALUES ($1, $2, $3)
            """
            
            async with self.transaction() as conn:
                for metric_name, metric_value in metrics.items():
                    await conn.execute(sql, metric_name, metric_value, datetime.utcnow())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {e}")
            return False
    
    async def get_performance_trends(self, metric_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance trends for a metric."""
        try:
            sql = """
                SELECT metric_value, recorded_at
                FROM performance_metrics
                WHERE metric_name = $1
                    AND recorded_at >= NOW() - INTERVAL '%s hours'
                ORDER BY recorded_at ASC
            """ % hours
            
            return await self.execute_query(sql, [metric_name])
            
        except Exception as e:
            logger.error(f"Failed to get performance trends: {e}")
            return []
    
    async def _initialize_schema(self):
        """Initialize database schema for orchestration system."""
        try:
            schema_sql = """
                -- Orchestration Events Table
                CREATE TABLE IF NOT EXISTS orchestration_events (
                    id SERIAL PRIMARY KEY,
                    event_type VARCHAR(100) NOT NULL,
                    event_data JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL
                );

                -- Create indexes for orchestration_events
                CREATE INDEX IF NOT EXISTS idx_orchestration_events_type ON orchestration_events (event_type);
                CREATE INDEX IF NOT EXISTS idx_orchestration_events_created ON orchestration_events (created_at);

                -- Task Results Table
                CREATE TABLE IF NOT EXISTS task_results (
                    task_id VARCHAR(255) PRIMARY KEY,
                    result_data JSONB NOT NULL,
                    completed_at TIMESTAMP WITH TIME ZONE NOT NULL
                );

                -- Create indexes for task_results
                CREATE INDEX IF NOT EXISTS idx_task_results_completed ON task_results (completed_at);

                -- Performance Metrics Table
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value DOUBLE PRECISION NOT NULL,
                    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL
                );

                -- Create indexes for performance_metrics
                CREATE INDEX IF NOT EXISTS idx_performance_metrics_name_time ON performance_metrics (metric_name, recorded_at);

                -- Agent Status Table
                CREATE TABLE IF NOT EXISTS agent_status (
                    agent_id VARCHAR(255) PRIMARY KEY,
                    agent_type VARCHAR(100) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    last_activity TIMESTAMP WITH TIME ZONE,
                    performance_score DOUBLE PRECISION DEFAULT 0.5,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                -- MCP Server Status Table
                CREATE TABLE IF NOT EXISTS mcp_server_status (
                    server_id VARCHAR(255) PRIMARY KEY,
                    server_type VARCHAR(100) NOT NULL,
                    endpoint VARCHAR(500) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    last_health_check TIMESTAMP WITH TIME ZONE,
                    response_time_avg DOUBLE PRECISION DEFAULT 0.0,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """
            
            # Execute schema creation
            async with self.pool.acquire() as conn:
                await conn.execute(schema_sql)
            
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise


class DatabaseIntegrationAdapter:
    """
    Adapter to integrate real database client with existing orchestration system.
    """
    
    def __init__(self, db_client: RealDatabaseClient):
        self.db_client = db_client
    
    async def execute_postgres_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute real PostgreSQL request for production database operations."""
        try:
            operation = request.get("operation", "")
            
            if operation == "query":
                sql = request.get("sql", "")
                params = request.get("params", [])
                
                results = await self.db_client.execute_query(sql, params)
                
                return {
                    "status": "success",
                    "rows": results,
                    "row_count": len(results),
                    "columns": list(results[0].keys()) if results else [],
                    "execution_time": 0.0  # Could add timing
                }
            
            elif operation == "execute":
                sql = request.get("sql", "")
                params = request.get("params", [])
                
                affected_rows = await self.db_client.execute_command(sql, params)
                
                return {
                    "status": "success",
                    "rows_affected": affected_rows,
                    "message": f"Command executed successfully"
                }
            
            elif operation == "log_event":
                event_type = request.get("event_type", "")
                event_data = request.get("event_data", {})
                
                success = await self.db_client.log_orchestration_event(event_type, event_data)
                
                return {
                    "status": "success" if success else "error",
                    "message": "Event logged" if success else "Failed to log event"
                }
            
            elif operation == "store_task_result":
                task_id = request.get("task_id", "")
                result_data = request.get("result_data", {})
                
                success = await self.db_client.store_task_result(task_id, result_data)
                
                return {
                    "status": "success" if success else "error",
                    "message": "Task result stored" if success else "Failed to store task result"
                }
            
            elif operation == "get_task_history":
                limit = request.get("limit", 100)
                
                history = await self.db_client.get_task_history(limit)
                
                return {
                    "status": "success",
                    "task_history": history,
                    "count": len(history)
                }
            
            else:
                return {
                    "status": "error",
                    "error": f"Unknown PostgreSQL operation: {operation}"
                }
                
        except Exception as e:
            logger.error(f"PostgreSQL request execution failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


# Integration Configuration
DATABASE_CONFIG = {
    "connection_string": "postgresql://postgres:postgres@localhost:54321/pygent_factory",
    "pool_size": 20,
    "timeout": 30
}


async def create_real_database_client() -> RealDatabaseClient:
    """Factory function to create real database client."""
    client = RealDatabaseClient(
        DATABASE_CONFIG["connection_string"],
        DATABASE_CONFIG["pool_size"]
    )
    
    success = await client.connect()
    if not success:
        raise ConnectionError("Failed to connect to PostgreSQL database")
    
    return client