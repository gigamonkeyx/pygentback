"""
PostgreSQL Vector Store Implementation

This module provides a PostgreSQL-based vector storage implementation using pgvector.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import json

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from .base import VectorStore, VectorDocument, VectorQuery, VectorSearchResult, DistanceMetric


logger = logging.getLogger(__name__)


class PostgreSQLVectorStore(VectorStore):
    """
    PostgreSQL vector store implementation using pgvector extension.
    
    This implementation provides high-performance vector storage and similarity
    search using PostgreSQL with the pgvector extension.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PostgreSQL vector store.
        
        Args:
            config: Configuration dictionary containing connection parameters
        """
        super().__init__(config)
        
        # Connection configuration
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 5432)
        self.database = config.get("database", "pygent_factory")
        self.username = config.get("username", "postgres")
        self.password = config.get("password", "")
        self.schema = config.get("schema", "vectors")
        
        # Connection pool
        self.pool: Optional[asyncpg.Pool] = None
        self.max_connections = config.get("max_connections", 10)
        self.min_connections = config.get("min_connections", 2)
        
        # Vector configuration
        self.default_dimension = config.get("default_dimension", 1536)
        self.table_prefix = config.get("table_prefix", "vector_")
    
    async def initialize(self) -> None:
        """Initialize the PostgreSQL vector store"""
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                min_size=self.min_connections,
                max_size=self.max_connections,
                init=self._init_connection
            )
            
            # Create schema if it doesn't exist
            async with self.pool.acquire() as conn:
                await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                
                # Create collections metadata table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.collections (
                        name VARCHAR(255) PRIMARY KEY,
                        dimension INTEGER NOT NULL,
                        distance_metric VARCHAR(50) DEFAULT 'cosine',
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index on collections
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_collections_name 
                    ON {self.schema}.collections(name)
                """)
            
            self._initialized = True
            logger.info("PostgreSQL vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL vector store: {str(e)}")
            raise
    
    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Initialize a new connection"""
        await register_vector(conn)
    
    async def close(self) -> None:
        """Close the PostgreSQL vector store connection"""
        if self.pool:
            await self.pool.close()
            self.pool = None
        self._initialized = False
        logger.info("PostgreSQL vector store closed")
    
    async def create_collection(self, collection_name: str, 
                               dimension: int, 
                               distance_metric: DistanceMetric = DistanceMetric.COSINE,
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection"""
        try:
            async with self.pool.acquire() as conn:
                # Insert collection metadata
                await conn.execute(f"""
                    INSERT INTO {self.schema}.collections 
                    (name, dimension, distance_metric, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (name) DO UPDATE SET
                        dimension = EXCLUDED.dimension,
                        distance_metric = EXCLUDED.distance_metric,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, collection_name, dimension, distance_metric.value, 
                json.dumps(metadata or {}))
                
                # Create table for the collection
                table_name = f"{self.table_prefix}{collection_name}"
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.{table_name} (
                        id VARCHAR(255) PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector({dimension}),
                        metadata JSONB DEFAULT '{{}}',
                        title VARCHAR(1000),
                        source VARCHAR(1000),
                        document_type VARCHAR(100),
                        chunk_index INTEGER,
                        chunk_count INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create vector index based on distance metric
                index_name = f"idx_{table_name}_embedding"
                if distance_metric == DistanceMetric.COSINE:
                    index_ops = "vector_cosine_ops"
                elif distance_metric == DistanceMetric.EUCLIDEAN:
                    index_ops = "vector_l2_ops"
                elif distance_metric == DistanceMetric.DOT_PRODUCT:
                    index_ops = "vector_ip_ops"
                else:
                    index_ops = "vector_cosine_ops"  # Default
                
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {self.schema}.{table_name} 
                    USING ivfflat (embedding {index_ops})
                    WITH (lists = 100)
                """)
                
                # Create additional indexes
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_metadata
                    ON {self.schema}.{table_name} USING GIN (metadata)
                """)
                
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_source
                    ON {self.schema}.{table_name}(source)
                """)
            
            logger.info(f"Created collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {str(e)}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            async with self.pool.acquire() as conn:
                # Drop the table
                table_name = f"{self.table_prefix}{collection_name}"
                await conn.execute(f"DROP TABLE IF EXISTS {self.schema}.{table_name}")
                
                # Remove from collections metadata
                await conn.execute(f"""
                    DELETE FROM {self.schema}.collections WHERE name = $1
                """, collection_name)
            
            logger.info(f"Deleted collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {str(e)}")
            return False
    
    async def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT name FROM {self.schema}.collections ORDER BY name
                """)
                return [row["name"] for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            return []
    
    async def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Add documents to the vector store"""
        if not documents:
            return []
        
        added_ids = []
        
        try:
            # Group documents by collection
            collections = {}
            for doc in documents:
                if doc.collection not in collections:
                    collections[doc.collection] = []
                collections[doc.collection].append(doc)
            
            async with self.pool.acquire() as conn:
                for collection_name, docs in collections.items():
                    table_name = f"{self.table_prefix}{collection_name}"
                    
                    # Prepare batch insert
                    values = []
                    for doc in docs:
                        values.append((
                            doc.id,
                            doc.content,
                            doc.embedding,
                            json.dumps(doc.metadata),
                            doc.title,
                            doc.source,
                            doc.document_type,
                            doc.chunk_index,
                            doc.chunk_count,
                            doc.created_at,
                            doc.updated_at
                        ))
                    
                    # Execute batch insert
                    await conn.executemany(f"""
                        INSERT INTO {self.schema}.{table_name}
                        (id, content, embedding, metadata, title, source, 
                         document_type, chunk_index, chunk_count, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            title = EXCLUDED.title,
                            source = EXCLUDED.source,
                            document_type = EXCLUDED.document_type,
                            chunk_index = EXCLUDED.chunk_index,
                            chunk_count = EXCLUDED.chunk_count,
                            updated_at = EXCLUDED.updated_at
                    """, values)
                    
                    added_ids.extend([doc.id for doc in docs])
            
            logger.info(f"Added {len(added_ids)} documents")
            return added_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return []
    
    async def update_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Update existing documents"""
        # For PostgreSQL, update is handled in add_documents with ON CONFLICT
        return await self.add_documents(documents)
    
    async def delete_documents(self, document_ids: List[str], 
                              collection: str = None) -> int:
        """Delete documents from the vector store"""
        if not document_ids:
            return 0
        
        try:
            deleted_count = 0
            
            async with self.pool.acquire() as conn:
                if collection:
                    # Delete from specific collection
                    table_name = f"{self.table_prefix}{collection}"
                    result = await conn.execute(f"""
                        DELETE FROM {self.schema}.{table_name} 
                        WHERE id = ANY($1)
                    """, document_ids)
                    deleted_count = int(result.split()[-1])
                else:
                    # Delete from all collections
                    collections = await self.list_collections()
                    for coll in collections:
                        table_name = f"{self.table_prefix}{coll}"
                        result = await conn.execute(f"""
                            DELETE FROM {self.schema}.{table_name} 
                            WHERE id = ANY($1)
                        """, document_ids)
                        deleted_count += int(result.split()[-1])
            
            logger.info(f"Deleted {deleted_count} documents")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            return 0
    
    async def get_document(self, document_id: str, 
                          collection: str = None) -> Optional[VectorDocument]:
        """Get a document by ID"""
        try:
            async with self.pool.acquire() as conn:
                collections_to_search = [collection] if collection else await self.list_collections()
                
                for coll in collections_to_search:
                    table_name = f"{self.table_prefix}{coll}"
                    row = await conn.fetchrow(f"""
                        SELECT * FROM {self.schema}.{table_name} WHERE id = $1
                    """, document_id)
                    
                    if row:
                        return VectorDocument(
                            id=row["id"],
                            content=row["content"],
                            embedding=list(row["embedding"]) if row["embedding"] else None,
                            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                            collection=coll,
                            title=row["title"],
                            source=row["source"],
                            document_type=row["document_type"],
                            chunk_index=row["chunk_index"],
                            chunk_count=row["chunk_count"],
                            created_at=row["created_at"],
                            updated_at=row["updated_at"]
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {str(e)}")
            return None
    
    async def search_similar(self, query: VectorQuery) -> List[VectorSearchResult]:
        """Search for similar documents"""
        try:
            if not query.query_vector:
                return []
            
            async with self.pool.acquire() as conn:
                table_name = f"{self.table_prefix}{query.collection}"
                
                # Build distance operator based on metric
                if query.distance_metric == DistanceMetric.COSINE:
                    distance_op = "<=>"
                elif query.distance_metric == DistanceMetric.EUCLIDEAN:
                    distance_op = "<->"
                elif query.distance_metric == DistanceMetric.DOT_PRODUCT:
                    distance_op = "<#>"
                else:
                    distance_op = "<=>"  # Default to cosine
                
                # Build WHERE clause for metadata filtering
                where_clause = ""
                params = [query.query_vector]
                param_count = 1
                
                if query.metadata_filter:
                    conditions = []
                    for key, value in query.metadata_filter.items():
                        param_count += 1
                        conditions.append(f"metadata->>'{key}' = ${param_count}")
                        params.append(str(value))
                    
                    if conditions:
                        where_clause = "WHERE " + " AND ".join(conditions)
                
                # Execute similarity search
                sql = f"""
                    SELECT *, embedding {distance_op} $1 as distance
                    FROM {self.schema}.{table_name}
                    {where_clause}
                    ORDER BY embedding {distance_op} $1
                    LIMIT {query.limit}
                    OFFSET {query.offset}
                """
                
                rows = await conn.fetch(sql, *params)
                
                results = []
                for i, row in enumerate(rows):
                    # Calculate similarity score from distance
                    distance = float(row["distance"])
                    if query.distance_metric == DistanceMetric.COSINE:
                        similarity = 1.0 - distance
                    elif query.distance_metric == DistanceMetric.DOT_PRODUCT:
                        similarity = distance  # Higher is better for dot product
                    else:
                        # For Euclidean and Manhattan, convert to similarity
                        similarity = 1.0 / (1.0 + distance)
                    
                    # Skip results below threshold
                    if similarity < query.similarity_threshold:
                        continue
                    
                    document = VectorDocument(
                        id=row["id"],
                        content=row["content"] if query.include_content else "",
                        embedding=list(row["embedding"]) if row["embedding"] else None,
                        metadata=json.loads(row["metadata"]) if query.include_metadata and row["metadata"] else {},
                        collection=query.collection,
                        title=row["title"],
                        source=row["source"],
                        document_type=row["document_type"],
                        chunk_index=row["chunk_index"],
                        chunk_count=row["chunk_count"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"]
                    )
                    
                    result = VectorSearchResult(
                        document=document,
                        similarity_score=similarity,
                        distance=distance,
                        rank=i + 1
                    )
                    
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search similar documents: {str(e)}")
            return []
    
    async def count_documents(self, collection: str = None) -> int:
        """Count documents in a collection"""
        try:
            async with self.pool.acquire() as conn:
                if collection:
                    table_name = f"{self.table_prefix}{collection}"
                    row = await conn.fetchrow(f"""
                        SELECT COUNT(*) as count FROM {self.schema}.{table_name}
                    """)
                    return row["count"]
                else:
                    # Count across all collections
                    total = 0
                    collections = await self.list_collections()
                    for coll in collections:
                        table_name = f"{self.table_prefix}{coll}"
                        row = await conn.fetchrow(f"""
                            SELECT COUNT(*) as count FROM {self.schema}.{table_name}
                        """)
                        total += row["count"]
                    return total
                    
        except Exception as e:
            logger.error(f"Failed to count documents: {str(e)}")
            return 0
