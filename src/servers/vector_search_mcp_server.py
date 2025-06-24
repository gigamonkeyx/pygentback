#!/usr/bin/env python3
"""
Vector Search MCP Server

A standalone service for comprehensive vector operations including:
- Multi-backend vector storage (PostgreSQL, FAISS, ChromaDB)
- Semantic search and similarity matching
- Collection management and optimization
- Performance monitoring and health checks

Compatible with PyGent Factory ecosystem and external clients.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import json
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import PyGent Factory components
import sys
import os
# Add the src directory to the path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

from storage.vector_store import VectorStoreManager, VectorDocument, VectorQuery, VectorSearchResult
from utils.embedding import get_embedding_service
from config.settings import get_settings

# Import distance metric enum
try:
    from storage.vector.base import DistanceMetric
except ImportError:
    # Fallback enum definition
    from enum import Enum
    class DistanceMetric(Enum):
        COSINE = "cosine"
        EUCLIDEAN = "euclidean"
        DOT_PRODUCT = "dot_product"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic Models
class DocumentAddRequest(BaseModel):
    """Request to add documents to a collection"""
    collection: str = Field(..., description="Collection name")
    documents: List[Dict[str, Any]] = Field(..., description="Documents to add")
    generate_embeddings: bool = Field(default=True, description="Generate embeddings automatically")


class DocumentSearchRequest(BaseModel):
    """Request to search for similar documents"""
    collection: str = Field(..., description="Collection name to search")
    query_text: Optional[str] = Field(default=None, description="Text query for semantic search")
    query_vector: Optional[List[float]] = Field(default=None, description="Vector query for direct search")
    limit: int = Field(default=10, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.0, description="Minimum similarity threshold")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter conditions")


class CollectionCreateRequest(BaseModel):
    """Request to create a new collection"""
    name: str = Field(..., description="Collection name")
    dimension: int = Field(..., description="Vector dimension")
    distance_metric: str = Field(default="cosine", description="Distance metric: cosine, euclidean, dot_product")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Collection metadata")
    store_name: Optional[str] = Field(default=None, description="Vector store name")


class CollectionInfo(BaseModel):
    """Collection information"""
    name: str
    dimension: int
    distance_metric: str
    document_count: int
    metadata: Dict[str, Any]
    store_name: str


class SearchResult(BaseModel):
    """Search result"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    collection: str


class VectorOperationResult(BaseModel):
    """Result of vector operation"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    service: Dict[str, str] = {
        "name": "Vector Search MCP Server",
        "version": "1.0.0",
        "description": "Multi-backend vector search and storage service"
    }
    stores: Dict[str, Any] = {}
    performance: Dict[str, Any] = {}


class VectorSearchServer:
    """Core vector search server implementation"""
    
    def __init__(self):
        self.start_time = time.time()
        self.stats = {
            'searches_performed': 0,
            'documents_added': 0,
            'collections_created': 0,
            'total_search_time': 0.0,
            'total_add_time': 0.0,
            'error_count': 0,
            'average_search_time': 0.0
        }
        
        self.vector_manager = None
        self.embedding_service = None
        self.settings = None
    
    async def initialize(self) -> bool:
        """Initialize the vector search server"""
        try:
            logger.info("Initializing Vector Search MCP Server...")
            
            # Initialize settings
            self.settings = get_settings()
            
            # Initialize embedding service
            self.embedding_service = get_embedding_service()
            if not self.embedding_service:
                logger.error("Failed to initialize embedding service")
                return False
            
            # Initialize vector store manager
            self.vector_manager = VectorStoreManager(self.settings)
            await self.vector_manager.initialize()
            
            # Test vector operations
            test_result = await self._test_vector_capabilities()
            if not test_result:
                logger.error("Vector capabilities test failed")
                return False
            
            logger.info("Vector Search MCP Server initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Vector Search Server: {e}")
            return False
    
    async def _test_vector_capabilities(self) -> bool:
        """Test core vector capabilities"""
        try:
            # Test embedding generation
            test_embedding = await self.embedding_service.generate_embedding("test vector capabilities")
            if not test_embedding or not test_embedding.embedding:
                logger.error("Embedding generation test failed")
                return False
            
            # Test vector store connectivity
            stores = self.vector_manager.list_stores()
            if not stores:
                logger.warning("No vector stores available")
            
            return True
            
        except Exception as e:
            logger.error(f"Vector capabilities test failed: {e}")
            return False
    
    async def create_collection(self, request: CollectionCreateRequest) -> VectorOperationResult:
        """Create a new vector collection"""
        start_time = time.time()
        
        try:
            logger.info(f"Creating collection: {request.name}")
            
            # Validate distance metric
            try:
                distance_metric = DistanceMetric(request.distance_metric.lower())
            except ValueError:
                return VectorOperationResult(
                    success=False,
                    message=f"Invalid distance metric: {request.distance_metric}",
                    processing_time=time.time() - start_time
                )
            
            # Create collection
            success = await self.vector_manager.create_collection(
                collection_name=request.name,
                dimension=request.dimension,
                distance_metric=distance_metric,
                metadata=request.metadata or {},
                store_name=request.store_name
            )
            
            processing_time = time.time() - start_time
            
            if success:
                self.stats['collections_created'] += 1
                return VectorOperationResult(
                    success=True,
                    message=f"Collection '{request.name}' created successfully",
                    data={
                        "collection_name": request.name,
                        "dimension": request.dimension,
                        "distance_metric": request.distance_metric,
                        "store_name": request.store_name or "default"
                    },
                    processing_time=processing_time
                )
            else:
                self.stats['error_count'] += 1
                return VectorOperationResult(
                    success=False,
                    message=f"Failed to create collection '{request.name}'",
                    processing_time=processing_time
                )
                
        except Exception as e:
            self.stats['error_count'] += 1
            processing_time = time.time() - start_time
            logger.error(f"Collection creation failed: {e}")
            return VectorOperationResult(
                success=False,
                message=f"Collection creation error: {str(e)}",
                processing_time=processing_time
            )
    
    async def add_documents(self, request: DocumentAddRequest) -> VectorOperationResult:
        """Add documents to a collection"""
        start_time = time.time()
        
        try:
            logger.info(f"Adding {len(request.documents)} documents to collection: {request.collection}")
            
            # Prepare vector documents
            vector_documents = []
            
            for doc_data in request.documents:
                # Generate embedding if requested and not provided
                embedding = None
                if request.generate_embeddings:
                    content = doc_data.get('content', '')
                    if content:
                        embedding_result = await self.embedding_service.generate_embedding(content)
                        if embedding_result and embedding_result.embedding:
                            embedding = embedding_result.embedding
                        else:
                            logger.warning(f"Failed to generate embedding for document")
                
                # Use provided embedding if available
                if 'embedding' in doc_data and doc_data['embedding']:
                    embedding = doc_data['embedding']
                
                if not embedding:
                    logger.warning("No embedding available for document, skipping")
                    continue
                
                # Create vector document
                vector_doc = VectorDocument(
                    id=doc_data.get('id', str(uuid.uuid4())),
                    content=doc_data.get('content', ''),
                    embedding=embedding,
                    metadata=doc_data.get('metadata', {}),
                    collection=request.collection
                )
                vector_documents.append(vector_doc)
            
            if not vector_documents:
                return VectorOperationResult(
                    success=False,
                    message="No valid documents to add (missing embeddings)",
                    processing_time=time.time() - start_time
                )
            
            # Add documents to vector store
            added_ids = await self.vector_manager.add_documents(vector_documents)
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats['documents_added'] += len(added_ids)
            self.stats['total_add_time'] += processing_time
            
            return VectorOperationResult(
                success=True,
                message=f"Added {len(added_ids)} documents to collection '{request.collection}'",
                data={
                    "collection": request.collection,
                    "added_count": len(added_ids),
                    "document_ids": added_ids
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            self.stats['error_count'] += 1
            processing_time = time.time() - start_time
            logger.error(f"Document addition failed: {e}")
            return VectorOperationResult(
                success=False,
                message=f"Document addition error: {str(e)}",
                processing_time=processing_time
            )
    
    async def search_documents(self, request: DocumentSearchRequest) -> List[SearchResult]:
        """Search for similar documents"""
        start_time = time.time()
        
        try:
            logger.info(f"Searching collection: {request.collection}")
            
            # Prepare query vector
            query_vector = None
            
            if request.query_text:
                # Generate embedding from text
                embedding_result = await self.embedding_service.generate_embedding(request.query_text)
                if embedding_result and embedding_result.embedding:
                    query_vector = embedding_result.embedding
                else:
                    raise ValueError("Failed to generate embedding for query text")
            elif request.query_vector:
                query_vector = request.query_vector
            else:
                raise ValueError("Either query_text or query_vector must be provided")
            
            # Create vector query
            vector_query = VectorQuery(
                query_vector=query_vector,
                collection=request.collection,
                limit=request.limit,
                similarity_threshold=request.similarity_threshold,
                metadata_filter=request.metadata_filter
            )
            
            # Perform search
            search_results = await self.vector_manager.search_similar(vector_query)
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats['searches_performed'] += 1
            self.stats['total_search_time'] += processing_time
            self.stats['average_search_time'] = (
                self.stats['total_search_time'] / self.stats['searches_performed']
            )
            
            # Convert to API format
            results = []
            for result in search_results:
                search_result = SearchResult(
                    document_id=result.document_id,
                    content=result.content,
                    metadata=result.metadata,
                    similarity_score=result.similarity_score,
                    collection=request.collection
                )
                results.append(search_result)
            
            logger.info(f"Search completed: {len(results)} results in {processing_time:.3f}s")
            return results
            
        except Exception as e:
            self.stats['error_count'] += 1
            processing_time = time.time() - start_time
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    
    async def list_collections(self, store_name: Optional[str] = None) -> List[CollectionInfo]:
        """List all collections"""
        try:
            collections = await self.vector_manager.list_collections(store_name)
            
            collection_infos = []
            for collection_name in collections:
                # Get collection stats
                stats = await self.vector_manager.get_collection_stats(collection_name, store_name)
                
                collection_info = CollectionInfo(
                    name=collection_name,
                    dimension=stats.get('dimension', 0),
                    distance_metric=stats.get('distance_metric', 'cosine'),
                    document_count=stats.get('document_count', 0),
                    metadata=stats.get('metadata', {}),
                    store_name=store_name or 'default'
                )
                collection_infos.append(collection_info)
            
            return collection_infos
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")
    
    async def delete_collection(self, collection_name: str, store_name: Optional[str] = None) -> VectorOperationResult:
        """Delete a collection"""
        start_time = time.time()
        
        try:
            success = await self.vector_manager.delete_collection(collection_name, store_name)
            processing_time = time.time() - start_time
            
            if success:
                return VectorOperationResult(
                    success=True,
                    message=f"Collection '{collection_name}' deleted successfully",
                    processing_time=processing_time
                )
            else:
                return VectorOperationResult(
                    success=False,
                    message=f"Failed to delete collection '{collection_name}'",
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Collection deletion failed: {e}")
            return VectorOperationResult(
                success=False,
                message=f"Collection deletion error: {str(e)}",
                processing_time=processing_time
            )
    
    async def get_health(self) -> HealthResponse:
        """Get server health status"""
        try:
            uptime = time.time() - self.start_time
            
            # Get vector store health
            stores_health = {}
            if self.vector_manager:
                for store_name in self.vector_manager.list_stores():
                    store = self.vector_manager.get_store(store_name)
                    if store:
                        store_health = await store.health_check()
                        stores_health[store_name] = store_health
            
            performance = {
                'uptime_seconds': round(uptime, 2),
                'searches_performed': self.stats['searches_performed'],
                'documents_added': self.stats['documents_added'],
                'collections_created': self.stats['collections_created'],
                'average_search_time_ms': round(self.stats['average_search_time'] * 1000, 2),
                'error_rate': round(
                    self.stats['error_count'] / max(
                        self.stats['searches_performed'] + self.stats['documents_added'], 1
                    ) * 100, 2
                )
            }
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.utcnow().isoformat(),
                stores=stores_health,
                performance=performance
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.utcnow().isoformat(),
                stores={},
                performance={}
            )


# Global server instance
vector_server: Optional[VectorSearchServer] = None


async def get_vector_server() -> VectorSearchServer:
    """Get the global vector search server instance"""
    global vector_server
    if vector_server is None:
        raise HTTPException(status_code=503, detail="Vector search server not initialized")
    return vector_server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global vector_server
    
    # Startup
    logger.info("Starting Vector Search MCP Server...")
    vector_server = VectorSearchServer()
    
    if not await vector_server.initialize():
        logger.error("Failed to initialize vector search server")
        raise Exception("Server initialization failed")
    
    logger.info("Vector Search MCP Server started successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down Vector Search MCP Server...")


# Create FastAPI application
app = FastAPI(
    title="Vector Search MCP Server",
    description="Multi-backend vector search and storage service with semantic search capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/collections", response_model=VectorOperationResult)
async def create_collection(
    request: CollectionCreateRequest,
    server: VectorSearchServer = Depends(get_vector_server)
) -> VectorOperationResult:
    """
    Create a new vector collection.
    
    Creates a collection with specified dimension and distance metric
    for storing and searching vector embeddings.
    """
    return await server.create_collection(request)


@app.get("/v1/collections", response_model=List[CollectionInfo])
async def list_collections(
    store_name: Optional[str] = Query(default=None, description="Vector store name"),
    server: VectorSearchServer = Depends(get_vector_server)
) -> List[CollectionInfo]:
    """
    List all vector collections.
    
    Returns information about all collections including document counts
    and configuration details.
    """
    return await server.list_collections(store_name)


@app.delete("/v1/collections/{collection_name}", response_model=VectorOperationResult)
async def delete_collection(
    collection_name: str,
    store_name: Optional[str] = Query(default=None, description="Vector store name"),
    server: VectorSearchServer = Depends(get_vector_server)
) -> VectorOperationResult:
    """
    Delete a vector collection.
    
    Permanently removes a collection and all its documents.
    """
    return await server.delete_collection(collection_name, store_name)


@app.post("/v1/collections/{collection_name}/documents", response_model=VectorOperationResult)
async def add_documents(
    collection_name: str,
    request: DocumentAddRequest,
    server: VectorSearchServer = Depends(get_vector_server)
) -> VectorOperationResult:
    """
    Add documents to a collection.
    
    Adds documents with automatic embedding generation or uses provided embeddings.
    """
    # Override collection name from URL
    request.collection = collection_name
    return await server.add_documents(request)


@app.post("/v1/collections/{collection_name}/search", response_model=List[SearchResult])
async def search_documents(
    collection_name: str,
    request: DocumentSearchRequest,
    server: VectorSearchServer = Depends(get_vector_server)
) -> List[SearchResult]:
    """
    Search for similar documents in a collection.
    
    Performs semantic search using text queries or direct vector queries
    with configurable similarity thresholds and metadata filtering.
    """
    # Override collection name from URL
    request.collection = collection_name
    return await server.search_documents(request)


@app.get("/health", response_model=HealthResponse)
async def health_check(
    server: VectorSearchServer = Depends(get_vector_server)
) -> HealthResponse:
    """Get server health status and performance metrics"""
    return await server.get_health()


@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "service": "Vector Search MCP Server",
        "version": "1.0.0",
        "description": "Multi-backend vector search and storage service with semantic search capabilities",
        "endpoints": {
            "collections": "/v1/collections",
            "create_collection": "/v1/collections",
            "add_documents": "/v1/collections/{collection_name}/documents",
            "search": "/v1/collections/{collection_name}/search",
            "health": "/health"
        },
        "capabilities": [
            "Multi-backend vector storage",
            "Semantic search and similarity matching",
            "Collection management",
            "Automatic embedding generation",
            "Metadata filtering",
            "Performance optimization"
        ]
    }


def main(host: str = "0.0.0.0", port: int = 8004):
    """Run the vector search MCP server"""
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8004
    
    main(host, port)
