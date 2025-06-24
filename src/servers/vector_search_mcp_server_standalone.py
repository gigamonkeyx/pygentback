#!/usr/bin/env python3
"""
Vector Search MCP Server (Standalone)

A standalone service for vector operations with simplified dependencies.
Focuses on core vector search functionality without complex modular imports.
"""

import asyncio
import logging
import time
import uuid
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic Models
class DocumentAddRequest(BaseModel):
    """Request to add documents to a collection"""
    collection: str = Field(..., description="Collection name")
    documents: List[Dict[str, Any]] = Field(..., description="Documents to add")


class DocumentSearchRequest(BaseModel):
    """Request to search for similar documents"""
    collection: str = Field(..., description="Collection name to search")
    query_text: str = Field(..., description="Text query for semantic search")
    limit: int = Field(default=10, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.0, description="Minimum similarity threshold")


class CollectionCreateRequest(BaseModel):
    """Request to create a new collection"""
    name: str = Field(..., description="Collection name")
    dimension: int = Field(default=384, description="Vector dimension")


class SearchResult(BaseModel):
    """Search result"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float


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
        "description": "Standalone vector search service"
    }
    performance: Dict[str, Any] = {}


class SimpleVectorStore:
    """Simple in-memory vector store for demonstration"""
    
    def __init__(self):
        self.collections = {}  # collection_name -> {"documents": [], "dimension": int}
        self.storage_dir = Path("data/vector_storage")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    async def create_collection(self, name: str, dimension: int) -> bool:
        """Create a new collection"""
        try:
            if name in self.collections:
                return False
            
            self.collections[name] = {
                "documents": [],
                "dimension": dimension,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Persist to disk
            await self._save_collection(name)
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            return False
    
    async def add_documents(self, collection: str, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to a collection"""
        try:
            if collection not in self.collections:
                raise ValueError(f"Collection {collection} does not exist")
            
            added_ids = []
            for doc in documents:
                doc_id = doc.get('id', str(uuid.uuid4()))
                
                # Validate embedding dimension
                embedding = doc.get('embedding', [])
                expected_dim = self.collections[collection]['dimension']
                
                if len(embedding) != expected_dim:
                    logger.warning(f"Document {doc_id} embedding dimension mismatch: {len(embedding)} != {expected_dim}")
                    continue
                
                document_entry = {
                    'id': doc_id,
                    'content': doc.get('content', ''),
                    'embedding': embedding,
                    'metadata': doc.get('metadata', {}),
                    'added_at': datetime.utcnow().isoformat()
                }
                
                self.collections[collection]['documents'].append(document_entry)
                added_ids.append(doc_id)
            
            # Persist to disk
            await self._save_collection(collection)
            return added_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to {collection}: {e}")
            return []
    
    async def search_similar(self, collection: str, query_embedding: List[float], 
                           limit: int = 10, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if collection not in self.collections:
                return []
            
            documents = self.collections[collection]['documents']
            if not documents:
                return []
            
            query_vector = np.array(query_embedding, dtype=np.float32)
            results = []
            
            for doc in documents:
                doc_vector = np.array(doc['embedding'], dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = np.dot(query_vector, doc_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
                )
                
                if similarity >= threshold:
                    results.append({
                        'document_id': doc['id'],
                        'content': doc['content'],
                        'metadata': doc['metadata'],
                        'similarity_score': float(similarity)
                    })
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Search failed in {collection}: {e}")
            return []
    
    async def list_collections(self) -> List[str]:
        """List all collections"""
        return list(self.collections.keys())
    
    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get collection statistics"""
        if collection not in self.collections:
            return {}
        
        coll_data = self.collections[collection]
        return {
            'name': collection,
            'dimension': coll_data['dimension'],
            'document_count': len(coll_data['documents']),
            'created_at': coll_data.get('created_at')
        }
    
    async def _save_collection(self, collection: str):
        """Save collection to disk"""
        try:
            file_path = self.storage_dir / f"{collection}.json"
            with open(file_path, 'w') as f:
                json.dump(self.collections[collection], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save collection {collection}: {e}")
    
    async def _load_collections(self):
        """Load collections from disk"""
        try:
            for file_path in self.storage_dir.glob("*.json"):
                collection_name = file_path.stem
                with open(file_path, 'r') as f:
                    self.collections[collection_name] = json.load(f)
                logger.info(f"Loaded collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to load collections: {e}")


class ProductionEmbeddingService:
    """Production embedding service using real PyGent Factory EmbeddingService"""

    def __init__(self):
        self.dimension = 384
        self._embedding_service = None
        self._fallback_model = None
        self._initialize_service()

    def _initialize_service(self):
        """Initialize the real embedding service"""
        try:
            # Try to import and use PyGent Factory's real embedding service
            from src.ai.embeddings.core import EmbeddingService
            from src.core.config import get_settings

            settings = get_settings()
            self._embedding_service = EmbeddingService(settings)
            logger.info("Production embedding service initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize production embedding service: {e}")
            logger.info("Initializing fast local embedding fallback")
            self._initialize_fallback()

    def _initialize_fallback(self):
        """Initialize fast local embedding fallback"""
        try:
            # Try sentence-transformers for fast, quality embeddings
            from sentence_transformers import SentenceTransformer
            self._fallback_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimension = 384  # all-MiniLM-L6-v2 dimension
            logger.info("Sentence-transformers fallback initialized")

        except ImportError:
            logger.warning("Sentence-transformers not available, using deterministic fallback")
            self._fallback_model = None

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate a real embedding using production service"""
        try:
            # First try: Use real PyGent Factory embedding service
            if self._embedding_service:
                result = await self._embedding_service.generate_embedding(text)
                if result and result.embedding:
                    return result.embedding

            # Second try: Use sentence-transformers fallback
            if self._fallback_model:
                embedding = self._fallback_model.encode(text, convert_to_tensor=False)
                return embedding.tolist()

            # Final fallback: Fast deterministic embedding (no artificial delays)
            import hashlib
            import numpy as np

            # Create fast deterministic embedding from text hash
            text_hash = hashlib.sha256(text.encode()).digest()
            # Convert to float array and normalize
            embedding = np.frombuffer(text_hash, dtype=np.uint8).astype(np.float32)
            embedding = embedding / 255.0  # Normalize to 0-1

            # Resize to target dimension
            if len(embedding) > self.dimension:
                embedding = embedding[:self.dimension]
            else:
                # Repeat pattern to reach target dimension
                repeats = (self.dimension // len(embedding)) + 1
                embedding = np.tile(embedding, repeats)[:self.dimension]

            # Normalize to unit vector for better similarity calculations
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vector as last resort
            return [0.0] * self.dimension


class VectorSearchServer:
    """Standalone vector search server"""
    
    def __init__(self):
        self.start_time = time.time()
        self.stats = {
            'searches_performed': 0,
            'documents_added': 0,
            'collections_created': 0,
            'total_search_time': 0.0,
            'error_count': 0
        }
        
        self.vector_store = SimpleVectorStore()
        self.embedding_service = ProductionEmbeddingService()
    
    async def initialize(self) -> bool:
        """Initialize the server"""
        try:
            logger.info("Initializing Vector Search MCP Server...")
            
            # Load existing collections
            await self.vector_store._load_collections()
            
            logger.info("Vector Search MCP Server initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            return False
    
    async def create_collection(self, request: CollectionCreateRequest) -> VectorOperationResult:
        """Create a new collection"""
        start_time = time.time()
        
        try:
            success = await self.vector_store.create_collection(request.name, request.dimension)
            processing_time = time.time() - start_time
            
            if success:
                self.stats['collections_created'] += 1
                return VectorOperationResult(
                    success=True,
                    message=f"Collection '{request.name}' created successfully",
                    data={"collection_name": request.name, "dimension": request.dimension},
                    processing_time=processing_time
                )
            else:
                return VectorOperationResult(
                    success=False,
                    message=f"Collection '{request.name}' already exists",
                    processing_time=processing_time
                )
                
        except Exception as e:
            self.stats['error_count'] += 1
            processing_time = time.time() - start_time
            return VectorOperationResult(
                success=False,
                message=f"Error creating collection: {str(e)}",
                processing_time=processing_time
            )
    
    async def add_documents(self, collection: str, request: DocumentAddRequest) -> VectorOperationResult:
        """Add documents to a collection"""
        start_time = time.time()
        
        try:
            # Generate embeddings for documents
            documents_with_embeddings = []
            
            for doc in request.documents:
                content = doc.get('content', '')
                if content:
                    embedding = await self.embedding_service.generate_embedding(content)
                    doc_with_embedding = {
                        'id': doc.get('id', str(uuid.uuid4())),
                        'content': content,
                        'embedding': embedding,
                        'metadata': doc.get('metadata', {})
                    }
                    documents_with_embeddings.append(doc_with_embedding)
            
            added_ids = await self.vector_store.add_documents(collection, documents_with_embeddings)
            processing_time = time.time() - start_time
            
            self.stats['documents_added'] += len(added_ids)
            
            return VectorOperationResult(
                success=True,
                message=f"Added {len(added_ids)} documents to collection '{collection}'",
                data={"collection": collection, "added_count": len(added_ids)},
                processing_time=processing_time
            )
            
        except Exception as e:
            self.stats['error_count'] += 1
            processing_time = time.time() - start_time
            return VectorOperationResult(
                success=False,
                message=f"Error adding documents: {str(e)}",
                processing_time=processing_time
            )
    
    async def search_documents(self, request: DocumentSearchRequest) -> List[SearchResult]:
        """Search for similar documents"""
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(request.query_text)
            
            # Search for similar documents
            results = await self.vector_store.search_similar(
                request.collection,
                query_embedding,
                request.limit,
                request.similarity_threshold
            )
            
            processing_time = time.time() - start_time
            self.stats['searches_performed'] += 1
            self.stats['total_search_time'] += processing_time
            
            # Convert to API format
            search_results = []
            for result in results:
                search_result = SearchResult(
                    document_id=result['document_id'],
                    content=result['content'],
                    metadata=result['metadata'],
                    similarity_score=result['similarity_score']
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
    
    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with stats"""
        try:
            collections = await self.vector_store.list_collections()
            collection_info = []
            
            for collection_name in collections:
                stats = await self.vector_store.get_collection_stats(collection_name)
                collection_info.append(stats)
            
            return collection_info
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    async def get_health(self) -> HealthResponse:
        """Get server health status"""
        try:
            uptime = time.time() - self.start_time
            avg_search_time = (
                self.stats['total_search_time'] / max(self.stats['searches_performed'], 1)
            )
            
            performance = {
                'uptime_seconds': round(uptime, 2),
                'searches_performed': self.stats['searches_performed'],
                'documents_added': self.stats['documents_added'],
                'collections_created': self.stats['collections_created'],
                'average_search_time_ms': round(avg_search_time * 1000, 2),
                'error_count': self.stats['error_count']
            }
            
            return HealthResponse(
                status="healthy",
                timestamp=datetime.utcnow().isoformat(),
                performance=performance
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.utcnow().isoformat(),
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
    description="Standalone vector search service with in-memory storage",
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
    """Create a new vector collection"""
    return await server.create_collection(request)


@app.get("/v1/collections")
async def list_collections(
    server: VectorSearchServer = Depends(get_vector_server)
) -> List[Dict[str, Any]]:
    """List all vector collections"""
    return await server.list_collections()


@app.post("/v1/collections/{collection_name}/documents", response_model=VectorOperationResult)
async def add_documents(
    collection_name: str,
    request: DocumentAddRequest,
    server: VectorSearchServer = Depends(get_vector_server)
) -> VectorOperationResult:
    """Add documents to a collection"""
    return await server.add_documents(collection_name, request)


@app.post("/v1/collections/{collection_name}/search", response_model=List[SearchResult])
async def search_documents(
    collection_name: str,
    request: DocumentSearchRequest,
    server: VectorSearchServer = Depends(get_vector_server)
) -> List[SearchResult]:
    """Search for similar documents in a collection"""
    request.collection = collection_name
    return await server.search_documents(request)


@app.get("/health", response_model=HealthResponse)
async def health_check(
    server: VectorSearchServer = Depends(get_vector_server)
) -> HealthResponse:
    """Get server health status"""
    return await server.get_health()


@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "service": "Vector Search MCP Server",
        "version": "1.0.0",
        "description": "Standalone vector search service with in-memory storage",
        "endpoints": {
            "collections": "/v1/collections",
            "add_documents": "/v1/collections/{collection_name}/documents",
            "search": "/v1/collections/{collection_name}/search",
            "health": "/health"
        },
        "capabilities": [
            "In-memory vector storage",
            "Semantic search and similarity matching",
            "Collection management",
            "Production embedding generation",
            "Multi-provider embedding support"
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
