"""
Embedding MCP Server - Standalone embedding service with OpenAI-compatible API

This server transforms PyGent Factory's internal EmbeddingService into a standalone,
interoperable MCP server that can be used by any agent or application.

Features:
- OpenAI-compatible /v1/embeddings endpoint
- Multiple embedding providers (OpenAI, SentenceTransformer, Ollama)
- Automatic provider fallback and selection
- Built-in caching and performance optimization
- Comprehensive error handling and logging
- Health monitoring and metrics
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import PyGent Factory components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embedding import EmbeddingService, get_embedding_service
from config.settings import get_settings, Settings

logger = logging.getLogger(__name__)


# OpenAI-compatible API models
class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request"""
    input: Union[str, List[str]] = Field(..., description="Text to embed")
    model: Optional[str] = Field(default="text-embedding-ada-002", description="Model to use")
    encoding_format: Optional[str] = Field(default="float", description="Encoding format")
    dimensions: Optional[int] = Field(default=None, description="Number of dimensions")
    user: Optional[str] = Field(default=None, description="User identifier")


class EmbeddingData(BaseModel):
    """Individual embedding data"""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Usage statistics"""
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class ServiceInfo(BaseModel):
    """Service information"""
    name: str = "Embedding MCP Server"
    version: str = "1.0.0"
    description: str = "OpenAI-compatible embedding service"


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    service: ServiceInfo = ServiceInfo()
    providers: Dict[str, Any] = {}
    performance: Dict[str, Any] = {}


class EmbeddingMCPServer:
    """
    Standalone Embedding MCP Server
    
    Provides OpenAI-compatible embedding API using PyGent Factory's
    EmbeddingService infrastructure.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.embedding_service: Optional[EmbeddingService] = None
        self.start_time = time.time()
        self.request_count = 0
        self.total_processing_time = 0.0
        
        # Performance tracking
        self.stats = {
            'requests_processed': 0,
            'total_embeddings_generated': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'provider_usage': {},
            'error_count': 0
        }
    
    async def initialize(self) -> bool:
        """Initialize the embedding service"""
        try:
            self.embedding_service = get_embedding_service(self.settings)
            
            # Test the service
            test_result = await self.embedding_service.generate_embedding("test")
            if not test_result or not test_result.embedding:
                raise Exception("Embedding service test failed")
            
            logger.info("Embedding MCP Server initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            return False
    
    async def generate_embeddings(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings with OpenAI-compatible response"""
        start_time = time.time()
        
        try:
            # Normalize input to list
            if isinstance(request.input, str):
                texts = [request.input]
            else:
                texts = request.input
            
            # Validate input
            if not texts or any(not text.strip() for text in texts):
                raise HTTPException(status_code=400, detail="Invalid input: empty text provided")
            
            if len(texts) > 100:  # Reasonable batch limit
                raise HTTPException(status_code=400, detail="Too many inputs: maximum 100 texts per request")
            
            # Generate embeddings
            embeddings_data = []
            total_tokens = 0
            
            for i, text in enumerate(texts):
                result = await self.embedding_service.generate_embedding(text)
                
                if not result or not result.embedding:
                    raise HTTPException(status_code=500, detail=f"Failed to generate embedding for text {i}")
                
                embeddings_data.append(EmbeddingData(
                    embedding=result.embedding,
                    index=i
                ))
                
                # Estimate tokens (rough approximation)
                total_tokens += len(text.split()) * 1.3  # Approximate token count
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['requests_processed'] += 1
            self.stats['total_embeddings_generated'] += len(texts)
            self.total_processing_time += processing_time
            self.stats['average_response_time'] = self.total_processing_time / self.stats['requests_processed']
            
            # Track provider usage
            provider = self.embedding_service.get_current_provider()
            self.stats['provider_usage'][provider] = self.stats['provider_usage'].get(provider, 0) + 1
            
            return EmbeddingResponse(
                data=embeddings_data,
                model=request.model or "text-embedding-ada-002",
                usage=EmbeddingUsage(
                    prompt_tokens=int(total_tokens),
                    total_tokens=int(total_tokens)
                )
            )
            
        except HTTPException:
            self.stats['error_count'] += 1
            raise
        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    async def get_health(self) -> HealthResponse:
        """Get server health status"""
        try:
            # Check if embedding service is initialized (no actual embedding test to avoid delays)
            service_healthy = self.embedding_service is not None
            
            # Get provider information
            providers = {}
            if self.embedding_service:
                providers = {
                    'current_provider': self.embedding_service.get_current_provider(),
                    'available_providers': list(self.embedding_service.providers.keys()),
                    'provider_count': len(self.embedding_service.providers)
                }
            
            # Performance metrics
            uptime = time.time() - self.start_time
            performance = {
                'uptime_seconds': round(uptime, 2),
                'request_count': self.stats['requests_processed'],
                'avg_response_time': round(self.stats['average_response_time'] * 1000, 2),
                'cache_hit_rate': round(self.stats.get('cache_hits', 0) / max(self.stats['requests_processed'], 1) * 100, 2),
                'total_embeddings': self.stats['total_embeddings_generated'],
                'error_rate': round(self.stats['error_count'] / max(self.stats['requests_processed'], 1) * 100, 2)
            }
            
            return HealthResponse(
                status="healthy" if service_healthy else "unhealthy",
                timestamp=datetime.utcnow().isoformat(),
                providers=providers,
                performance=performance
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.utcnow().isoformat(),
                providers={},
                performance={}
            )


# Global server instance
embedding_server: Optional[EmbeddingMCPServer] = None


async def get_embedding_server() -> EmbeddingMCPServer:
    """Get the global embedding server instance"""
    global embedding_server
    if embedding_server is None:
        raise HTTPException(status_code=503, detail="Embedding server not initialized")
    return embedding_server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global embedding_server
    
    # Startup
    logger.info("Starting Embedding MCP Server...")
    embedding_server = EmbeddingMCPServer()
    
    if not await embedding_server.initialize():
        logger.error("Failed to initialize embedding server")
        raise Exception("Server initialization failed")
    
    logger.info("Embedding MCP Server started successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down Embedding MCP Server...")


# Create FastAPI application
app = FastAPI(
    title="Embedding MCP Server",
    description="OpenAI-compatible embedding service powered by PyGent Factory",
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


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    server: EmbeddingMCPServer = Depends(get_embedding_server)
) -> EmbeddingResponse:
    """
    Create embeddings for the given input text(s).

    Compatible with OpenAI's embeddings API.
    """
    return await server.generate_embeddings(request)


@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings_legacy(
    request: EmbeddingRequest,
    server: EmbeddingMCPServer = Depends(get_embedding_server)
) -> EmbeddingResponse:
    """
    Legacy endpoint for OpenAI SDK compatibility.

    Some versions of the OpenAI SDK use /embeddings instead of /v1/embeddings.
    """
    return await server.generate_embeddings(request)


@app.get("/health", response_model=HealthResponse)
async def health_check(
    server: EmbeddingMCPServer = Depends(get_embedding_server)
) -> HealthResponse:
    """Get server health status and metrics"""
    return await server.get_health()


@app.get("/")
async def root():
    """Root endpoint with server information"""
    return {
        "service": "Embedding MCP Server",
        "version": "1.0.0",
        "description": "OpenAI-compatible embedding service powered by PyGent Factory",
        "endpoints": {
            "embeddings": "/v1/embeddings",
            "health": "/health"
        }
    }


def main(host: str = "0.0.0.0", port: int = 8001):
    """Run the embedding MCP server"""
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
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8001
    
    main(host, port)
