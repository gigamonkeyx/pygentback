"""
FastAPI Main Application

Entry point for the PyGent Factory API server.
Handles application lifecycle, dependency injection, and route configuration.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config.settings import get_settings
from ..database.manager import initialize_database
from ..storage.vector_store_manager import VectorStoreManager
from ..mcp.discovery.manager import MCPDiscoveryManager
from ..memory.manager import MemoryManager
from ..core.agent_factory import AgentFactory
from ..communication.message_bus import MessageBus
from ..protocols.registry import ProtocolRegistry
from ..rag.retrieval_system import RetrievalSystem
from ..search.embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global application state
app_state: Dict[str, Any] = {
    "settings": None,
    "db_manager": None,
    "vector_store_manager": None,
    "memory_manager": None,
    "mcp_manager": None,
    "mcp_discovery_manager": None,
    "mcp_discovery_results": None,
    "agent_factory": None,
    "message_bus": None,
    "protocol_manager": None,
    "retrieval_system": None,
    "embedding_service": None,
    "ollama_manager": None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown of all system components
    in the correct order with proper error handling.
    """
    settings = get_settings()
    
    try:
        logger.info("Starting PyGent Factory application...")

        # Store settings in app state
        app_state["settings"] = settings

        # 0. Initialize Ollama service first (critical dependency)
        logger.info("Initializing Ollama service...")
        from ..core.ollama_manager import get_ollama_manager
        ollama_manager = get_ollama_manager()
        app_state["ollama_manager"] = ollama_manager

        if not await ollama_manager.start():
            logger.error("Failed to start Ollama service - this will affect AI functionality")
            # Continue startup but log the issue
        else:
            logger.info("Ollama service ready")

        # 1. Initialize database
        logger.info("Initializing database...")
        db_manager = await initialize_database(settings)
        app_state["db_manager"] = db_manager

        # 2. Initialize vector store manager
        logger.info("Initializing vector store manager...")
        vector_store_manager = VectorStoreManager(settings)
        await vector_store_manager.initialize()
        app_state["vector_store_manager"] = vector_store_manager

        # 3. Initialize embedding service
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService(settings)
        await embedding_service.initialize()
        app_state["embedding_service"] = embedding_service

        # 4. Initialize memory manager
        logger.info("Initializing memory manager...")
        memory_manager = MemoryManager(
            db_manager=db_manager,
            vector_store_manager=vector_store_manager,
            embedding_service=embedding_service,
            settings=settings
        )
        await memory_manager.initialize()
        app_state["memory_manager"] = memory_manager

        # 5. Initialize MCP manager
        logger.info("Initializing MCP manager...")
        from ..mcp.manager import MCPManager
        mcp_manager = MCPManager(settings)
        await mcp_manager.initialize()
        app_state["mcp_manager"] = mcp_manager

        # 6. Initialize MCP discovery manager
        logger.info("Initializing MCP discovery manager...")
        mcp_discovery_manager = MCPDiscoveryManager(db_manager, settings)
        await mcp_discovery_manager.initialize()
        app_state["mcp_discovery_manager"] = mcp_discovery_manager

        # 7. Initialize message bus
        logger.info("Initializing message bus...")
        message_bus = MessageBus(settings)
        await message_bus.initialize()
        app_state["message_bus"] = message_bus

        # 8. Initialize protocol registry
        logger.info("Initializing protocol registry...")
        protocol_manager = ProtocolRegistry()
        await protocol_manager.initialize()
        app_state["protocol_manager"] = protocol_manager

        # 9. Initialize agent factory
        logger.info("Initializing agent factory...")
        agent_factory = AgentFactory(mcp_manager, memory_manager, settings, ollama_manager)
        app_state["agent_factory"] = agent_factory
        
        # 10. Initialize RAG retrieval system
        logger.info("Initializing RAG retrieval system...")
        retrieval_system = RetrievalSystem(settings)
        await retrieval_system.initialize(vector_store_manager, db_manager)
        app_state["retrieval_system"] = retrieval_system

        # 11. Set all route dependencies now that all components are initialized
        logger.info("Setting route dependencies...")
        
        # Set health dependencies
        from .routes.health import set_health_dependencies
        set_health_dependencies(
            app_state.get("db_manager"),
            app_state.get("agent_factory"),
            app_state.get("memory_manager"),
            app_state.get("mcp_manager"),
            app_state.get("retrieval_system"),
            app_state.get("message_bus"),
            app_state.get("ollama_manager")
        )

        # Set other route dependencies
        from .routes.agents import set_agent_factory
        from .routes.memory import set_memory_manager
        from .routes.mcp import set_mcp_manager
        from .routes.rag import set_retrieval_system
        from .routes.models import set_db_manager

        if app_state.get("agent_factory"):
            set_agent_factory(app_state["agent_factory"])

            # Also set it in the dependencies module for workflows
            try:
                from .dependencies import set_agent_factory as set_deps_agent_factory
                set_deps_agent_factory(app_state["agent_factory"])
            except ImportError:
                logger.warning("Dependencies module not available for agent factory")

        if app_state.get("memory_manager"):
            set_memory_manager(app_state["memory_manager"])

        if app_state.get("mcp_manager"):
            set_mcp_manager(app_state["mcp_manager"])

        if app_state.get("retrieval_system"):
            set_retrieval_system(app_state["retrieval_system"])

        if app_state.get("db_manager"):
            set_db_manager(app_state["db_manager"])

        logger.info("PyGent Factory application started successfully!")
        
        # Application is ready
        yield

    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
    
    finally:
        # Shutdown sequence
        logger.info("Shutting down PyGent Factory application...")
        
        # Shutdown components in reverse order
        if app_state.get("retrieval_system"):
            try:
                await app_state["retrieval_system"].shutdown()
            except Exception as e:
                logger.error(f"Error shutting down retrieval system: {e}")

        if app_state.get("agent_factory"):
            try:
                await app_state["agent_factory"].shutdown()
            except Exception as e:
                logger.error(f"Error shutting down agent factory: {e}")

        if app_state.get("protocol_manager"):
            try:
                await app_state["protocol_manager"].shutdown()
            except Exception as e:
                logger.error(f"Error shutting down protocol manager: {e}")

        if app_state.get("message_bus"):
            try:
                await app_state["message_bus"].shutdown()
            except Exception as e:
                logger.error(f"Error shutting down message bus: {e}")

        if app_state.get("mcp_discovery_manager"):
            try:
                await app_state["mcp_discovery_manager"].shutdown()
            except Exception as e:
                logger.error(f"Error shutting down MCP discovery manager: {e}")

        if app_state.get("mcp_manager"):
            try:
                await app_state["mcp_manager"].shutdown()
            except Exception as e:
                logger.error(f"Error shutting down MCP manager: {e}")

        if app_state.get("memory_manager"):
            try:
                await app_state["memory_manager"].shutdown()
            except Exception as e:
                logger.error(f"Error shutting down memory manager: {e}")

        if app_state.get("embedding_service"):
            try:
                await app_state["embedding_service"].shutdown()
            except Exception as e:
                logger.error(f"Error shutting down embedding service: {e}")

        if app_state.get("vector_store_manager"):
            try:
                await app_state["vector_store_manager"].shutdown()
            except Exception as e:
                logger.error(f"Error shutting down vector store manager: {e}")

        if app_state.get("db_manager"):
            try:
                await app_state["db_manager"].shutdown()
            except Exception as e:
                logger.error(f"Error shutting down database manager: {e}")

        if app_state.get("ollama_manager"):
            try:
                await app_state["ollama_manager"].shutdown()
            except Exception as e:
                logger.error(f"Error shutting down Ollama manager: {e}")

        logger.info("PyGent Factory application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="PyGent Factory API",
    description="Advanced AI Agent Factory with Genetic Algorithms and Multi-Agent Collaboration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from .routes import (
    health,
    agents,
    memory,
    mcp,
    rag,
    models,
    workflows,
    research,
    evolution
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(agents.router, prefix="/agents", tags=["agents"])
app.include_router(memory.router, prefix="/memory", tags=["memory"])
app.include_router(mcp.router, prefix="/mcp", tags=["mcp"])
app.include_router(rag.router, prefix="/rag", tags=["rag"])
app.include_router(models.router, prefix="/models", tags=["models"])
app.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
app.include_router(research.router, prefix="/research", tags=["research"])
app.include_router(evolution.router, prefix="/evolution", tags=["evolution"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Global exception handler caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "PyGent Factory API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=1  # Use 1 worker for development
    )
