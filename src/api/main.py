"""
FastAPI Application Setup

This module sets up the main FastAPI application for PyGent Factory,
including middleware, routing, error handling, and application lifecycle
management with proper async initialization.
"""

import asyncio
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from ..config.settings import get_settings
from ..database.connection import initialize_database, close_database
from ..storage.vector_store import VectorStoreManager
from ..memory.memory_manager import MemoryManager
from ..mcp.server_registry import MCPServerManager
from ..core.agent_factory import AgentFactory
from ..core.message_system import MessageBus
from ..communication.protocols import ProtocolManager
from ..rag.retrieval_system import RetrievalSystem
from ..utils.embedding import get_embedding_service

logger = logging.getLogger(__name__)

# Global application state
app_state = {
    "db_manager": None,
    "vector_store_manager": None,
    "memory_manager": None,
    "mcp_manager": None,
    "mcp_discovery_results": None,    "agent_factory": None,
    "message_bus": None,
    "protocol_manager": None,
    "retrieval_system": None,
    "embedding_service": None,
    "task_dispatcher": None,
    "orchestration_manager": None,
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
        logger.info("Starting PyGent Factory application...")        # Store settings in app state
        app_state["settings"] = settings

        # 1. Initialize database (both async and sync)
        logger.info("Initializing database...")
        db_manager = await initialize_database(settings)
        app_state["db_manager"] = db_manager
        
        # Initialize synchronous database for legacy services
        from ..database.connection import init_sync_database
        init_sync_database(settings)
        
        # Set database manager in models module
        from .routes.models import set_db_manager
        set_db_manager(db_manager)
        
        # Initialize user service and create default users
        logger.info("Initializing user service...")
        from ..services.user_service import get_user_service
        user_service = get_user_service()
        await user_service.create_default_users()
        app_state["user_service"] = user_service
        
        # 2. Initialize vector store manager
        logger.info("Initializing vector store manager...")
        vector_store_manager = VectorStoreManager(settings, db_manager)
        app_state["vector_store_manager"] = vector_store_manager
        
        # 3. Initialize embedding service
        logger.info("Initializing embedding service...")
        embedding_service = get_embedding_service(settings)
        app_state["embedding_service"] = embedding_service
        
        # 4. Initialize memory manager
        logger.info("Initializing memory manager...")
        memory_manager = MemoryManager(vector_store_manager, settings)
        await memory_manager.start()
        app_state["memory_manager"] = memory_manager
        
        # 5. Initialize MCP server manager
        logger.info("Initializing MCP server manager...")
        mcp_manager = MCPServerManager(settings)
        await mcp_manager.start()
        app_state["mcp_manager"] = mcp_manager
        
        # 5a. Load real MCP servers instead of auto-discovery
        logger.info("Loading real MCP servers...")
        try:
            from ..mcp.real_server_loader import load_real_mcp_servers
            server_results = await load_real_mcp_servers(mcp_manager)
            app_state["mcp_discovery_results"] = server_results

            if server_results.get("success", False):
                logger.info(f"Real MCP servers loaded: {server_results.get('servers_loaded', 0)} servers")
            else:
                logger.warning("Real MCP server loading completed with issues")
        except Exception as e:
            logger.warning(f"Real MCP server loading failed: {e}")
            app_state["mcp_discovery_results"] = {"success": False, "error": str(e)}

        # 6. Initialize message bus
        logger.info("Initializing message bus...")
        message_bus = MessageBus()
        await message_bus.start()
        app_state["message_bus"] = message_bus
        logger.info("Message bus initialized successfully")

        # 7. Initialize protocol manager
        logger.info("Initializing protocol manager...")
        protocol_manager = ProtocolManager(settings)
        await protocol_manager.initialize()
        app_state["protocol_manager"] = protocol_manager          # 8. Initialize agent factory
        logger.info("Initializing agent factory...")
        agent_factory = AgentFactory(
            mcp_manager, 
            memory_manager, 
            settings
        )
        
        # Initialize providers through the agent factory
        await agent_factory.initialize(
            enable_ollama=True,
            enable_openrouter=True
        )
        
        app_state["agent_factory"] = agent_factory
        
        # 8a. Initialize agent service
        logger.info("Initializing agent service...")
        from ..services.agent_service import AgentService, set_agent_service
        agent_service = AgentService(agent_factory)
        set_agent_service(agent_service)
        app_state["agent_service"] = agent_service
        
        # 8b. Initialize document service  
        logger.info("Initializing document service...")
        from ..services.document_service import DocumentService, set_document_service
        document_service = DocumentService()
        set_document_service(document_service)
        app_state["document_service"] = document_service
        
        # 9. Initialize RAG retrieval system
        logger.info("Initializing RAG retrieval system...")
        retrieval_system = RetrievalSystem(settings)
        await retrieval_system.initialize(vector_store_manager, db_manager)
        app_state["retrieval_system"] = retrieval_system        

        # 10. Set all route dependencies now that all components are initialized
        logger.info("Setting route dependencies...")          # Set health dependencies
        from .routes.health import set_health_dependencies
        set_health_dependencies(
            app_state.get("db_manager"),
            app_state.get("agent_factory"),
            app_state.get("memory_manager"),
            app_state.get("mcp_manager"),
            app_state.get("retrieval_system"),
            app_state.get("message_bus")
        )
          # Set other route dependencies
        from .routes.agents import set_agent_factory
        from .routes.memory import set_memory_manager
        from .routes.mcp import set_mcp_manager
        from .routes.rag import set_retrieval_system

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
        try:            # Shutdown in reverse order
            if app_state["agent_factory"]:
                await app_state["agent_factory"].shutdown()
            if app_state["protocol_manager"]:
                await app_state["protocol_manager"].shutdown()
            if app_state["message_bus"]:
                await app_state["message_bus"].stop()
            if app_state.get("orchestration_manager"):
                await app_state["orchestration_manager"].shutdown()
            if app_state["memory_manager"]:
                await app_state["memory_manager"].stop()
            if app_state["mcp_manager"]:
                await app_state["mcp_manager"].stop()
            if app_state["db_manager"]:
                await close_database()

            logger.info("PyGent Factory application shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    settings = get_settings()
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title=settings.app.APP_NAME,
        version=settings.app.APP_VERSION,
        description="PyGent Factory - MCP-Compliant Agent Factory System",
        docs_url="/docs" if settings.app.DEBUG else None,
        redoc_url="/redoc" if settings.app.DEBUG else None,
        openapi_url="/openapi.json" if settings.app.DEBUG else None,
        lifespan=lifespan
    )
    
    # Add middleware
    setup_middleware(app, settings)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    # Include routes
    setup_routes(app)
    
    return app

def is_origin_allowed(origin: str, allowed_patterns: List[str]) -> bool:
    """Check if origin matches any of the allowed patterns (supports wildcards)"""
    for pattern in allowed_patterns:
        if pattern == "*":
            return True
        elif "*" in pattern:
            # Convert wildcard pattern to regex
            regex_pattern = pattern.replace("*", ".*").replace(".", r"\.")
            if re.match(f"^{regex_pattern}$", origin):
                return True
        elif pattern == origin:
            return True
    return False

async def custom_cors_middleware(request: Request, call_next):
    """Custom CORS middleware with wildcard support"""
    
    # Get origin from request
    origin = request.headers.get("origin")
    
    # Process the request
    response = await call_next(request)

    # Get allowed origins from config manager
    from src.config.config_manager import get_config_manager
    config_manager = get_config_manager()
    cors_origins = config_manager.get('security.CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001,http://localhost:5173,http://localhost:5174,http://localhost:5175,https://timpayne.net,https://*.trycloudflare.com')
    allowed_origins = [origin.strip() for origin in cors_origins.split(',')]

    # Check if origin is allowed
    if origin and is_origin_allowed(origin, allowed_origins):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        logger.debug(f"✅ CORS allowed for origin: {origin}")
    else:
        logger.debug(f"❌ CORS denied for origin: {origin}")
    
    # Handle preflight requests
    if request.method == "OPTIONS":
        response.headers["Access-Control-Max-Age"] = "86400"
    
    return response

def setup_middleware(app: FastAPI, settings) -> None:
    """Set up FastAPI middleware"""
    
    # Add custom CORS middleware with wildcard support
    @app.middleware("http")
    async def cors_middleware(request: Request, call_next):
        return await custom_cors_middleware(request, call_next)
    
    # Trusted host middleware (if not in debug mode)
    if not settings.app.DEBUG:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", settings.app.HOST]
        )
    
    # Custom middleware for request logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = asyncio.get_event_loop().time()
        
        # Process request
        response = await call_next(request)
        
        # Log request
        process_time = asyncio.get_event_loop().time() - start_time
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.3f}s"
        )
        
        return response

def setup_exception_handlers(app: FastAPI) -> None:
    """Set up global exception handlers"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "path": str(request.url.path)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "path": str(request.url.path)
            }
        )

def setup_routes(app: FastAPI) -> None:
    """Set up application routes"""
    print(">>> SETUP_ROUTES CALLED <<<")    # Import route modules
    from .routes.agents import router as agents_router
    from .routes.memory import router as memory_router
    from .routes.mcp import router as mcp_router
    from .routes.rag import router as rag_router
    from .routes.health import router as health_router
    from .routes.models import router as models_router
    from .routes.ollama import router as ollama_router
    from .routes.auth import router as auth_router
    from .routes.documentation import router as documentation_router

    # A2A Protocol router
    try:
        from .routes.a2a import router as a2a_router
        A2A_ROUTER_AVAILABLE = True
        print(">>> A2A ROUTER IMPORTED SUCCESSFULLY <<<")
    except ImportError as e:
        A2A_ROUTER_AVAILABLE = False
        print(f">>> A2A ROUTER IMPORT FAILED: {e} <<<")
    print(">>> IMPORTING CORE ROUTER <<<")
    from .routes.core import router as core_router
    print(f">>> CORE ROUTER IMPORTED: {core_router} <<<")
    print(f">>> CORE ROUTER ROUTES: {[route.path for route in core_router.routes]} <<<")
    
    try:
        from .routes.workflows import router as workflows_router
    except Exception as e:
        logger.warning(f"Failed to import workflows router: {e}")
        from fastapi import APIRouter
        workflows_router = APIRouter(prefix="/workflows", tags=["workflows"])
        
        error_detail = str(e)  # Capture the error for later use
        
        @workflows_router.get("/error")
        async def workflows_error():
            return {"error": "Workflows router failed to import", "detail": error_detail}

    from .routes.websocket import router as websocket_router
    
    # Include routers with prefixes
    print(">>> REGISTERING CORE ROUTER <<<")
    app.include_router(core_router, prefix="/api", tags=["Core"])
    print(">>> REGISTERING HEALTH ROUTER <<<")
    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(agents_router, prefix="/api/v1/agents", tags=["Agents"])
    app.include_router(memory_router, prefix="/api/v1/memory", tags=["Memory"])
    print(">>> REGISTERING MCP ROUTER <<<")
    app.include_router(mcp_router, prefix="/api/mcp", tags=["MCP"])
    app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])
    app.include_router(models_router, prefix="/api/v1/models", tags=["Model Performance"])
    app.include_router(ollama_router, prefix="/api/v1/ollama", tags=["Ollama"])
    app.include_router(workflows_router, prefix="/api/v1", tags=["Workflows"])
    app.include_router(websocket_router, tags=["WebSocket"])
    app.include_router(auth_router, prefix="/auth", tags=["OAuth Authentication"])

    # A2A Protocol router
    if A2A_ROUTER_AVAILABLE:
        app.include_router(a2a_router, tags=["A2A Protocol"])
        print(">>> A2A ROUTER INCLUDED IN APP <<<")
    else:
        print(">>> A2A ROUTER NOT AVAILABLE - SKIPPING <<<")
    
    # Documentation router - add both prefixes for compatibility
    app.include_router(documentation_router, prefix="/api/documentation", tags=["API Documentation"])
    
    # Debug: Print all registered routes
    print(">>> ALL REGISTERED ROUTES <<<")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            methods = getattr(route, 'methods', [])
            print(f"  {route.path} [{', '.join(methods)}]")
        elif hasattr(route, 'path'):
            print(f"  {route.path} [MOUNT/OTHER]")
    print(">>> END ROUTES LIST <<<")
    
    # Root endpoints
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint"""
        return {
            "message": "PyGent Factory API",
            "version": get_settings().app.APP_VERSION,
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    

    @app.get("/health", include_in_schema=False)
    async def simple_health():
        """Simple health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "pygent-factory"
        }

    @app.get("/api/v1/health", include_in_schema=False)
    async def api_v1_health():
        """API v1 health check endpoint (for frontend compatibility)"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "pygent-factory"
        }

    @app.options("/api/v1/health", include_in_schema=False)
    async def api_v1_health_options(request: Request):
        """OPTIONS handler for /api/v1/health (CORS preflight)"""
        response = JSONResponse(content={"status": "ok"})
        origin = request.headers.get("origin")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Max-Age"] = "86400"
        return response

# Create the app instance
app = create_app()

def run_server():
    """Run the FastAPI server"""
    import sys
    settings = get_settings()
    
    # Check for --no-reload command line argument
    disable_reload = "--no-reload" in sys.argv
    enable_reload = settings.app.RELOAD and settings.app.DEBUG and not disable_reload
    
    if disable_reload:
        print("🔧 File watching disabled via --no-reload flag")
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.app.HOST,
        port=settings.app.PORT,
        reload=enable_reload,
        log_level=settings.app.LOG_LEVEL.lower(),
        access_log=settings.app.DEBUG
    )

if __name__ == "__main__":
    run_server()