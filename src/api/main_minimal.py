"""
Minimal FastAPI Application for Development

This is a simplified version that skips complex initialization
to get the basic API and WebSocket endpoints running quickly.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_minimal_app() -> FastAPI:
    """Create a minimal FastAPI app for development"""
    
    app = FastAPI(
        title="PyGent Factory API (Minimal)",
        description="Minimal development server for PyGent Factory",
        version="1.0.0-dev",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
      # Basic health endpoint
    @app.get("/api/v1/health")
    async def health():
        return {"status": "ok", "service": "minimal"}
    
    # Handle OPTIONS requests for CORS preflight
    @app.options("/api/v1/health")
    async def health_options():
        return {"status": "ok"}
    
    @app.options("/api/v1/{path:path}")
    async def handle_options(path: str):
        return {"status": "ok"}
    
    # MCP discovery status endpoint (mock)
    @app.get("/api/v1/mcp/discovery/status")
    async def mcp_discovery_status():
        return {
            "status": "running",
            "servers_discovered": 3,
            "servers_loaded": 3,
            "last_discovery": "2025-06-08T11:00:00Z"
        }
      # Basic WebSocket endpoint
    from fastapi import WebSocket, WebSocketDisconnect
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        logger.info("WebSocket connection established")
        
        try:
            while True:
                data = await websocket.receive_text()
                logger.info(f"Received WebSocket message: {data}")
                
                # Parse and handle the message properly
                try:
                    import json
                    import time
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get("type") == "ping":
                        response = {"type": "pong", "timestamp": message.get("timestamp")}
                    elif message.get("type") == "subscribe":
                        response = {"type": "subscribed", "channel": message.get("channel", "default")}
                    else:
                        # For other messages, send a generic acknowledgment
                        response = {
                            "type": "ack",
                            "received": message.get("type", "unknown"),
                            "timestamp": int(time.time() * 1000)
                        }
                    
                    await websocket.send_text(json.dumps(response))
                    
                except json.JSONDecodeError:
                    # If not valid JSON, send error response
                    import time
                    error_response = {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": int(time.time() * 1000)
                    }
                    await websocket.send_text(json.dumps(error_response))
                
        except WebSocketDisconnect:
            logger.info("WebSocket connection closed")
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "PyGent Factory Minimal API",
            "version": "1.0.0-dev",
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    
    return app

# Create the app
app = create_minimal_app()

def run_minimal_server():
    """Run the minimal FastAPI server"""
    logger.info("🚀 Starting PyGent Factory Minimal API Server...")
    
    uvicorn.run(
        "src.api.main_minimal:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_minimal_server()
