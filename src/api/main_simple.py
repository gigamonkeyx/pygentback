"""
Ultra-minimal FastAPI backend for development testing
No complex dependencies, just basic WebSocket and API endpoints
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="PyGent Factory Simple API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "PyGent Factory Simple API", "status": "running"}

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Simple backend is running"}

@app.get("/api/v1/mcp/discovery/status")
async def mcp_discovery_status():
    """Mock MCP discovery status endpoint"""
    return {
        "status": "active",
        "servers": [
            {"name": "python", "status": "connected"},
            {"name": "context7", "status": "connected"},
            {"name": "github", "status": "connected"}
        ]
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Simple WebSocket endpoint"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        # Send welcome message
        welcome_msg = {
            "type": "connection",
            "status": "connected",
            "message": "WebSocket connected to simple backend"
        }
        await websocket.send_text(json.dumps(welcome_msg))
        
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            
            try:
                # Parse incoming message
                message = json.loads(data)
                
                # Echo back with response
                response = {
                    "type": "response",
                    "original": message,
                    "timestamp": "2025-06-08T11:22:00Z",
                    "status": "received"
                }
                
                await websocket.send_text(json.dumps(response))
                
            except json.JSONDecodeError:
                # Handle non-JSON messages
                error_response = {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "received": data
                }
                await websocket.send_text(json.dumps(error_response))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            error_msg = {
                "type": "error",
                "message": str(e)
            }
            await websocket.send_text(json.dumps(error_msg))
        except Exception:
            pass

if __name__ == "__main__":
    logger.info("Starting simple FastAPI server...")
    uvicorn.run(
        "src.api.main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
