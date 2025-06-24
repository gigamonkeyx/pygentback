#!/usr/bin/env python3
"""Simple backend with API and WebSocket support"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from datetime import datetime
import asyncio

app = FastAPI(title="PyGent Factory API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Simple connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

@app.get("/")
async def root():
    return {"message": "PyGent Factory API", "version": "1.0.0"}

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon configured"}

@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/v1/mcp/discovery/status")
async def mcp_status():
    return {
        "status": "active",
        "servers": [
            {"name": "python", "status": "connected"},
            {"name": "context7", "status": "connected"}
        ]
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                print(f"Received WebSocket message: {message}")
                
                # Handle different message types
                if message.get('type') == 'chat_message':
                    await handle_chat_message(message, websocket)
                else:
                    # Echo back other message types
                    response = {
                        "type": "response",
                        "data": {
                            "message": f"Received: {message.get('type', 'unknown')}",
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    await manager.send_personal_message(response, websocket)
                
            except json.JSONDecodeError:
                # Handle non-JSON messages
                response = {
                    "type": "response", 
                    "data": {"message": f"Received text: {data}"}
                }
                await manager.send_personal_message(response, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
