#!/usr/bin/env python3
"""
Simple A2A Server

Minimal FastAPI server with A2A protocol endpoints for testing.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import hashlib

def create_app() -> FastAPI:
    """Create minimal FastAPI app with A2A endpoints"""
    
    app = FastAPI(
        title="PyGent Factory A2A Protocol",
        description="Agent-to-Agent Protocol Implementation",
        version="1.0.0",
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
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "service": "pygent-factory-a2a",
            "version": "1.0.0",
            "a2a_enabled": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # A2A Well-known endpoint
    @app.get("/a2a/v1/.well-known/agent.json")
    async def well_known_agent():
        return {
            "name": "PyGent Factory",
            "description": "Multi-agent research and development platform with A2A protocol support",
            "url": "http://localhost:8000",
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": True
            },
            "skills": [
                {
                    "id": "research",
                    "name": "Research Analysis",
                    "description": "Comprehensive research and analysis capabilities",
                    "tags": ["research", "analysis", "academic"]
                },
                {
                    "id": "agent_management",
                    "name": "Agent Management",
                    "description": "Create and manage AI agents",
                    "tags": ["agents", "management", "orchestration"]
                },
                {
                    "id": "a2a_communication",
                    "name": "A2A Communication",
                    "description": "Agent-to-agent communication and coordination",
                    "tags": ["a2a", "communication", "protocol"]
                }
            ],
            "provider": {
                "name": "PyGent Factory",
                "organization": "Open Source",
                "description": "Open source multi-agent platform"
            },
            "endpoints": {
                "discover": "/a2a/v1/agents/discover",
                "message": "/a2a/v1/message/send",
                "health": "/a2a/v1/health"
            },
            "metadata": {
                "agent_id": "pygent_factory_main",
                "version": "1.0.0",
                "protocol_version": "1.0",
                "deployment": "production"
            }
        }
    
    # A2A Discovery endpoint
    @app.get("/a2a/v1/agents/discover")
    async def discover_agents():
        return {
            "agents": [
                {
                    "id": "pygent_factory_main",
                    "name": "PyGent Factory",
                    "url": "http://localhost:8000",
                    "status": "active",
                    "capabilities": ["research", "agent_management", "a2a_communication"],
                    "last_seen": datetime.utcnow().isoformat()
                }
            ],
            "total": 1,
            "timestamp": datetime.utcnow().isoformat(),
            "discovery_method": "local"
        }
    
    # A2A Message send endpoint
    @app.post("/a2a/v1/message/send")
    async def send_message(message_data: dict):
        # Generate message ID
        message_str = str(message_data) + str(datetime.utcnow())
        message_id = f"msg_{hashlib.md5(message_str.encode()).hexdigest()[:8]}"
        
        return {
            "status": "received",
            "message_id": message_id,
            "timestamp": datetime.utcnow().isoformat(),
            "recipient": "pygent_factory_main",
            "sender": message_data.get("sender", "unknown"),
            "message_type": message_data.get("message_type", "general")
        }
    
    # A2A Health endpoint
    @app.get("/a2a/v1/health")
    async def a2a_health():
        return {
            "status": "healthy",
            "a2a_protocol": "operational",
            "endpoints": ["well-known", "discover", "message", "health"],
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": "active",
            "version": "1.0.0"
        }
    
    # Additional A2A endpoints for completeness
    @app.get("/a2a/v1/agents/{agent_id}/card")
    async def get_agent_card(agent_id: str):
        if agent_id == "pygent_factory_main":
            return {
                "id": agent_id,
                "name": "PyGent Factory",
                "description": "Multi-agent research and development platform",
                "capabilities": ["research", "agent_management", "a2a_communication"],
                "status": "active",
                "url": "http://localhost:8000",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {"error": "Agent not found", "agent_id": agent_id}
    
    @app.post("/a2a/v1/message/stream")
    async def stream_message(message_data: dict):
        return {
            "status": "streaming_not_implemented",
            "message": "Streaming endpoint placeholder",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.get("/a2a/v1/tasks/{task_id}")
    async def get_task(task_id: str):
        return {
            "task_id": task_id,
            "status": "completed",
            "message": "Task endpoint placeholder",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @app.post("/a2a/v1/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str):
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task cancellation placeholder",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return app

def main():
    """Main server function"""
    print("🚀 Starting Simple A2A Protocol Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔗 A2A Well-known: http://localhost:8000/a2a/v1/.well-known/agent.json")
    
    app = create_app()
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
