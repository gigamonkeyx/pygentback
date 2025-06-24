"""
Health Check API Routes

This module provides health check endpoints for monitoring the status
of PyGent Factory components including database, MCP servers, agents,
and system resources.
"""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import psutil

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...config.settings import get_settings


logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances (will be set by main.py)
_db_manager = None
_agent_factory = None
_memory_manager = None
_mcp_manager = None
_retrieval_system = None
_message_bus = None
_ollama_manager = None

def set_health_dependencies(db_manager, agent_factory, memory_manager, mcp_manager, retrieval_system, message_bus, ollama_manager=None):
    """Set all health check dependencies"""
    global _db_manager, _agent_factory, _memory_manager, _mcp_manager, _retrieval_system, _message_bus, _ollama_manager
    _db_manager = db_manager
    _agent_factory = agent_factory
    _memory_manager = memory_manager
    _mcp_manager = mcp_manager
    _retrieval_system = retrieval_system
    _message_bus = message_bus
    _ollama_manager = ollama_manager

def get_db_manager():
    """Get database manager dependency"""
    if _db_manager is None:
        raise HTTPException(status_code=503, detail="Database manager not available")
    return _db_manager

def get_agent_factory():
    """Get agent factory dependency"""
    if _agent_factory is None:
        raise HTTPException(status_code=503, detail="Agent factory not available")
    return _agent_factory

def get_memory_manager():
    """Get memory manager dependency"""
    if _memory_manager is None:
        raise HTTPException(status_code=503, detail="Memory manager not available")
    return _memory_manager

def get_mcp_manager():
    """Get MCP manager dependency"""
    if _mcp_manager is None:
        raise HTTPException(status_code=503, detail="MCP manager not available")
    return _mcp_manager

def get_retrieval_system():
    """Get retrieval system dependency"""
    if _retrieval_system is None:
        raise HTTPException(status_code=503, detail="Retrieval system not available")
    return _retrieval_system

def get_message_bus():
    """Get message bus dependency"""
    if _message_bus is None:
        raise HTTPException(status_code=503, detail="Message bus not available")
    return _message_bus

def get_ollama_manager():
    """Get Ollama manager dependency (optional)"""
    # Return None if not available instead of raising exception
    # This allows health check to work even if Ollama is not configured
    return _ollama_manager


class HealthStatus(BaseModel):
    """Health status response model"""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    version: str
    uptime_seconds: float
    components: Dict[str, Dict[str, Any]]
    system_info: Dict[str, Any]


class ComponentHealth(BaseModel):
    """Individual component health model"""
    status: str
    message: str
    details: Dict[str, Any] = {}
    last_check: datetime


# Application start time for uptime calculation
app_start_time = datetime.utcnow()


@router.head("/health")
async def head_health_status():
    """
    HEAD request for health status - returns just headers for connectivity check.
    """
    return None

@router.get("/health", response_model=HealthStatus)
async def get_health_status(
    db_manager=Depends(get_db_manager),
    agent_factory=Depends(get_agent_factory),
    memory_manager=Depends(get_memory_manager),
    mcp_manager=Depends(get_mcp_manager),
    retrieval_system=Depends(get_retrieval_system),
    message_bus=Depends(get_message_bus),
    ollama_manager=Depends(get_ollama_manager)
):
    """
    Get comprehensive health status of all system components.
    
    Returns detailed health information including:
    - Overall system status
    - Individual component status
    - System resource usage
    - Performance metrics
    """
    try:
        settings = get_settings()
        
        # Calculate uptime
        uptime = (datetime.utcnow() - app_start_time).total_seconds()
        
        # Check all components
        components = {}
        
        # Database health
        components["database"] = await check_database_health(db_manager)
        
        # Agent factory health
        components["agent_factory"] = await check_agent_factory_health(agent_factory)
        
        # Memory manager health
        components["memory_manager"] = await check_memory_manager_health(memory_manager)
        
        # MCP manager health
        components["mcp_manager"] = await check_mcp_manager_health(mcp_manager)
        
        # Retrieval system health
        components["retrieval_system"] = await check_retrieval_system_health(retrieval_system)
        
        # Message bus health
        components["message_bus"] = await check_message_bus_health(message_bus)

        # Ollama health
        components["ollama"] = await check_ollama_health(ollama_manager)

        # System resource health
        components["system_resources"] = await check_system_resources()
        
        # Determine overall status
        overall_status = determine_overall_status(components)
        
        # Get system information
        system_info = get_system_info()
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version=settings.app.APP_VERSION,
            uptime_seconds=uptime,
            components=components,
            system_info=system_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/health/database")
async def get_database_health(db_manager=Depends(get_db_manager)):
    """Get detailed database health information"""
    try:
        health_info = await check_database_health(db_manager)
        return health_info
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Database health check failed")


@router.get("/health/agents")
async def get_agents_health(agent_factory=Depends(get_agent_factory)):
    """Get detailed agent system health information"""
    try:
        health_info = await check_agent_factory_health(agent_factory)
        return health_info
    except Exception as e:
        logger.error(f"Agent health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Agent health check failed")


@router.get("/health/mcp")
async def get_mcp_health(mcp_manager=Depends(get_mcp_manager)):
    """Get detailed MCP system health information"""
    try:
        health_info = await check_mcp_manager_health(mcp_manager)
        return health_info
    except Exception as e:
        logger.error(f"MCP health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="MCP health check failed")


@router.get("/health/memory")
async def get_memory_health(memory_manager=Depends(get_memory_manager)):
    """Get detailed memory system health information"""
    try:
        health_info = await check_memory_manager_health(memory_manager)
        return health_info
    except Exception as e:
        logger.error(f"Memory health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Memory health check failed")


@router.get("/health/rag")
async def get_rag_health(retrieval_system=Depends(get_retrieval_system)):
    """Get detailed RAG system health information"""
    try:
        health_info = await check_retrieval_system_health(retrieval_system)
        return health_info
    except Exception as e:
        logger.error(f"RAG health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="RAG health check failed")


@router.get("/health/ollama")
async def get_ollama_health(ollama_manager=Depends(get_ollama_manager)):
    """Get detailed Ollama system health information"""
    try:
        health_info = await check_ollama_health(ollama_manager)
        return health_info
    except Exception as e:
        logger.error(f"Ollama health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Ollama health check failed")


# Health check functions for individual components

async def check_database_health(db_manager) -> Dict[str, Any]:
    """Check database health"""
    try:
        # Test database connection
        is_healthy = await db_manager.health_check()
        
        # Get connection info
        connection_info = await db_manager.get_connection_info()
        
        if is_healthy:
            return {
                "status": "healthy",
                "message": "Database connection is working",
                "details": connection_info,
                "last_check": datetime.utcnow()
            }
        else:
            return {
                "status": "unhealthy",
                "message": "Database connection failed",
                "details": connection_info,
                "last_check": datetime.utcnow()
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Database health check error: {str(e)}",
            "details": {},
            "last_check": datetime.utcnow()
        }


async def check_agent_factory_health(agent_factory) -> Dict[str, Any]:
    """Check agent factory health"""
    try:
        # Get agent factory health
        health_info = await agent_factory.health_check()
        
        if health_info.get("error_agents", 0) == 0:
            status = "healthy"
            message = f"Agent factory operational with {health_info.get('active_agents', 0)} active agents"
        elif health_info.get("active_agents", 0) > 0:
            status = "degraded"
            message = f"Some agents in error state: {health_info.get('error_agents', 0)} errors"
        else:
            status = "unhealthy"
            message = "No active agents"
        
        return {
            "status": status,
            "message": message,
            "details": health_info,
            "last_check": datetime.utcnow()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Agent factory health check error: {str(e)}",
            "details": {},
            "last_check": datetime.utcnow()
        }


async def check_memory_manager_health(memory_manager) -> Dict[str, Any]:
    """Check memory manager health"""
    try:
        # Get memory statistics
        stats = await memory_manager.get_global_stats()
        
        status = "healthy"
        message = f"Memory manager operational with {stats.get('total_agents', 0)} agent memory spaces"
        
        return {
            "status": status,
            "message": message,
            "details": stats,
            "last_check": datetime.utcnow()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Memory manager health check error: {str(e)}",
            "details": {},
            "last_check": datetime.utcnow()
        }


async def check_mcp_manager_health(mcp_manager) -> Dict[str, Any]:
    """Check MCP manager health"""
    try:
        # Get MCP server status
        servers = await mcp_manager.list_servers()
        connected_count = await mcp_manager.get_connected_servers_count()
        total_count = len(servers)
        
        if connected_count == total_count and total_count > 0:
            status = "healthy"
            message = f"All {total_count} MCP servers connected"
        elif connected_count > 0:
            status = "degraded"
            message = f"{connected_count}/{total_count} MCP servers connected"
        else:
            status = "unhealthy"
            message = "No MCP servers connected"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "connected_servers": connected_count,
                "total_servers": total_count,
                "servers": servers
            },
            "last_check": datetime.utcnow()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"MCP manager health check error: {str(e)}",
            "details": {},
            "last_check": datetime.utcnow()
        }


async def check_retrieval_system_health(retrieval_system) -> Dict[str, Any]:
    """Check RAG retrieval system health"""
    try:
        # Get retrieval statistics
        stats = await retrieval_system.get_retrieval_stats()
        
        status = "healthy"
        message = "RAG retrieval system operational"
        
        return {
            "status": status,
            "message": message,
            "details": stats,
            "last_check": datetime.utcnow()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"RAG retrieval system health check error: {str(e)}",
            "details": {},
            "last_check": datetime.utcnow()
        }


async def check_message_bus_health(message_bus) -> Dict[str, Any]:
    """Check message bus health"""
    try:
        # Get message bus metrics
        metrics = message_bus.get_metrics()

        if metrics.get("running", False):
            status = "healthy"
            message = "Message bus operational"
        else:
            status = "unhealthy"
            message = "Message bus not running"

        return {
            "status": status,
            "message": message,
            "details": metrics,
            "last_check": datetime.utcnow()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Message bus health check error: {str(e)}",
            "details": {},
            "last_check": datetime.utcnow()
        }


async def check_ollama_health(ollama_manager) -> Dict[str, Any]:
    """Check Ollama service health"""
    try:
        # Handle case where Ollama manager is not available
        if ollama_manager is None:
            return {
                "status": "unavailable",
                "message": "Ollama manager not configured",
                "details": {
                    "is_ready": False,
                    "available_models": [],
                    "model_count": 0,
                    "process_running": False,
                    "url": "",
                    "executable_path": ""
                },
                "last_check": datetime.utcnow()
            }

        # Get Ollama status
        ollama_status = ollama_manager.get_status()

        # Check if Ollama is ready and has models
        is_ready = ollama_status.get("is_ready", False)
        available_models = ollama_status.get("available_models", [])
        process_running = ollama_status.get("process_running", False)

        if is_ready and len(available_models) > 0:
            status = "healthy"
            message = f"Ollama operational with {len(available_models)} models available"
        elif is_ready and len(available_models) == 0:
            status = "degraded"
            message = "Ollama running but no models available"
        elif process_running:
            status = "degraded"
            message = "Ollama process running but not ready"
        else:
            status = "unhealthy"
            message = "Ollama service not running"

        return {
            "status": status,
            "message": message,
            "details": {
                "is_ready": is_ready,
                "available_models": available_models,
                "model_count": len(available_models),
                "process_running": process_running,
                "url": ollama_status.get("url", ""),
                "executable_path": ollama_status.get("executable_path", "")
            },
            "last_check": datetime.utcnow()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Ollama health check error: {str(e)}",
            "details": {},
            "last_check": datetime.utcnow()
        }


async def check_system_resources() -> Dict[str, Any]:
    """Check system resource usage"""
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        
        # Determine status based on resource usage
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = "unhealthy"
            message = "High resource usage detected"
        elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 80:
            status = "degraded"
            message = "Moderate resource usage"
        else:
            status = "healthy"
            message = "Resource usage normal"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "last_check": datetime.utcnow()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"System resource check error: {str(e)}",
            "details": {},
            "last_check": datetime.utcnow()
        }


def determine_overall_status(components: Dict[str, Dict[str, Any]]) -> str:
    """Determine overall system status from component statuses"""
    statuses = [comp.get("status", "unknown") for comp in components.values()]
    
    if all(status == "healthy" for status in statuses):
        return "healthy"
    elif any(status == "unhealthy" for status in statuses):
        return "unhealthy"
    else:
        return "degraded"


def get_system_info() -> Dict[str, Any]:
    """Get general system information"""
    try:
        return {
            "platform": psutil.LINUX if hasattr(psutil, 'LINUX') else "unknown",
            "cpu_count": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": f"{psutil.version_info.major}.{psutil.version_info.minor}",
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
    except Exception:
        return {"error": "Unable to retrieve system information"}
