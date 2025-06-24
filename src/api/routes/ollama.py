"""
Ollama Management API Routes

This module provides REST API endpoints for managing Ollama service
and models including status monitoring, model management, and metrics.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from ...core.ollama_manager import get_ollama_manager

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class ModelPullRequest(BaseModel):
    model_name: str


class ModelPullResponse(BaseModel):
    success: bool
    message: str
    model_name: str


class OllamaStatusResponse(BaseModel):
    is_ready: bool
    url: str
    available_models: List[str]
    process_running: bool
    executable_path: str = None


class OllamaModelResponse(BaseModel):
    name: str
    size: int
    digest: str
    modified_at: str
    details: Dict[str, Any] = None


class OllamaModelsResponse(BaseModel):
    models: List[OllamaModelResponse]
    total_count: int


class OllamaMetricsResponse(BaseModel):
    total_models: int
    total_size: int
    memory_usage: int
    gpu_utilization: Optional[float] = None
    active_models: List[str]
    last_updated: datetime


def get_ollama_manager_dependency():
    """Get Ollama manager dependency"""
    try:
        return get_ollama_manager()
    except Exception as e:
        logger.error(f"Failed to get Ollama manager: {e}")
        raise HTTPException(status_code=503, detail="Ollama service not available")


@router.get("/status", response_model=OllamaStatusResponse)
async def get_ollama_status(
    ollama_manager=Depends(get_ollama_manager_dependency)
):
    """Get Ollama service status and health information"""
    try:
        status = ollama_manager.get_status()
        
        return OllamaStatusResponse(
            is_ready=status.get("is_ready", False),
            url=status.get("url", ""),
            available_models=status.get("available_models", []),
            process_running=status.get("process_running", False),
            executable_path=status.get("executable_path")
        )
        
    except Exception as e:
        logger.error(f"Error getting Ollama status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Ollama status: {str(e)}")


@router.get("/models", response_model=OllamaModelsResponse)
async def get_ollama_models(
    ollama_manager=Depends(get_ollama_manager_dependency)
):
    """Get list of available Ollama models"""
    try:
        # Get available models from Ollama manager
        model_names = await ollama_manager.get_available_models()
        
        # For now, return basic model info
        # In a full implementation, you'd get detailed model info from Ollama API
        models = []
        total_size = 0
        
        for model_name in model_names:
            # Basic model info - in production, fetch from Ollama API
            model_info = OllamaModelResponse(
                name=model_name,
                size=0,  # Would fetch actual size from Ollama
                digest="",  # Would fetch actual digest
                modified_at=datetime.utcnow().isoformat(),
                details={}
            )
            models.append(model_info)
        
        return OllamaModelsResponse(
            models=models,
            total_count=len(models)
        )
        
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Ollama models: {str(e)}")


@router.post("/models/pull", response_model=ModelPullResponse)
async def pull_ollama_model(
    request: ModelPullRequest,
    ollama_manager=Depends(get_ollama_manager_dependency)
):
    """Pull a new model from Ollama registry"""
    try:
        model_name = request.model_name.strip()
        if not model_name:
            raise HTTPException(status_code=400, detail="Model name is required")
        
        logger.info(f"Pulling Ollama model: {model_name}")
        
        # Use Ollama manager to pull the model
        success = await ollama_manager.ensure_model_available(model_name)
        
        if success:
            return ModelPullResponse(
                success=True,
                message=f"Model '{model_name}' pulled successfully",
                model_name=model_name
            )
        else:
            return ModelPullResponse(
                success=False,
                message=f"Failed to pull model '{model_name}'",
                model_name=model_name
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pulling model {request.model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pull model: {str(e)}")


@router.delete("/models/{model_name}")
async def delete_ollama_model(
    model_name: str,
    ollama_manager=Depends(get_ollama_manager_dependency)
):
    """Delete an Ollama model"""
    try:
        if not model_name.strip():
            raise HTTPException(status_code=400, detail="Model name is required")
        
        logger.info(f"Deleting Ollama model: {model_name}")
        
        # Note: OllamaManager doesn't have a delete method yet
        # This would need to be implemented in the OllamaManager class
        # For now, return a not implemented error
        raise HTTPException(
            status_code=501, 
            detail="Model deletion not yet implemented in Ollama manager"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.get("/metrics", response_model=OllamaMetricsResponse)
async def get_ollama_metrics(
    ollama_manager=Depends(get_ollama_manager_dependency)
):
    """Get Ollama performance metrics"""
    try:
        # Get basic status
        status = ollama_manager.get_status()
        available_models = status.get("available_models", [])
        
        # Calculate basic metrics
        total_models = len(available_models)
        
        # For now, return basic metrics
        # In a full implementation, you'd gather actual performance data
        return OllamaMetricsResponse(
            total_models=total_models,
            total_size=0,  # Would calculate actual total size
            memory_usage=0,  # Would get actual memory usage
            gpu_utilization=None,  # Would get GPU utilization if available
            active_models=available_models,
            last_updated=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting Ollama metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Ollama metrics: {str(e)}")


@router.get("/health")
async def get_ollama_health(
    ollama_manager=Depends(get_ollama_manager_dependency)
):
    """Get Ollama health check information"""
    try:
        # Perform health check
        is_healthy = await ollama_manager.health_check()
        status = ollama_manager.get_status()
        
        return {
            "healthy": is_healthy,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking Ollama health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check Ollama health: {str(e)}")
