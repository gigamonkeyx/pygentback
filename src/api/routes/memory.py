"""
Memory Management API Routes

This module provides REST API endpoints for agent memory management
including memory storage, retrieval, and statistics.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ...memory.memory_manager import MemoryManager, MemoryType, MemoryImportance
from ...security.auth import get_current_user, require_memory_read, require_memory_write, User


logger = logging.getLogger(__name__)

router = APIRouter()

# Global memory manager instance (will be set by main.py)
_memory_manager: Optional[MemoryManager] = None

def set_memory_manager(manager: MemoryManager):
    """Set the global memory manager instance"""
    global _memory_manager
    _memory_manager = manager

def get_memory_manager() -> MemoryManager:
    """Get the memory manager dependency"""
    if _memory_manager is None:
        raise HTTPException(status_code=500, detail="Memory manager not initialized")
    return _memory_manager


# Request/Response models
class StoreMemoryRequest(BaseModel):
    content: str
    memory_type: str = "short_term"
    metadata: Dict[str, Any] = {}
    importance: str = "medium"


class MemorySearchRequest(BaseModel):
    query: str
    memory_types: Optional[List[str]] = None
    limit: int = 10
    similarity_threshold: float = 0.7


class MemoryResponse(BaseModel):
    id: str
    agent_id: str
    memory_type: str
    content: str
    metadata: Dict[str, Any]
    importance: str
    created_at: datetime
    last_accessed: datetime
    access_count: int


@router.post("/{agent_id}/store", response_model=MemoryResponse)
async def store_memory(
    agent_id: str,
    request: StoreMemoryRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: User = Depends(require_memory_write)
):
    """Store a memory entry for an agent"""
    try:
        # Get memory space
        memory_space = await memory_manager.get_memory_space(agent_id)
        if not memory_space:
            raise HTTPException(status_code=404, detail="Agent memory space not found")
        
        # Parse memory type and importance
        try:
            memory_type = MemoryType(request.memory_type)
            importance = MemoryImportance[request.importance.upper()]
        except (ValueError, KeyError):
            raise HTTPException(status_code=400, detail="Invalid memory type or importance")
        
        # Store memory
        memory_id = await memory_space.store_memory(
            content=request.content,
            memory_type=memory_type,
            metadata=request.metadata,
            importance=importance
        )
        
        # Get stored memory for response
        memory = await memory_space.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=500, detail="Failed to retrieve stored memory")
        
        return MemoryResponse(
            id=memory.id,
            agent_id=memory.agent_id,
            memory_type=memory.memory_type.value,
            content=memory.content,
            metadata=memory.metadata,
            importance=memory.importance.name.lower(),
            created_at=memory.created_at,
            last_accessed=memory.last_accessed,
            access_count=memory.access_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to store memory for agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store memory: {str(e)}")


@router.post("/{agent_id}/search")
async def search_memories(
    agent_id: str,
    request: MemorySearchRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: User = Depends(require_memory_read)
):
    """Search agent memories"""
    try:
        # Get memory space
        memory_space = await memory_manager.get_memory_space(agent_id)
        if not memory_space:
            raise HTTPException(status_code=404, detail="Agent memory space not found")
        
        # Parse memory types
        memory_types = None
        if request.memory_types:
            try:
                memory_types = [MemoryType(mt) for mt in request.memory_types]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid memory type")
        
        # Search memories
        results = await memory_space.retrieve_memories(
            query=request.query,
            memory_types=memory_types,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        
        return {
            "agent_id": agent_id,
            "query": request.query,
            "results": [
                {
                    "memory": MemoryResponse(
                        id=result.memory.id,
                        agent_id=result.memory.agent_id,
                        memory_type=result.memory.memory_type.value,
                        content=result.memory.content,
                        metadata=result.memory.metadata,
                        importance=result.memory.importance.name.lower(),
                        created_at=result.memory.created_at,
                        last_accessed=result.memory.last_accessed,
                        access_count=result.memory.access_count
                    ),
                    "relevance_score": result.relevance_score,
                    "similarity_score": result.similarity_score
                }
                for result in results
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search memories for agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to search memories: {str(e)}")


@router.get("/{agent_id}/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    agent_id: str,
    memory_id: str,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: User = Depends(require_memory_read)
):
    """Get a specific memory by ID"""
    try:
        # Get memory space
        memory_space = await memory_manager.get_memory_space(agent_id)
        if not memory_space:
            raise HTTPException(status_code=404, detail="Agent memory space not found")
        
        # Get memory
        memory = await memory_space.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return MemoryResponse(
            id=memory.id,
            agent_id=memory.agent_id,
            memory_type=memory.memory_type.value,
            content=memory.content,
            metadata=memory.metadata,
            importance=memory.importance.name.lower(),
            created_at=memory.created_at,
            last_accessed=memory.last_accessed,
            access_count=memory.access_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory {memory_id} for agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory: {str(e)}")


@router.delete("/{agent_id}/memories/{memory_id}")
async def delete_memory(
    agent_id: str,
    memory_id: str,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: User = Depends(require_memory_write)
):
    """Delete a memory entry"""
    try:
        # Get memory space
        memory_space = await memory_manager.get_memory_space(agent_id)
        if not memory_space:
            raise HTTPException(status_code=404, detail="Agent memory space not found")
        
        # Delete memory
        success = await memory_space.delete_memory(memory_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return {"message": f"Memory {memory_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete memory {memory_id} for agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")


@router.get("/{agent_id}/stats")
async def get_memory_stats(
    agent_id: str,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: User = Depends(require_memory_read)
):
    """Get memory statistics for an agent"""
    try:
        # Get memory space
        memory_space = await memory_manager.get_memory_space(agent_id)
        if not memory_space:
            raise HTTPException(status_code=404, detail="Agent memory space not found")
        
        # Get stats
        stats = await memory_space.get_memory_stats()
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory stats for agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")


@router.post("/{agent_id}/consolidate")
async def consolidate_memories(
    agent_id: str,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: User = Depends(require_memory_write)
):
    """Trigger memory consolidation for an agent"""
    try:
        # Get memory space
        memory_space = await memory_manager.get_memory_space(agent_id)
        if not memory_space:
            raise HTTPException(status_code=404, detail="Agent memory space not found")
        
        # Consolidate memories
        await memory_space.consolidate_memories()
        
        return {"message": f"Memory consolidation completed for agent {agent_id}"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to consolidate memories for agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to consolidate memories: {str(e)}")


@router.get("/global/stats")
async def get_global_memory_stats(
    memory_manager: MemoryManager = Depends(get_memory_manager),
    current_user: User = Depends(require_memory_read)
):
    """Get global memory statistics"""
    try:
        stats = await memory_manager.get_global_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get global memory stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get global memory stats: {str(e)}")


@router.get("/types")
async def get_memory_types(
    current_user: User = Depends(require_memory_read)
):
    """Get available memory types"""
    return {
        "memory_types": [mt.value for mt in MemoryType],
        "importance_levels": [imp.name.lower() for imp in MemoryImportance]
    }
