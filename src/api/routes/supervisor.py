"""
Supervisor Agent API Routes
FastAPI routes for supervisor agent monitoring and management.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...agents.supervisor_agent import SupervisorAgent
from ...agents.quality_evaluator import QualityEvaluator
from ...agents.teaching_system import TeachingSystem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/supervisor", tags=["supervisor"])


# Response models
class SupervisorStatsResponse(BaseModel):
    """Supervisor statistics response"""
    total_tasks: int = Field(..., description="Total tasks supervised")
    successful_tasks: int = Field(..., description="Successfully completed tasks")
    success_rate: float = Field(..., description="Success rate (0.0-1.0)")
    active_tasks: List[str] = Field(..., description="Currently active task IDs")


class TaskStatusResponse(BaseModel):
    """Task status response"""
    task_id: str = Field(..., description="Task identifier")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Task analysis")
    quality_score: Optional[Dict[str, Any]] = Field(None, description="Quality assessment")
    feedback: Optional[str] = Field(None, description="Supervisor feedback")
    requires_retry: bool = Field(..., description="Whether task needs retry")
    timestamp: float = Field(..., description="Task timestamp")


class TeachingStatsResponse(BaseModel):
    """Teaching system statistics response"""
    total_interactions: int = Field(..., description="Total teaching interactions")
    successful_interactions: int = Field(..., description="Successful learning interactions")
    improvement_rate: float = Field(..., description="Agent improvement rate")
    most_common_issues: List[str] = Field(..., description="Most common issues found")
    task_type_distribution: Dict[str, int] = Field(..., description="Distribution of task types")


# Dependency injection
async def get_supervisor_agent() -> SupervisorAgent:
    """Get supervisor agent instance"""
    return SupervisorAgent()


async def get_quality_evaluator() -> QualityEvaluator:
    """Get quality evaluator instance"""
    return QualityEvaluator()


async def get_teaching_system() -> TeachingSystem:
    """Get teaching system instance"""
    return TeachingSystem()


# Supervisor monitoring endpoints
@router.get("/health")
async def supervisor_health():
    """Check supervisor agent health"""
    try:
        supervisor = SupervisorAgent()
        return {
            "status": "healthy",
            "supervisor_active": True,
            "timestamp": "2025-01-27T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Supervisor health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Supervisor unhealthy: {str(e)}")


@router.get("/stats", response_model=SupervisorStatsResponse)
async def get_supervisor_stats(
    supervisor: SupervisorAgent = Depends(get_supervisor_agent)
):
    """Get supervisor agent statistics"""
    try:
        stats = supervisor.get_supervision_stats()
        return SupervisorStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get supervisor stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    supervisor: SupervisorAgent = Depends(get_supervisor_agent)
):
    """Get status of a supervised task"""
    try:
        task_status = supervisor.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return TaskStatusResponse(
            task_id=task_id,
            analysis=task_status.get("analysis").__dict__ if task_status.get("analysis") else None,
            quality_score=task_status.get("quality_score").__dict__ if task_status.get("quality_score") else None,
            feedback=task_status.get("feedback"),
            requires_retry=task_status.get("requires_retry", True),
            timestamp=task_status.get("timestamp", 0.0)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def list_supervised_tasks(
    supervisor: SupervisorAgent = Depends(get_supervisor_agent)
):
    """List all supervised tasks"""
    try:
        stats = supervisor.get_supervision_stats()
        return {
            "active_tasks": stats.get("active_tasks", []),
            "total_tasks": stats.get("total_tasks", 0),
            "success_rate": stats.get("success_rate", 0.0)
        }
    except Exception as e:
        logger.error(f"Failed to list supervised tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Teaching system endpoints
@router.get("/teaching/stats", response_model=TeachingStatsResponse)
async def get_teaching_stats(
    teaching_system: TeachingSystem = Depends(get_teaching_system)
):
    """Get teaching system statistics"""
    try:
        stats = teaching_system.get_teaching_stats()
        return TeachingStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get teaching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/teaching/patterns")
async def get_success_patterns(
    task_type: Optional[str] = None,
    teaching_system: TeachingSystem = Depends(get_teaching_system)
):
    """Get success patterns for task types"""
    try:
        patterns = teaching_system.get_success_patterns(task_type)
        return {
            "success_patterns": patterns,
            "task_type_filter": task_type
        }
    except Exception as e:
        logger.error(f"Failed to get success patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/teaching/issues")
async def get_common_issues(
    task_type: Optional[str] = None,
    teaching_system: TeachingSystem = Depends(get_teaching_system)
):
    """Get common issues for task types"""
    try:
        issues = teaching_system.get_common_issues(task_type)
        return {
            "common_issues": issues,
            "task_type_filter": task_type
        }
    except Exception as e:
        logger.error(f"Failed to get common issues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/teaching/history")
async def get_learning_history(
    agent_id: Optional[str] = None,
    teaching_system: TeachingSystem = Depends(get_teaching_system)
):
    """Get learning history for agents"""
    try:
        history = teaching_system.get_learning_history(agent_id)
        
        # Convert learning records to dict format
        history_data = []
        for record in history:
            history_data.append({
                "agent_id": record.agent_id,
                "task_type": record.task_type,
                "original_output": record.original_output,
                "feedback_provided": record.feedback_provided,
                "improvement_achieved": record.improvement_achieved,
                "timestamp": record.timestamp.isoformat()
            })
        
        return {
            "learning_history": history_data,
            "agent_filter": agent_id,
            "total_records": len(history_data)
        }
    except Exception as e:
        logger.error(f"Failed to get learning history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Quality evaluation endpoints
@router.get("/quality/standards")
async def get_quality_standards(
    quality_evaluator: QualityEvaluator = Depends(get_quality_evaluator)
):
    """Get current quality standards"""
    try:
        return {
            "quality_standards": quality_evaluator.quality_standards,
            "description": "Minimum thresholds for quality metrics"
        }
    except Exception as e:
        logger.error(f"Failed to get quality standards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality/standards")
async def update_quality_standards(
    standards: Dict[str, float],
    quality_evaluator: QualityEvaluator = Depends(get_quality_evaluator)
):
    """Update quality standards"""
    try:
        quality_evaluator.update_standards(standards)
        return {
            "message": "Quality standards updated successfully",
            "new_standards": quality_evaluator.quality_standards
        }
    except Exception as e:
        logger.error(f"Failed to update quality standards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Test endpoint for supervisor functionality
@router.post("/test/supervise")
async def test_supervision(
    task_description: str,
    agent_output: str,
    supervisor: SupervisorAgent = Depends(get_supervisor_agent)
):
    """Test supervisor functionality with sample data"""
    try:
        task_id = f"test_{datetime.now().timestamp()}"
        
        result = await supervisor.supervise_task(
            task_id=task_id,
            task_description=task_description,
            agent_output=agent_output
        )
        
        return {
            "test_result": result,
            "message": "Supervision test completed"
        }
    except Exception as e:
        logger.error(f"Supervision test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
