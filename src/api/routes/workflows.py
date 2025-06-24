"""
Workflow API Routes

Provides endpoints for automated workflows including Research-to-Analysis
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
from datetime import datetime

from ...workflows.research_analysis_orchestrator import (
    ResearchAnalysisOrchestrator, 
    WorkflowStatus,
    WorkflowProgress
)
from ...core.agent_factory import AgentFactory
from ..dependencies import get_agent_factory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["workflows"])

# Global storage for workflow progress (in production, use Redis or database)
workflow_progress_store: Dict[str, WorkflowProgress] = {}
workflow_results_store: Dict[str, Dict[str, Any]] = {}


class ResearchAnalysisRequest(BaseModel):
    """Request model for Research-to-Analysis workflow"""
    query: str = Field(..., description="Research query to investigate")
    analysis_model: str = Field(default="deepseek-r1:8b", description="Model for analysis phase")
    max_papers: int = Field(default=15, ge=1, le=50, description="Maximum papers to retrieve")
    analysis_depth: int = Field(default=3, ge=1, le=5, description="Depth of analysis reasoning")
    export_format: str = Field(default="markdown", description="Export format (markdown, pdf, html)")


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status"""
    workflow_id: str
    status: str
    current_step: str
    progress_percentage: float
    research_papers_found: int = 0
    analysis_confidence: float = 0.0
    estimated_time_remaining: Optional[float] = None
    error_message: Optional[str] = None


class ResearchAnalysisResponse(BaseModel):
    """Response model for completed Research-to-Analysis workflow"""
    workflow_id: str
    success: bool
    query: str
    research_summary: str
    analysis_summary: str
    formatted_output: str
    citations: list
    metadata: dict
    execution_time: float
    error_message: Optional[str] = None


@router.post("/research-analysis", response_model=dict)
async def start_research_analysis_workflow(
    request: ResearchAnalysisRequest,
    background_tasks: BackgroundTasks,
    agent_factory: AgentFactory = Depends(get_agent_factory)
):
    """
    Start an automated Research-to-Analysis workflow
    
    This endpoint initiates a complete pipeline that:
    1. Searches academic databases for relevant papers
    2. Analyzes the research using AI reasoning
    3. Formats results with academic citations
    
    Returns immediately with a workflow_id for tracking progress.
    """
    try:
        # Generate unique workflow ID
        workflow_id = f"research_analysis_{int(datetime.now().timestamp())}_{hash(request.query) % 10000}"
        
        logger.info(f"Starting Research-to-Analysis workflow: {workflow_id}")
        
        # Initialize progress tracking
        workflow_progress_store[workflow_id] = WorkflowProgress(
            status=WorkflowStatus.PENDING,
            current_step="Initializing workflow...",
            progress_percentage=0.0
        )
        
        # Create orchestrator with progress callback
        async def progress_callback(wf_id: str, progress: WorkflowProgress):
            workflow_progress_store[wf_id] = progress
        
        orchestrator = ResearchAnalysisOrchestrator(
            agent_factory=agent_factory,
            progress_callback=progress_callback
        )
        
        # Start workflow in background
        background_tasks.add_task(
            execute_workflow_background,
            orchestrator,
            workflow_id,
            request
        )
        
        return {
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Research-to-Analysis workflow initiated",
            "estimated_duration": "2-5 minutes",
            "tracking_url": f"/api/v1/workflows/research-analysis/{workflow_id}/status"
        }
        
    except Exception as e:
        logger.error(f"Failed to start Research-to-Analysis workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")


async def execute_workflow_background(
    orchestrator: ResearchAnalysisOrchestrator,
    workflow_id: str,
    request: ResearchAnalysisRequest
):
    """Execute the workflow in the background"""
    try:
        result = await orchestrator.execute_workflow(
            query=request.query,
            analysis_model=request.analysis_model,
            max_papers=request.max_papers,
            analysis_depth=request.analysis_depth,
            workflow_id=workflow_id
        )
        
        # Store results
        workflow_results_store[workflow_id] = {
            "result": result,
            "request": request.dict(),
            "completed_at": datetime.now().isoformat()
        }
        
        logger.info(f"Workflow {workflow_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")
        
        # Update progress with error
        workflow_progress_store[workflow_id] = WorkflowProgress(
            status=WorkflowStatus.FAILED,
            current_step=f"Workflow failed: {str(e)}",
            progress_percentage=0.0,
            error_message=str(e)
        )


@router.get("/research-analysis/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """
    Get the current status of a Research-to-Analysis workflow
    """
    if workflow_id not in workflow_progress_store:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    progress = workflow_progress_store[workflow_id]
    
    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        status=progress.status.value,
        current_step=progress.current_step,
        progress_percentage=progress.progress_percentage,
        research_papers_found=progress.research_papers_found,
        analysis_confidence=progress.analysis_confidence,
        estimated_time_remaining=progress.estimated_time_remaining,
        error_message=progress.error_message
    )


@router.get("/research-analysis/{workflow_id}/result", response_model=ResearchAnalysisResponse)
async def get_workflow_result(workflow_id: str):
    """
    Get the complete results of a Research-to-Analysis workflow
    """
    if workflow_id not in workflow_results_store:
        # Check if workflow is still running
        if workflow_id in workflow_progress_store:
            progress = workflow_progress_store[workflow_id]
            if progress.status != WorkflowStatus.COMPLETED:
                raise HTTPException(
                    status_code=202, 
                    detail=f"Workflow still running: {progress.current_step}"
                )
        
        raise HTTPException(status_code=404, detail="Workflow results not found")
    
    workflow_data = workflow_results_store[workflow_id]
    result = workflow_data["result"]
    
    return ResearchAnalysisResponse(
        workflow_id=workflow_id,
        success=result.success,
        query=result.query,
        research_summary=result.research_data.get("response", "")[:500] + "...",
        analysis_summary=result.analysis_data.get("response", {}).get("solution", "")[:500] + "...",
        formatted_output=result.formatted_output,
        citations=result.citations,
        metadata=result.metadata,
        execution_time=result.execution_time,
        error_message=result.error_message
    )


@router.get("/research-analysis/{workflow_id}/export/{format}")
async def export_workflow_result(workflow_id: str, format: str):
    """
    Export workflow results in various formats (markdown, html, pdf)
    """
    if workflow_id not in workflow_results_store:
        raise HTTPException(status_code=404, detail="Workflow results not found")
    
    workflow_data = workflow_results_store[workflow_id]
    result = workflow_data["result"]
    
    if format.lower() == "markdown":
        content = result.formatted_output
        media_type = "text/markdown"
        filename = f"research_analysis_{workflow_id}.md"
        
    elif format.lower() == "html":
        # Convert markdown to HTML (simplified)
        import markdown
        html_content = markdown.markdown(result.formatted_output)
        content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Research Analysis Report</title>
    <style>
        body {{ font-family: 'Times New Roman', serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .citation {{ font-style: italic; margin-left: 20px; }}
        .metadata {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
        media_type = "text/html"
        filename = f"research_analysis_{workflow_id}.html"
        
    elif format.lower() == "json":
        content = json.dumps({
            "workflow_id": workflow_id,
            "query": result.query,
            "research_data": result.research_data,
            "analysis_data": result.analysis_data,
            "citations": result.citations,
            "metadata": result.metadata,
            "formatted_output": result.formatted_output
        }, indent=2)
        media_type = "application/json"
        filename = f"research_analysis_{workflow_id}.json"
        
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")
    
    # Return as downloadable file
    return StreamingResponse(
        iter([content.encode()]),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/research-analysis/{workflow_id}/stream")
async def stream_workflow_progress(workflow_id: str):
    """
    Stream real-time progress updates for a workflow (Server-Sent Events)
    """
    async def generate_progress_stream():
        while True:
            if workflow_id in workflow_progress_store:
                progress = workflow_progress_store[workflow_id]
                
                # Send progress update
                data = {
                    "workflow_id": workflow_id,
                    "status": progress.status.value,
                    "current_step": progress.current_step,
                    "progress_percentage": progress.progress_percentage,
                    "research_papers_found": progress.research_papers_found,
                    "analysis_confidence": progress.analysis_confidence,
                    "timestamp": datetime.now().isoformat()
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                
                # Stop streaming if workflow is completed or failed
                if progress.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                    break
                    
            await asyncio.sleep(1)  # Update every second
    
    return StreamingResponse(
        generate_progress_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/research-analysis/active", response_model=list)
async def get_active_workflows():
    """
    Get list of all active Research-to-Analysis workflows
    """
    active_workflows = []
    
    for workflow_id, progress in workflow_progress_store.items():
        if progress.status not in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
            active_workflows.append({
                "workflow_id": workflow_id,
                "status": progress.status.value,
                "current_step": progress.current_step,
                "progress_percentage": progress.progress_percentage
            })
    
    return active_workflows


@router.delete("/research-analysis/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """
    Cancel a running workflow and clean up resources
    """
    if workflow_id not in workflow_progress_store:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Mark as cancelled (in a real implementation, you'd stop the background task)
    workflow_progress_store[workflow_id] = WorkflowProgress(
        status=WorkflowStatus.FAILED,
        current_step="Workflow cancelled by user",
        progress_percentage=0.0,
        error_message="Cancelled by user"
    )
    
    # Clean up stored data
    if workflow_id in workflow_results_store:
        del workflow_results_store[workflow_id]
    
    return {"message": f"Workflow {workflow_id} cancelled successfully"}
