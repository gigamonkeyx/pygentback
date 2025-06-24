"""
Model Performance API Routes

This module provides REST API endpoints for managing AI model performance
metrics, usefulness scores, and recommendations.
"""

import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from sqlalchemy import desc, and_, or_

from ..models import (
    ModelPerformanceData, ModelPerformanceResponse, ModelPerformanceListResponse,
    ModelRecommendationRequest, ModelRecommendationResponse,
    ModelRatingRequest, ModelRatingResponse, ErrorResponse
)
from ...database.models import ModelPerformance

logger = logging.getLogger(__name__)

router = APIRouter()

# Global state for dependencies
_db_manager = None


def set_db_manager(db_manager):
    """Set the database manager for this module"""
    global _db_manager
    _db_manager = db_manager


async def get_db():
    """Get database session dependency"""
    if not _db_manager:
        raise HTTPException(status_code=503, detail="Database not available")

    async with _db_manager.get_session() as session:
        yield session


@router.get("/", response_model=ModelPerformanceListResponse)
async def list_models(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
    architecture: Optional[str] = Query(None, description="Filter by architecture"),
    min_usefulness: Optional[float] = Query(None, ge=0, le=100, description="Minimum usefulness score"),
    sort_by: str = Query("usefulness_score", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    db: AsyncSession = Depends(get_db)
):
    """List all model performance records with filtering and pagination"""
    try:
        # Build query
        query = select(ModelPerformance)

        # Apply filters
        if architecture:
            query = query.where(ModelPerformance.architecture == architecture)

        if min_usefulness is not None:
            query = query.where(ModelPerformance.usefulness_score >= min_usefulness)

        # Apply sorting
        sort_column = getattr(ModelPerformance, sort_by, ModelPerformance.usefulness_score)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(sort_column)

        # Get total count
        count_query = select(func.count(ModelPerformance.id))
        if architecture:
            count_query = count_query.where(ModelPerformance.architecture == architecture)
        if min_usefulness is not None:
            count_query = count_query.where(ModelPerformance.usefulness_score >= min_usefulness)

        count_result = await db.execute(count_query)
        total_count = count_result.scalar()

        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        # Execute query
        result = await db.execute(query)
        models = result.scalars().all()
        
        # Convert to response models
        model_responses = []
        for model in models:
            model_responses.append(ModelPerformanceResponse(
                id=model.id,
                model_name=model.model_name,
                model_size_gb=model.model_size_gb,
                usefulness_score=model.usefulness_score,
                speed_rating=model.speed_rating,
                speed_seconds=model.speed_seconds,
                gpu_utilization=model.gpu_utilization,
                gpu_layers_offloaded=model.gpu_layers_offloaded,
                gpu_layers_total=model.gpu_layers_total,
                context_window=model.context_window,
                parameters_billions=model.parameters_billions,
                architecture=model.architecture,
                best_use_cases=model.best_use_cases,
                cost_per_token=model.cost_per_token,
                last_tested=model.last_tested,
                test_results=model.test_results,
                user_ratings=model.user_ratings,
                performance_metrics=model.performance_metrics,
                created_at=model.created_at,
                updated_at=model.updated_at
            ))
        
        return ModelPerformanceListResponse(
            models=model_responses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/{model_name}", response_model=ModelPerformanceResponse)
async def get_model(model_name: str, db: AsyncSession = Depends(get_db)):
    """Get specific model performance data"""
    try:
        query = select(ModelPerformance).where(ModelPerformance.model_name == model_name)
        result = await db.execute(query)
        model = result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
        
        return ModelPerformanceResponse(
            id=model.id,
            model_name=model.model_name,
            model_size_gb=model.model_size_gb,
            usefulness_score=model.usefulness_score,
            speed_rating=model.speed_rating,
            speed_seconds=model.speed_seconds,
            gpu_utilization=model.gpu_utilization,
            gpu_layers_offloaded=model.gpu_layers_offloaded,
            gpu_layers_total=model.gpu_layers_total,
            context_window=model.context_window,
            parameters_billions=model.parameters_billions,
            architecture=model.architecture,
            best_use_cases=model.best_use_cases,
            cost_per_token=model.cost_per_token,
            last_tested=model.last_tested,
            test_results=model.test_results,
            user_ratings=model.user_ratings,
            performance_metrics=model.performance_metrics,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model: {str(e)}")


@router.post("/", response_model=ModelPerformanceResponse)
async def create_model_performance(
    model_data: ModelPerformanceData,
    db: AsyncSession = Depends(get_db)
):
    """Create or update model performance record"""
    try:
        # Check if model already exists
        query = select(ModelPerformance).where(ModelPerformance.model_name == model_data.model_name)
        result = await db.execute(query)
        existing_model = result.scalar_one_or_none()

        if existing_model:
            # Update existing model
            for field, value in model_data.dict().items():
                setattr(existing_model, field, value)
            existing_model.last_tested = datetime.utcnow()
            model = existing_model
        else:
            # Create new model
            model = ModelPerformance(
                **model_data.dict(),
                last_tested=datetime.utcnow()
            )
            db.add(model)

        await db.commit()
        await db.refresh(model)
        
        return ModelPerformanceResponse(
            id=model.id,
            model_name=model.model_name,
            model_size_gb=model.model_size_gb,
            usefulness_score=model.usefulness_score,
            speed_rating=model.speed_rating,
            speed_seconds=model.speed_seconds,
            gpu_utilization=model.gpu_utilization,
            gpu_layers_offloaded=model.gpu_layers_offloaded,
            gpu_layers_total=model.gpu_layers_total,
            context_window=model.context_window,
            parameters_billions=model.parameters_billions,
            architecture=model.architecture,
            best_use_cases=model.best_use_cases,
            cost_per_token=model.cost_per_token,
            last_tested=model.last_tested,
            test_results=model.test_results,
            user_ratings=model.user_ratings,
            performance_metrics=model.performance_metrics,
            created_at=model.created_at,
            updated_at=model.updated_at
        )
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating/updating model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save model performance: {str(e)}")


@router.post("/recommend", response_model=ModelRecommendationResponse)
async def recommend_models(
    request: ModelRecommendationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Get model recommendations based on task requirements"""
    try:
        # Build query based on requirements
        query = select(ModelPerformance)

        # Apply filters
        if request.max_size_gb:
            query = query.where(ModelPerformance.model_size_gb <= request.max_size_gb)

        if request.min_usefulness_score:
            query = query.where(ModelPerformance.usefulness_score >= request.min_usefulness_score)

        # Task-specific filtering
        if request.task_type.lower() == "coding":
            query = query.where(
                or_(
                    ModelPerformance.architecture.in_(["deepseek-coder", "codellama"]),
                    ModelPerformance.best_use_cases.contains(["coding"])
                )
            )
        elif request.task_type.lower() == "reasoning":
            query = query.where(
                or_(
                    ModelPerformance.architecture.in_(["deepseek-r1", "qwen3"]),
                    ModelPerformance.best_use_cases.contains(["reasoning"])
                )
            )

        # Priority-based sorting
        if request.priority == "speed":
            query = query.order_by(ModelPerformance.speed_seconds.asc())
        elif request.priority == "quality":
            query = query.order_by(desc(ModelPerformance.usefulness_score))
        else:  # balanced
            # Weighted score: 60% usefulness, 40% speed (inverted)
            query = query.order_by(desc(ModelPerformance.usefulness_score))

        query = query.limit(5)  # Top 5 recommendations
        result = await db.execute(query)
        models = result.scalars().all()
        
        # Convert to response models
        recommended_models = []
        for model in models:
            recommended_models.append(ModelPerformanceResponse(
                id=model.id,
                model_name=model.model_name,
                model_size_gb=model.model_size_gb,
                usefulness_score=model.usefulness_score,
                speed_rating=model.speed_rating,
                speed_seconds=model.speed_seconds,
                gpu_utilization=model.gpu_utilization,
                gpu_layers_offloaded=model.gpu_layers_offloaded,
                gpu_layers_total=model.gpu_layers_total,
                context_window=model.context_window,
                parameters_billions=model.parameters_billions,
                architecture=model.architecture,
                best_use_cases=model.best_use_cases,
                cost_per_token=model.cost_per_token,
                last_tested=model.last_tested,
                test_results=model.test_results,
                user_ratings=model.user_ratings,
                performance_metrics=model.performance_metrics,
                created_at=model.created_at,
                updated_at=model.updated_at
            ))
        
        # Generate reasoning
        reasoning = f"Recommended {len(recommended_models)} models for {request.task_type} tasks "
        reasoning += f"with {request.priority} priority. "
        if request.max_size_gb:
            reasoning += f"Filtered by max size {request.max_size_gb}GB. "
        if request.min_usefulness_score:
            reasoning += f"Minimum usefulness score {request.min_usefulness_score}. "
        
        return ModelRecommendationResponse(
            recommended_models=recommended_models,
            reasoning=reasoning,
            task_type=request.task_type,
            priority=request.priority
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@router.post("/{model_name}/rate", response_model=ModelRatingResponse)
async def rate_model(
    model_name: str,
    rating_request: ModelRatingRequest,
    db: AsyncSession = Depends(get_db)
):
    """Add user rating for a model"""
    try:
        # Get the model
        query = select(ModelPerformance).where(ModelPerformance.model_name == model_name)
        result = await db.execute(query)
        model = result.scalar_one_or_none()

        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        # Add new rating
        new_rating = {
            "rating": rating_request.rating,
            "task_type": rating_request.task_type,
            "feedback": rating_request.feedback,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Update user ratings
        if not model.user_ratings:
            model.user_ratings = []

        model.user_ratings.append(new_rating)

        # Calculate new average rating
        ratings = [r["rating"] for r in model.user_ratings]
        new_average = sum(ratings) / len(ratings)

        # Update usefulness score based on user feedback
        # Weight: 70% original score, 30% user ratings
        original_weight = 0.7
        user_weight = 0.3
        model.usefulness_score = (
            model.usefulness_score * original_weight +
            (new_average * 20) * user_weight  # Convert 1-5 scale to 0-100
        )

        await db.commit()

        return ModelRatingResponse(
            success=True,
            message=f"Rating added successfully for {model_name}",
            new_average_rating=new_average
        )

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error rating model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rate model: {str(e)}")


@router.delete("/{model_name}")
async def delete_model(model_name: str, db: AsyncSession = Depends(get_db)):
    """Delete model performance record"""
    try:
        query = select(ModelPerformance).where(ModelPerformance.model_name == model_name)
        result = await db.execute(query)
        model = result.scalar_one_or_none()

        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        await db.delete(model)
        await db.commit()

        return {"message": f"Model '{model_name}' deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting model {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.get("/stats/summary")
async def get_model_stats(db: AsyncSession = Depends(get_db)):
    """Get model performance statistics summary"""
    try:
        query = select(ModelPerformance)
        result = await db.execute(query)
        models = result.scalars().all()

        if not models:
            return {
                "total_models": 0,
                "average_usefulness": 0,
                "best_model": None,
                "fastest_model": None,
                "architectures": {}
            }

        # Calculate statistics
        total_models = len(models)
        average_usefulness = sum(m.usefulness_score for m in models) / total_models

        # Find best and fastest models
        best_model = max(models, key=lambda m: m.usefulness_score)
        fastest_model = min(
            [m for m in models if m.speed_seconds],
            key=lambda m: m.speed_seconds,
            default=None
        )

        # Architecture distribution
        architectures = {}
        for model in models:
            arch = model.architecture
            if arch not in architectures:
                architectures[arch] = {"count": 0, "avg_usefulness": 0}
            architectures[arch]["count"] += 1

        # Calculate average usefulness per architecture
        for arch in architectures:
            arch_models = [m for m in models if m.architecture == arch]
            architectures[arch]["avg_usefulness"] = sum(
                m.usefulness_score for m in arch_models
            ) / len(arch_models)

        return {
            "total_models": total_models,
            "average_usefulness": round(average_usefulness, 2),
            "best_model": {
                "name": best_model.model_name,
                "usefulness_score": best_model.usefulness_score,
                "architecture": best_model.architecture
            },
            "fastest_model": {
                "name": fastest_model.model_name,
                "speed_seconds": fastest_model.speed_seconds,
                "architecture": fastest_model.architecture
            } if fastest_model else None,
            "architectures": architectures
        }

    except Exception as e:
        logger.error(f"Error getting model stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model stats: {str(e)}")
