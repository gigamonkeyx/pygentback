#!/usr/bin/env python3
"""
Feature Registry API Integration

Adds API endpoints for the feature registry system to the FastAPI backend.
This allows the frontend to display and manage discovered features.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional
import logging

from ...feature_registry.core import FeatureRegistry, FeatureType, FeatureStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/features", tags=["Feature Registry"])

# Global registry instance
feature_registry = FeatureRegistry(".")


@router.get("/", response_model=Dict[str, Any])
async def get_all_features():
    """Get all discovered features"""
    try:
        await feature_registry.load_registry()
        features = {name: feature.to_dict() for name, feature in feature_registry.features.items()}
        return {
            "total_features": len(features),
            "features": features
        }
    except Exception as e:
        logger.error(f"Error getting features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-type/{feature_type}")
async def get_features_by_type(feature_type: str):
    """Get features filtered by type"""
    try:
        await feature_registry.load_registry()
        
        # Validate feature type
        try:
            filter_type = FeatureType(feature_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid feature type: {feature_type}")
        
        filtered_features = {
            name: feature.to_dict() 
            for name, feature in feature_registry.features.items()
            if feature.type == filter_type
        }
        
        return {
            "feature_type": feature_type,
            "count": len(filtered_features),
            "features": filtered_features
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error filtering features by type: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_feature_health():
    """Get feature health analysis"""
    try:
        await feature_registry.load_registry()
        health_analysis = await feature_registry.analyze_feature_health()
        return health_analysis
    except Exception as e:
        logger.error(f"Error analyzing feature health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/discover")
async def discover_features(background_tasks: BackgroundTasks):
    """Trigger feature discovery (runs in background)"""
    try:
        # Run discovery in background
        background_tasks.add_task(run_feature_discovery)
        
        return {
            "message": "Feature discovery started",
            "status": "running"
        }
    except Exception as e:
        logger.error(f"Error starting feature discovery: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documentation")
async def get_feature_documentation():
    """Get generated feature documentation"""
    try:
        await feature_registry.load_registry()
        documentation = await feature_registry.generate_documentation()
        
        return {
            "documentation": documentation,
            "format": "markdown"
        }
    except Exception as e:
        logger.error(f"Error generating documentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_feature_types():
    """Get all available feature types"""
    return {
        "feature_types": [feature_type.value for feature_type in FeatureType]
    }


@router.get("/statuses")
async def get_feature_statuses():
    """Get all available feature statuses"""
    return {
        "feature_statuses": [status.value for status in FeatureStatus]
    }


@router.get("/search")
async def search_features(q: str, feature_type: Optional[str] = None):
    """Search features by name or description"""
    try:
        await feature_registry.load_registry()
        
        # Filter by type if specified
        features_to_search = feature_registry.features
        if feature_type:
            try:
                filter_type = FeatureType(feature_type)
                features_to_search = {
                    name: feature for name, feature in features_to_search.items()
                    if feature.type == filter_type
                }
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid feature type: {feature_type}")
        
        # Search by name or description
        matching_features = {}
        query_lower = q.lower()
        
        for name, feature in features_to_search.items():
            if (query_lower in name.lower() or 
                query_lower in feature.description.lower() or
                query_lower in feature.file_path.lower()):
                matching_features[name] = feature.to_dict()
        
        return {
            "query": q,
            "feature_type_filter": feature_type,
            "matches": len(matching_features),
            "features": matching_features
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_feature_discovery():
    """Background task to run feature discovery"""
    try:
        logger.info("Starting background feature discovery")
        
        # Load existing registry
        await feature_registry.load_registry()
        
        # Discover all features
        await feature_registry.discover_all_features()
        
        # Save updated registry
        await feature_registry.save_registry()
        
        # Generate documentation
        documentation = await feature_registry.generate_documentation()
        
        # Save documentation
        from pathlib import Path
        import aiofiles
        
        doc_path = Path("docs/COMPLETE_FEATURE_REGISTRY.md")
        doc_path.parent.mkdir(exist_ok=True)
        async with aiofiles.open(doc_path, 'w') as f:
            await f.write(documentation)
        
        logger.info("Background feature discovery completed")
        
    except Exception as e:
        logger.error(f"Error in background feature discovery: {e}")


# Include router in main FastAPI app
def setup_feature_registry_routes(app):
    """Setup feature registry routes in the main FastAPI app"""
    app.include_router(router)
