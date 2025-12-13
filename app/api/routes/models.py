"""
Models API routes for trained model management and HuggingFace integration.
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, List
from pydantic import BaseModel
import uuid
from app.storage.model_registry import model_registry
from app.storage.model_cache import model_cache
from app.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])


# Request/Response schemas
class ModelDownloadRequest(BaseModel):
    force_download: bool = False


class ModelDownloadResponse(BaseModel):
    run_id: str
    model_id: str
    status: str
    message: str


@router.get("/search")
async def search_models(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, le=100, description="Maximum results"),
    tags: Optional[str] = Query(None, description="Comma-separated tags filter")
):
    """Search HuggingFace Hub for models."""
    try:
        filter_tags = tags.split(",") if tags else None
        
        results = await model_cache.search_hf_models(
            query=q,
            limit=limit,
            filter_tags=filter_tags
        )
        
        return {
            "query": q,
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        logger.error("Model search failed", query=q, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cached")
async def list_cached_models():
    """List all models cached in MinIO."""
    try:
        models = await model_cache.list_cached_models()
        
        return {
            "models": models,
            "count": len(models)
        }
    
    except Exception as e:
        logger.error("Failed to list cached models", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download/{model_id:path}")
async def download_model_endpoint(
    model_id: str,
    request: ModelDownloadRequest = ModelDownloadRequest(),
    background_tasks: BackgroundTasks = None
):
    """
    Download a HuggingFace model and cache in MinIO.
    Returns immediately with a run_id for tracking progress via logs.
    """
    try:
        # Generate run ID for tracking
        run_id = f"model_download_{uuid.uuid4().hex[:8]}"
        
        logger.info("Model download requested", model_id=model_id, run_id=run_id)
        
        # Start download (this will stream progress via LogStream)
        result = await model_cache.download_hf_model(
            model_id=model_id,
            run_id=run_id,
            force_download=request.force_download
        )
        
        return {
            "run_id": run_id,
            "model_id": model_id,
            "status": result["status"],
            "s3_path": result["s3_path"],
            "size_gb": result["size_gb"],
            "message": f"Model {'cached' if result['status'] == 'cached' else 'downloaded'} successfully"
        }
    
    except Exception as e:
        logger.error("Model download failed", model_id=model_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_models(
    limit: int = Query(20, le=100),
    offset: int = Query(0, ge=0),
    base_model: Optional[str] = None
):
    """List all trained models with pagination."""
    try:
        models = model_registry.list_models(prefix="")
        
        # Apply base_model filter if specified
        if base_model:
            models = [m for m in models if m.get("base_model") == base_model]
        
        # Pagination
        total = len(models)
        paginated_models = models[offset:offset + limit]
        
        return {
            "models": paginated_models,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
    
    except Exception as e:
        logger.error("Failed to list models", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}")
async def get_model(run_id: str):
    """Get model details by run ID."""
    try:
        model_metadata = model_registry.get_model_metadata(run_id)
        
        if not model_metadata:
            raise HTTPException(status_code=404, detail=f"Model for run {run_id} not found")
        
        return model_metadata
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/card")
async def get_model_card(run_id: str):
    """Get model card in Markdown format."""
    try:
        model_metadata = model_registry.get_model_metadata(run_id)
        
        if not model_metadata:
            raise HTTPException(status_code=404, detail=f"Model for run {run_id} not found")
        
        card = model_registry.generate_model_card(run_id, model_metadata)
        
        return {"content": card, "format": "markdown"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model card", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/download")
async def download_model(run_id: str, format: str = "adapter"):
    """Get download URL for model."""
    try:
        model_metadata = model_registry.get_model_metadata(run_id)
        
        if not model_metadata:
            raise HTTPException(status_code=404, detail=f"Model for run {run_id} not found")
        
        exports = model_metadata.get("exports", {})
        
        if format not in exports:
            raise HTTPException(
                status_code=404,
                detail=f"Export format '{format}' not available. Available: {list(exports.keys())}"
            )
        
        export_path = exports[format]["path"]
        
        return {
            "format": format,
            "path": export_path,
            "note": "Download from MinIO using this path"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download model", run_id=run_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
