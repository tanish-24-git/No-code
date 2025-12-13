"""
HuggingFace model cache management.
Downloads models from HF Hub and caches them in MinIO.
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from huggingface_hub import HfApi, snapshot_download, model_info
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from app.storage.object_store import object_store
from app.infra.logging_stream import LogStream
from app.infra.redis import redis_client
from app.utils.logging import get_logger
from app.utils.exceptions import StorageException

logger = get_logger(__name__)


class ModelCache:
    """
    Manages HuggingFace model downloads and MinIO caching.
    Provides search, download, and listing capabilities.
    """
    
    def __init__(self):
        self.hf_api = HfApi()
        self.temp_dir = Path("/tmp/hf_models")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    async def search_hf_models(
        self,
        query: str,
        limit: int = 20,
        filter_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search HuggingFace Hub for models.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            filter_tags: Optional tags to filter by (e.g., ['text-generation'])
        
        Returns:
            List of model metadata dictionaries
        """
        try:
            logger.info("Searching HuggingFace Hub", query=query, limit=limit)
            
            # Search models
            models = self.hf_api.list_models(
                search=query,
                limit=limit,
                filter=filter_tags,
                sort="downloads",
                direction=-1
            )
            
            results = []
            for model in models:
                try:
                    # Get detailed model info
                    info = model_info(model.id)
                    
                    results.append({
                        "id": model.id,
                        "name": model.id.split("/")[-1] if "/" in model.id else model.id,
                        "author": model.id.split("/")[0] if "/" in model.id else "unknown",
                        "downloads": getattr(model, "downloads", 0),
                        "likes": getattr(model, "likes", 0),
                        "tags": getattr(model, "tags", []),
                        "pipeline_tag": getattr(model, "pipeline_tag", None),
                        "library": getattr(info, "library_name", "transformers"),
                        "created_at": getattr(model, "created_at", None)
                    })
                except Exception as e:
                    logger.warn("Failed to get model info", model_id=model.id, error=str(e))
                    continue
            
            logger.info("Search completed", results_count=len(results))
            return results
        
        except Exception as e:
            logger.error("HuggingFace search failed", error=str(e))
            raise StorageException(f"Failed to search HuggingFace Hub: {str(e)}")
    
    async def check_model_exists(self, model_id: str) -> bool:
        """
        Check if model exists in MinIO cache.
        
        Args:
            model_id: HuggingFace model ID (e.g., 'Qwen/Qwen2-0.5B-Instruct')
        
        Returns:
            True if model is cached, False otherwise
        """
        try:
            # Normalize model_id to object path
            object_prefix = model_id.replace("/", "--")
            
            # List objects in models bucket
            objects = object_store.list_objects(prefix=object_prefix, bucket_type='models')
            
            return len(objects) > 0
        
        except Exception as e:
            logger.error("Failed to check model existence", model_id=model_id, error=str(e))
            return False
    
    async def get_model_size(self, model_id: str) -> Optional[int]:
        """
        Get cached model size in bytes.
        
        Args:
            model_id: HuggingFace model ID
        
        Returns:
            Size in bytes, or None if not cached
        """
        try:
            object_prefix = model_id.replace("/", "--")
            
            # Get all objects for this model
            bucket_name = object_store.buckets['models']
            objects = object_store.client.list_objects(bucket_name, prefix=object_prefix)
            
            total_size = sum(obj.size for obj in objects)
            return total_size if total_size > 0 else None
        
        except Exception as e:
            logger.error("Failed to get model size", model_id=model_id, error=str(e))
            return None
    
    async def download_hf_model(
        self,
        model_id: str,
        run_id: str,
        force_download: bool = False
    ) -> Dict[str, Any]:
        """
        Download HuggingFace model and cache in MinIO.
        Streams progress via LogStream.
        
        Args:
            model_id: HuggingFace model ID
            run_id: Run ID for logging
            force_download: Force re-download even if cached
        
        Returns:
            Dictionary with model metadata and S3 path
        """
        log_stream = LogStream(redis_client.client)
        
        try:
            # Check if already cached
            if not force_download and await self.check_model_exists(model_id):
                await log_stream.publish_log(
                    run_id=run_id,
                    agent="model_cache",
                    level="INFO",
                    message=f"Model {model_id} already cached"
                )
                
                size_bytes = await self.get_model_size(model_id)
                object_prefix = model_id.replace("/", "--")
                
                return {
                    "model_id": model_id,
                    "s3_path": f"s3://models/{object_prefix}",
                    "size_bytes": size_bytes,
                    "size_gb": round(size_bytes / (1024**3), 2) if size_bytes else 0,
                    "status": "cached"
                }
            
            await log_stream.publish_log(
                run_id=run_id,
                agent="model_cache",
                level="INFO",
                message=f"Starting download of {model_id} from HuggingFace Hub"
            )
            
            # Download to temp directory
            temp_model_path = self.temp_dir / model_id.replace("/", "--")
            
            await log_stream.publish_log(
                run_id=run_id,
                agent="model_cache",
                level="INFO",
                message="Downloading model files...",
                metadata={"step": "download"}
            )
            
            # Download model snapshot
            local_path = snapshot_download(
                repo_id=model_id,
                local_dir=str(temp_model_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            await log_stream.publish_log(
                run_id=run_id,
                agent="model_cache",
                level="INFO",
                message="Download complete, uploading to MinIO...",
                metadata={"step": "upload"}
            )
            
            # Upload all files to MinIO
            object_prefix = model_id.replace("/", "--")
            uploaded_files = []
            total_size = 0
            
            for file_path in Path(local_path).rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    object_name = f"{object_prefix}/{relative_path}"
                    
                    # Upload file
                    object_store.upload_file(
                        file_path=str(file_path),
                        object_name=object_name,
                        bucket_type='models'
                    )
                    
                    uploaded_files.append(str(relative_path))
                    total_size += file_path.stat().st_size
            
            # Clean up temp directory
            shutil.rmtree(temp_model_path, ignore_errors=True)
            
            await log_stream.publish_log(
                run_id=run_id,
                agent="model_cache",
                level="INFO",
                message=f"Model {model_id} cached successfully",
                metadata={
                    "files_uploaded": len(uploaded_files),
                    "total_size_gb": round(total_size / (1024**3), 2)
                }
            )
            
            return {
                "model_id": model_id,
                "s3_path": f"s3://models/{object_prefix}",
                "size_bytes": total_size,
                "size_gb": round(total_size / (1024**3), 2),
                "files_count": len(uploaded_files),
                "status": "downloaded"
            }
        
        except RepositoryNotFoundError:
            await log_stream.publish_log(
                run_id=run_id,
                agent="model_cache",
                level="ERROR",
                message=f"Model {model_id} not found on HuggingFace Hub"
            )
            raise StorageException(f"Model {model_id} not found on HuggingFace Hub")
        
        except Exception as e:
            await log_stream.publish_log(
                run_id=run_id,
                agent="model_cache",
                level="ERROR",
                message=f"Failed to download model: {str(e)}"
            )
            logger.error("Model download failed", model_id=model_id, error=str(e))
            raise StorageException(f"Failed to download model: {str(e)}")
    
    async def list_cached_models(self) -> List[Dict[str, Any]]:
        """
        List all cached models in MinIO.
        
        Returns:
            List of cached model metadata
        """
        try:
            bucket_name = object_store.buckets['models']
            
            # Get all objects in models bucket
            objects = object_store.client.list_objects(bucket_name, recursive=False)
            
            # Group by model prefix
            models_dict = {}
            for obj in objects:
                # Extract model prefix (before first /)
                parts = obj.object_name.split("/")
                if len(parts) > 0:
                    model_prefix = parts[0]
                    
                    if model_prefix not in models_dict:
                        models_dict[model_prefix] = {
                            "model_id": model_prefix.replace("--", "/"),
                            "s3_path": f"s3://models/{model_prefix}",
                            "files": [],
                            "total_size": 0
                        }
                    
                    models_dict[model_prefix]["files"].append(obj.object_name)
                    models_dict[model_prefix]["total_size"] += obj.size
            
            # Convert to list and add size in GB
            models = []
            for model_data in models_dict.values():
                model_data["size_gb"] = round(model_data["total_size"] / (1024**3), 2)
                model_data["files_count"] = len(model_data["files"])
                del model_data["files"]  # Don't return full file list
                models.append(model_data)
            
            logger.info("Listed cached models", count=len(models))
            return models
        
        except Exception as e:
            logger.error("Failed to list cached models", error=str(e))
            raise StorageException(f"Failed to list cached models: {str(e)}")


# Global model cache instance
model_cache = ModelCache()
