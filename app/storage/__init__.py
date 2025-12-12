"""Storage package initialization."""
from app.storage.object_store import object_store, ObjectStore
from app.storage.model_registry import model_registry, ModelRegistry

__all__ = [
    "object_store",
    "ObjectStore",
    "model_registry",
    "ModelRegistry"
]
