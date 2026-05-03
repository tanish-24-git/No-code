"""Storage package — JSON-on-disk store. Legacy MinIO/registry modules
remain in this directory for reference but are no longer auto-imported."""
from app.storage import store

__all__ = ["store"]
