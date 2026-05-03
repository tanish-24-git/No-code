"""Utility package. Importing the package only loads the settings singleton;
specific submodules are imported directly by callers."""
from app.utils.config import settings

__all__ = ["settings"]
