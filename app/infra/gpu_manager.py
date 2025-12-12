"""
GPU resource management and device allocation.
"""
import os
import torch
from typing import Optional, List
from app.utils.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class GPUManager:
    """
    GPU resource management for training jobs.
    Handles device detection, allocation, and fallback to CPU.
    """
    
    def __init__(self):
        self.gpu_enabled = settings.gpu_enabled
        self.cuda_devices = settings.cuda_visible_devices
        self._available_devices: Optional[List[int]] = None
        
        # Set CUDA_VISIBLE_DEVICES if specified
        if self.cuda_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_devices
        
        self._detect_devices()
    
    def _detect_devices(self):
        """Detect available GPU devices."""
        if not self.gpu_enabled:
            logger.info("GPU disabled by configuration")
            self._available_devices = []
            return
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self._available_devices = []
            return
        
        device_count = torch.cuda.device_count()
        self._available_devices = list(range(device_count))
        
        logger.info(
            "GPU devices detected",
            count=device_count,
            devices=self._available_devices
        )
        
        # Log device info
        for i in self._available_devices:
            props = torch.cuda.get_device_properties(i)
            logger.info(
                "GPU device info",
                device_id=i,
                name=props.name,
                memory_gb=round(props.total_memory / 1024**3, 2),
                compute_capability=f"{props.major}.{props.minor}"
            )
    
    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return len(self._available_devices) > 0 if self._available_devices else False
    
    def get_device(self, device_id: Optional[int] = None) -> str:
        """
        Get device string for PyTorch.
        
        Args:
            device_id: Specific GPU device ID (optional)
        
        Returns:
            Device string ('cuda:0', 'cuda:1', or 'cpu')
        """
        if not self.has_gpu():
            logger.info("Using CPU device")
            return "cpu"
        
        if device_id is not None and device_id in self._available_devices:
            logger.info("Using GPU device", device_id=device_id)
            return f"cuda:{device_id}"
        
        # Default to first available GPU
        default_device = self._available_devices[0]
        logger.info("Using default GPU device", device_id=default_device)
        return f"cuda:{default_device}"
    
    def get_device_map(self, model_name: str = None) -> Optional[str]:
        """
        Get device map for model loading.
        
        Args:
            model_name: Model name (for logging)
        
        Returns:
            'auto' for multi-GPU, None for single GPU/CPU
        """
        if not self.has_gpu():
            return None
        
        if len(self._available_devices) > 1:
            logger.info(
                "Using automatic device mapping for multi-GPU",
                model=model_name,
                gpus=len(self._available_devices)
            )
            return "auto"
        
        return None
    
    def get_memory_stats(self, device_id: int = 0) -> dict:
        """
        Get GPU memory statistics.
        
        Args:
            device_id: GPU device ID
        
        Returns:
            Dictionary with memory stats
        """
        if not self.has_gpu():
            return {"error": "No GPU available"}
        
        if device_id not in self._available_devices:
            return {"error": f"Invalid device ID: {device_id}"}
        
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory
        
        return {
            "device_id": device_id,
            "allocated_gb": round(allocated / 1024**3, 2),
            "reserved_gb": round(reserved / 1024**3, 2),
            "total_gb": round(total / 1024**3, 2),
            "free_gb": round((total - reserved) / 1024**3, 2)
        }
    
    def clear_cache(self, device_id: Optional[int] = None):
        """Clear GPU cache."""
        if self.has_gpu():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared", device_id=device_id)


# Global GPU manager instance
gpu_manager = GPUManager()
