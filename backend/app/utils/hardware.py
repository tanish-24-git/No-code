"""Cheap, dependency-tolerant hardware detection. Used by the agent to suggest
configs and by the API for /health."""
from __future__ import annotations

import platform
from typing import Any


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": "cpu",
        "gpu_name": None,
        "vram_gb": None,
        "cuda_version": None,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "recommended_trainer": "lora",
    }
    try:
        import torch  # type: ignore
    except Exception:
        info["recommended_trainer"] = "lora"
        return info

    try:
        if torch.cuda.is_available():
            idx = 0
            props = torch.cuda.get_device_properties(idx)
            vram_gb = round(props.total_memory / (1024**3), 2)
            info.update(
                device="cuda",
                gpu_name=props.name,
                vram_gb=vram_gb,
                cuda_version=getattr(torch.version, "cuda", None),
            )
            if vram_gb < 6:
                info["recommended_trainer"] = "qlora"
            elif vram_gb < 16:
                info["recommended_trainer"] = "lora"
            else:
                info["recommended_trainer"] = "lora"  # full requires careful setup; default to LoRA
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info.update(device="mps", gpu_name="Apple Silicon")
            info["recommended_trainer"] = "lora"
    except Exception:
        # Torch present but unhappy — stay on CPU defaults rather than crash.
        pass
    return info
