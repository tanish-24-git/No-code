# services/ops/shared/safeguards_ops.py
from typing import List, Dict, Any, Tuple
import psutil
import pandas as pd
import anyio
OP_REGISTRY = {}

def file_size_check(file_path: str, max_size_mb: int = 100, **kwargs) -> Tuple[Dict, Dict, List[str]]:
    size = os.path.getsize(file_path) / (1024*1024)
    if size > max_size_mb:
        raise ValueError(f"File too large: {size}MB")
    return {}, {"size_ok": True}, []

def memory_guard(df: pd.DataFrame, threshold_gb: float = 2.0, **kwargs) -> Tuple[pd.DataFrame, Dict, List[str]]:
    mem = psutil.virtual_memory().available / (1024**3)
    if mem < threshold_gb:
        raise MemoryError("Low memory")
    return df, {"mem_ok": True}, []

def cpu_throttling(**kwargs) -> Tuple[Dict, Dict, List[str]]:
    cpu = psutil.cpu_percent(interval=1)
    if cpu > 90:
        anyio.sleep(1)  # Throttle
    return {}, {"cpu_throttled": cpu}, []

def timeout_operation(func, timeout: int = 300, **kwargs) -> Any:
    # Wrap in anyio.move_on_after
    return anyio.move_on_after(timeout, func)

def kill_switch(**kwargs) -> Tuple[Dict, Dict, List[str]]:
    # Placeholder for signal handler
    return {}, {}, []

def disk_usage_check(upload_dir: str, threshold_gb: float = 10, **kwargs) -> Tuple[Dict, Dict, List[str]]:
    usage = psutil.disk_usage(upload_dir).used / (1024**3)
    if usage > threshold_gb:
        raise RuntimeError("Low disk")
    return {}, {"disk_ok": True}, []

OP_REGISTRY = {
    "file_size_check": file_size_check,
    "memory_guard": memory_guard,
    "cpu_throttling": cpu_throttling,
    "timeout_operation": timeout_operation,
    "kill_switch": kill_switch,
    "disk_usage_check": disk_usage_check,
}