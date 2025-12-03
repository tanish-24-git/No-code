# services/ops/shared/artifact_ops.py
from typing import List, Dict, Any, Tuple
import shutil
OP_REGISTRY = {}

def atomic_writes(dest: Path, **kwargs) -> Tuple[Dict, Dict, List[str]]:
    # Engine handles
    return {}, {"atomic": True}, []

def temporary_backups(file_path: str, **kwargs) -> Tuple[Dict, Dict, List[str]]:
    backup = Path(file_path).with_suffix('.backup')
    shutil.copy(file_path, backup)
    return {}, {"backup_created": str(backup)}, []

def gzip_archived(dest: Path, **kwargs) -> Tuple[Dict, Dict, List[str]]:
    # Engine handles gzip
    return {}, {"gzipped": True}, []

def restore_last_stable(backup_path: Path, target: Path, **kwargs) -> Tuple[Dict, Dict, List[str]]:
    if backup_path.exists():
        shutil.move(str(backup_path), str(target))
    return {}, {"restored": True}, []

def metadata_transform(dest: Path, plan: Dict, **kwargs) -> Tuple[Dict, Dict, List[str]]:
    # Engine sidecar
    return {}, {"meta_applied": plan}, []

OP_REGISTRY = {
    "atomic_writes": atomic_writes,
    "temporary_backups": temporary_backups,
    "gzip_archived": gzip_archived,
    "restore_last_stable": restore_last_stable,
    "metadata_transform": metadata_transform,
}