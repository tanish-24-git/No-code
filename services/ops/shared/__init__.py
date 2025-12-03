# services/ops/shared/__init__.py
from .parsing_ops import OP_REGISTRY as parsing_registry
from .safeguards_ops import OP_REGISTRY as safeguards_registry
from .artifact_ops import OP_REGISTRY as artifact_registry

OP_REGISTRY = {
    **parsing_registry,
    **safeguards_registry,
    **artifact_registry
}