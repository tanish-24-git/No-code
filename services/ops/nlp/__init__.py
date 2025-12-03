# services/ops/nlp/__init__.py
from .cleaning_ops import OP_REGISTRY as cleaning_registry
from .tokenization_ops import OP_REGISTRY as tokenization_registry
from .prompt_ops import OP_REGISTRY as prompt_registry
from .chunking_ops import OP_REGISTRY as chunking_registry
from .dedup_ops import OP_REGISTRY as dedup_registry
from .pii_ops import OP_REGISTRY as pii_registry
from .jsonl_ops import OP_REGISTRY as jsonl_registry
from .splitting_ops import OP_REGISTRY as splitting_registry
from .metadata_ops import OP_REGISTRY as metadata_registry

OP_REGISTRY = {
    **cleaning_registry,
    **tokenization_registry,
    **prompt_registry,
    **chunking_registry,
    **dedup_registry,
    **pii_registry,
    **jsonl_registry,
    **splitting_registry,
    **metadata_registry
}