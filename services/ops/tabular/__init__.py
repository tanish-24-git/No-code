# services/ops/tabular/__init__.py
from .missing_ops import OP_REGISTRY as missing_registry
from .type_fix_ops import OP_REGISTRY as type_fix_registry
from .encoding_ops import OP_REGISTRY as encoding_registry
from .scaling_ops import OP_REGISTRY as scaling_registry
from .fe_ops import OP_REGISTRY as fe_registry
from .selection_ops import OP_REGISTRY as selection_registry
from .row_ops import OP_REGISTRY as row_registry
from .cleaning_ops import OP_REGISTRY as cleaning_registry

OP_REGISTRY = {
    **missing_registry,
    **type_fix_registry,
    **encoding_registry,
    **scaling_registry,
    **fe_registry,
    **selection_registry,
    **row_registry,
    **cleaning_registry
}