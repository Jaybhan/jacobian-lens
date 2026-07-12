"""Steps 2-3: decompose LoRA write directions into the J-lens concept
dictionary and measure alignment-vs-depth.

Tests whether steering (IFT) adapters write into the workspace — sparse
combinations of lens concept directions, concentrated mid-network — while
skill (CPT) adapters do not. CPU-only once a fitted lens + unembed cache
exist (see exp/lens_fit/).
"""

from exp.decompose.adapters import (
    ALL_MODULES,
    RESIDUAL_WRITE_MODULES,
    AdapterWrites,
    ModuleWrites,
    load_adapter_writes,
    svd_of_lowrank,
)
from exp.decompose.align import alignment_for_adapter, module_alignment, random_floor
from exp.decompose.dictionary import (
    build_dictionary,
    dictionary_for_layer,
    export_unembed,
    load_unembed,
)
from exp.decompose.pursuit import (
    PursuitResult,
    nonneg_alignment,
    nonneg_mp,
    omp,
    subspace_projection,
)

__all__ = [
    "ALL_MODULES",
    "RESIDUAL_WRITE_MODULES",
    "AdapterWrites",
    "ModuleWrites",
    "PursuitResult",
    "alignment_for_adapter",
    "build_dictionary",
    "dictionary_for_layer",
    "export_unembed",
    "load_adapter_writes",
    "load_unembed",
    "module_alignment",
    "nonneg_alignment",
    "nonneg_mp",
    "omp",
    "random_floor",
    "subspace_projection",
    "svd_of_lowrank",
]
