"""Step-4 δ experiment: directional scatter of per-example pullback vectors.

Tests whether the per-example backward propagators (the training pressure a
LoRA write matrix B would feel) concentrate in a few shared directions on
steering-type data (instruction tuning) but scatter on skill-type data
(code/math continued pretraining). See the plan in the repo root and the
module docstrings for conventions shared with ``jlens``.
"""

from exp.delta.data import CONDITIONS, PreparedExample, iter_condition
from exp.delta.metrics import (
    mean_direction_norm,
    participation_ratio,
    spectrum_from_second_moment,
    topk_energy_fractions,
)
from exp.delta.pullbacks import (
    DeltaAccumulator,
    extract_condition,
    pullbacks_for_example,
)

__all__ = [
    "CONDITIONS",
    "DeltaAccumulator",
    "PreparedExample",
    "extract_condition",
    "iter_condition",
    "mean_direction_norm",
    "participation_ratio",
    "pullbacks_for_example",
    "spectrum_from_second_moment",
    "topk_energy_fractions",
]
