"""Positive control: pullback eigendirections through the decompose path.

Phase 3 scored the adapters at the random floor with no wrong-layer contrast,
which has two readings: (a) the adapters genuinely write outside the J-frame,
or (b) the lens dictionary has no resolving power at these layers. This module
separates them with directions whose theoretical status is known.

The δ-screen accumulated per-example pullbacks ``u = Φᵀ W_Uᵀ ε`` — the exact
vectors whose span confines a LoRA ``B`` under (S)GD (writeup Lemmas 1-2), and
under Jacobian concentration each is approximately a sparse combination of
J-lens dictionary atoms ``J_lᵀ (γ ⊙ W_U[t])``. The top eigendirections of the
pullback second moment ``C_l`` are the shared core of that span: the
best-possible-case client for the dictionary. Decomposing them through the
IDENTICAL code path as the adapters (same OMP, same floor, same wrong-layer
grid) gives the decision rule:

- control HIGH, adapters at floor -> the dictionary resolves J-frame content;
  the Phase-3 adapter null is real (e.g. AdamW breaking span preservation).
- control AT FLOOR too -> the dictionary/lens lacks resolving power here and
  the Phase-3 null is uninformative about the hypothesis.

Eigendirections are wrapped as an :class:`~exp.decompose.adapters.AdapterWrites`
under the module name :data:`CONTROL_MODULE` so
:func:`~exp.decompose.align.alignment_for_adapter` runs unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from exp.decompose.adapters import AdapterWrites, ModuleWrites

#: Module name for the wrapped eigendirections. Not a real nn.Module; keep the
#: control artifacts in their own --out directory so exp.decompose.analyze
#: (which expects o_proj/down_proj) never mixes them with adapter results.
CONTROL_MODULE = "pullback"


def eigendirections(
    second_moment: torch.Tensor, n_vectors: int, *, n_directions: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top eigendirections of ``C/n`` with sqrt-eigenvalue weights.

    Returns ``(U, S)`` shaped like a :class:`ModuleWrites` factor pair:
    ``U [d, n]`` eigenvectors as columns (descending eigenvalue) and
    ``S = sqrt(λ) [n]``, so the Σ²-weighted aggregation in
    ``exp.decompose.align`` weights each direction by its energy fraction —
    the exact analogue of the adapters' singular-value weighting. Since
    ``trace(C/n) = 1`` for unit-normalized pullbacks, ``Σ S²`` is the fraction
    of directional energy the control directions capture.
    """
    if n_vectors <= 0:
        raise ValueError("n_vectors must be positive")
    n_directions = min(n_directions, second_moment.shape[0])
    eigvals, eigvecs = torch.linalg.eigh(second_moment.double() / n_vectors)
    U = eigvecs[:, -n_directions:].flip(1).float()  # ascending -> descending
    S = eigvals[-n_directions:].flip(0).clamp(min=0.0).sqrt().float()
    return U, S


def control_writes(
    state: dict[str, Any], *, n_directions: int = 16
) -> AdapterWrites:
    """Wrap a saved δ-screen accumulator (``exp.delta`` ``{condition}.pt``
    state dict) as an :class:`AdapterWrites` of pullback eigendirections, one
    ``(layer, CONTROL_MODULE)`` entry per accumulated layer.

    ``r`` is set to ``n_directions`` and ``scale`` to 1.0; per-layer
    ``ModuleWrites.energy`` is the captured directional-energy fraction.
    """
    name = state.get("meta", {}).get("condition", "delta-control")
    adapter = AdapterWrites(name=f"pullback-eig:{name}", r=n_directions, scale=1.0)
    for layer in state["layers"]:
        n = state["n_vectors"][layer]
        if n <= 0:
            continue
        U, S = eigendirections(
            state["second_moment"][layer], n, n_directions=n_directions
        )
        adapter.writes[(layer, CONTROL_MODULE)] = ModuleWrites(U=U, S=S, V=U)
    if not adapter.writes:
        raise ValueError("accumulator has no populated layers")
    adapter.n_layers = len(adapter.layers())
    return adapter
