"""Alignment of adapter write directions with the lens dictionary, per depth.

For each (layer, residual-frame module): decompose the ΔW left singular
directions with signed OMP (primary), the non-negative variant (paper-faithful
check), and the top-k J-subspace projection (ungameable check); aggregate as
the **Σ²-weighted mean** so directions the adapter barely uses barely count.

Controls produced alongside, never after:
- **random floor** — Gaussian directions through the same dictionary/OMP,
  matched n. The overcomplete-dictionary inflation baseline.
- **wrong-layer grid** — layer-l writes scored against layer-l′ dictionaries.
  If the diagonal doesn't dominate, the frame/indexing is broken (the
  silent-failure mode), and no other number from this module means anything.

Frame caveat (also in code where it applies): ``down_proj`` output lands
exactly at the block-l output that ``J_l`` is defined on; ``o_proj`` output
enters mid-block (before the MLP read), so ``J_l`` is a first-order
approximation for it.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from exp.decompose.adapters import RESIDUAL_WRITE_MODULES, AdapterWrites, ModuleWrites
from exp.decompose.dictionary import build_dictionary
from exp.decompose.pursuit import nonneg_alignment, omp, subspace_projection
from jlens.lens import JacobianLens

logger = logging.getLogger(__name__)

DEFAULT_K = 25
DEFAULT_PROJECTION_KS = (16, 64, 256)
DEFAULT_WRONG_LAYER_OFFSETS = (-8, -4, 4, 8)


def _weighted(values: torch.Tensor, S: torch.Tensor) -> float:
    """Σ²-weighted mean of per-direction values."""
    weights = (S**2) / (S**2).sum()
    return float((values * weights).sum())


def module_alignment(
    module_writes: ModuleWrites,
    dictionary: torch.Tensor,
    J_l: torch.Tensor,
    *,
    k: int = DEFAULT_K,
    projection_ks: tuple[int, ...] = DEFAULT_PROJECTION_KS,
) -> dict[str, Any]:
    """All alignment metrics for one (layer, module)."""
    directions = module_writes.U.T  # [r, d]
    signed = omp(directions, dictionary, k=k)
    nonneg = nonneg_alignment(directions, dictionary, k=k)
    projections = subspace_projection(directions, J_l, ks=projection_ks)
    S = module_writes.S
    return {
        "curve": signed.curve,  # [r, k+1]
        "atoms": signed.atoms,  # [r, k]
        "S": S,
        "signed": _weighted(signed.alignment(), S),
        "nonneg": _weighted(nonneg, S),
        "projection": {pk: _weighted(v, S) for pk, v in projections.items()},
        "per_direction_signed": signed.alignment(),
    }


def random_floor(
    dictionary: torch.Tensor,
    *,
    n: int,
    d: int,
    k: int = DEFAULT_K,
    seed: int = 0,
) -> dict[str, float]:
    """Signed-OMP alignment of ``n`` Gaussian directions (the inflation floor)."""
    generator = torch.Generator().manual_seed(seed)
    gaussian = torch.randn(n, d, generator=generator)
    result = omp(gaussian, dictionary, k=k)
    alignment = result.alignment()
    return {"mean": float(alignment.mean()), "std": float(alignment.std())}


def alignment_for_adapter(
    adapter: AdapterWrites,
    lens: JacobianLens,
    W_U: torch.Tensor,
    gamma: torch.Tensor,
    *,
    k: int = DEFAULT_K,
    modules: tuple[str, ...] = RESIDUAL_WRITE_MODULES,
    projection_ks: tuple[int, ...] = DEFAULT_PROJECTION_KS,
    wrong_layer_probe_every: int = 4,
    wrong_layer_offsets: tuple[int, ...] = DEFAULT_WRONG_LAYER_OFFSETS,
    floor_seed: int = 0,
) -> dict[str, Any]:
    """Full per-layer alignment analysis for one adapter against one lens.

    Layers analyzed are the intersection of the lens's fitted layers and the
    adapter's layers. Returns a torch.save-able dict (see keys below).
    """
    layers = sorted(set(lens.source_layers) & set(adapter.layers()))
    if not layers:
        raise ValueError("lens and adapter share no layers")
    d = W_U.shape[1]

    result: dict[str, Any] = {
        "adapter": adapter.name,
        "r": adapter.r,
        "scale": adapter.scale,
        "k": k,
        "coverage": adapter.coverage(),
        "residual_coverage": adapter.residual_coverage(),
        "layers": layers,
        "modules": list(modules),
        "per_layer": {},
        "floor": {},
        "wrong_layer": {},
    }

    for layer in layers:
        J_l = lens.jacobians[layer]
        dictionary = build_dictionary(J_l, W_U, gamma)
        per_module: dict[str, Any] = {}
        for module in modules:
            writes = adapter.writes.get((layer, module))
            if writes is None:
                continue
            per_module[module] = module_alignment(
                writes, dictionary, J_l, k=k, projection_ks=projection_ks
            )
        result["per_layer"][layer] = per_module
        result["floor"][layer] = random_floor(
            dictionary, n=adapter.r, d=d, k=k, seed=floor_seed
        )
        logger.info(
            "layer %d: %s  floor=%.3f",
            layer,
            {m: f"{v['signed']:.3f}" for m, v in per_module.items()},
            result["floor"][layer]["mean"],
        )
        del dictionary

    # Wrong-layer null on a probe grid. Diagonal (offset 0, above) must beat
    # these or the frame/indexing is broken.
    probe_layers = layers[::wrong_layer_probe_every]
    for layer in probe_layers:
        grid: dict[int, dict[str, float]] = {}
        for offset in wrong_layer_offsets:
            other = layer + offset
            if other not in set(lens.source_layers):
                continue
            other_dictionary = build_dictionary(lens.jacobians[other], W_U, gamma)
            entry: dict[str, float] = {}
            for module in modules:
                writes = adapter.writes.get((layer, module))
                if writes is None:
                    continue
                pursuit = omp(writes.U.T, other_dictionary, k=k)
                entry[module] = _weighted(pursuit.alignment(), writes.S)
            grid[offset] = entry
            del other_dictionary
        result["wrong_layer"][layer] = grid
    return result
