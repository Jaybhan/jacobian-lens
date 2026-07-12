"""The lens concept dictionary: token directions in layer-l residual space.

The J-lens reads layer-l residual ``h`` as logits ``W_U · norm(J_l · h)``. The
paper's steering direction for token ``t`` at layer ``l`` (the construction in
data/experiments/README.md, verbal-introspection) is the unit-normalized
transpose row: ``d_t = normalize( J_lᵀ · (γ ⊙ W_U[t]) )``, where ``γ`` is the
final RMSNorm's elementwise weight. Stacked over the vocabulary these rows are
the dictionary ``D_l [V, d]`` that write directions are decomposed against.

RMSNorm's data-dependent ``1/rms`` scalar is dropped in this linearization —
row normalization absorbs any per-token scale, so only the direction matters.

``unembed.pt`` (written by exp/lens_fit/export_unembed.py) carries
``W_U`` + ``γ`` so nothing here ever needs the full model.
"""

from __future__ import annotations

from typing import Any

import torch

from jlens.lens import JacobianLens


def export_unembed(
    W_U: torch.Tensor, gamma: torch.Tensor, path: str, *, meta: dict[str, Any] | None = None
) -> None:
    """Save the unembedding matrix and final-norm weight (fp16) to ``path``."""
    torch.save(
        {"W_U": W_U.half().cpu(), "gamma": gamma.half().cpu(), "meta": meta or {}},
        path,
    )


def load_unembed(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ``(W_U [V, d], gamma [d])`` as fp32 CPU tensors."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state["W_U"].float(), state["gamma"].float()


def build_dictionary(
    J_l: torch.Tensor, W_U: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """Dictionary ``D_l [V, d]``: row t is ``normalize(J_lᵀ (γ ⊙ W_U[t]))``.

    Computed as ``normalize_rows((W_U ⊙ γ) @ J_l)`` — one [V, d] @ [d, d]
    matmul (~0.5 GB fp32 for V=32000, d=4096; build one layer at a time).
    """
    rows = (W_U * gamma) @ J_l.float()
    norms = rows.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return rows / norms


def dictionary_for_layer(
    lens: JacobianLens, layer: int, W_U: torch.Tensor, gamma: torch.Tensor
) -> torch.Tensor:
    """Convenience: :func:`build_dictionary` for a fitted lens layer."""
    return build_dictionary(lens.jacobians[layer], W_U, gamma)
