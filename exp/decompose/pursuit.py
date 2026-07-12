"""Sparse pursuit of write directions in the lens dictionary.

Primary metric: **signed orthogonal matching pursuit** — greedy atom selection
by absolute correlation with least-squares refit each step. Treats a write and
its negation symmetrically (a LoRA write can suppress a concept).

Secondary, paper-faithful variant: **non-negative matching pursuit** — only
positively-correlated atoms, non-negative coefficients (plain MP, atoms may
repeat). The workspace paper defines J-space membership via ~25-atom
non-negative combinations; callers run it on ±w and take the max.

Robustness metric no greedy search can game: :func:`subspace_projection`,
the energy fraction of a write in the span of the top-k right singular vectors
of ``J_l``. The dictionary is overcomplete (V≈32k atoms in d≈4k dims), so
*absolute* pursuit alignments are inflated; conclusions rest on the IFT-vs-CPT
contrast against a matched random floor, plus this deterministic projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PursuitResult:
    """Batched pursuit output for ``n`` directions.

    Attributes:
        curve: ``[n, k+1]`` fraction of each direction's energy explained
            after 0..k steps (monotone, in [0, 1]).
        atoms: ``[n, k]`` selected dictionary row indices per step.
        coeffs: ``[n, k]`` coefficient of each selected atom (final refit
            values for OMP; per-pick values for non-negative MP).
    """

    curve: torch.Tensor
    atoms: torch.Tensor
    coeffs: torch.Tensor

    def alignment(self, k: int | None = None) -> torch.Tensor:
        """``[n]`` energy fraction explained at step ``k`` (default: last)."""
        return self.curve[:, -1 if k is None else k]


def omp(directions: torch.Tensor, dictionary: torch.Tensor, *, k: int = 25) -> PursuitResult:
    """Signed orthogonal matching pursuit, batched over rows of ``directions``.

    Args:
        directions: ``[n, d]``; need not be normalized (the curve is relative
            to each row's own energy).
        dictionary: ``[V, d]``, rows unit-normalized.
        k: Number of atoms.

    Returns:
        :class:`PursuitResult` with least-squares-refit coefficients.
    """
    W = directions.float()
    D = dictionary.float()
    n, d = W.shape
    k = min(k, d)
    residual = W.clone()
    energy = (W**2).sum(dim=1).clamp(min=1e-30)

    curve = torch.zeros(n, k + 1)
    atoms = torch.zeros(n, k, dtype=torch.long)
    coeffs = torch.zeros(n, k)
    selected_mask = torch.zeros(n, D.shape[0], dtype=torch.bool)

    for step in range(k):
        correlation = residual @ D.T  # [n, V]
        correlation[selected_mask] = 0.0  # never re-select (belt & braces)
        chosen = correlation.abs().argmax(dim=1)  # [n]
        atoms[:, step] = chosen
        selected_mask[torch.arange(n), chosen] = True

        # Per-row least-squares refit on the selected atom set, then update
        # the residual. Atom sets differ per row, so loop (n and k are tiny).
        for i in range(n):
            basis = D[atoms[i, : step + 1]]  # [step+1, d]
            solution = torch.linalg.lstsq(basis.T, W[i]).solution
            coeffs[i, : step + 1] = solution
            residual[i] = W[i] - solution @ basis
        curve[:, step + 1] = 1.0 - (residual**2).sum(dim=1) / energy

    return PursuitResult(curve=curve, atoms=atoms, coeffs=coeffs)


def nonneg_mp(
    directions: torch.Tensor, dictionary: torch.Tensor, *, k: int = 25
) -> PursuitResult:
    """Non-negative (plain) matching pursuit: positive-correlation atoms only.

    Coefficients are the positive correlations at pick time; atoms may repeat
    (standard MP). Rows whose best remaining correlation is ≤ 0 stop early
    (their curve flattens). Run on ``+w`` and ``-w`` and take the max to score
    membership in the non-negative cone either way.
    """
    W = directions.float()
    D = dictionary.float()
    n, d = W.shape
    k = min(k, d)
    residual = W.clone()
    energy = (W**2).sum(dim=1).clamp(min=1e-30)

    curve = torch.zeros(n, k + 1)
    atoms = torch.full((n, k), -1, dtype=torch.long)
    coeffs = torch.zeros(n, k)

    for step in range(k):
        correlation = residual @ D.T
        best, chosen = correlation.max(dim=1)  # positive side only
        active = best > 0
        atoms[active, step] = chosen[active]
        coeffs[active, step] = best[active]
        residual[active] -= best[active, None] * D[chosen[active]]
        curve[:, step + 1] = 1.0 - (residual**2).sum(dim=1) / energy

    return PursuitResult(curve=curve, atoms=atoms, coeffs=coeffs)


def nonneg_alignment(
    directions: torch.Tensor, dictionary: torch.Tensor, *, k: int = 25
) -> torch.Tensor:
    """``[n]`` max of non-negative-MP alignment over ``±w`` at step k."""
    plus = nonneg_mp(directions, dictionary, k=k).alignment()
    minus = nonneg_mp(-directions, dictionary, k=k).alignment()
    return torch.maximum(plus, minus)


def subspace_projection(
    directions: torch.Tensor, J_l: torch.Tensor, *, ks: tuple[int, ...] = (16, 64, 256)
) -> dict[int, torch.Tensor]:
    """Energy fraction of each direction in the top-k right singular subspace
    of ``J_l`` — the deterministic robustness metric.

    Returns:
        ``{k: Tensor[n]}``.
    """
    W = directions.float()
    unit = W / W.norm(dim=1, keepdim=True).clamp(min=1e-30)
    # Right singular vectors of J_l: eigenvectors of J_lᵀ J_l.
    _, _, Vh = torch.linalg.svd(J_l.float(), full_matrices=False)
    out: dict[int, torch.Tensor] = {}
    for k in ks:
        basis = Vh[:k]  # [k, d]
        out[k] = ((unit @ basis.T) ** 2).sum(dim=1)
    return out
