"""δ metrics: all scale-free functions of the pullback direction distribution.

Everything is computed from either the streaming second moment ``C_l``
(exact, all vectors) or the reservoir sample (for example-level jackknife
error bars and cross-condition subspace angles). CPU, seconds.
"""

from __future__ import annotations

import torch

DEFAULT_TOP_KS = (1, 5, 25)


def spectrum_from_second_moment(
    second_moment: torch.Tensor, n_vectors: int
) -> torch.Tensor:
    """Eigenvalues (descending) of ``C/n`` — the second moment of unit vectors.

    Eigenvalues sum to ~1 (unit-normalized inputs), so each is the fraction of
    directional energy along its eigenvector.
    """
    if n_vectors <= 0:
        raise ValueError("n_vectors must be positive")
    eigvals = torch.linalg.eigvalsh(second_moment.double() / n_vectors)
    return eigvals.flip(0).clamp(min=0.0)


def participation_ratio(eigvals: torch.Tensor) -> float:
    """Effective dimension ``(sum λ)² / sum λ²``.

    1 when all vectors share one direction; d for isotropic scatter. This is
    the primary δ proxy: low = concentrated (workspace-like), high = scattered.
    """
    total = float(eigvals.sum())
    if total <= 0:
        raise ValueError("empty spectrum")
    return total**2 / float((eigvals**2).sum())


def topk_energy_fractions(
    eigvals: torch.Tensor, ks: tuple[int, ...] = DEFAULT_TOP_KS
) -> dict[int, float]:
    """Fraction of directional energy in the top-k eigendirections."""
    total = float(eigvals.sum())
    return {k: float(eigvals[:k].sum()) / total for k in ks}


def mean_direction_norm(mean_sum: torch.Tensor, n_vectors: int) -> float:
    """``‖mean(v̂)‖`` in [0, 1]: 1 = perfectly aligned, 0 = symmetric scatter."""
    return float(mean_sum.norm()) / n_vectors


def second_moment_of(vectors: torch.Tensor) -> torch.Tensor:
    """``VᵀV`` of a stack of (already unit-normalized) vectors, in fp32."""
    v = vectors.float()
    return v.T @ v


def participation_ratio_from_moment(moment: torch.Tensor) -> float:
    """PR straight from a second-moment matrix: ``trace(M)² / ‖M‖_F²``.

    Identical to eigendecomposing first (``Σλ = tr M`` and
    ``Σλ² = tr M² = ‖M‖_F²`` for symmetric ``M``) at O(d²) instead of
    O(d³); the ``/n`` normalization cancels in the ratio.
    """
    m = moment.double()
    total = float(m.trace())
    if total <= 0:
        raise ValueError("empty spectrum")
    return total**2 / float((m * m).sum())


def jackknife_participation_ratio(
    vectors: torch.Tensor,
    example_ids: torch.Tensor,
    *,
    n_shards: int = 10,
) -> tuple[float, float]:
    """Delete-one-shard jackknife of PR over the reservoir sample.

    Shards are formed over **unique example ids**, never positions: pullbacks
    from the same sequence are correlated and must leave together, or the
    error bars are fake.

    Returns:
        ``(pr_full, standard_error)``.
    """
    unique_ids = example_ids.unique()
    if len(unique_ids) < n_shards:
        n_shards = max(2, len(unique_ids))
    # Deterministic shard assignment by id order.
    shard_of_id = {int(uid): i % n_shards for i, uid in enumerate(unique_ids)}
    shard = torch.tensor([shard_of_id[int(e)] for e in example_ids])

    full_moment = second_moment_of(vectors)
    pr_full = participation_ratio_from_moment(full_moment)
    estimates = []
    for s in range(n_shards):
        keep = shard != s
        if int(keep.sum()) == 0:
            continue
        moment = full_moment - second_moment_of(vectors[~keep])
        estimates.append(participation_ratio_from_moment(moment))
    estimates_t = torch.tensor(estimates, dtype=torch.float64)
    n = len(estimates_t)
    # Jackknife SE over leave-one-shard-out estimates.
    se = float(((n - 1) / n * ((estimates_t - estimates_t.mean()) ** 2).sum()).sqrt())
    return pr_full, se


def topk_basis(moment: torch.Tensor, n: int, *, k: int = 25) -> torch.Tensor:
    """Top-``k`` eigenbasis of a second-moment matrix (the O(d³) step,
    exposed separately so callers comparing one basis against several
    others can compute it once)."""
    eigvals, eigvecs = torch.linalg.eigh(moment.double() / n)
    return eigvecs[:, -k:]  # ascending order -> last k are largest


def principal_angles_between(
    basis_a: torch.Tensor, basis_b: torch.Tensor
) -> torch.Tensor:
    """Principal angles (radians, ascending) between two orthonormal bases.
    0 = shared subspace, π/2 = orthogonal."""
    singular = torch.linalg.svdvals(basis_a.T @ basis_b).clamp(-1.0, 1.0)
    return singular.acos().flip(0)


def principal_angles(
    moment_a: torch.Tensor,
    n_a: int,
    moment_b: torch.Tensor,
    n_b: int,
    *,
    k: int = 25,
) -> torch.Tensor:
    """Principal angles (radians, ascending) between the top-``k`` eigenspaces
    of two second-moment matrices. 0 = shared subspace, π/2 = orthogonal."""
    return principal_angles_between(
        topk_basis(moment_a, n_a, k=k), topk_basis(moment_b, n_b, k=k)
    )


def random_baseline_pr(
    d_model: int, n_vectors: int, *, seed: int = 0
) -> float:
    """PR of unit-normalized isotropic Gaussian vectors, through the same code
    path — the calibration ceiling (≈ min(n, d) up to finite-sample effects)."""
    generator = torch.Generator().manual_seed(seed)
    vectors = torch.randn(n_vectors, d_model, generator=generator)
    vectors = vectors / vectors.norm(dim=1, keepdim=True)
    moment = second_moment_of(vectors)
    return participation_ratio(spectrum_from_second_moment(moment, n_vectors))
