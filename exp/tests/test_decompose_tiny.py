# Sanity harness for the Steps 2-3 decomposition pipeline. CPU, no network.
# No alignment number from exp/decompose is believed unless these pass.

import json

import pytest
import torch

from exp.decompose.adapters import (
    AdapterWrites,
    ModuleWrites,
    load_adapter_writes,
    svd_of_lowrank,
)
from exp.decompose.align import alignment_for_adapter, random_floor
from exp.decompose.dictionary import (
    build_dictionary,
    export_unembed,
    load_unembed,
)
from exp.decompose.pursuit import nonneg_alignment, nonneg_mp, omp, subspace_projection
from jlens.fitting import fit
from tests.tiny import TinyDecoder


def unit_rows(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.norm(dim=1, keepdim=True)


# ------------------------------------------------------------- svd_of_lowrank


def test_svd_of_lowrank_reconstructs_and_is_gauge_invariant():
    torch.manual_seed(0)
    B = torch.randn(24, 4)
    A = torch.randn(4, 12)
    scale = 2.0
    U, S, V = svd_of_lowrank(B, A, scale=scale)
    # Exact reconstruction of scale*B@A.
    torch.testing.assert_close(U @ torch.diag(S) @ V.T, scale * B @ A, atol=1e-4, rtol=1e-4)
    assert (S[:-1] >= S[1:]).all()  # descending
    assert torch.allclose(U.T @ U, torch.eye(4), atol=1e-5)

    # Gauge change: ΔW = (B R)(R⁻¹ A) is the same operator -> same S, same
    # singular subspaces (columns equal up to sign for distinct S).
    R = torch.randn(4, 4) + 3 * torch.eye(4)  # well-conditioned invertible
    U2, S2, V2 = svd_of_lowrank(B @ R, torch.linalg.solve(R, A), scale=scale)
    torch.testing.assert_close(S, S2, atol=1e-4, rtol=1e-4)
    cosines = (U.T @ U2).diagonal().abs()
    torch.testing.assert_close(cosines, torch.ones(4), atol=1e-4, rtol=0)


def test_load_adapter_writes_from_local_dir(tmp_path):
    """Parse a locally fabricated PEFT-layout adapter (exact key format
    verified against the real LoRA-TMLR-2024 safetensors header)."""
    from safetensors.torch import save_file

    torch.manual_seed(1)
    d, d_mlp, r = 8, 12, 3
    tensors = {}
    for layer in range(2):
        for module, d_out, d_in in [
            ("q_proj", d, d),
            ("o_proj", d, d),
            ("gate_proj", d_mlp, d),
            ("down_proj", d, d_mlp),
        ]:
            prefix = f"base_model.model.model.layers.{layer}"
            parent = "self_attn" if module in ("q_proj", "o_proj") else "mlp"
            tensors[f"{prefix}.{parent}.{module}.lora_A.weight"] = torch.randn(r, d_in)
            tensors[f"{prefix}.{parent}.{module}.lora_B.weight"] = torch.randn(d_out, r)
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": 6})
    )

    adapter = load_adapter_writes(str(tmp_path))
    assert adapter.r == 3 and adapter.scale == 2.0
    assert adapter.layers() == [0, 1]
    assert set(m for _, m in adapter.writes) == {"q_proj", "o_proj", "gate_proj", "down_proj"}
    assert adapter.writes[(0, "o_proj")].U.shape == (d, r)
    assert adapter.writes[(0, "down_proj")].U.shape == (d, r)  # residual frame
    assert adapter.writes[(0, "gate_proj")].U.shape == (d_mlp, r)
    coverage = adapter.coverage()
    assert sum(coverage.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < adapter.residual_coverage() < 1.0


# ------------------------------------------------------------------ dictionary


def test_dictionary_identity_jacobian_matches_unembed_rows():
    """J = I  =>  dictionary rows are exactly the normalized γ⊙W_U rows (the
    late-layer / logit-lens limit)."""
    torch.manual_seed(2)
    W_U = torch.randn(64, 16)
    gamma = torch.rand(16) + 0.5
    D = build_dictionary(torch.eye(16), W_U, gamma)
    expected = unit_rows(W_U * gamma)
    torch.testing.assert_close(D, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(D.norm(dim=1), torch.ones(64), atol=1e-5, rtol=0)


def test_unembed_export_roundtrip(tmp_path):
    W_U = torch.randn(32, 8)
    gamma = torch.rand(8)
    path = str(tmp_path / "unembed.pt")
    export_unembed(W_U, gamma, path, meta={"model": "tiny"})
    W2, g2 = load_unembed(path)
    assert W2.dtype == torch.float32
    torch.testing.assert_close(W2, W_U, atol=2e-3, rtol=0)  # fp16 roundtrip
    torch.testing.assert_close(g2, gamma, atol=1e-3, rtol=0)


# --------------------------------------------------------------------- pursuit


def test_omp_self_reconstruction():
    """A synthetic sparse combination of dictionary atoms must decompose back
    to itself: alignment ~ 1 and the true atoms recovered."""
    torch.manual_seed(3)
    D = unit_rows(torch.randn(64, 32))
    true_atoms = torch.tensor([3, 17, 29, 41, 55])
    coeffs = torch.tensor([2.0, -1.5, 1.0, 0.7, -0.5])
    w = (coeffs[:, None] * D[true_atoms]).sum(dim=0, keepdim=True)

    result = omp(w, D, k=10)
    assert float(result.alignment()) > 0.999
    assert set(true_atoms.tolist()) <= set(result.atoms[0].tolist())


def test_omp_curve_properties_and_random_floor():
    torch.manual_seed(4)
    D = unit_rows(torch.randn(128, 16))
    W = torch.randn(6, 16)
    result = omp(W, D, k=16)  # k = d -> full-rank dictionary explains ~all
    assert result.curve.shape == (6, 17)
    assert (result.curve >= -1e-5).all() and (result.curve <= 1 + 1e-5).all()
    assert (result.curve[:, 1:] - result.curve[:, :-1] >= -1e-5).all()  # monotone
    assert (result.alignment() > 0.999).all()

    # Random directions at small k explain much less than self-reconstruction.
    floor = omp(W, D, k=2).alignment()
    assert (floor < 0.9).all()

    stats = random_floor(D, n=8, d=16, k=2, seed=0)
    assert 0.0 < stats["mean"] < 0.9
    assert stats["std"] >= 0.0


def test_nonneg_mp_sign_symmetry_via_max():
    torch.manual_seed(5)
    D = unit_rows(torch.randn(64, 16))
    atom = D[7:8]
    plus = nonneg_mp(atom, D, k=3).alignment()
    assert float(plus) > 0.999  # the atom itself: one positive pick suffices
    # Its negation is unreachable with nonneg coefficients on +w alone...
    minus_only = nonneg_mp(-atom, D, k=1).alignment()
    # ...unless another atom happens to correlate; it must do worse anyway.
    assert float(minus_only) < 0.999
    # The ±max wrapper restores symmetry.
    both = nonneg_alignment(-atom, D, k=3)
    assert float(both) > 0.999
    # Curves stay monotone and bounded.
    curve = nonneg_mp(torch.randn(4, 16), D, k=8).curve
    assert (curve >= -1e-5).all() and (curve <= 1 + 1e-5).all()
    assert (curve[:, 1:] - curve[:, :-1] >= -1e-5).all()


def test_subspace_projection_extremes():
    torch.manual_seed(6)
    # J with a known dominant right subspace: diag(10, 9, ..., 0-ish).
    d = 16
    Q = torch.linalg.qr(torch.randn(d, d))[0]
    singulars = torch.linspace(10, 0.01, d)
    J = Q @ torch.diag(singulars) @ torch.eye(d)  # right basis = standard
    inside = torch.zeros(2, d)
    inside[0, 0], inside[1, 1] = 1.0, 1.0  # in the top-2 right subspace
    outside = torch.zeros(1, d)
    outside[0, -1] = 1.0  # in the bottom direction
    proj_in = subspace_projection(inside, J, ks=(2,))
    proj_out = subspace_projection(outside, J, ks=(2,))
    torch.testing.assert_close(proj_in[2], torch.ones(2), atol=1e-4, rtol=0)
    assert float(proj_out[2].max()) < 1e-4


# ------------------------------------------------------- wrong-layer + runner


def _synthetic_lens(d: int, layers: dict[int, torch.Tensor]):
    from jlens.lens import JacobianLens

    return JacobianLens(jacobians=layers, n_prompts=1, d_model=d)


def test_wrong_layer_null_with_separated_rotations():
    """Writes built from layer-0 atoms must score ~1 against the layer-0
    dictionary and much lower against a well-separated layer-2 dictionary —
    the check that catches frame/off-by-one bugs."""
    torch.manual_seed(7)
    # k << d, like the real setting (k/d = 25/4096): at k/d ~ 0.25 even a
    # wrong dictionary explains most of a direction and the null has no teeth.
    d, vocab = 64, 256
    Q0 = torch.linalg.qr(torch.randn(d, d))[0]
    Q2 = torch.linalg.qr(torch.randn(d, d))[0]
    W_U = torch.randn(vocab, d)
    gamma = torch.ones(d)
    D0 = build_dictionary(Q0, W_U, gamma)
    D2 = build_dictionary(Q2, W_U, gamma)

    writes = D0[torch.tensor([5, 21, 40, 250])]  # 4 layer-0 atoms
    right = omp(writes, D0, k=4).alignment()
    wrong = omp(writes, D2, k=4).alignment()
    assert (right > 0.999).all()
    assert float(right.mean() - wrong.mean()) > 0.4


def test_alignment_for_adapter_end_to_end_tiny():
    """Full pipeline against a real fitted (tiny) lens: fit -> dictionary ->
    fabricated adapter -> per-layer metrics, floor, wrong-layer grid."""
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = fit(
        model,
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        source_layers=[0, 1, 2],
        dim_batch=4,
        max_seq_len=64,
    )
    W_U = model.lm_head.weight.detach().clone()  # [32, 8]
    gamma = model.norm.weight.detach().clone()  # [8]

    torch.manual_seed(8)
    adapter = AdapterWrites(name="fabricated", r=3, scale=2.0)
    for layer in [0, 1, 2]:
        for module in ("o_proj", "down_proj"):
            U, S, V = svd_of_lowrank(torch.randn(8, 3), torch.randn(3, 8), scale=2.0)
            adapter.writes[(layer, module)] = ModuleWrites(U=U, S=S, V=V)
    adapter.n_layers = 3

    result = alignment_for_adapter(
        adapter,
        lens,
        W_U,
        gamma,
        k=6,
        projection_ks=(2, 4),
        wrong_layer_probe_every=2,
        wrong_layer_offsets=(-2, -1, 1, 2),
    )
    assert result["layers"] == [0, 1, 2]
    for layer in [0, 1, 2]:
        for module in ("o_proj", "down_proj"):
            stats = result["per_layer"][layer][module]
            assert 0.0 <= stats["signed"] <= 1.0
            assert 0.0 <= stats["nonneg"] <= 1.0
            assert stats["curve"].shape == (3, 7)
            assert set(stats["projection"]) == {2, 4}
        assert 0.0 < result["floor"][layer]["mean"] < 1.0
    # Wrong-layer grid exists for probe layers and contains valid offsets only.
    assert set(result["wrong_layer"]) == {0, 2}
    assert set(result["wrong_layer"][0]) <= {1, 2}
    for grid in result["wrong_layer"].values():
        for entry in grid.values():
            for value in entry.values():
                assert 0.0 <= value <= 1.0
