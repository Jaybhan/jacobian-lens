# Sanity harness for the pullback-eigendirection positive control. CPU, no
# network. The control is only trustworthy if directions with KNOWN dictionary
# structure come out aligned and structureless ones do not.

import pytest
import torch

from exp.decompose.align import alignment_for_adapter
from exp.decompose.control import CONTROL_MODULE, control_writes, eigendirections
from exp.decompose.dictionary import build_dictionary
from exp.decompose.pursuit import omp
from exp.decompose.run_decompose import summarize
from jlens.fitting import fit
from tests.tiny import TinyDecoder


def _moment_state(vectors_by_layer: dict[int, torch.Tensor], *, condition="test"):
    """Fake δ-screen accumulator state from stacks of unit vectors."""
    layers = sorted(vectors_by_layer)
    return {
        "layers": layers,
        "second_moment": {
            l: vectors_by_layer[l].T @ vectors_by_layer[l] for l in layers
        },
        "n_vectors": {l: vectors_by_layer[l].shape[0] for l in layers},
        "meta": {"condition": condition},
    }


def unit_rows(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.norm(dim=1, keepdim=True)


# ------------------------------------------------------------ eigendirections


def test_eigendirections_shapes_weights_and_energy():
    torch.manual_seed(0)
    vectors = unit_rows(torch.randn(200, 8))
    moment = vectors.T @ vectors
    U, S = eigendirections(moment, 200, n_directions=4)
    assert U.shape == (8, 4) and S.shape == (4,)
    assert (S[:-1] >= S[1:]).all()  # descending
    torch.testing.assert_close(U.T @ U, torch.eye(4), atol=1e-4, rtol=0)
    # trace(C/n) == 1 for unit vectors, so ΣS² over ALL d directions is 1 and
    # the top-4 capture strictly less.
    U_all, S_all = eigendirections(moment, 200, n_directions=8)
    assert float((S_all**2).sum()) == pytest.approx(1.0, abs=1e-4)
    assert 0.0 < float((S**2).sum()) < 1.0
    # A rank-one moment concentrates all weight on the first direction.
    one = unit_rows(torch.randn(1, 8)).expand(50, 8)
    U1, S1 = eigendirections(one.T @ one, 50, n_directions=3)
    assert float(S1[0] ** 2) == pytest.approx(1.0, abs=1e-4)
    assert float((S1[1:] ** 2).sum()) == pytest.approx(0.0, abs=1e-4)
    cosine = float((U1[:, 0] @ one[0]).abs())
    assert cosine == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError):
        eigendirections(moment, 0)


def test_control_writes_wraps_state_and_coverage_is_safe():
    torch.manual_seed(1)
    state = _moment_state(
        {l: unit_rows(torch.randn(100, 8)) for l in (0, 1, 2)}, condition="fake"
    )
    adapter = control_writes(state, n_directions=4)
    assert adapter.name == "pullback-eig:fake"
    assert adapter.r == 4 and adapter.layers() == [0, 1, 2]
    assert adapter.writes[(0, CONTROL_MODULE)].U.shape == (8, 4)
    # The synthetic module name must not break the coverage bookkeeping
    # (regression: coverage() used to KeyError on unknown modules).
    coverage = adapter.coverage()
    assert coverage[CONTROL_MODULE] == pytest.approx(1.0, abs=1e-6)
    assert adapter.residual_coverage() == 0.0

    empty = _moment_state({0: unit_rows(torch.randn(10, 8))})
    empty["n_vectors"][0] = 0
    with pytest.raises(ValueError):
        control_writes(empty)


# ---------------------------------------------------------------- end-to-end


def test_control_separates_atom_built_from_random_moments():
    """Pullbacks fabricated as sparse combinations of dictionary atoms must
    come out highly aligned through the control path; isotropic pullbacks must
    not. This is the property that makes the control a control."""
    torch.manual_seed(2)
    # k << d, like the real setting (25/4096): at k comparable to d, OMP
    # explains even isotropic directions and the contrast vanishes.
    d, vocab = 64, 512
    J = torch.linalg.qr(torch.randn(d, d))[0]
    W_U = torch.randn(vocab, d)
    gamma = torch.ones(d)
    D = build_dictionary(J, W_U, gamma)

    atoms = D[torch.tensor([3, 40, 90])]  # the "workspace concepts"
    coeffs = torch.rand(400, 3) + 0.1  # sparse nonneg combos, all 3 active
    structured = unit_rows(coeffs @ atoms + 0.01 * torch.randn(400, d))
    isotropic = unit_rows(torch.randn(400, d))

    aligned, scattered = (
        omp(eigendirections(v.T @ v, 400, n_directions=3)[0].T, D, k=4).alignment()
        for v in (structured, isotropic)
    )
    assert float(aligned[0]) > 0.95  # top eigendirection: in the atom span
    assert float(aligned.mean() - scattered.mean()) > 0.3


def test_control_end_to_end_with_fitted_tiny_lens():
    """Full runner path: fitted lens -> control_writes -> alignment_for_adapter
    with modules=(CONTROL_MODULE,) -> summarize. Pins the result schema the
    pod run and the comparison table rely on."""
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = fit(
        model,
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        source_layers=[0, 1, 2],
        dim_batch=4,
        max_seq_len=64,
    )
    torch.manual_seed(3)
    # Layer 3 in the state but not the lens: must be dropped, not crash.
    state = _moment_state(
        {l: unit_rows(torch.randn(120, 8)) for l in (0, 1, 2, 3)}
    )
    adapter = control_writes(state, n_directions=4)
    result = alignment_for_adapter(
        adapter,
        lens,
        model.lm_head.weight.detach().clone(),
        model.norm.weight.detach().clone(),
        k=6,
        modules=(CONTROL_MODULE,),
        projection_ks=(2, 4),
        wrong_layer_probe_every=2,
        wrong_layer_offsets=(-1, 1),
    )
    assert result["layers"] == [0, 1, 2]
    for layer in result["layers"]:
        stats = result["per_layer"][layer][CONTROL_MODULE]
        assert 0.0 <= stats["signed"] <= 1.0
        assert stats["curve"].shape == (4, 7)
        assert 0.0 < result["floor"][layer]["mean"] < 1.0
    assert set(result["wrong_layer"]) == {0, 2}
    summary = summarize(result)  # must serialize with the synthetic module
    assert summary["signed"]["0"][CONTROL_MODULE] == pytest.approx(
        result["per_layer"][0][CONTROL_MODULE]["signed"]
    )
