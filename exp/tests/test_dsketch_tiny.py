# CPU tests for the fixed-cotangent δ-sketch against the tiny decoder in
# tests/tiny.py. No network, no GPU.

import pytest
import torch

from exp.delta.data import prepare_cpt
from exp.delta.metrics import (
    participation_ratio_from_moment,
    second_moment_of,
)
from exp.delta.pullbacks import (
    pullbacks_for_example,
    pullbacks_for_example_multi,
)
from exp.dsketch.accumulate import SketchAccumulator
from exp.dsketch.extract import extract_condition_sketch
from exp.dsketch.probes import ProbeSet, build_probes, probe_loss_fns
from tests.tiny import TinyDecoder

SKIP = 4  # tiny prompts; the default 16 is for real models


def make_model() -> TinyDecoder:
    model = TinyDecoder(n_layers=4, d_model=8)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def cpt_example(model, text="the quick brown fox jumps over the lazy dog " * 2):
    example = prepare_cpt(
        model.tokenizer, text, max_seq_len=64, min_tokens=10, skip_first=SKIP
    )
    assert example is not None
    return example


def make_probe_set(vectors: torch.Tensor, seed: int = 0) -> ProbeSet:
    n = vectors.shape[0]
    return ProbeSet(
        vectors=vectors,
        kinds=("gauss",) * n,
        tokens=(None,) * n,
        token_ids=(None,) * n,
        seed=seed,
    )


# ------------------------------------------------------------------- multi


def test_multi_matches_single_pullbacks():
    """pullbacks_for_example_multi == P separate pullbacks_for_example calls,
    including one exact one-hot probe (chains to the jlens parity anchor)."""
    model = make_model()
    example = cpt_example(model)
    layers = [0, 1, 2]

    torch.manual_seed(1)
    dense = torch.randn(2, 8)
    dense = dense / dense.norm(dim=1, keepdim=True)
    one_hot = torch.zeros(1, 8)
    one_hot[0, 5] = 1.0
    vectors = torch.cat([dense, one_hot])
    loss_fns = probe_loss_fns(vectors, skip_first=SKIP)

    per_loss, positions_multi, stats = pullbacks_for_example_multi(
        model, example, layers, loss_fns, skip_first=SKIP
    )
    assert len(per_loss) == 3
    for j, loss_fn in enumerate(loss_fns):
        single, positions_single, single_stats = pullbacks_for_example(
            model, example, layers, skip_first=SKIP, loss_fn=loss_fn
        )
        assert torch.equal(positions_multi, positions_single)
        assert stats[f"loss_{j}"] == pytest.approx(single_stats["loss"], abs=1e-6)
        for layer in layers:
            torch.testing.assert_close(
                per_loss[j][layer], single[layer], rtol=0, atol=1e-6
            )


def test_multi_rejects_empty_loss_fns():
    model = make_model()
    with pytest.raises(ValueError, match="non-empty"):
        pullbacks_for_example_multi(model, cpt_example(model), [0], [], skip_first=SKIP)


# ------------------------------------------------------------------- probes


def test_probe_determinism_and_fallback():
    model = make_model()
    # (a) determinism: same seed -> bitwise identical; different seed differs.
    first = build_probes(model, n_gauss=3, seed=7)
    second = build_probes(model, n_gauss=3, seed=7)
    assert torch.equal(first.vectors, second.vectors)
    other = build_probes(model, n_gauss=3, seed=8)
    assert not torch.equal(first.vectors, other.vectors)
    assert torch.allclose(first.vectors.norm(dim=1), torch.ones(3), atol=1e-5)

    # (b) TinyDecoder lacks _lm_head/_final_norm -> gaussian-only fallback.
    assert len(first) == 3
    assert first.kinds == ("gauss",) * 3

    # (c) instance-attached readout attrs -> unembed probes appear.
    model._lm_head = model.lm_head
    model._final_norm = model.norm
    with_unembed = build_probes(model, n_gauss=2, tokens=("ab",), seed=7)
    assert len(with_unembed) == 3
    assert with_unembed.kinds == ("gauss", "gauss", "unembed")
    token_id = with_unembed.token_ids[2]
    expected = model.norm.weight.float() * model.lm_head.weight[token_id].float()
    expected = expected / expected.norm()
    torch.testing.assert_close(with_unembed.vectors[2], expected, rtol=0, atol=1e-6)
    # state_dict round-trips the descriptive fields.
    state = with_unembed.state_dict()
    assert state["kinds"] == ["gauss", "gauss", "unembed"]
    assert state["tokens"][2] == "ab"
    assert state["seed"] == 7


# -------------------------------------------------------------- accumulator


def test_example_mean_metric_sanity():
    """Identical directions -> mdn ~ 1, PR ~ 1; random -> mdn small, PR high."""
    torch.manual_seed(0)
    direction = torch.randn(8)
    direction = direction / direction.norm()

    aligned = SketchAccumulator(1, [0], d_model=8, reservoir_cap=16)
    for i in range(20):
        aligned.update([{0: direction.expand(4, 8) * (1.0 + i)}], i)
    rows = aligned.state_dict()["example_mean"][0][0].float()
    assert rows.shape == (20, 8)
    assert float(rows.mean(0).norm()) == pytest.approx(1.0, abs=1e-2)
    assert participation_ratio_from_moment(second_moment_of(rows)) == pytest.approx(
        1.0, abs=1e-2
    )

    scattered = SketchAccumulator(1, [0], d_model=8, reservoir_cap=16)
    for i in range(200):
        scattered.update([{0: torch.randn(4, 8)}], i)
    rows = scattered.state_dict()["example_mean"][0][0].float()
    assert float(rows.mean(0).norm()) < 0.35  # positions average, N=200
    assert participation_ratio_from_moment(second_moment_of(rows)) > 6.0


def test_extract_sketch_end_to_end(tmp_path):
    model = make_model()
    texts = [
        "the quick brown fox jumps over the lazy dog " * 2,
        "pack my box with five dozen liquor jugs today " * 2,
        "sphinx of black quartz judge my vow tonight ok " * 2,
    ]
    examples = [cpt_example(model, t) for t in texts]
    torch.manual_seed(2)
    vectors = torch.randn(2, 8)
    vectors = vectors / vectors.norm(dim=1, keepdim=True)
    probes = make_probe_set(vectors)

    accumulator = extract_condition_sketch(
        model, examples, [0, 1, 2], probes,
        skip_first=SKIP, reservoir_cap=16, log_every=1,
    )
    assert accumulator.n_examples == 3
    state = accumulator.state_dict()
    assert state["schema"] == "dsketch-v1"
    assert state["n_probes"] == 2 and state["layers"] == [0, 1, 2]

    for j in range(2):
        # Inner per-probe dicts follow the δ-screen accumulator key layout.
        inner = state["per_probe"][j]
        for key in ("layers", "second_moment", "mean_sum", "n_vectors",
                    "reservoir", "reservoir_example"):
            assert key in inner
        for layer in (0, 1, 2):
            reservoir = inner["reservoir"][layer].float()
            assert torch.allclose(
                reservoir.norm(dim=1), torch.ones(reservoir.shape[0]), atol=1e-2
            )
            means = state["example_mean"][j][layer]
            assert means.dtype == torch.float16 and means.shape == (3, 8)
            assert torch.allclose(
                means.float().norm(dim=1), torch.ones(3), atol=1e-2
            )
            assert state["example_mean_ids"][j][layer].tolist() == [0, 1, 2]

    # save/load round-trip with probes + meta present.
    out = tmp_path / "wikitext.pt"
    accumulator.save(
        str(out), probes=probes.state_dict(), meta={"condition": "wikitext"}
    )
    loaded = torch.load(out, weights_only=False)
    assert loaded["meta"]["condition"] == "wikitext"
    assert torch.equal(loaded["probes"]["vectors"], vectors)
    assert loaded["example_mean"][1][2].shape == (3, 8)
    assert len(loaded["example_stats"]) == 3
    assert all("loss_1" in s for s in loaded["example_stats"])


def test_final_layer_probe_pullback_is_v():
    """At the final block, dL/dh_final IS the probe at every valid position —
    the degeneracy that motivates excluding layer n_layers-1 by default."""
    model = make_model()
    example = cpt_example(model)
    final = model.n_layers - 1
    torch.manual_seed(3)
    v = torch.randn(1, 8)
    v = v / v.norm()
    per_loss, _, _ = pullbacks_for_example_multi(
        model, example, [final], probe_loss_fns(v, skip_first=SKIP),
        skip_first=SKIP,
    )
    rows = per_loss[0][final]
    for row in rows:
        torch.testing.assert_close(row, v[0], rtol=0, atol=1e-6)
