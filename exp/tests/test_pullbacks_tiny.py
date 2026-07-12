# CPU tests for the δ pipeline against the tiny end-to-end decoder in
# tests/tiny.py. No network, no GPU.

import pytest
import torch
import torch.nn.functional as F

from exp.delta.data import prepare_cpt, prepare_ift
from exp.delta.metrics import (
    jackknife_participation_ratio,
    mean_direction_norm,
    participation_ratio,
    principal_angles,
    random_baseline_pr,
    second_moment_of,
    spectrum_from_second_moment,
    topk_energy_fractions,
)
from exp.delta.pullbacks import (
    DeltaAccumulator,
    extract_condition,
    pullbacks_for_example,
)
from jlens.fitting import jacobian_for_prompt, valid_position_mask
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


# ---------------------------------------------------------------- pullbacks


def test_pullback_matches_finite_differences():
    """dL/dh_l from the extraction path == central finite differences of the
    loss under a perturbation of block l's output.

    Few scored targets + a perturbation along the pullback itself keep the
    directional-derivative signal well above fp32 forward-pass noise."""
    model = make_model()
    example = cpt_example(model)
    keep = example.loss_positions.nonzero(as_tuple=True)[0][:5]
    example.loss_positions[:] = False
    example.loss_positions[keep] = True
    layers = [0, 1, 2]
    pullbacks, source_positions, _ = pullbacks_for_example(
        model, example, layers, skip_first=SKIP
    )

    positions = example.loss_positions.nonzero(as_tuple=True)[0]
    targets = example.input_ids[0, positions + 1]

    def loss_with_perturbation(layer: int, delta: torch.Tensor) -> float:
        def hook(module, inputs, output):
            return output + delta

        handle = model.layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                hidden = model.forward(example.input_ids).last_hidden_state
                logits = model.unembed(hidden)
                return float(
                    F.cross_entropy(logits[0, positions], targets, reduction="sum")
                )
        finally:
            handle.remove()

    eps = 5e-2
    checked = 0
    for layer in layers:
        for pick in [0, len(source_positions) // 2]:
            vector = pullbacks[layer][pick]
            if float(vector.norm()) < 1e-6:  # position after the last target
                continue
            p = int(source_positions[pick])
            direction = vector / vector.norm()
            delta = torch.zeros(1, example.input_ids.shape[1], 8)
            delta[0, p] = eps * direction
            fd = (
                loss_with_perturbation(layer, delta)
                - loss_with_perturbation(layer, -delta)
            ) / (2 * eps)
            analytic = float(vector @ direction)  # == ||vector||
            assert fd == pytest.approx(analytic, rel=3e-2)
            checked += 1
    assert checked >= 3


def test_pullback_frame_parity_with_jlens_estimator():
    """With the cotangent used by the lens fit (sum of h_final[:, k] over valid
    positions), the position-mean of our pullbacks must equal row k of
    jacobian_for_prompt's J_l exactly — pins layer indexing, position masking,
    and orientation to the jlens frame."""
    model = make_model()
    prompt = "the quick brown fox " * 4
    layers = [0, 1, 2]
    jacobians, seq_len, n_valid = jacobian_for_prompt(
        model, prompt, layers, dim_batch=4, max_seq_len=64
    )

    input_ids = model.encode(prompt, max_length=64)
    assert input_ids.shape[1] == seq_len
    mask = valid_position_mask(seq_len)  # jlens default skip: match the fit
    from exp.delta.data import PreparedExample

    example = PreparedExample(
        input_ids=input_ids, loss_positions=mask, meta={"n_targets": -1}
    )
    for k in (0, 5):
        pullbacks, source_positions, _ = pullbacks_for_example(
            model,
            example,
            layers,
            skip_first=16,  # jlens default, matches the jacobian_for_prompt call
            loss_fn=lambda h_final, k=k: h_final[
                0, mask.nonzero(as_tuple=True)[0], k
            ].sum(),
        )
        assert len(source_positions) == n_valid
        for layer in layers:
            torch.testing.assert_close(
                pullbacks[layer].mean(dim=0),
                jacobians[layer][k],
                rtol=0,
                atol=1e-5,
            )


def test_loss_mask_localizes_gradient():
    """A single scored target at position p0 must produce zero pullback at
    every other source position (TinyDecoder has no attention, so influence is
    strictly per-position)."""
    model = make_model()
    example = cpt_example(model)
    p0 = 10
    example.loss_positions[:] = False
    example.loss_positions[p0] = True

    pullbacks, source_positions, _ = pullbacks_for_example(
        model, example, [0, 2], skip_first=SKIP
    )
    for layer in (0, 2):
        norms = pullbacks[layer].norm(dim=1)
        at_p0 = source_positions == p0
        assert norms[at_p0].item() > 1e-6
        assert norms[~at_p0].max().item() == 0.0


def test_response_mask_is_subset_and_alltok_control_differs():
    model = make_model()
    instruction = "add two numbers " * 3
    response = "def add(a, b) return a plus b " * 3
    # 256: the Alpaca template alone is ~140 byte-tokens under ByteTokenizer.
    masked = prepare_ift(
        model.tokenizer, instruction, response,
        max_seq_len=256, min_tokens=10, skip_first=SKIP,
    )
    alltok = prepare_ift(
        model.tokenizer, instruction, response,
        max_seq_len=256, min_tokens=10, skip_first=SKIP, mask_to_response=False,
    )
    assert masked is not None and alltok is not None
    valid = valid_position_mask(masked.input_ids.shape[1], skip_first=SKIP)
    # Masked positions are a strict subset of the all-token (valid) positions.
    assert bool((masked.loss_positions & ~valid).sum() == 0)
    assert int(masked.loss_positions.sum()) < int(alltok.loss_positions.sum())
    # No scored target before the response starts.
    prompt_len = masked.meta["prompt_len"]
    assert bool(masked.loss_positions[: prompt_len - 1].sum() == 0)


def test_prepare_filters():
    model = make_model()
    assert prepare_cpt(model.tokenizer, "x", max_seq_len=64, min_tokens=10) is None
    # Instruction so long the response is truncated away entirely.
    assert (
        prepare_ift(
            model.tokenizer, "a" * 300, "response", max_seq_len=64,
            min_tokens=10, skip_first=SKIP,
        )
        is None
    )


# ------------------------------------------------------------------ metrics


def test_pr_rank_one_and_isotropic():
    direction = torch.randn(8)
    repeated = (direction / direction.norm()).expand(100, 8)
    eigvals = spectrum_from_second_moment(second_moment_of(repeated), 100)
    assert participation_ratio(eigvals) == pytest.approx(1.0, abs=1e-6)
    assert topk_energy_fractions(eigvals)[1] == pytest.approx(1.0, abs=1e-6)
    assert mean_direction_norm(repeated.sum(dim=0), 100) == pytest.approx(1.0, abs=1e-5)

    torch.manual_seed(0)
    gaussian = torch.randn(2000, 8)
    gaussian = gaussian / gaussian.norm(dim=1, keepdim=True)
    eigvals = spectrum_from_second_moment(second_moment_of(gaussian), 2000)
    assert participation_ratio(eigvals) > 6.0
    assert mean_direction_norm(gaussian.sum(dim=0), 2000) < 0.1
    assert random_baseline_pr(8, 2000) > 6.0


def test_jackknife_shards_by_example():
    torch.manual_seed(0)
    vectors = torch.randn(200, 8)
    vectors = vectors / vectors.norm(dim=1, keepdim=True)
    example_ids = torch.arange(200) // 5  # 40 examples x 5 positions
    pr, se = jackknife_participation_ratio(vectors, example_ids, n_shards=8)
    assert 6.0 < pr <= 8.0
    assert 0.0 < se < 1.0


def test_principal_angles_extremes():
    moment = torch.eye(8)
    zero_angles = principal_angles(moment, 1, moment.clone(), 1, k=3)
    assert float(zero_angles.max()) == pytest.approx(0.0, abs=1e-5)

    a = torch.zeros(8, 8)
    a[0, 0] = a[1, 1] = 1.0
    b = torch.zeros(8, 8)
    b[2, 2] = b[3, 3] = 1.0
    orthogonal = principal_angles(a, 1, b, 1, k=2)
    assert float(orthogonal.min()) == pytest.approx(torch.pi / 2, abs=1e-5)


# ------------------------------------------------- accumulator / extraction


def test_accumulator_counts_and_reservoir():
    torch.manual_seed(0)
    accumulator = DeltaAccumulator([0, 1], d_model=8, reservoir_cap=8, seed=0)
    fed: list[torch.Tensor] = []
    for example_index in range(3):
        vectors = torch.randn(4, 8)
        fed.append(vectors / vectors.norm(dim=1, keepdim=True))
        accumulator.update({0: vectors, 1: vectors * 2.0}, example_index)
    assert accumulator.n_examples == 3
    assert accumulator.n_vectors[0] == accumulator.n_vectors[1] == 12
    # Second moment of unit vectors has trace == n, invariant to input scale.
    for layer in (0, 1):
        assert float(accumulator.second_moment[layer].trace()) == pytest.approx(
            12.0, abs=1e-4
        )
    # Reservoir is full; eviction may hit any slot, but every stored row must
    # be one of the vectors actually fed for its recorded example id.
    state = accumulator.state_dict()
    assert state["reservoir"][0].shape == (8, 8)
    assert set(state["reservoir_example"][0].tolist()) <= {0, 1, 2}
    for slot in range(8):
        example_id = int(state["reservoir_example"][0][slot])
        distances = (fed[example_id] - state["reservoir"][0][slot].float()).norm(dim=1)
        assert float(distances.min()) < 1e-3


def test_extract_condition_end_to_end():
    model = make_model()
    texts = [
        "the quick brown fox jumps over the lazy dog " * 2,
        "pack my box with five dozen liquor jugs today " * 2,
        "sphinx of black quartz judge my vow tonight ok " * 2,
    ]
    examples = [cpt_example(model, t) for t in texts]
    layers = list(range(model.n_layers))
    accumulator = extract_condition(
        model, examples, layers, skip_first=SKIP, reservoir_cap=64, log_every=1
    )
    assert accumulator.n_examples == 3
    for layer in layers:
        n = accumulator.n_vectors[layer]
        assert n > 0
        eigvals = spectrum_from_second_moment(
            accumulator.second_moment[layer].cpu(), n
        )
        pr = participation_ratio(eigvals)
        assert 1.0 <= pr <= 8.0
        assert torch.isfinite(accumulator.second_moment[layer]).all()
    assert len(accumulator.example_stats) == 3
    assert all(s["loss"] > 0 for s in accumulator.example_stats)
