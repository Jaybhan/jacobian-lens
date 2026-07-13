# CPU tests for the SGD-vs-AdamW gradient-confinement experiment, against the
# tiny decoder in tests/tiny.py. The containment pair (tests 3-4) runs in
# float64 so exact-arithmetic claims separate cleanly from Adam's violation.
#
# Frame note: for the tiny block ``h + linear(h)``, dL/d(module output) equals
# dL/d(block output) — the residual add has identity Jacobian w.r.t. the
# linear branch. This is the tiny analog of down_proj's exact frame.

import json

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from exp.decompose.adapters import load_adapter_writes, svd_of_lowrank
from exp.delta.metrics import topk_basis
from exp.sgdtest.lora import (
    attach_lora,
    detach_lora,
    lora_parameters,
    save_adapter,
)
from exp.sgdtest.span_test import energy_stats
from exp.sgdtest.train import compute_gate
from tests.tiny import TinyDecoder


def make_model() -> TinyDecoder:
    model = TinyDecoder(n_layers=4, d_model=8).double()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def tiny_loss(model: TinyDecoder, input_ids: torch.Tensor, positions: torch.Tensor):
    """Mean next-token CE at ``positions`` — all in the model's dtype (fp64)."""
    hidden = model.forward(input_ids).last_hidden_state
    logits = model.lm_head(model.norm(hidden))
    targets = input_ids[0, positions + 1]
    return F.cross_entropy(logits[0, positions], targets, reduction="mean")


def example_ids(model: TinyDecoder, text: str = "the quick brown fox " * 3):
    return model.encode(text, max_length=32)


# ---------------------------------------------------------------- lora hooks


def test_lora_hook_forward_delta():
    model = make_model()
    target = model.layers[1].linear
    x = torch.randn(1, 5, 8, dtype=torch.float64)
    out_base = target(x).clone()

    sites = attach_lora(
        [model.layers[1]], r=2, alpha=4, modules=("linear",), seed=3,
        dtype=torch.float64,
    )
    with torch.no_grad():
        sites[0].B.copy_(torch.randn(8, 2, dtype=torch.float64))
    A, B, scale = sites[0].A, sites[0].B, 4 / 2

    out_hooked = target(x)
    expected = out_base + scale * (x @ A.T) @ B.T
    torch.testing.assert_close(out_hooked, expected, rtol=0, atol=1e-12)

    detach_lora(sites)
    assert torch.equal(target(x), out_base)


def test_first_step_grads():
    """B=0 init: A's gradient is exactly zero on step one, B's is not."""
    model = make_model()
    sites = attach_lora(
        model.layers, r=2, alpha=4, modules=("linear",), seed=0,
        dtype=torch.float64,
    )
    ids = example_ids(model)
    positions = torch.tensor([6, 11])
    loss = tiny_loss(model, ids, positions)
    loss.backward()
    for site in sites:
        assert float(site.A.grad.abs().max()) == 0.0
        assert float(site.B.grad.abs().max()) > 0.0

    # After one SGD step (B becomes nonzero), A starts receiving gradient.
    opt = torch.optim.SGD(lora_parameters(sites), lr=0.5)
    opt.step()
    opt.zero_grad(set_to_none=True)
    tiny_loss(model, ids, positions).backward()
    assert any(float(s.A.grad.abs().max()) > 0.0 for s in sites)
    detach_lora(sites)


# ------------------------------------------------------- Lemma-2 containment


def _train_and_record(optimizer_factory, n_steps: int = 2):
    """Train one tiny site while recording per-step module-output pullbacks.

    Two scored targets x two steps => recorded span rank <= 4 < d=8, so the
    containment assertion cannot be vacuously true (asserted explicitly).
    """
    model = make_model()
    sites = attach_lora(
        [model.layers[2]], r=2, alpha=4, modules=("linear",), seed=1,
        dtype=torch.float64, record_output=True,
    )
    site = sites[0]
    params = lora_parameters(sites)
    opt = optimizer_factory(params)
    ids = example_ids(model)
    positions = torch.tensor([6, 11])  # 2 scored targets

    recorded: list[torch.Tensor] = []
    for _ in range(n_steps):
        loss = tiny_loss(model, ids, positions)
        loss.backward()
        rows = site.recorded_output.grad[0]  # [seq, d]
        recorded.append(rows[rows.norm(dim=1) > 1e-30].clone())
        torch.nn.utils.clip_grad_norm_(params, 1e-3)  # scalar clip: span-safe
        opt.step()
        opt.zero_grad(set_to_none=True)
    detach_lora(sites)

    pullbacks = torch.cat(recorded)  # [<=4, 8]
    rank = int(torch.linalg.matrix_rank(pullbacks, tol=1e-20))
    assert rank < 8, "recorded span covers all of R^d — test is vacuous"
    # Orthonormal basis of the recorded span.
    Q, _ = torch.linalg.qr(pullbacks.T)
    Q = Q[:, : pullbacks.shape[0]]
    residuals = []
    for j in range(site.B.shape[1]):
        b = site.B.detach()[:, j]
        if float(b.norm()) < 1e-12:
            continue
        residuals.append(float((b - Q @ (Q.T @ b)).norm() / b.norm()))
    assert residuals, "B stayed identically zero — training did nothing"
    return max(residuals)


def test_sgd_containment_lemma2():
    """Plain SGD from B=0: every B column stays in the recorded pullback span
    (exact induction; fp64 residual ~ machine precision). Clipping is active
    and does not break it (a global scalar preserves the span)."""
    worst = _train_and_record(lambda p: torch.optim.SGD(p, lr=0.5, momentum=0.0))
    assert worst < 1e-8


def test_adam_breaks_containment():
    """AdamW's per-coordinate preconditioning knocks B out of the span."""
    worst = _train_and_record(
        lambda p: torch.optim.AdamW(p, lr=0.5, weight_decay=0.0)
    )
    assert worst > 1e-3


# --------------------------------------------------------- save / span / gate


class _FakeLlamaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.down_proj = nn.Linear(12, 8, bias=False)


def test_save_roundtrip_peft_layout(tmp_path):
    torch.manual_seed(4)
    blocks = [_FakeLlamaBlock(), _FakeLlamaBlock()]
    sites = attach_lora(blocks, r=3, alpha=6, seed=4)
    with torch.no_grad():
        for site in sites:
            site.B.copy_(torch.randn_like(site.B))
    save_adapter(sites, str(tmp_path), r=3, alpha=6)
    detach_lora(sites)

    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config == {"r": 3, "lora_alpha": 6}
    adapter = load_adapter_writes(str(tmp_path))
    assert adapter.r == 3 and adapter.scale == 2.0
    assert set(adapter.writes) == {
        (l, m) for l in (0, 1) for m in ("o_proj", "down_proj")
    }
    # Parsed SVD matches the direct SVD of the site's factors (up to sign).
    site = next(s for s in sites if s.layer == 0 and s.module == "down_proj")
    U, S, _ = svd_of_lowrank(site.B.detach(), site.A.detach(), scale=2.0)
    parsed = adapter.writes[(0, "down_proj")]
    torch.testing.assert_close(parsed.S, S, rtol=1e-4, atol=1e-5)
    cosines = (parsed.U.T @ U).diagonal().abs()
    torch.testing.assert_close(cosines, torch.ones(3), rtol=0, atol=1e-4)


def test_span_energy_metric():
    d = 8
    moment = torch.diag(torch.tensor([10.0, 5.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]))
    basis = topk_basis(moment, 100, k=2).float()  # spans e0, e1
    inside = torch.zeros(d, 2)
    inside[0, 0], inside[7, 1] = 1.0, 1.0  # col 0 in-span, col 1 orthogonal

    both = energy_stats(inside, torch.tensor([1.0, 1.0]), basis)
    assert both["max"] == pytest.approx(1.0, abs=1e-6)
    assert both["min"] == pytest.approx(0.0, abs=1e-6)
    assert both["unweighted_mean"] == pytest.approx(0.5, abs=1e-6)
    # S²-weighting: all weight on the in-span direction.
    weighted = energy_stats(inside, torch.tensor([1.0, 0.0]), basis)
    assert weighted["weighted_mean"] == pytest.approx(1.0, abs=1e-6)


def test_gate():
    def records(losses):
        return [{"step": i, "loss": v} for i, v in enumerate(losses)]

    good = records([2.0] * 20 + [1.0] * 20)
    flat = records([2.0] * 40)
    bad = records([2.0] * 20 + [float("nan")] * 20)

    assert compute_gate({"sgd": good, "adamw": good})["pass"] is True
    assert compute_gate({"sgd": flat, "adamw": good})["pass"] is False
    assert compute_gate({"sgd": bad, "adamw": good})["pass"] is False
