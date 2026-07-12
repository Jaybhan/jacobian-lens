"""Pullback-vector extraction and streaming accumulation.

The pullback vector for one example at layer ``l`` and source position ``p``
is ``v_l[p] = dL/dh_l[p]`` — the cross-entropy output error carried backward
into layer-``l`` coordinates. It is the training pressure a LoRA write matrix
``B`` at layer ``l`` would feel, and the single-example, un-averaged cousin of
the J-lens's defining Jacobian (which injects one-hot cotangents per output
dimension instead; see ``jlens.fitting``). One forward + **one** backward per
example.

Frame: ``h_l`` is the output of decoder block ``l`` (post-residual-add,
pre-final-norm), captured with :class:`jlens.hooks.ActivationRecorder` —
identical to the frame the lens is fitted in.

Aggregation is streaming, so nothing scales with the corpus:

- per-layer second moment ``C_l = sum v̂ v̂ᵀ`` over unit-normalized pullbacks
  (direction, not length — a few high-loss tokens must not dominate),
- per-layer mean of ``v̂``,
- a per-layer reservoir sample of raw ``v̂`` (fp16) with example ids, for
  jackknife error bars and cross-condition subspace angles offline.

The reservoir update is vectorized per example: rows drawn for the same slot
in one batch overwrite each other rather than chaining, so it is an
*approximate* reservoir sample — bias is negligible at reservoir sizes ≫
positions-per-example.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from exp.delta.data import PreparedExample
from jlens.fitting import SKIP_FIRST_N_POSITIONS, valid_position_mask
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)

#: Positions whose pullback norm falls below this are dropped (e.g. positions
#: after the last scored target receive exactly zero gradient).
NORM_EPS = 1e-20


def pullbacks_for_example(
    model: LensModel,
    example: PreparedExample,
    layers: Sequence[int],
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    loss_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, dict[str, float]]:
    """Per-position pullback vectors for one example: one forward, one backward.

    Args:
        model: The (frozen) model.
        example: Tokenized example with its loss-position mask.
        layers: Block indices to extract pullbacks at.
        skip_first: Leading source positions to exclude, as in lens fitting.
        loss_fn: Optional override mapping the final-layer residual
            ``[1, seq, d]`` to a scalar; used by tests to pin parity with the
            ``jlens`` fitting estimator. Default: summed next-token
            cross-entropy over ``example.loss_positions``.

    Returns:
        ``(pullbacks, source_positions, stats)`` where ``pullbacks[l]`` is
        ``[n_source_positions, d_model]`` fp32 (raw, un-normalized, on the
        layer's device) and ``source_positions`` indexes the sequence.
    """
    input_ids = example.input_ids.to(model.input_device)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    source_positions = position_mask.nonzero(as_tuple=True)[0]
    final_layer = model.n_layers - 1

    with (
        ActivationRecorder(
            model.layers,
            at=[*layers, final_layer],
            start_graph_at=min(layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids)
        h_final = recorder.activations[final_layer]
        source_activations = [recorder.activations[l] for l in layers]

        if loss_fn is not None:
            loss = loss_fn(h_final)
        else:
            positions = example.loss_positions.nonzero(as_tuple=True)[0]
            logits = model.unembed(h_final).float()  # [1, seq, vocab]
            targets = input_ids[0, (positions + 1).to(input_ids.device)]
            loss = F.cross_entropy(
                logits[0, positions.to(logits.device)],
                targets.to(logits.device),
                reduction="sum",
            )

        grads = torch.autograd.grad(loss, source_activations)

    pullbacks = {
        layer: grad[0, source_positions.to(grad.device), :].float()
        for layer, grad in zip(layers, grads, strict=True)
    }
    stats = {
        "loss": float(loss.detach()),
        "seq_len": float(seq_len),
        "n_targets": float(example.meta.get("n_targets", -1)),
        "n_source_positions": float(len(source_positions)),
    }
    return pullbacks, source_positions, stats


def pullbacks_for_example_multi(
    model: LensModel,
    example: PreparedExample,
    layers: Sequence[int],
    loss_fns: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[list[dict[int, torch.Tensor]], torch.Tensor, dict[str, float]]:
    """Pullbacks for several scalar losses of ``h_final``: ONE forward, then
    one ``torch.autograd.grad`` per loss (``retain_graph`` on all but the
    last). Used by the fixed-cotangent δ-sketch, where each loss dots the
    final residual against a shared probe vector.

    Args:
        model: The (frozen) model.
        example: Tokenized example (its ``loss_positions`` are ignored here —
            each ``loss_fn`` defines its own cotangent).
        layers: Block indices to extract pullbacks at.
        loss_fns: Non-empty sequence of maps from the final-layer residual
            ``[1, seq, d]`` to a scalar, as in :func:`pullbacks_for_example`.
        skip_first: Leading source positions to exclude, as in lens fitting.

    Returns:
        ``(per_loss, source_positions, stats)`` where ``per_loss[j][l]`` is
        ``[n_source_positions, d_model]`` fp32 for loss ``j`` at layer ``l``,
        and ``stats`` carries flat floats ``loss_0..loss_{P-1}``.
    """
    if not loss_fns:
        raise ValueError("loss_fns must be non-empty")

    input_ids = example.input_ids.to(model.input_device)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    source_positions = position_mask.nonzero(as_tuple=True)[0]
    final_layer = model.n_layers - 1

    per_loss: list[dict[int, torch.Tensor]] = []
    losses: list[float] = []
    with (
        ActivationRecorder(
            model.layers,
            at=[*layers, final_layer],
            start_graph_at=min(layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids)
        h_final = recorder.activations[final_layer]
        source_activations = [recorder.activations[l] for l in layers]

        for j, loss_fn in enumerate(loss_fns):
            loss = loss_fn(h_final)
            grads = torch.autograd.grad(
                loss, source_activations, retain_graph=j < len(loss_fns) - 1
            )
            per_loss.append(
                {
                    layer: grad[0, source_positions.to(grad.device), :].float()
                    for layer, grad in zip(layers, grads, strict=True)
                }
            )
            losses.append(float(loss.detach()))

    stats: dict[str, float] = {
        "seq_len": float(seq_len),
        "n_targets": float(example.meta.get("n_targets", -1)),
        "n_source_positions": float(len(source_positions)),
    }
    for j, loss_value in enumerate(losses):
        stats[f"loss_{j}"] = loss_value
    return per_loss, source_positions, stats


class DeltaAccumulator:
    """Streaming per-layer accumulators over unit-normalized pullbacks."""

    def __init__(
        self,
        layers: Sequence[int],
        d_model: int,
        *,
        device: torch.device | str = "cpu",
        reservoir_cap: int = 8192,
        seed: int = 0,
    ) -> None:
        self.layers = sorted(layers)
        self.d_model = d_model
        self.reservoir_cap = reservoir_cap
        self.second_moment = {
            l: torch.zeros(d_model, d_model, dtype=torch.float32, device=device)
            for l in self.layers
        }
        self.mean_sum = {
            l: torch.zeros(d_model, dtype=torch.float32, device=device)
            for l in self.layers
        }
        self.n_vectors = dict.fromkeys(self.layers, 0)
        self.reservoir = {
            l: torch.zeros(reservoir_cap, d_model, dtype=torch.float16)
            for l in self.layers
        }
        self.reservoir_example = {
            l: torch.full((reservoir_cap,), -1, dtype=torch.long) for l in self.layers
        }
        self.n_examples = 0
        self.example_stats: list[dict[str, float]] = []
        self._generator = torch.Generator().manual_seed(seed)

    def update(self, pullbacks: dict[int, torch.Tensor], example_index: int) -> None:
        """Fold one example's per-position pullbacks into the accumulators."""
        for layer in self.layers:
            vectors = pullbacks[layer]
            norms = vectors.norm(dim=1)
            kept = vectors[norms > NORM_EPS]
            if kept.shape[0] == 0:
                continue
            unit = kept / kept.norm(dim=1, keepdim=True)
            moment_device = self.second_moment[layer].device
            unit_there = unit.to(moment_device)
            self.second_moment[layer] += unit_there.T @ unit_there
            self.mean_sum[layer] += unit_there.sum(dim=0)
            self._reservoir_update(layer, unit, example_index)
            self.n_vectors[layer] += unit.shape[0]
        self.n_examples += 1

    def _reservoir_update(
        self, layer: int, unit: torch.Tensor, example_index: int
    ) -> None:
        n_new = unit.shape[0]
        cap = self.reservoir_cap
        seen = self.n_vectors[layer]  # vectors folded in before this batch
        unit_cpu = unit.detach().to("cpu", torch.float16)
        if seen < cap:
            n_fill = min(cap - seen, n_new)
            self.reservoir[layer][seen : seen + n_fill] = unit_cpu[:n_fill]
            self.reservoir_example[layer][seen : seen + n_fill] = example_index
            unit_cpu = unit_cpu[n_fill:]
            n_new -= n_fill
            seen += n_fill
        if n_new == 0:
            return
        # Approximate reservoir step (see module docstring).
        global_idx = seen + torch.arange(n_new)
        slots = (
            torch.rand(n_new, generator=self._generator) * (global_idx + 1).float()
        ).long()
        replace = slots < cap
        self.reservoir[layer][slots[replace]] = unit_cpu[replace]
        self.reservoir_example[layer][slots[replace]] = example_index

    def state_dict(self) -> dict[str, Any]:
        n_res = {l: min(self.n_vectors[l], self.reservoir_cap) for l in self.layers}
        return {
            "layers": self.layers,
            "d_model": self.d_model,
            "second_moment": {
                l: self.second_moment[l].cpu() for l in self.layers
            },
            "mean_sum": {l: self.mean_sum[l].cpu() for l in self.layers},
            "n_vectors": self.n_vectors,
            "reservoir": {l: self.reservoir[l][: n_res[l]] for l in self.layers},
            "reservoir_example": {
                l: self.reservoir_example[l][: n_res[l]] for l in self.layers
            },
            "n_examples": self.n_examples,
            "example_stats": self.example_stats,
        }

    def save(self, path: str, *, meta: dict[str, Any] | None = None) -> None:
        state = self.state_dict()
        state["meta"] = meta or {}
        torch.save(state, path)


def extract_condition(
    model: LensModel,
    examples: Iterable[PreparedExample],
    layers: Sequence[int],
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    reservoir_cap: int = 8192,
    seed: int = 0,
    accumulator_device: torch.device | str | None = None,
    log_every: int = 100,
) -> DeltaAccumulator:
    """Run the extraction loop over a stream of examples.

    ``accumulator_device`` defaults to the device of the first pullback batch
    (keep the second-moment matmuls on GPU when there is one).
    """
    accumulator: DeltaAccumulator | None = None
    start_time = time.perf_counter()
    n_failed = 0
    for index, example in enumerate(examples):
        try:
            pullbacks, _, stats = pullbacks_for_example(
                model, example, layers, skip_first=skip_first
            )
        except ValueError as exc:  # e.g. prompt too short for the mask
            logger.warning("  skipping example %d: %s", index, exc)
            n_failed += 1
            continue
        if accumulator is None:
            device = (
                accumulator_device
                if accumulator_device is not None
                else pullbacks[layers[0]].device
            )
            accumulator = DeltaAccumulator(
                layers,
                model.d_model,
                device=device,
                reservoir_cap=reservoir_cap,
                seed=seed,
            )
        accumulator.update(pullbacks, example_index=index)
        stats["example_index"] = float(index)
        accumulator.example_stats.append(stats)
        if (index + 1) % log_every == 0:
            elapsed = time.perf_counter() - start_time
            logger.info(
                "  example %d  %.2f s/example  loss=%.1f seq=%d",
                index + 1,
                elapsed / (index + 1),
                stats["loss"],
                int(stats["seq_len"]),
            )
    if accumulator is None:
        raise ValueError("no examples were extractable")
    logger.info(
        "extracted %d examples (%d skipped), %d vectors/layer, %.0f s total",
        accumulator.n_examples,
        n_failed,
        accumulator.n_vectors[layers[0]],
        time.perf_counter() - start_time,
    )
    return accumulator
