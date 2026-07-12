"""Streaming accumulation for the δ-sketch: per (probe, layer).

Wraps one :class:`exp.delta.pullbacks.DeltaAccumulator` per probe for
position-level statistics (second moment / mean / reservoir — the same
schema the δ-screen analyzer consumes), and adds the sketch's primary store:
per-(probe, layer) EXAMPLE-MEAN unit directions. Across-example scatter of
those means is the δ statistic; position-level stats are kept for continuity
with the δ-screen numbers.

All per-probe accumulators share one reservoir seed, so every probe samples
the SAME positions — deliberate, for cross-probe comparability.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from exp.delta.pullbacks import NORM_EPS, DeltaAccumulator

logger = logging.getLogger(__name__)


class SketchAccumulator:
    def __init__(
        self,
        n_probes: int,
        layers: list[int],
        d_model: int,
        *,
        device: str | torch.device = "cpu",
        reservoir_cap: int = 4096,
        seed: int = 0,
    ) -> None:
        self.n_probes = n_probes
        self.layers = sorted(layers)
        self.d_model = d_model
        self.per_probe = [
            DeltaAccumulator(
                self.layers, d_model, device=device,
                reservoir_cap=reservoir_cap, seed=seed,
            )
            for _ in range(n_probes)
        ]
        # Built as fp32 CPU lists; stacked to fp16 at save time.
        self._example_mean: dict[tuple[int, int], list[torch.Tensor]] = {
            (j, l): [] for j in range(n_probes) for l in self.layers
        }
        self._example_mean_ids: dict[tuple[int, int], list[int]] = {
            (j, l): [] for j in range(n_probes) for l in self.layers
        }
        self.example_stats: list[dict[str, float]] = []
        self.n_examples = 0

    def update(
        self, per_loss: list[dict[int, torch.Tensor]], example_index: int
    ) -> None:
        """Fold one example's per-probe pullbacks in.

        ``per_loss[j][l]`` is ``[n_source_positions, d_model]`` fp32 (raw),
        as returned by ``pullbacks_for_example_multi``.
        """
        if len(per_loss) != self.n_probes:
            raise ValueError(
                f"expected {self.n_probes} probes, got {len(per_loss)}"
            )
        for j, pullbacks in enumerate(per_loss):
            self.per_probe[j].update(pullbacks, example_index)
            for layer in self.layers:
                rows = pullbacks[layer]
                norms = rows.norm(dim=1)
                kept = rows[norms > NORM_EPS]
                if kept.shape[0] == 0:
                    continue
                unit = kept / kept.norm(dim=1, keepdim=True)
                mean = unit.mean(dim=0).float()
                mean_norm = float(mean.norm())
                if mean_norm <= NORM_EPS:
                    continue
                self._example_mean[(j, layer)].append((mean / mean_norm).cpu())
                self._example_mean_ids[(j, layer)].append(example_index)
        self.n_examples += 1

    def state_dict(self) -> dict[str, Any]:
        example_mean: dict[int, dict[int, torch.Tensor]] = {}
        example_mean_ids: dict[int, dict[int, torch.Tensor]] = {}
        for j in range(self.n_probes):
            example_mean[j] = {}
            example_mean_ids[j] = {}
            for layer in self.layers:
                rows = self._example_mean[(j, layer)]
                example_mean[j][layer] = (
                    torch.stack(rows).half()
                    if rows
                    else torch.empty(0, self.d_model, dtype=torch.float16)
                )
                example_mean_ids[j][layer] = torch.tensor(
                    self._example_mean_ids[(j, layer)], dtype=torch.long
                )
        return {
            "schema": "dsketch-v1",
            "layers": self.layers,
            "d_model": self.d_model,
            "n_probes": self.n_probes,
            "per_probe": [acc.state_dict() for acc in self.per_probe],
            "example_mean": example_mean,
            "example_mean_ids": example_mean_ids,
            "example_stats": self.example_stats,
            "n_examples": self.n_examples,
        }

    def save(
        self,
        path: str,
        *,
        probes: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        state = self.state_dict()
        state["probes"] = probes or {}
        state["meta"] = meta or {}
        torch.save(state, path)
