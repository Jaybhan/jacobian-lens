"""δ-sketch extraction loop: fixed probes pulled back through each example."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence

from exp.delta.data import PreparedExample
from exp.delta.pullbacks import pullbacks_for_example_multi
from exp.dsketch.accumulate import SketchAccumulator
from exp.dsketch.probes import ProbeSet, probe_loss_fns
from jlens.fitting import SKIP_FIRST_N_POSITIONS
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)


def extract_condition_sketch(
    model: LensModel,
    examples: Iterable[PreparedExample],
    layers: Sequence[int],
    probes: ProbeSet,
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    reservoir_cap: int = 4096,
    seed: int = 0,
    accumulator_device: str | None = None,
    log_every: int = 25,
) -> SketchAccumulator:
    """One forward + ``len(probes)`` backwards per example, streamed into a
    :class:`SketchAccumulator`. Mirrors ``exp.delta.pullbacks.extract_condition``."""
    layers = sorted(layers)
    if model.n_layers - 1 in layers:
        logger.warning(
            "layer %d is the final block: its probe pullback is identically "
            "the probe vector (Φ = I) and measures nothing",
            model.n_layers - 1,
        )
    loss_fns = probe_loss_fns(probes.vectors, skip_first=skip_first)

    accumulator: SketchAccumulator | None = None
    n_failed = 0
    started = time.time()
    for example_index, example in enumerate(examples):
        try:
            per_loss, _, stats = pullbacks_for_example_multi(
                model, example, layers, loss_fns, skip_first=skip_first
            )
        except ValueError as exc:
            logger.warning("  skipping example %d: %s", example_index, exc)
            n_failed += 1
            continue
        if accumulator is None:
            device = accumulator_device or per_loss[0][layers[0]].device
            accumulator = SketchAccumulator(
                len(probes), list(layers), model.d_model,
                device=device, reservoir_cap=reservoir_cap, seed=seed,
            )
        accumulator.update(per_loss, example_index)
        stats["example_index"] = float(example_index)
        accumulator.example_stats.append(stats)
        done = accumulator.n_examples
        if done % log_every == 0:
            logger.info(
                "  example %d (%d ok, %d skipped)  %.0fs",
                example_index + 1, done, n_failed, time.time() - started,
            )
    if accumulator is None:
        raise ValueError("no examples were extractable")
    logger.info(
        "condition done: %d examples (%d skipped) in %.0fs",
        accumulator.n_examples, n_failed, time.time() - started,
    )
    return accumulator
