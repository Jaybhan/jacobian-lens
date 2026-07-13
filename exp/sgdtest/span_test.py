"""CLI: the lens-free span test — is B where Lemma 2 says it must be?

    python -m exp.sgdtest.span_test \
        --moment out/delta/math-ift.pt \
        --adapters out/sgdtest/math-sgd out/sgdtest/math-adamw

Projects each adapter's write directions (left singular vectors of ΔW,
Σ²-weighted) onto the top-k eigenspace of the δ-screen's pullback second
moment — the measurable, energy-weighted version of the span that plain-SGD
training provably confines B to. No lens, no dictionary, no corpus confound.

Prediction: SGD ≫ AdamW ≈ random floor (k/d). Frame note: the moment lives at
block outputs, so ``down_proj``'s pullback frame is exact (the residual add
has identity Jacobian) while ``o_proj``'s is approximate ((I+J_mlp)ᵀ off) —
the two module types are reported separately and down_proj is the headline.
The moment comes from the *base* model while pullbacks drift during training,
so even SGD will not reach 1.0; the claim rests on the three-way contrast and
the per-checkpoint energy-vs-step curve (``--checkpoints``).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from typing import Any

import torch

import jlens
from exp.decompose.adapters import RESIDUAL_WRITE_MODULES, load_adapter_writes
from exp.delta.metrics import topk_basis

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moment", default="out/delta/math-ift.pt")
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--n-random", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="*", default=None,
                        help="Subset of layers (default: all in the moment file).")
    parser.add_argument("--checkpoints", action="store_true",
                        help="Also analyze checkpoints/step-*/ under each adapter.")
    parser.add_argument("--out", default="out/sgdtest")
    return parser.parse_args()


def load_moment(path: str) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["second_moment"], state["n_vectors"]


def bases_for_layer(
    moment: torch.Tensor, n_vectors: int, ks: tuple[int, ...]
) -> dict[int, torch.Tensor]:
    """One eigh at max(ks); smaller k's are slices of the same basis."""
    big = topk_basis(moment, n_vectors, k=max(ks)).float()  # [d, kmax], ascending
    return {k: big[:, -k:] for k in ks}


def energy_stats(U: torch.Tensor, S: torch.Tensor, basis: torch.Tensor) -> dict[str, float]:
    """Energy fraction of each (unit) write direction inside ``basis``'s span."""
    energy = ((U.T.float() @ basis) ** 2).sum(dim=1)  # [r], each in [0, 1]
    weights = (S.float() ** 2) / (S.float() ** 2).sum()
    return {
        "weighted_mean": float((energy * weights).sum()),
        "unweighted_mean": float(energy.mean()),
        "max": float(energy.max()),
        "min": float(energy.min()),
    }


def random_floor(
    d: int, ks: tuple[int, ...], n_random: int, seed: int,
    bases: dict[int, torch.Tensor],
) -> dict[int, dict[str, float]]:
    """Analytic k/d plus an empirical sample through the same code path."""
    generator = torch.Generator().manual_seed(seed)
    gauss = torch.randn(n_random, d, generator=generator)
    gauss = gauss / gauss.norm(dim=1, keepdim=True)
    out: dict[int, dict[str, float]] = {}
    for k, basis in bases.items():
        energy = ((gauss @ basis) ** 2).sum(dim=1)
        out[k] = {
            "expected": k / d,
            "empirical_mean": float(energy.mean()),
            "p05": float(energy.quantile(0.05)),
            "p95": float(energy.quantile(0.95)),
        }
    return out


def analyze_adapter(
    adapter_dir: str,
    bases_by_layer: dict[int, dict[int, torch.Tensor]],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    adapter = load_adapter_writes(adapter_dir)
    result: dict[str, Any] = {"per_layer": {}, "aggregate": {}}
    sums: dict[tuple[str, int], list[float]] = {}
    for layer, bases in bases_by_layer.items():
        per_module: dict[str, Any] = {}
        for module in RESIDUAL_WRITE_MODULES:
            writes = adapter.writes.get((layer, module))
            if writes is None:
                continue
            per_module[module] = {
                k: energy_stats(writes.U, writes.S, bases[k]) for k in ks
            }
            for k in ks:
                sums.setdefault((module, k), []).append(
                    per_module[module][k]["weighted_mean"]
                )
        if per_module:
            result["per_layer"][layer] = per_module
    for (module, k), values in sums.items():
        result["aggregate"].setdefault(module, {})[k] = {
            "mean_over_layers": sum(values) / len(values),
            "min_over_layers": min(values),
            "max_over_layers": max(values),
        }
    return result


def _print_tables(
    results: dict[str, dict], floor: dict[int, dict[str, float]],
    layers: list[int], ks: tuple[int, ...],
) -> None:
    for module in RESIDUAL_WRITE_MODULES:
        for k in ks:
            print(f"\nenergy in top-{k} pullback eigenspace — {module} "
                  f"(floor={floor[k]['expected']:.4f}):")
            print("%-16s %s" % ("adapter", " ".join(f"L{l:>2}" for l in layers)))
            for name, result in results.items():
                cells = []
                for layer in layers:
                    stats = result["per_layer"].get(layer, {}).get(module)
                    cells.append(f"{stats[k]['weighted_mean']:>5.3f}"[:5]
                                 if stats else "    -")
                print("%-16s %s" % (name, " ".join(cells)))


def plot(
    results: dict[str, dict], by_step: dict[str, dict[int, dict]],
    floor: dict[int, dict[str, float]], layers: list[int],
    ks: tuple[int, ...], out_path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_mid = ks[min(1, len(ks) - 1)]
    n_panels = 2 + (1 if by_step else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8.5, 3.6 * n_panels))
    axes = list(axes) if n_panels > 1 else [axes]
    colors = {"sgd": "#1baf7a", "adamw": "#2a78d6"}

    for ax, module in zip(axes[:2], RESIDUAL_WRITE_MODULES):
        for name, result in results.items():
            color = next((c for tag, c in colors.items() if tag in name), "#898781")
            values = [
                result["per_layer"].get(l, {}).get(module, {}).get(k_mid, {})
                .get("weighted_mean") for l in layers
            ]
            ax.plot(layers, values, linewidth=2, color=color, label=name)
        ax.axhline(floor[k_mid]["expected"], color="#898781",
                   linestyle=(0, (2, 2)), linewidth=1.5)
        ax.set_ylabel(f"{module}: energy@k={k_mid}")
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("layer (block output)")

    if by_step:
        ax = axes[-1]
        for name, steps in by_step.items():
            color = next((c for tag, c in colors.items() if tag in name), "#898781")
            xs = sorted(steps)
            ys = [steps[s]["down_proj"][k_mid]["mean_over_layers"] for s in xs]
            ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=name)
        ax.axhline(floor[k_mid]["expected"], color="#898781",
                   linestyle=(0, (2, 2)), linewidth=1.5)
        ax.set_xlabel("training step")
        ax.set_ylabel(f"down_proj energy@k={k_mid}")
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle("Span test: write-direction energy in the pullback eigenspace\n"
                 "prediction: SGD >> AdamW ~ floor", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"wrote {out_path}")


def main() -> None:
    jlens.configure_logging()
    logging.getLogger("exp").setLevel(logging.INFO)
    args = parse_args()
    ks = tuple(sorted(args.ks))
    os.makedirs(args.out, exist_ok=True)

    moments, n_vectors = load_moment(args.moment)
    layers = sorted(args.layers if args.layers else moments.keys())
    d = moments[layers[0]].shape[0]

    logger.info("eigendecomposing %d layers (k=%d)...", len(layers), max(ks))
    bases_by_layer: dict[int, dict[int, torch.Tensor]] = {}
    for layer in layers:
        bases_by_layer[layer] = bases_for_layer(moments[layer], n_vectors[layer], ks)
        logger.info("  layer %d done", layer)
    floor = random_floor(d, ks, args.n_random, args.seed, bases_by_layer[layers[0]])

    results: dict[str, dict] = {}
    by_step: dict[str, dict[int, dict]] = {}
    for adapter_dir in args.adapters:
        name = os.path.basename(os.path.normpath(adapter_dir))
        logger.info("analyzing %s", name)
        results[name] = analyze_adapter(adapter_dir, bases_by_layer, ks)
        if args.checkpoints:
            steps: dict[int, dict] = {}
            for ckpt in sorted(glob.glob(os.path.join(adapter_dir, "checkpoints", "step-*"))):
                step = int(os.path.basename(ckpt).split("-")[1])
                if step == 0:
                    continue  # B = 0: write directions undefined
                steps[step] = analyze_adapter(ckpt, bases_by_layer, ks)["aggregate"]
            if steps:
                by_step[name] = steps

    _print_tables(results, floor, layers, ks)

    gate_path = os.path.join(args.out, "gate.json")
    gate = json.load(open(gate_path)) if os.path.exists(gate_path) else None
    if gate and not gate.get("pass", False):
        print("\nWARNING: training gate FAILED — SGD numbers below are NOT "
              "interpretable (loss did not drop comparably to AdamW).")

    summary = {
        "moment": args.moment, "ks": list(ks), "seed": args.seed,
        "gate": gate, "random_floor": floor,
        "adapters": {
            name: {"per_layer": result["per_layer"],
                   "aggregate": result["aggregate"],
                   "by_step": by_step.get(name)}
            for name, result in results.items()
        },
    }
    summary_path = os.path.join(args.out, "span_test.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"wrote {summary_path}")

    plot(results, by_step, floor, layers, ks,
         os.path.join(args.out, "span_test.png"))


if __name__ == "__main__":
    main()
