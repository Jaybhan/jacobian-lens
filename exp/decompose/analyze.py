"""CLI: alignment-vs-depth figure + decoded-atoms report from saved artifacts.

    python -m exp.decompose.analyze --dir out/decompose \
        [--tokenizer LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32]

Reads every ``{adapter}.pt`` written by ``run_decompose`` and writes into
``--dir``: ``alignment_vs_depth.png``, ``atoms_report.md`` (decoded top atoms
for middle-band directions; needs ``--tokenizer``, else ids only), and prints
the summary table including the wrong-layer diagnostics.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any

import torch

# Color follows the dataset entity, consistent with exp/delta/analyze.py
# (validated palette): magicoder blue, metamath aqua, starcoder yellow,
# openwebmath green. Solid = IFT (steering), dashed only for controls.
_SERIES: dict[str, dict[str, Any]] = {
    "magicoder": {"color": "#2a78d6", "label": "code IFT (Magicoder)"},
    "metamath": {"color": "#1baf7a", "label": "math IFT (MetaMath)"},
    "starcoder": {"color": "#eda100", "label": "code CPT (StarCoder py)"},
    "openwebmath": {"color": "#008300", "label": "math CPT (OpenWebMath)"},
}
_INK, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
_MODULES = ("o_proj", "down_proj")


def load_results(directory: str) -> dict[str, dict]:
    results = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.pt"))):
        result = torch.load(path, map_location="cpu", weights_only=False)
        name = os.path.basename(path)[: -len(".pt")]
        results[name] = result
    if not results:
        raise SystemExit(f"no .pt artifacts in {directory}")
    return results


def plot(results: dict[str, dict], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(
        len(_MODULES), 1, figsize=(8.5, 8), sharex=True, facecolor=_SURFACE
    )
    for ax, module in zip(axes, _MODULES, strict=True):
        for name, result in sorted(results.items()):
            spec = _SERIES.get(name, {"color": _MUTED, "label": name})
            layers = [
                l for l in result["layers"] if module in result["per_layer"][l]
            ]
            values = [result["per_layer"][l][module]["signed"] for l in layers]
            ax.plot(
                layers, values, color=spec["color"], linewidth=2,
                label=spec["label"] if module == _MODULES[0] else None,
            )
            ax.annotate(
                spec["label"], (layers[-1], values[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, color=_INK, va="center",
            )
        # Random floor (same for every adapter up to seed; take the first).
        first = next(iter(results.values()))
        floor_layers = sorted(first["floor"])
        floor = [first["floor"][l]["mean"] for l in floor_layers]
        ax.plot(
            floor_layers, floor, color=_MUTED, linestyle=(0, (2, 2)),
            linewidth=1.5,
        )
        ax.annotate(
            "random-B floor", (floor_layers[0], floor[0]),
            xytext=(0, 5), textcoords="offset points", fontsize=8, color=_MUTED,
        )
        ax.set_ylabel(f"{module}: energy in {first['k']} atoms", color=_INK)
        ax.set_ylim(0, 1)
        ax.set_facecolor(_SURFACE)
        ax.grid(True, color=_GRID, linewidth=0.75)
        ax.tick_params(colors=_MUTED)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.margins(x=0.14)
    axes[-1].set_xlabel("layer (block output)", color=_INK)
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", fontsize=8, frameon=False,
        ncol=2, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "LoRA write directions decomposed in the J-lens dictionary\n"
        "prediction: IFT high in the middle band, CPT low/smeared",
        color=_INK, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=200, facecolor=_SURFACE)
    print(f"wrote {out_path}")


def atoms_report(
    results: dict[str, dict], out_path: str, tokenizer_name: str | None
) -> None:
    """Decode the selected atoms of the strongest middle-band directions."""
    tokenizer = None
    if tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    lines = ["# Decoded lens atoms per adapter (middle band)\n"]
    for name, result in sorted(results.items()):
        layers = result["layers"]
        band = [l for l in layers if layers[len(layers) // 3] <= l <= layers[2 * len(layers) // 3]]
        lines.append(f"\n## {name}  (residual coverage {result['residual_coverage']:.2f})\n")
        for layer in band:
            for module, stats in result["per_layer"][layer].items():
                # Strongest direction by S; its atoms in selection order.
                strongest = int(stats["S"].argmax())
                atom_ids = stats["atoms"][strongest].tolist()
                if tokenizer is not None:
                    atoms = [repr(tokenizer.decode([a])) for a in atom_ids[:10]]
                else:
                    atoms = [str(a) for a in atom_ids[:10]]
                lines.append(
                    f"- L{layer} {module} (align {stats['per_direction_signed'][strongest]:.2f}): "
                    + ", ".join(atoms)
                )
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


def print_table(results: dict[str, dict]) -> None:
    first = next(iter(results.values()))
    probe = [l for l in first["layers"] if l % 4 == 0]
    for module in _MODULES:
        print(f"\nsigned alignment@{first['k']} — {module}")
        print(f"{'adapter':<14} " + " ".join(f"L{l:>2}" for l in probe))
        for name, result in sorted(results.items()):
            cells = []
            for l in probe:
                stats = result["per_layer"].get(l, {}).get(module)
                cells.append(f"{stats['signed']:.2f}" if stats else "  - ")
            print(f"{name:<14} " + " ".join(cells))
    print("\nwrong-layer check (signed alignment, offset -> value; 0-offset must dominate):")
    for name, result in sorted(results.items()):
        for layer, grid in sorted(result["wrong_layer"].items())[:2]:
            own = result["per_layer"][layer]
            own_str = {m: f"{s['signed']:.2f}" for m, s in own.items()}
            off_str = {
                offset: {m: f"{v:.2f}" for m, v in entry.items()}
                for offset, entry in sorted(grid.items())
            }
            print(f"  {name} L{layer}: own={own_str} wrong={off_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="out/decompose")
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()

    results = load_results(args.dir)
    print_table(results)
    plot(results, os.path.join(args.dir, "alignment_vs_depth.png"))
    atoms_report(results, os.path.join(args.dir, "atoms_report.md"), args.tokenizer)


if __name__ == "__main__":
    main()
