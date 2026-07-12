"""CLI: turn saved pullback accumulators into the δ-vs-layer figure + tables.

    python -m exp.delta.analyze --dir out/delta

Reads every ``{condition}.pt`` written by ``run_extract``, computes the δ
metrics per layer per condition, and writes into ``--dir``:

- ``delta_vs_layer.png`` — the gate figure (participation ratio + top-25
  energy vs depth, jackknife bands, random-baseline reference),
- ``summary.json`` — all metrics, spectra digests, token stats, subspace
  angles between steering and skill conditions,
- printed per-layer table.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

import torch

from exp.delta.metrics import (
    jackknife_participation_ratio,
    mean_direction_norm,
    participation_ratio,
    principal_angles_between,
    random_baseline_pr,
    spectrum_from_second_moment,
    topk_basis,
    topk_energy_fractions,
)

# Color follows the dataset entity (fixed categorical slots, validated);
# linestyle carries the loss regime (solid = as-trained, dashed = all-token
# regime control). Palette per references/palette.md, light mode.
_SERIES: dict[str, dict[str, Any]] = {
    "code-ift": {"color": "#2a78d6", "dash": "solid", "label": "code IFT (Magicoder)"},
    "code-ift-alltok": {"color": "#2a78d6", "dash": (0, (5, 3)), "label": "code IFT, all-token loss"},
    "math-ift": {"color": "#1baf7a", "dash": "solid", "label": "math IFT (MetaMath)"},
    "math-ift-alltok": {"color": "#1baf7a", "dash": (0, (5, 3)), "label": "math IFT, all-token loss"},
    "code-cpt": {"color": "#eda100", "dash": "solid", "label": "code CPT (StarCoder py)"},
    "math-cpt": {"color": "#008300", "dash": "solid", "label": "math CPT (OpenWebMath)"},
    "wikitext": {"color": "#4a3aa7", "dash": "solid", "label": "WikiText (neutral)"},
}
_INK, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

_STEERING = ("code-ift", "math-ift")
_SKILL = ("code-cpt", "math-cpt")


def analyze_condition(path: str) -> dict[str, Any]:
    """Per-layer metrics for one saved accumulator."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    layers = state["layers"]
    result: dict[str, Any] = {
        "condition": state["meta"].get("condition", os.path.basename(path)),
        "layers": layers,
        "n_examples": state["n_examples"],
        "meta": state["meta"],
        "per_layer": {},
        "token_stats": _token_stats(state["example_stats"]),
        "_second_moment": state["second_moment"],  # kept for cross-condition angles
        "_n_vectors": state["n_vectors"],
    }
    for layer in layers:
        n = state["n_vectors"][layer]
        eigvals = spectrum_from_second_moment(state["second_moment"][layer], n)
        pr_jack, pr_se = jackknife_participation_ratio(
            state["reservoir"][layer].float(), state["reservoir_example"][layer]
        )
        result["per_layer"][layer] = {
            "n_vectors": n,
            "participation_ratio": participation_ratio(eigvals),
            "pr_reservoir": pr_jack,
            "pr_se": pr_se,
            "topk_energy": topk_energy_fractions(eigvals),
            "mean_direction_norm": mean_direction_norm(state["mean_sum"][layer], n),
            "top_eigvals": eigvals[:50].tolist(),
        }
    return result


def _token_stats(example_stats: list[dict[str, float]]) -> dict[str, float]:
    if not example_stats:
        return {}
    seq = torch.tensor([s["seq_len"] for s in example_stats])
    targets = torch.tensor([s["n_targets"] for s in example_stats])
    loss = torch.tensor([s["loss"] for s in example_stats])
    return {
        "n_examples": len(example_stats),
        "seq_len_mean": float(seq.mean()),
        "seq_len_p10": float(seq.quantile(0.1)),
        "seq_len_p90": float(seq.quantile(0.9)),
        "targets_mean": float(targets.mean()),
        "loss_per_target_mean": float((loss / targets.clamp(min=1)).mean()),
    }


def _cross_condition_angles(results: dict[str, dict], k: int = 25) -> dict[str, Any]:
    """Median principal angle (degrees) between top-k pullback subspaces for
    each steering x skill pair, per layer."""
    angles: dict[str, Any] = {}
    # Each condition's basis is compared against every partner: compute the
    # O(d³) eigenbasis once per (condition, layer), not once per pairing.
    bases: dict[tuple[str, int], torch.Tensor] = {}

    def basis_for(name: str, layer: int) -> torch.Tensor:
        key = (name, layer)
        if key not in bases:
            bases[key] = topk_basis(
                results[name]["_second_moment"][layer],
                results[name]["_n_vectors"][layer],
                k=k,
            )
        return bases[key]

    for a in _STEERING:
        for b in _SKILL:
            if a not in results or b not in results:
                continue
            per_layer = {}
            for layer in results[a]["layers"]:
                theta = principal_angles_between(
                    basis_for(a, layer), basis_for(b, layer)
                )
                per_layer[layer] = float(theta.median() * 180 / torch.pi)
            angles[f"{a}|{b}"] = per_layer
    return angles


def plot(results: dict[str, dict], baseline_pr: float, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_pr, ax_energy) = plt.subplots(
        2, 1, figsize=(8.5, 8), sharex=True, facecolor=_SURFACE
    )
    for name, result in sorted(results.items()):
        spec = _SERIES.get(name, {"color": _MUTED, "dash": "solid", "label": name})
        layers = result["layers"]
        pr = [result["per_layer"][l]["participation_ratio"] for l in layers]
        se = [result["per_layer"][l]["pr_se"] for l in layers]
        e25 = [result["per_layer"][l]["topk_energy"][25] for l in layers]
        ax_pr.plot(
            layers, pr, color=spec["color"], linestyle=spec["dash"], linewidth=2,
            label=spec["label"],
        )
        ax_pr.fill_between(
            layers,
            [p - 2 * s for p, s in zip(pr, se, strict=True)],
            [p + 2 * s for p, s in zip(pr, se, strict=True)],
            color=spec["color"], alpha=0.15, linewidth=0,
        )
        ax_energy.plot(
            layers, e25, color=spec["color"], linestyle=spec["dash"], linewidth=2,
        )
        # Direct end labels on the PR panel only (they collide where the
        # energy curves converge): the relief for sub-3:1 series colors.
        ax_pr.annotate(
            spec["label"], (layers[-1], pr[-1]),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, color=_INK, va="center",
        )
    ax_pr.axhline(
        baseline_pr, color=_MUTED, linestyle=(0, (2, 2)), linewidth=1.5
    )
    ax_pr.annotate(
        f"random baseline (PR={baseline_pr:.0f})",
        (0.01, baseline_pr), xycoords=("axes fraction", "data"),
        xytext=(0, 4), textcoords="offset points", fontsize=8, color=_MUTED,
    )
    ax_pr.set_yscale("log")
    ax_pr.set_ylabel("participation ratio (effective dim of pullbacks)", color=_INK)
    ax_energy.set_ylabel("energy in top-25 directions", color=_INK)
    ax_energy.set_xlabel("layer (block output)", color=_INK)
    ax_energy.set_ylim(0, 1)
    from matplotlib.ticker import MaxNLocator

    ax_energy.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Legend outside the axes (below the figure) so it never covers data.
    handles, labels = ax_pr.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", fontsize=8, frameon=False,
        ncol=2, bbox_to_anchor=(0.5, 0.0),
    )
    for ax in (ax_pr, ax_energy):
        ax.set_facecolor(_SURFACE)
        ax.grid(True, color=_GRID, linewidth=0.75)
        ax.tick_params(colors=_MUTED)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.margins(x=0.12)  # room for direct labels
    fig.suptitle(
        "δ screen: directional scatter of per-example pullback vectors\n"
        "prediction: IFT (steering) low, CPT (skill) high, gap peaks mid-network",
        color=_INK, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=200, facecolor=_SURFACE)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="out/delta")
    parser.add_argument("--top-k-angles", type=int, default=25)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.pt")))
    if not paths:
        raise SystemExit(f"no .pt accumulators found in {args.dir}")
    results = {}
    for i, path in enumerate(paths):
        print(f"[{i + 1}/{len(paths)}] analyzing {os.path.basename(path)}", flush=True)
        result = analyze_condition(path)
        results[result["condition"]] = result

    any_result = next(iter(results.values()))
    d_model = any_result["_second_moment"][any_result["layers"][0]].shape[0]
    n_reference = min(8192, min(r["_n_vectors"][r["layers"][0]] for r in results.values()))
    baseline_pr = random_baseline_pr(d_model, n_reference)

    # Printed table: one row per (condition, layer) at a few depths.
    probe_layers = [l for l in any_result["layers"] if l % 4 == 0]
    header = f"{'condition':<22} " + " ".join(f"L{l:>2}" for l in probe_layers)
    print("\nparticipation ratio by depth (random baseline "
          f"~{baseline_pr:.0f} at n={n_reference}):\n" + header)
    for name, result in sorted(results.items()):
        row = " ".join(
            f"{result['per_layer'][l]['participation_ratio']:>4.0f}"
            if l in result["per_layer"] else "   -"
            for l in probe_layers
        )
        print(f"{name:<22} {row}")

    print("computing cross-condition angles...", flush=True)
    summary = {
        "baseline_pr": baseline_pr,
        "baseline_n": n_reference,
        "cross_condition_angles_deg": _cross_condition_angles(
            results, k=args.top_k_angles
        ),
        "conditions": {
            name: {k: v for k, v in r.items() if not k.startswith("_")}
            for name, r in results.items()
        },
    }
    summary_path = os.path.join(args.dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"wrote {summary_path}")

    plot(results, baseline_pr, os.path.join(args.dir, "delta_vs_layer.png"))


if __name__ == "__main__":
    main()
