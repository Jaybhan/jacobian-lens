"""CLI: turn saved δ-sketch accumulators into the concentration figure + tables.

    python -m exp.dsketch.analyze --dir out/dsketch

Reads every ``{condition}.pt`` written by ``exp.dsketch.run_extract`` and
writes into ``--dir``:

- ``dsketch_vs_layer.png`` — mean-direction norm and example-level PR vs
  depth per condition (median over probes, min/max band, baselines),
- ``summary.json`` — per (condition, probe, layer) metrics, aggregates,
  steering×skill transport angles, probe metadata,
- printed per-layer tables.

The primary statistic is the across-example scatter of per-example MEAN
pullback directions of each fixed probe: ``mean_direction_norm`` near 1 (and
``centered_pr`` near 1) means every example transports the probe the same way
— the small-δ regime. ``centered_pr`` guards the residual-path risk: a large
shared ``c·Jᵀv`` component can push raw mdn toward 1 for every condition;
the centered scatter is the heterogeneity (``Eᵀv``) component itself.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any

import torch

from exp.delta.metrics import (
    jackknife_participation_ratio,
    mean_direction_norm,
    participation_ratio_from_moment,
    random_baseline_pr,
    second_moment_of,
)

# Colors follow the dataset entity, matching exp/delta/analyze.py's palette
# (copied, not imported, to keep the modules decoupled).
_SERIES: dict[str, dict[str, Any]] = {
    "code-ift": {"color": "#2a78d6", "label": "code IFT (Magicoder)"},
    "math-ift": {"color": "#1baf7a", "label": "math IFT (MetaMath)"},
    "code-cpt": {"color": "#eda100", "label": "code CPT (StarCoder py)"},
    "math-cpt": {"color": "#008300", "label": "math CPT (OpenWebMath)"},
    "wikitext": {"color": "#4a3aa7", "label": "WikiText (neutral)"},
}
_INK, _MUTED, _GRID, _SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

_STEERING = ("code-ift", "math-ift")
_SKILL = ("code-cpt", "math-cpt")


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    cos = float((a @ b) / (a.norm() * b.norm()))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def analyze_condition(path: str) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    layers = state["layers"]
    n_probes = state["n_probes"]
    result: dict[str, Any] = {
        "condition": state["meta"].get("condition", os.path.basename(path)),
        "layers": layers,
        "n_examples": state["n_examples"],
        "n_probes": n_probes,
        "meta": state["meta"],
        "probe_kinds": list(state["probes"].get("kinds", [])),
        "probe_tokens": list(state["probes"].get("tokens", [])),
        "per_probe_layer": {},
        "_probe_vectors": state["probes"]["vectors"].float(),
        "_mean_dir": {},  # (j, layer) -> unit mean direction, for angles
    }
    for j in range(n_probes):
        result["per_probe_layer"][j] = {}
        inner = state["per_probe"][j]
        for layer in layers:
            rows = state["example_mean"][j][layer].float()
            if rows.shape[0] < 2:
                continue
            ids = state["example_mean_ids"][j][layer]
            mean = rows.mean(dim=0)
            mdn = float(mean.norm())
            pr, pr_se = jackknife_participation_ratio(rows, ids)
            centered = rows - mean
            centered_moment = second_moment_of(centered)
            # Zero heterogeneity (all example-means identical) leaves nothing
            # to take a PR of — report NaN, filtered from aggregates.
            centered_pr = (
                participation_ratio_from_moment(centered_moment)
                if float(centered_moment.double().trace()) > 1e-12
                else float("nan")
            )
            n_positions = inner["n_vectors"][layer]
            result["per_probe_layer"][j][layer] = {
                "mean_direction_norm": mdn,
                "delta_sq_fraction": 1.0 - mdn * mdn,
                "pr": pr,
                "pr_se": pr_se,
                "centered_pr": centered_pr,
                "n_examples": rows.shape[0],
                "position_pr": participation_ratio_from_moment(
                    inner["second_moment"][layer]
                ),
                "position_mean_dir_norm": mean_direction_norm(
                    inner["mean_sum"][layer], n_positions
                ),
            }
            result["_mean_dir"][(j, layer)] = mean / mdn
    return result


def _aggregate(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Median + range over probes per layer, pooled and by probe kind."""
    aggregate: dict[int, dict[str, Any]] = {}
    kinds = result["probe_kinds"] or ["gauss"] * result["n_probes"]
    for layer in result["layers"]:
        per_metric: dict[str, dict[str, list[float]]] = {}
        for j, stats in result["per_probe_layer"].items():
            if layer not in stats:
                continue
            kind = kinds[j] if j < len(kinds) else "gauss"
            for metric in ("mean_direction_norm", "pr", "centered_pr"):
                value = stats[layer][metric]
                if not math.isfinite(value):
                    continue
                per_metric.setdefault(metric, {}).setdefault("all", []).append(value)
                per_metric[metric].setdefault(kind, []).append(value)
        if not per_metric:
            continue
        entry: dict[str, Any] = {}
        for metric, groups in per_metric.items():
            for group, values in groups.items():
                tensor = torch.tensor(values, dtype=torch.float64)
                key = metric if group == "all" else f"{metric}:{group}"
                entry[f"{key}_median"] = float(tensor.median())
                entry[f"{key}_range"] = [float(tensor.min()), float(tensor.max())]
        aggregate[layer] = entry
    return aggregate


def _cross_condition_angles(results: dict[str, dict]) -> dict[str, Any]:
    """Angle between the two conditions' mean transported directions, per
    (probe, layer), for each steering x skill pair."""
    angles: dict[str, Any] = {}
    for a in _STEERING:
        for b in _SKILL:
            if a not in results or b not in results:
                continue
            per_probe: dict[int, dict[int, float]] = {}
            for (j, layer), dir_a in results[a]["_mean_dir"].items():
                dir_b = results[b]["_mean_dir"].get((j, layer))
                if dir_b is None:
                    continue
                per_probe.setdefault(j, {})[layer] = _angle_deg(dir_a, dir_b)
            angles[f"{a}|{b}"] = per_probe
    return angles


def _probe_transport(results: dict[str, dict]) -> dict[str, Any]:
    """How far transport rotates each probe: angle(condition mean dir, v)."""
    transport: dict[str, Any] = {}
    for name, result in results.items():
        vectors = result["_probe_vectors"]
        per_probe: dict[int, dict[int, float]] = {}
        for (j, layer), mean_dir in result["_mean_dir"].items():
            per_probe.setdefault(j, {})[layer] = _angle_deg(mean_dir, vectors[j])
        transport[name] = per_probe
    return transport


def _print_tables(results: dict[str, dict], baselines: dict[str, float]) -> None:
    any_result = next(iter(results.values()))
    layers = any_result["layers"]
    for metric, title in (
        ("mean_direction_norm", "mean-direction norm (1 = zero heterogeneity)"),
        ("pr", "example-level participation ratio"),
        ("centered_pr", "centered PR (heterogeneity component)"),
    ):
        print(f"\n{title}, median over probes "
              f"(isotropic mdn~{baselines['mean_dir_norm_isotropic']:.3f}, "
              f"random PR~{baselines['pr_random']:.0f}):")
        print("%-22s " % "condition" + " ".join(f"L{l:>2}" for l in layers))
        for name, result in sorted(results.items()):
            aggregate = result["aggregate"]
            cells = []
            for layer in layers:
                value = aggregate.get(layer, {}).get(f"{metric}_median")
                if value is None:
                    cells.append("    -")
                elif metric == "mean_direction_norm":
                    cells.append(f"{value:>5.3f}"[:5])
                else:
                    cells.append(f"{value:>5.0f}")
            print("%-22s " % name + " ".join(cells))


def plot(results: dict[str, dict], baselines: dict[str, float], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, (ax_mdn, ax_pr) = plt.subplots(
        2, 1, figsize=(8.5, 8), sharex=True, facecolor=_SURFACE
    )
    for name, result in sorted(results.items()):
        spec = _SERIES.get(name, {"color": _MUTED, "label": name})
        layers = [l for l in result["layers"] if l in result["aggregate"]]
        aggregate = result["aggregate"]
        mdn = [aggregate[l]["mean_direction_norm_median"] for l in layers]
        mdn_lo = [aggregate[l]["mean_direction_norm_range"][0] for l in layers]
        mdn_hi = [aggregate[l]["mean_direction_norm_range"][1] for l in layers]
        pr = [aggregate[l]["pr_median"] for l in layers]
        pr_lo = [aggregate[l]["pr_range"][0] for l in layers]
        pr_hi = [aggregate[l]["pr_range"][1] for l in layers]
        ax_mdn.plot(layers, mdn, color=spec["color"], linewidth=2,
                    label=spec["label"])
        ax_mdn.fill_between(layers, mdn_lo, mdn_hi, color=spec["color"],
                            alpha=0.15, linewidth=0)
        ax_pr.plot(layers, pr, color=spec["color"], linewidth=2)
        ax_pr.fill_between(layers, pr_lo, pr_hi, color=spec["color"],
                           alpha=0.15, linewidth=0)
    iso = baselines["mean_dir_norm_isotropic"]
    ax_mdn.axhline(iso, color=_MUTED, linestyle=(0, (2, 2)), linewidth=1.5)
    ax_mdn.annotate(f"isotropic (~{iso:.3f})", (0.01, iso),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, 4), textcoords="offset points",
                    fontsize=8, color=_MUTED)
    ax_mdn.set_ylim(0, 1.02)
    ax_mdn.set_ylabel("mean-direction norm of probe pullbacks", color=_INK)
    ax_pr.axhline(baselines["pr_random"], color=_MUTED,
                  linestyle=(0, (2, 2)), linewidth=1.5)
    ax_pr.annotate(f"random baseline (PR={baselines['pr_random']:.0f})",
                   (0.01, baselines["pr_random"]),
                   xycoords=("axes fraction", "data"),
                   xytext=(0, 4), textcoords="offset points",
                   fontsize=8, color=_MUTED)
    ax_pr.set_yscale("log")
    ax_pr.set_ylabel("example-level PR of probe pullbacks", color=_INK)
    ax_pr.set_xlabel("layer (block output)", color=_INK)
    ax_pr.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = ax_mdn.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=8, frameon=False,
               ncol=2, bbox_to_anchor=(0.5, 0.0))
    for ax in (ax_mdn, ax_pr):
        ax.set_facecolor(_SURFACE)
        ax.grid(True, color=_GRID, linewidth=0.75)
        ax.tick_params(colors=_MUTED)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
    fig.suptitle(
        "δ-sketch: across-example scatter of FIXED-cotangent pullbacks\n"
        "prediction: steering data concentrated (high mdn / low PR), "
        "skill data scattered",
        color=_INK, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_path, dpi=200, facecolor=_SURFACE)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="out/dsketch")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.pt")))
    if not paths:
        raise SystemExit(f"no .pt accumulators found in {args.dir}")
    results: dict[str, dict] = {}
    for i, path in enumerate(paths):
        print(f"[{i + 1}/{len(paths)}] analyzing {os.path.basename(path)}",
              flush=True)
        result = analyze_condition(path)
        result["aggregate"] = _aggregate(result)
        results[result["condition"]] = result

    # Probes must be identical across conditions — that's the design.
    reference = next(iter(results.values()))["_probe_vectors"]
    for name, result in results.items():
        if not torch.allclose(result["_probe_vectors"], reference, atol=1e-6):
            print(f"WARNING: {name} was extracted with DIFFERENT probes — "
                  "cross-condition comparison is invalid", flush=True)

    d_model = reference.shape[1]
    n_reference = min(
        stats["n_examples"]
        for result in results.values()
        for per_layer in result["per_probe_layer"].values()
        for stats in per_layer.values()
    )
    baselines = {
        "pr_random": random_baseline_pr(d_model, n_reference),
        "n": n_reference,
        "mean_dir_norm_isotropic": 1.0 / math.sqrt(n_reference),
    }

    _print_tables(results, baselines)

    summary = {
        "baseline": baselines,
        "probes": {
            "kinds": next(iter(results.values()))["probe_kinds"],
            "tokens": next(iter(results.values()))["probe_tokens"],
        },
        "conditions": {
            name: {
                "layers": r["layers"],
                "n_examples": r["n_examples"],
                "per_probe_layer": r["per_probe_layer"],
                "aggregate": r["aggregate"],
            }
            for name, r in results.items()
        },
        "cross_condition_angles_deg": _cross_condition_angles(results),
        "probe_transport_angles_deg": _probe_transport(results),
    }
    summary_path = os.path.join(args.dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"wrote {summary_path}")

    plot(results, baselines, os.path.join(args.dir, "dsketch_vs_layer.png"))


if __name__ == "__main__":
    main()
