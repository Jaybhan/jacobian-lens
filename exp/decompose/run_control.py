"""CLI: positive control — pullback eigendirections through the decompose path.

    python -m exp.decompose.run_control \
        --delta-dir out/delta --lens out/lens/lens.pt \
        --unembed out/lens/unembed.pt --out out/control

Reads each ``{delta-dir}/{condition}.pt`` δ-screen accumulator, extracts the
top eigendirections of the pullback second moment per layer, and scores them
against the lens dictionary with the identical code path as the adapters
(same signed OMP, random floor, wrong-layer grid). See
``exp/decompose/control.py`` for what the outcomes mean.

Writes ``{out}/{condition}.pt`` / ``.json`` (same schema as ``run_decompose``,
module name ``"pullback"``) plus ``captured_energy`` per layer — the fraction
of pullback directional energy the control directions carry. Keep ``--out``
separate from the adapter artifacts; ``exp.decompose.analyze`` expects
o_proj/down_proj and would choke on these. Ends with a printed comparison
against any adapter summaries found in ``--decompose-dir``.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time

import torch

import jlens
from exp.decompose.align import alignment_for_adapter
from exp.decompose.control import CONTROL_MODULE, control_writes
from exp.decompose.dictionary import load_unembed
from exp.decompose.run_decompose import summarize
from jlens.lens import JacobianLens

logger = logging.getLogger(__name__)


def _print_comparison(
    control_results: dict[str, dict], decompose_dir: str
) -> None:
    """Control vs adapter signed alignment vs floor, at a few depths."""
    adapter_summaries: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(decompose_dir, "*.json"))):
        with open(path) as f:
            adapter_summaries[os.path.basename(path)[: -len(".json")]] = json.load(f)

    any_control = next(iter(control_results.values()))
    probe = [l for l in any_control["layers"] if l % 4 == 0]
    print(f"\nsigned alignment@{any_control['k']}: control (pullback eigendirections)"
          " vs adapters vs floor")
    print(f"{'series':<26} " + " ".join(f"L{l:>2}" for l in probe))
    for name, result in sorted(control_results.items()):
        cells = [
            f"{result['per_layer'][l][CONTROL_MODULE]['signed']:.2f}"
            if l in result["per_layer"] else "  - "
            for l in probe
        ]
        print(f"control:{name:<18} " + " ".join(cells))
    for name, summary in sorted(adapter_summaries.items()):
        cells = [
            f"{summary['signed'][str(l)]['down_proj']:.2f}"
            if str(l) in summary["signed"] else "  - "
            for l in probe
        ]
        print(f"adapter:{name:<18} " + " ".join(cells))
    floor = next(iter(control_results.values()))["floor"]
    cells = [f"{floor[l]['mean']:.2f}" if l in floor else "  - " for l in probe]
    print(f"{'random floor':<26} " + " ".join(cells))


def main() -> None:
    jlens.configure_logging()
    # jlens.configure_logging only handles the "jlens" logger; give "exp" a
    # real handler too or its INFO records die in logging.lastResort (WARNING+).
    logging.basicConfig(level=logging.INFO, format="[exp] %(message)s")
    logging.getLogger("exp").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["code-ift", "math-ift", "code-cpt", "math-cpt", "wikitext"],
    )
    parser.add_argument("--delta-dir", default="out/delta")
    parser.add_argument("--lens", required=True, help="lens.pt from exp.lens_fit")
    parser.add_argument("--unembed", required=True, help="unembed.pt from exp.lens_fit")
    parser.add_argument("--out", default="out/control")
    parser.add_argument("--decompose-dir", default="out/decompose",
                        help="adapter summaries to print a comparison against")
    parser.add_argument("--n-directions", type=int, default=16,
                        help="eigendirections per layer (default matches adapter r)")
    parser.add_argument("--k", type=int, default=25)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lens = JacobianLens.from_pretrained(args.lens)
    W_U, gamma = load_unembed(args.unembed)
    logger.info("lens: %s | unembed: %s", lens, tuple(W_U.shape))

    results: dict[str, dict] = {}
    for condition in args.conditions:
        out_pt = os.path.join(args.out, f"{condition}.pt")
        if os.path.exists(out_pt):
            logger.info("skipping %s: exists", out_pt)
            results[condition] = torch.load(
                out_pt, map_location="cpu", weights_only=False
            )
            continue
        delta_path = os.path.join(args.delta_dir, f"{condition}.pt")
        if not os.path.exists(delta_path):
            logger.warning("no accumulator at %s, skipping", delta_path)
            continue
        start = time.time()
        logger.info("=== %s ===", condition)
        state = torch.load(delta_path, map_location="cpu", weights_only=False)
        adapter = control_writes(state, n_directions=args.n_directions)
        del state
        result = alignment_for_adapter(
            adapter, lens, W_U, gamma, k=args.k, modules=(CONTROL_MODULE,)
        )
        result["captured_energy"] = {
            layer: adapter.writes[(layer, CONTROL_MODULE)].energy
            for layer in result["layers"]
        }
        torch.save(result, out_pt)
        summary = summarize(result)
        summary["captured_energy"] = {
            str(layer): energy for layer, energy in result["captured_energy"].items()
        }
        with open(os.path.join(args.out, f"{condition}.json"), "w") as f:
            json.dump(summary, f, indent=1)
        results[condition] = result
        logger.info("saved %s (%.0f s)", out_pt, time.time() - start)

    if results:
        _print_comparison(results, args.decompose_dir)


if __name__ == "__main__":
    main()
