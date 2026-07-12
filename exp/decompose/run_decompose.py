"""CLI: decompose adapters against a fitted lens; save alignment artifacts.

    python -m exp.decompose.run_decompose \
        --lens out/lens/lens.pt --unembed out/lens/unembed.pt \
        --out out/decompose

CPU-only; minutes per adapter. Writes ``{out}/{short-name}.pt`` (full curves,
atom ids) and ``{out}/{short-name}.json`` (summary numbers). Analyze with
``python -m exp.decompose.analyze --dir out/decompose``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import torch

import jlens
from exp.decompose.adapters import load_adapter_writes
from exp.decompose.align import alignment_for_adapter
from exp.decompose.dictionary import load_unembed
from jlens.lens import JacobianLens

logger = logging.getLogger(__name__)

DEFAULT_ADAPTERS = (
    "LoRA-TMLR-2024/magicoder-lora-rank-16-alpha-32",
    "LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32",
    "LoRA-TMLR-2024/starcoder-lora-rank-16-20B-tokens",
    "LoRA-TMLR-2024/openwebmath-lora-rank-16-20B-tokens",
)


def short_name(repo: str) -> str:
    return repo.split("/")[-1].split("-lora-")[0]


def summarize(result: dict) -> dict:
    """JSON-safe summary: per layer per module scalar metrics only."""
    return {
        "adapter": result["adapter"],
        "r": result["r"],
        "k": result["k"],
        "coverage": result["coverage"],
        "residual_coverage": result["residual_coverage"],
        "layers": result["layers"],
        "signed": {
            str(layer): {m: s["signed"] for m, s in mods.items()}
            for layer, mods in result["per_layer"].items()
        },
        "nonneg": {
            str(layer): {m: s["nonneg"] for m, s in mods.items()}
            for layer, mods in result["per_layer"].items()
        },
        "projection": {
            str(layer): {
                m: {str(k): v for k, v in s["projection"].items()}
                for m, s in mods.items()
            }
            for layer, mods in result["per_layer"].items()
        },
        "floor": {str(layer): f for layer, f in result["floor"].items()},
        "wrong_layer": {
            str(layer): {
                str(offset): entry for offset, entry in grid.items()
            }
            for layer, grid in result["wrong_layer"].items()
        },
    }


def main() -> None:
    jlens.configure_logging()
    logging.getLogger("exp").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapters", nargs="+", default=list(DEFAULT_ADAPTERS))
    parser.add_argument("--lens", required=True, help="lens.pt from exp.lens_fit")
    parser.add_argument("--unembed", required=True, help="unembed.pt from exp.lens_fit")
    parser.add_argument("--out", default="out/decompose")
    parser.add_argument("--k", type=int, default=25)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lens = JacobianLens.from_pretrained(args.lens)
    W_U, gamma = load_unembed(args.unembed)
    logger.info("lens: %s | unembed: %s", lens, tuple(W_U.shape))

    for repo in args.adapters:
        name = short_name(repo)
        out_pt = os.path.join(args.out, f"{name}.pt")
        if os.path.exists(out_pt):
            logger.info("skipping %s: exists", out_pt)
            continue
        start = time.time()
        logger.info("=== %s ===", repo)
        adapter = load_adapter_writes(repo)
        logger.info(
            "  r=%d scale=%.1f layers=%d residual_coverage=%.2f",
            adapter.r,
            adapter.scale,
            adapter.n_layers,
            adapter.residual_coverage(),
        )
        result = alignment_for_adapter(adapter, lens, W_U, gamma, k=args.k)
        torch.save(result, out_pt)
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(summarize(result), f, indent=1)
        logger.info("saved %s (%.0f s)", out_pt, time.time() - start)


if __name__ == "__main__":
    main()
