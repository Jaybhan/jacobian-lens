"""CLI: fixed-cotangent δ-sketch extraction (Phase 1.5).

Usage (on the GPU box):

    python -m exp.dsketch.run_extract \
        --model NousResearch/Llama-2-7b-hf \
        --conditions code-ift math-ift code-cpt math-cpt wikitext \
        --n-examples 500 --out out/dsketch

Pulls a FIXED probe set (shared across all examples and conditions) back
through each example: one forward + one backward per probe. With the
cotangent fixed, across-example scatter isolates propagator heterogeneity
(the theory's Assumption-A ``δ``) from the ε diversity that the δ-screen
conflates it with. Loss masking is irrelevant here, so the ``*-alltok``
conditions are definitionally identical to their bases — run the 5 base
conditions. Writes ``{out}/{condition}.pt``; analyze with
``python -m exp.dsketch.analyze --dir out/dsketch``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import jlens
from exp.delta.data import CONDITIONS, iter_condition
from exp.delta.run_extract import load_model
from exp.dsketch.extract import extract_condition_sketch
from exp.dsketch.probes import DEFAULT_PROBE_TOKENS, build_probes

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["code-ift", "math-ift", "code-cpt", "math-cpt", "wikitext"],
        choices=sorted(CONDITIONS),
    )
    parser.add_argument("--n-examples", type=int, default=500)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--out", default="out/dsketch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Block indices; default: every 4th layer (the final block is "
        "excluded — its probe pullback is degenerate).",
    )
    parser.add_argument("--n-gauss", type=int, default=4)
    parser.add_argument(
        "--probe-tokens", nargs="+", default=list(DEFAULT_PROBE_TOKENS)
    )
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--reservoir-cap", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    jlens.configure_logging()
    logging.getLogger("exp").setLevel(logging.INFO)
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = load_model(args.model, args.device, args.dtype)
    layers = args.layers if args.layers else list(range(0, model.n_layers, 4))

    # One probe set, shared verbatim across every condition — that sharing is
    # what makes the across-condition comparison meaningful.
    probes = build_probes(
        model, n_gauss=args.n_gauss, tokens=args.probe_tokens, seed=args.probe_seed
    )
    logger.info(
        "probes: %d (%s)", len(probes),
        ", ".join(
            k if t is None else f"{k}:{t!r}"
            for k, t in zip(probes.kinds, probes.tokens, strict=True)
        ),
    )

    for condition in args.conditions:
        out_path = os.path.join(args.out, f"{condition}.pt")
        if os.path.exists(out_path):
            logger.info("skipping %s: %s exists", condition, out_path)
            continue
        logger.info("=== condition %s -> %s ===", condition, out_path)
        start = time.time()
        examples = iter_condition(
            condition,
            model.tokenizer,
            n_examples=args.n_examples,
            max_seq_len=args.max_seq_len,
        )
        accumulator = extract_condition_sketch(
            model,
            examples,
            layers,
            probes,
            reservoir_cap=args.reservoir_cap,
            seed=args.seed,
        )
        accumulator.save(
            out_path,
            probes=probes.state_dict(),
            meta={
                "condition": condition,
                "model": args.model,
                "n_examples_requested": args.n_examples,
                "max_seq_len": args.max_seq_len,
                "dtype": args.dtype,
                "layers": list(layers),
                "seconds": time.time() - start,
                "argv": json.dumps(vars(args), default=str),
            },
        )
        logger.info("saved %s (%.0f s)", out_path, time.time() - start)


if __name__ == "__main__":
    main()
