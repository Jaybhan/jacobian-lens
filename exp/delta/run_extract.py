"""CLI: extract pullback accumulators for one or more conditions.

Usage (on the GPU box):

    python -m exp.delta.run_extract \
        --model NousResearch/Llama-2-7b-hf \
        --conditions code-ift math-ift code-cpt math-cpt wikitext \
        --n-examples 2000 --out out/delta

Writes ``{out}/{condition}.pt`` per condition (second moments, means,
reservoirs, per-example stats, run metadata). Analyze with
``python -m exp.delta.analyze --dir out/delta``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import torch

import jlens
from exp.delta.data import CONDITIONS, iter_condition
from exp.delta.pullbacks import extract_condition

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
    parser.add_argument("--n-examples", type=int, default=2000)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--out", default="out/delta")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Block indices; default: every layer.",
    )
    parser.add_argument("--reservoir-cap", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_model(name: str, device: str, dtype: str) -> jlens.HFLensModel:
    import transformers

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    logger.info("loading %s (%s, %s)", name, device, dtype)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch_dtype,
        # Eager attention is mandatory: fused SDPA/Flash kernels can fail (or
        # silently misbehave) for grads w.r.t. intermediate activations.
        attn_implementation="eager",
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return jlens.from_hf(hf, tokenizer)


def main() -> None:
    jlens.configure_logging()
    logging.getLogger("exp").setLevel(logging.INFO)
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = load_model(args.model, args.device, args.dtype)
    layers = args.layers if args.layers else list(range(model.n_layers))

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
        accumulator = extract_condition(
            model,
            examples,
            layers,
            reservoir_cap=args.reservoir_cap,
            seed=args.seed,
        )
        accumulator.save(
            out_path,
            meta={
                "condition": condition,
                "model": args.model,
                "n_examples_requested": args.n_examples,
                "max_seq_len": args.max_seq_len,
                "dtype": args.dtype,
                "seconds": time.time() - start,
                "argv": json.dumps(vars(args), default=str),
            },
        )
        logger.info("saved %s (%.0f s)", out_path, time.time() - start)


if __name__ == "__main__":
    main()
