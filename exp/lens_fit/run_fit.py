"""CLI: fit the J-lens on the base model (Phase 2; GPU, hours).

    python -m exp.lens_fit.run_fit \
        --model NousResearch/Llama-2-7b-hf \
        --n-prompts 100 --dim-batch 16 \
        --checkpoint /workspace/lens_ckpt.pt --out out/lens/lens.pt

Default corpus is WikiText (general). ``--corpus metamath`` refits on-
distribution (MetaMathQA, Alpaca-formatted as exp.delta's math-ift condition)
— the follow-up that separates "the workspace theory is wrong" from "the
WikiText-fitted lens is pointed at the wrong subspace to see it." Use a
different --checkpoint/--out per corpus; they are not interchangeable.

Resumable: rerunning with the same --checkpoint continues where it stopped
(jlens.fit checkpoints every prompt, atomically). Ends with two acceptance
checks: a mid-layer readability example, and late-layer agreement between the
J-lens and the plain logit lens (where they must coincide) — the frame sanity
check. Failures WARN rather than abort; judge the numbers yourself.
"""

from __future__ import annotations

import argparse
import logging
import os

import torch

import jlens
from exp.lens_fit.math_prompts import load_metamath_prompts
from jlens.examples import load_wikitext_prompts

logger = logging.getLogger(__name__)

ACCEPTANCE_PROMPT = (
    "Fact: The capital of Japan is Tokyo.\n"
    "Fact: The currency used in the country shaped like a boot is"
)


def load_model(name: str, device: str) -> jlens.HFLensModel:
    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # mandatory for grads w.r.t. activations
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return jlens.from_hf(hf, tokenizer)


def acceptance(model: jlens.HFLensModel, lens: jlens.JacobianLens) -> None:
    """Mid-layer readability + late-layer logit-lens agreement."""
    tokenizer = model.tokenizer
    mid_layers = [l for l in lens.source_layers if model.n_layers // 4 <= l <= 3 * model.n_layers // 4]
    lens_logits, model_logits, _ = lens.apply(
        model, ACCEPTANCE_PROMPT, layers=mid_layers, positions=[-1]
    )
    logger.info("mid-layer readout at final position (expect Italy/lira-ish mid-band):")
    for layer in mid_layers[:: max(1, len(mid_layers) // 8)]:
        top = [tokenizer.decode([t]) for t in lens_logits[layer][0].topk(5).indices]
        logger.info("  L%-2d %s", layer, top)
    logger.info("model output: %s",
                [tokenizer.decode([t]) for t in model_logits[0].topk(5).indices])

    late = max(lens.source_layers)
    j_top = set(
        lens.apply(model, ACCEPTANCE_PROMPT, layers=[late], positions=[-1])[0][late][0]
        .topk(10).indices.tolist()
    )
    logit_top = set(
        lens.apply(
            model, ACCEPTANCE_PROMPT, layers=[late], positions=[-1], use_jacobian=False
        )[0][late][0].topk(10).indices.tolist()
    )
    overlap = len(j_top & logit_top) / 10
    level = logging.WARNING if overlap < 0.6 else logging.INFO
    logger.log(level, "late-layer (L%d) J-lens vs logit-lens top-10 overlap: %.1f "
               "(< 0.6 suggests a frame problem)", late, overlap)


def main() -> None:
    jlens.configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--checkpoint", default="out/lens/lens_ckpt.pt")
    parser.add_argument("--out", default="out/lens/lens.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--source-layers", type=int, nargs="*", default=None,
        help="Default: every layer below the final one.",
    )
    parser.add_argument(
        "--corpus", default="wikitext", choices=["wikitext", "metamath"],
        help="wikitext: general corpus (original Phase 2). metamath: "
        "on-distribution refit for the decompose-null follow-up.",
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    model = load_model(args.model, args.device)
    logger.info("%s", model)
    if args.corpus == "metamath":
        prompts = load_metamath_prompts(args.n_prompts)
    else:
        prompts = load_wikitext_prompts(args.n_prompts)
    logger.info("fitting on %d %s prompts", len(prompts), args.corpus)

    lens = jlens.fit(
        model,
        prompts,
        source_layers=args.source_layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=args.checkpoint,
    )
    lens.save(args.out)
    logger.info("saved %s", args.out)
    acceptance(model, lens)


if __name__ == "__main__":
    main()
