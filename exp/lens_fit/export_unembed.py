"""CLI: export {W_U, γ} from the base model to unembed.pt (one-time, on pod).

    python -m exp.lens_fit.export_unembed \
        --model NousResearch/Llama-2-7b-hf --out out/lens/unembed.pt

After this (plus lens.pt), all of exp/decompose runs without the 13GB model.
"""

from __future__ import annotations

import argparse
import os

import torch

import jlens
from exp.decompose.dictionary import export_unembed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--out", default="out/lens/unembed.pt")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tokenizer)
    # jlens's layout detection already located the norm and lm_head; reuse it
    # rather than hardcoding attribute paths (private members, our own exp code).
    W_U = model._lm_head.weight.detach()
    gamma = model._final_norm.weight.detach()
    export_unembed(W_U, gamma, args.out, meta={"model": args.model})
    print(f"wrote {args.out}: W_U {tuple(W_U.shape)}, gamma {tuple(gamma.shape)}")


if __name__ == "__main__":
    main()
