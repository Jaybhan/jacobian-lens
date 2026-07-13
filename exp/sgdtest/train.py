"""CLI: train paired LoRA adapters (plain SGD vs AdamW) for the span test.

    python -m exp.sgdtest.train --out out/sgdtest

Trains rank-16 adapters on o_proj/down_proj of a frozen Llama-2-7B, on
MetaMathQA (Alpaca-formatted, response-masked mean CE — the δ-screen's
math-ift condition), twice: once with plain SGD (momentum 0) and once with
AdamW. Everything else — seed, A init, batch order, clipping — is identical,
so the arms differ ONLY in Adam's per-coordinate preconditioning (the one
span-breaking operation; momentum/global-clip/weight-decay are all scalar or
linear and span-preserving, which is why they are held at 0/1.0/0).

Both arms run in one process over one materialized example list: identical
batch order is true by construction, not by trust in dataset streaming.

SGD's learning rate comes from a short sweep (plain SGD needs a far larger
LR than Adam); the run is only interpretable if the HARD GATE passes — SGD's
loss drop must reach at least ``--gate-factor`` of AdamW's. Gate status is
stamped into train_log.json and must be checked before reading span numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import torch
import torch.nn.functional as F

import jlens
from exp.delta.data import PreparedExample, iter_condition
from exp.sgdtest.lora import attach_lora, detach_lora, lora_parameters, save_adapter

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--condition", default="math-ift")
    parser.add_argument("--optimizers", nargs="+", default=["sgd", "adamw"],
                        choices=["sgd", "adamw"])
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--adamw-lr", type=float, default=1e-4)
    parser.add_argument("--sweep-lrs", type=float, nargs="+",
                        default=[3e-2, 1e-1, 3e-1])
    parser.add_argument("--sweep-steps", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--gate-factor", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="out/sgdtest")
    return parser.parse_args()


def load_model(name: str, device: str):
    """Frozen bf16 base with default (SDPA) attention.

    Training only needs parameter grads through a standard backward — the
    eager-attention requirement in exp/delta applies to grads w.r.t.
    hook-captured intermediate activations, not here.
    """
    import transformers

    logger.info("loading %s (%s, bfloat16, default attention)", name, device)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16
    ).to(device)
    hf.eval()
    for param in hf.parameters():
        param.requires_grad_(False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return hf, tokenizer


def materialize_examples(
    tokenizer, condition: str, n: int, max_seq_len: int
) -> list[PreparedExample]:
    """One shared list => bit-identical batch order across arms."""
    started = time.time()
    examples = list(
        iter_condition(condition, tokenizer, n_examples=n, max_seq_len=max_seq_len)
    )
    logger.info("materialized %d %s examples (%.0f s)",
                len(examples), condition, time.time() - started)
    return examples


def loss_for_example(hf_model, example: PreparedExample, device: str) -> torch.Tensor:
    """Response-masked next-token CE, mean over scored positions.

    Position/target construction matches exp/delta/pullbacks.py exactly; the
    mean (vs the δ-screen's sum) is a per-example scalar multiple — span-
    preserving — chosen to decouple the effective LR from n_targets.
    """
    input_ids = example.input_ids.to(device)
    out = hf_model(input_ids, use_cache=False)
    positions = example.loss_positions.nonzero(as_tuple=True)[0]
    targets = input_ids[0, (positions + 1).to(device)]
    return F.cross_entropy(
        out.logits[0, positions.to(device)].float(), targets, reduction="mean"
    )


def make_optimizer(params, name: str, lr: float) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    raise ValueError(f"unknown optimizer {name!r}")


def train_arm(
    hf_model,
    examples: list[PreparedExample],
    *,
    optimizer_name: str,
    lr: float,
    steps: int,
    r: int,
    alpha: int,
    seed: int,
    clip: float,
    checkpoint_every: int,
    device: str,
    out_dir: str | None,
) -> list[dict[str, float]]:
    """Train one arm from scratch; returns per-step records."""
    sites = attach_lora(
        hf_model.model.layers, r=r, alpha=alpha, seed=seed, device=device
    )
    params = lora_parameters(sites)
    optimizer = make_optimizer(params, optimizer_name, lr)
    records: list[dict[str, float]] = []
    started = time.time()
    try:
        if out_dir:
            save_adapter(sites, os.path.join(out_dir, "checkpoints", "step-000000"),
                         r=r, alpha=alpha)
        for step in range(1, steps + 1):
            example = examples[(step - 1) % len(examples)]
            loss = loss_for_example(hf_model, example, device)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            records.append({
                "step": step,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "seconds": time.time() - started,
            })
            if step % 50 == 0:
                logger.info("  [%s lr=%g] step %d/%d  loss=%.4f",
                            optimizer_name, lr, step, steps, records[-1]["loss"])
            if out_dir and checkpoint_every and step % checkpoint_every == 0:
                save_adapter(
                    sites, os.path.join(out_dir, "checkpoints", f"step-{step:06d}"),
                    r=r, alpha=alpha,
                )
        if out_dir:
            save_adapter(sites, out_dir, r=r, alpha=alpha)
    finally:
        detach_lora(sites)
    return records


def lr_sweep(
    hf_model, examples, lrs, n_steps, **arm_kwargs
) -> tuple[float, dict[str, list[float]]]:
    """Short SGD sweep; score = mean loss over the last third of steps."""
    curves: dict[str, list[float]] = {}
    best_lr, best_score = None, float("inf")
    for lr in lrs:
        records = train_arm(
            hf_model, examples[:n_steps], optimizer_name="sgd", lr=lr,
            steps=n_steps, checkpoint_every=0, out_dir=None, **arm_kwargs,
        )
        losses = [rec["loss"] for rec in records]
        curves[str(lr)] = losses
        tail = losses[-max(1, n_steps // 3):]
        score = sum(tail) / len(tail)
        finite = all(map(lambda v: v == v and abs(v) != float("inf"), losses))
        logger.info("sweep lr=%g  tail-mean=%.4f  finite=%s", lr, score, finite)
        if finite and score < best_score:
            best_lr, best_score = lr, score
    if best_lr is None:
        raise RuntimeError("no finite SGD sweep candidate — widen --sweep-lrs")
    return best_lr, curves


def compute_gate(
    records_by_arm: dict[str, list[dict]], *, window: int = 20, factor: float = 0.5
) -> dict[str, float | bool]:
    """SGD interpretable only if its loss drop reaches ``factor`` of AdamW's."""
    def drop(records: list[dict]) -> float:
        losses = [rec["loss"] for rec in records]
        w = min(window, len(losses) // 2)
        return sum(losses[:w]) / w - sum(losses[-w:]) / w

    delta_sgd = drop(records_by_arm["sgd"]) if "sgd" in records_by_arm else float("nan")
    delta_adamw = (
        drop(records_by_arm["adamw"]) if "adamw" in records_by_arm else float("nan")
    )
    finite = delta_sgd == delta_sgd and delta_adamw == delta_adamw
    gate_pass = bool(finite and delta_adamw > 0 and delta_sgd >= factor * delta_adamw)
    return {
        "delta_sgd": delta_sgd,
        "delta_adamw": delta_adamw,
        "factor": factor,
        "pass": gate_pass,
    }


def main() -> None:
    jlens.configure_logging()
    logging.getLogger("exp").setLevel(logging.INFO)
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    hf, tokenizer = load_model(args.model, args.device)
    examples = materialize_examples(
        tokenizer, args.condition, args.steps, args.max_seq_len
    )
    arm_kwargs = dict(
        r=args.r, alpha=args.alpha, seed=args.seed, clip=args.clip,
        device=args.device,
    )

    lrs = {"adamw": args.adamw_lr}
    if "sgd" in args.optimizers:
        sgd_lr, curves = lr_sweep(
            hf, examples, args.sweep_lrs, args.sweep_steps, **arm_kwargs
        )
        lrs["sgd"] = sgd_lr
        with open(os.path.join(args.out, "sweep_sgd.json"), "w") as f:
            json.dump({"curves": curves, "chosen_lr": sgd_lr}, f, indent=1)
        logger.info("sweep chose SGD lr=%g", sgd_lr)

    records_by_arm: dict[str, list[dict]] = {}
    condition_slug = args.condition.split("-")[0]
    for name in args.optimizers:
        arm_dir = os.path.join(args.out, f"{condition_slug}-{name}")
        if os.path.exists(os.path.join(arm_dir, "adapter_model.safetensors")):
            logger.info("skipping %s: adapter exists", arm_dir)
            log_path = os.path.join(arm_dir, "train_log.json")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    records_by_arm[name] = json.load(f)["records"]
            continue
        logger.info("=== arm %s (lr=%g) -> %s ===", name, lrs[name], arm_dir)
        records = train_arm(
            hf, examples, optimizer_name=name, lr=lrs[name], steps=args.steps,
            checkpoint_every=args.checkpoint_every, out_dir=arm_dir, **arm_kwargs,
        )
        records_by_arm[name] = records
        with open(os.path.join(arm_dir, "train_log.json"), "w") as f:
            json.dump({
                "meta": {"optimizer": name, "lr": lrs[name], "steps": args.steps,
                         "seed": args.seed, "clip": args.clip,
                         "condition": args.condition, "model": args.model,
                         "argv": json.dumps(vars(args), default=str)},
                "records": records,
            }, f, indent=1)

    if {"sgd", "adamw"} <= set(records_by_arm):
        gate = compute_gate(records_by_arm, factor=args.gate_factor)
        with open(os.path.join(args.out, "gate.json"), "w") as f:
            json.dump(gate, f, indent=1)
        level = logging.INFO if gate["pass"] else logging.WARNING
        logger.log(level, "GATE %s: dL_sgd=%.4f dL_adamw=%.4f (factor %.2f)",
                   "PASS" if gate["pass"] else "FAIL — SGD span numbers are "
                   "NOT interpretable; retune the LR",
                   gate["delta_sgd"], gate["delta_adamw"], gate["factor"])


if __name__ == "__main__":
    main()
