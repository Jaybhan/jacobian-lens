"""Hook-based LoRA for the gradient-confinement test.

Attaches rank-``r`` adapters to target ``nn.Linear`` modules via forward
hooks — no module surgery, no ``peft`` dependency. The hook computes the
delta in fp32 from the (possibly bf16) input and adds it back in the stream
dtype, so ``A``/``B`` are fp32 master weights updated in fp32 while the base
model runs in bf16. A hook returning a tensor replaces the module output for
all downstream consumers and is autograd-transparent, so ``loss.backward()``
reaches ``A`` and ``B`` like any leaf (the same mechanism as the perturbation
hook in exp/tests/test_pullbacks_tiny.py).

Init matches standard LoRA: ``A`` kaiming-uniform, ``B = 0`` — the theory's
assumption. With ``B = 0``, ``A`` receives exactly zero gradient on the first
step (``dL/dA = scale·Bᵀ v xᵀ``).

Adapters save into the PEFT file layout that
:func:`exp.decompose.adapters.load_adapter_writes` parses from a local
directory: ``adapter_config.json`` + ``adapter_model.safetensors`` with keys
``base_model.model.model.layers.{i}.{self_attn|mlp}.{mod}.lora_{A|B}.weight``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

#: Parent attribute holding each target module in a Llama decoder layer.
LLAMA_PARENT = {"o_proj": "self_attn", "down_proj": "mlp"}

_PEFT_KEY = "base_model.model.model.layers.{layer}.{parent}.{module}.lora_{factor}.weight"


@dataclass
class LoRASite:
    """One attached adapter: fp32 ``A [r, d_in]`` / ``B [d_out, r]`` + its hook."""

    layer: int
    module: str
    A: nn.Parameter
    B: nn.Parameter
    handle: Any
    #: Post-LoRA module output of the most recent forward (only populated
    #: when attached with ``record_output=True``; used by the containment test).
    recorded_output: torch.Tensor | None = field(default=None, repr=False)


def _find_target(block: nn.Module, module: str) -> nn.Linear:
    """Locate ``module`` on a decoder block: as a direct attribute (tiny test
    models) or under its Llama parent (``self_attn``/``mlp``)."""
    if hasattr(block, module):
        return getattr(block, module)
    parent_name = LLAMA_PARENT.get(module)
    if parent_name is not None and hasattr(block, parent_name):
        return getattr(getattr(block, parent_name), module)
    raise AttributeError(f"block {type(block).__name__} has no module {module!r}")


def attach_lora(
    decoder_layers: Any,
    *,
    r: int,
    alpha: int,
    modules: tuple[str, ...] = ("o_proj", "down_proj"),
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    record_output: bool = False,
) -> list[LoRASite]:
    """Attach LoRA to every (layer, module) target; returns the sites.

    ``A`` is initialized from a CPU generator seeded with ``seed`` so two
    arms attached with the same seed start bit-identical. ``B`` starts at
    exactly zero.
    """
    generator = torch.Generator().manual_seed(seed)
    scale = alpha / r
    sites: list[LoRASite] = []
    for layer_index, block in enumerate(decoder_layers):
        for module in modules:
            target = _find_target(block, module)
            d_out, d_in = target.weight.shape
            # kaiming_uniform_(a=sqrt(5)) on [r, d_in], as in standard LoRA.
            bound = 1.0 / math.sqrt(d_in)
            A = nn.Parameter(
                (torch.rand(r, d_in, generator=generator) * 2 - 1).mul_(bound)
                .to(device=device, dtype=dtype)
            )
            B = nn.Parameter(torch.zeros(d_out, r, device=device, dtype=dtype))
            site = LoRASite(layer=layer_index, module=module, A=A, B=B, handle=None)

            def hook(
                mod: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                site: LoRASite = site,
                scale: float = scale,
            ) -> torch.Tensor:
                x = inputs[0].to(site.A.dtype)
                delta = nn.functional.linear(nn.functional.linear(x, site.A), site.B)
                out = output + (scale * delta).to(output.dtype)
                if record_output:
                    out.retain_grad()
                    site.recorded_output = out
                return out

            site.handle = target.register_forward_hook(hook)
            sites.append(site)
    return sites


def detach_lora(sites: list[LoRASite]) -> None:
    for site in sites:
        site.handle.remove()


def lora_parameters(sites: list[LoRASite]) -> list[nn.Parameter]:
    return [p for site in sites for p in (site.A, site.B)]


def adapter_state_dict(
    sites: list[LoRASite], *, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """PEFT-layout tensors (Llama key format), contiguous on CPU."""
    state: dict[str, torch.Tensor] = {}
    for site in sites:
        parent = LLAMA_PARENT.get(site.module, "self_attn")
        for factor, tensor in (("A", site.A), ("B", site.B)):
            key = _PEFT_KEY.format(
                layer=site.layer, parent=parent, module=site.module, factor=factor
            )
            state[key] = tensor.detach().to("cpu", dtype).contiguous()
    return state


def save_adapter(sites: list[LoRASite], out_dir: str, *, r: int, alpha: int) -> None:
    """Write ``adapter_config.json`` + ``adapter_model.safetensors`` (fp32)."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump({"r": r, "lora_alpha": alpha}, f)
    save_file(adapter_state_dict(sites), os.path.join(out_dir, "adapter_model.safetensors"))
