"""Loading LoRA adapters and extracting their gauge-invariant write directions.

A PEFT adapter stores, per (layer, module), factors ``A [r, d_in]`` and
``B [d_out, r]`` with ``ΔW = (α/r)·B·A``. Raw B columns are gauge-dependent
(``ΔW = (B R)(R⁻¹ A)`` for any invertible R gives the identical adapter), so
the well-defined write directions are the **left singular vectors of ΔW**,
with the singular values saying how much each direction is actually used.
:func:`svd_of_lowrank` computes that SVD exactly without materializing ΔW.

Frame note (the silent-failure trap): a B column lives in the *output space of
its module*. Only ``o_proj`` and ``down_proj`` outputs are added to the
residual stream — the frame the J-lens reads. ``q/k/v_proj`` outputs are
head-space (4096-dim but a *different* basis: dimension match ≠ frame match)
and ``gate/up_proj`` are 11008-dim MLP-internal. Alignment analysis therefore
uses only :data:`RESIDUAL_WRITE_MODULES`; the others are parsed anyway so the
per-module energy coverage can be reported honestly.

Verified layout of the LoRA-TMLR-2024 files (read from the real safetensors
header): keys ``base_model.model.model.layers.{i}.{self_attn|mlp}.{mod}.lora_{A|B}.weight``,
BF16, r=16, α=32.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import torch

#: Modules whose output is added directly to the residual stream.
RESIDUAL_WRITE_MODULES = ("o_proj", "down_proj")

#: All adapted modules in the LoRA-TMLR-2024 runs.
ALL_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

_KEY_RE = re.compile(
    r"layers\.(?P<layer>\d+)\.(?:self_attn|mlp)\.(?P<module>\w+_proj)\.lora_(?P<factor>[AB])\.weight$"
)


def svd_of_lowrank(
    B: torch.Tensor, A: torch.Tensor, *, scale: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact SVD of ``scale·B·A`` without forming the [d_out, d_in] product.

    QR-decompose both thin factors, SVD the small r×r core:
    ``B A = Q_b (R_b R_aᵀ) Q_aᵀ``. Gauge-invariant by construction.

    Args:
        B: ``[d_out, r]`` write factor.
        A: ``[r, d_in]`` read factor.
        scale: LoRA scaling ``α/r``.

    Returns:
        ``(U, S, V)`` with ``U [d_out, r]``, ``S [r]`` descending,
        ``V [d_in, r]``, satisfying ``scale·B·A == U @ diag(S) @ V.T``.
    """
    B32, A32 = B.float(), A.float()
    Qb, Rb = torch.linalg.qr(B32)  # [d_out, r], [r, r]
    Qa, Ra = torch.linalg.qr(A32.T)  # [d_in, r], [r, r]
    core = scale * (Rb @ Ra.T)  # [r, r]
    Uc, S, Vhc = torch.linalg.svd(core)
    return Qb @ Uc, S, Qa @ Vhc.T


@dataclass
class ModuleWrites:
    """Write-direction SVD for one (layer, module).

    Attributes:
        U: ``[d_out, r]`` orthonormal write directions (left singular vectors).
        S: ``[r]`` singular values, descending. ``S²`` is the energy weight.
        V: ``[d_in, r]`` read directions (kept for completeness/debugging).
    """

    U: torch.Tensor
    S: torch.Tensor
    V: torch.Tensor

    @property
    def energy(self) -> float:
        """Total ΔW energy ``‖ΔW‖_F² = Σ S²`` for this module."""
        return float((self.S**2).sum())


@dataclass
class AdapterWrites:
    """All write-direction SVDs for one adapter, plus config metadata.

    Attributes:
        name: Adapter repo id or local path.
        r: LoRA rank.
        scale: ``lora_alpha / r``.
        writes: ``{(layer, module): ModuleWrites}`` over every adapted module.
        n_layers: Number of distinct layers seen.
    """

    name: str
    r: int
    scale: float
    writes: dict[tuple[int, str], ModuleWrites] = field(default_factory=dict)
    n_layers: int = 0

    def layers(self) -> list[int]:
        return sorted({layer for layer, _ in self.writes})

    def coverage(self) -> dict[str, float]:
        """Fraction of total ΔW energy per module type (sums to 1).

        The honesty number: alignment is only measured on
        :data:`RESIDUAL_WRITE_MODULES`, and this says how much of the adapter
        that primary claim covers.
        """
        per_module = dict.fromkeys(ALL_MODULES, 0.0)
        for (_, module), mw in self.writes.items():
            per_module[module] += mw.energy
        total = sum(per_module.values())
        return {m: e / total for m, e in per_module.items() if e > 0}

    def residual_coverage(self) -> float:
        cov = self.coverage()
        return sum(cov.get(m, 0.0) for m in RESIDUAL_WRITE_MODULES)


def load_adapter_writes(
    name_or_path: str, *, modules: tuple[str, ...] = ALL_MODULES
) -> AdapterWrites:
    """Download (or open locally) a PEFT adapter and SVD every module's ΔW.

    Args:
        name_or_path: HF repo id (e.g. ``LoRA-TMLR-2024/metamath-lora-rank-16-alpha-32``)
            or a local directory containing ``adapter_config.json`` +
            ``adapter_model.safetensors``.
        modules: Module types to keep.

    Returns:
        :class:`AdapterWrites` with fp32 SVD factors on CPU.
    """
    import os

    from safetensors.torch import load_file

    if os.path.isdir(name_or_path):
        config_path = os.path.join(name_or_path, "adapter_config.json")
        weights_path = os.path.join(name_or_path, "adapter_model.safetensors")
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(name_or_path, "adapter_config.json")
        weights_path = hf_hub_download(name_or_path, "adapter_model.safetensors")

    with open(config_path) as f:
        config = json.load(f)
    r, alpha = int(config["r"]), float(config["lora_alpha"])
    scale = alpha / r

    tensors = load_file(weights_path)
    factors: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        match = _KEY_RE.search(key)
        if match is None:
            continue
        layer, module, factor = (
            int(match["layer"]),
            match["module"],
            match["factor"],
        )
        if module not in modules:
            continue
        factors.setdefault((layer, module), {})[factor] = tensor

    result = AdapterWrites(name=name_or_path, r=r, scale=scale)
    for (layer, module), pair in sorted(factors.items()):
        if set(pair) != {"A", "B"}:
            raise ValueError(f"incomplete A/B pair at layer {layer} {module}")
        U, S, V = svd_of_lowrank(pair["B"], pair["A"], scale=scale)
        result.writes[(layer, module)] = ModuleWrites(U=U, S=S, V=V)
    result.n_layers = len(result.layers())
    if not result.writes:
        raise ValueError(f"no LoRA factors matched in {weights_path}")
    return result
