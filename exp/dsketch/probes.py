"""Fixed cotangent probes for the δ-sketch.

A probe is a unit vector ``v`` in final-residual coordinates, shared across
ALL examples and conditions. Pulling ``v`` back through example ``n`` yields
``w⁽ⁿ⁾ = Φ⁽ⁿ⁾ᵀ v`` per source position — with ``v`` fixed, the across-example
scatter of ``w`` measures propagator heterogeneity (the theory's Assumption-A
``δ``) with the error-vector (ε) diversity of the δ-screen excluded by
construction.

Two probe kinds:

- ``gauss``: seeded isotropic unit vectors — unbiased directions.
- ``unembed``: ``normalize(γ ⊙ W_U[token_id])`` for a fixed short token list —
  the *linear part* of the readout cotangent for that token (the norm's 1/rms
  scalar is dropped, as in ``exp/decompose/dictionary.py``). These are
  "on-theory": real training cotangents live in the row space of ``W_U``.

Falls back to Gaussian-only when the model does not expose ``_lm_head`` /
``_final_norm`` (e.g. the tiny test decoder).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from jlens.fitting import valid_position_mask

logger = logging.getLogger(__name__)

#: Fixed default token list for unembed-flavored probes: a function word, an
#: IFT-ish content word, a code keyword, and an operator.
DEFAULT_PROBE_TOKENS: tuple[str, ...] = (" the", " answer", " def", " =")


@dataclass(frozen=True)
class ProbeSet:
    """A fixed, self-describing set of cotangent probes."""

    vectors: torch.Tensor  # [P, d] fp32, CPU, unit rows
    kinds: tuple[str, ...]  # "gauss" | "unembed" per probe
    tokens: tuple[str | None, ...]  # None for gauss probes
    token_ids: tuple[int | None, ...]
    seed: int

    def __len__(self) -> int:
        return self.vectors.shape[0]

    def state_dict(self) -> dict[str, Any]:
        return {
            "vectors": self.vectors.clone(),
            "kinds": list(self.kinds),
            "tokens": list(self.tokens),
            "token_ids": list(self.token_ids),
            "seed": self.seed,
        }


def _token_id(tokenizer: Any, text: str) -> int | None:
    """Last token id of ``text``; ``add_special_tokens=False`` when supported.

    Taking the last id handles SentencePiece leading-space pieces; the plain
    ``tokenizer(text)`` fallback (used by the tiny test tokenizer) may include
    a BOS at position 0, which taking the last id also sidesteps.
    """
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except (AttributeError, TypeError):
        try:
            ids = tokenizer(text).input_ids
            if hasattr(ids, "tolist"):
                ids = ids[0].tolist() if ids.dim() > 1 else ids.tolist()
        except Exception:  # noqa: BLE001 - probe construction must not abort
            return None
    ids = list(ids)
    return int(ids[-1]) if ids else None


def build_probes(
    model: Any,
    *,
    n_gauss: int = 4,
    tokens: Sequence[str] = DEFAULT_PROBE_TOKENS,
    seed: int = 0,
) -> ProbeSet:
    """Construct the fixed probe set: ``n_gauss`` Gaussian + one unembed-
    flavored probe per token (skipped, with a warning, when unavailable)."""
    d_model = model.d_model
    generator = torch.Generator().manual_seed(seed)
    gauss = torch.randn(n_gauss, d_model, generator=generator)
    gauss = gauss / gauss.norm(dim=1, keepdim=True)

    vectors = [gauss]
    kinds: list[str] = ["gauss"] * n_gauss
    token_names: list[str | None] = [None] * n_gauss
    token_ids: list[int | None] = [None] * n_gauss

    lm_head = getattr(model, "_lm_head", None)
    final_norm = getattr(model, "_final_norm", None)
    if lm_head is None or final_norm is None or not hasattr(final_norm, "weight"):
        if tokens:
            logger.warning(
                "model lacks _lm_head/_final_norm: unembed probes skipped, "
                "using %d gaussian probes only", n_gauss,
            )
        tokens = ()
    else:
        unembed_weight = lm_head.weight.detach().float().cpu()  # [vocab, d]
        gamma = final_norm.weight.detach().float().cpu()  # [d]
        for text in tokens:
            token_id = _token_id(model.tokenizer, text)
            if token_id is None or not 0 <= token_id < unembed_weight.shape[0]:
                logger.warning("probe token %r: id lookup failed, skipping", text)
                continue
            direction = gamma * unembed_weight[token_id]
            norm = float(direction.norm())
            if norm <= 0:
                logger.warning("probe token %r: zero direction, skipping", text)
                continue
            vectors.append((direction / norm).unsqueeze(0))
            kinds.append("unembed")
            token_names.append(text)
            token_ids.append(token_id)

    stacked = torch.cat(vectors, dim=0).float()
    assert torch.allclose(
        stacked.norm(dim=1), torch.ones(stacked.shape[0]), atol=1e-5
    )
    return ProbeSet(
        vectors=stacked,
        kinds=tuple(kinds),
        tokens=tuple(token_names),
        token_ids=tuple(token_ids),
        seed=seed,
    )


def probe_loss_fns(
    vectors: torch.Tensor, *, skip_first: int
) -> list[Callable[[torch.Tensor], torch.Tensor]]:
    """One scalar loss per probe row ``v``: ``Σ_{p valid} ⟨v, h_final[p]⟩``.

    The valid-position mask is recomputed from ``h_final.shape[1]`` inside
    each call, so one list of closures serves examples of any length. The
    example's ``loss_positions`` are deliberately unused (ε-free cotangent).
    """

    def make(v: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        def loss_fn(h_final: torch.Tensor) -> torch.Tensor:
            mask = valid_position_mask(h_final.shape[1], skip_first=skip_first)
            positions = mask.nonzero(as_tuple=True)[0].to(h_final.device)
            return (h_final[0, positions, :].float() @ v.to(h_final.device)).sum()

        return loss_fn

    return [make(vectors[j]) for j in range(vectors.shape[0])]
