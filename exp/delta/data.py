"""Dataset conditions for the δ experiment: loading, formatting, loss masks.

Each *condition* pairs a HuggingFace dataset with a loss regime:

- ``ift``  — instruction data, cross-entropy scored on **response tokens only**
  (Alpaca-style template; an approximation of the LoRA-TMLR-2024 training
  format, documented as such).
- ``cpt``  — raw text, cross-entropy scored on **all valid tokens** (continued
  pretraining regime).

The ``*-alltok`` variants rescore the IFT datasets with the CPT regime — the
control that separates instruction *content* from loss-masking *regime*.

Position conventions match ``jlens.fitting``: loss targets and pullback source
positions are both restricted to :func:`jlens.fitting.valid_position_mask`
(skip the first ``skip_first`` attention-sink positions, exclude the final
position, which has no next-token target).

Sampling is "first N that pass the length filter" from a streamed dataset:
deterministic and cheap, but not a uniform sample of the corpus — fine for the
screen, noted here for honesty.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from jlens.fitting import SKIP_FIRST_N_POSITIONS, valid_position_mask

logger = logging.getLogger(__name__)

#: Alpaca-style prompt template (approximates the LoRA-TMLR-2024 IFT format).
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

#: Minimum tokenized length for an example to be used (length matching across
#: conditions), and minimum number of scored response targets for IFT examples.
MIN_TOKENS = 96
MIN_RESPONSE_TARGETS = 8


@dataclass(frozen=True)
class Condition:
    """One dataset x loss-regime cell.

    Attributes:
        dataset: HF dataset repo id.
        kind: ``"ift"`` (response-masked loss) or ``"cpt"`` (all-token loss).
        fields: ``(instruction_field, response_field)`` for ift;
            ``(text_field,)`` for cpt.
        config: HF dataset config name (``None`` for default).
        data_dir: HF dataset data_dir (e.g. ``"python"`` for starcoderdata).
        split: Dataset split to stream.
    """

    dataset: str
    kind: str
    fields: tuple[str, ...]
    config: str | None = None
    data_dir: str | None = None
    split: str = "train"


CONDITIONS: dict[str, Condition] = {
    "code-ift": Condition(
        "ise-uiuc/Magicoder-Evol-Instruct-110K", "ift", ("instruction", "response")
    ),
    "math-ift": Condition("meta-math/MetaMathQA", "ift", ("query", "response")),
    "code-cpt": Condition(
        "bigcode/starcoderdata", "cpt", ("content",), data_dir="python"
    ),
    "math-cpt": Condition("open-web-math/open-web-math", "cpt", ("text",)),
    "wikitext": Condition(
        "Salesforce/wikitext", "cpt", ("text",), config="wikitext-103-raw-v1"
    ),
    # Regime controls: IFT data, CPT (all-token) loss.
    "code-ift-alltok": Condition(
        "ise-uiuc/Magicoder-Evol-Instruct-110K", "cpt-on-ift", ("instruction", "response")
    ),
    "math-ift-alltok": Condition(
        "meta-math/MetaMathQA", "cpt-on-ift", ("query", "response")
    ),
}


@dataclass
class PreparedExample:
    """A tokenized example ready for pullback extraction.

    Attributes:
        input_ids: ``[1, seq_len]`` token ids.
        loss_positions: Boolean ``[seq_len]``; position ``p`` is True when the
            next-token prediction at ``p`` (target token ``p+1``) contributes
            to the loss. Always a subset of ``valid_position_mask``.
        meta: Bookkeeping (sequence length, target count, prompt length, ...).
    """

    input_ids: torch.Tensor
    loss_positions: torch.Tensor
    meta: dict[str, Any]


def _encode(tokenizer: Any, text: str, max_seq_len: int) -> torch.Tensor:
    return tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_seq_len
    ).input_ids


def prepare_cpt(
    tokenizer: Any,
    text: str,
    *,
    max_seq_len: int = 256,
    min_tokens: int = MIN_TOKENS,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> PreparedExample | None:
    """All-token-loss example, or ``None`` if it fails the length filter."""
    input_ids = _encode(tokenizer, text, max_seq_len)
    seq_len = input_ids.shape[1]
    if seq_len < min_tokens:
        return None
    loss_positions = valid_position_mask(seq_len, skip_first=skip_first)
    return PreparedExample(
        input_ids=input_ids,
        loss_positions=loss_positions,
        meta={"seq_len": seq_len, "n_targets": int(loss_positions.sum())},
    )


def prepare_ift(
    tokenizer: Any,
    instruction: str,
    response: str,
    *,
    max_seq_len: int = 256,
    min_tokens: int = MIN_TOKENS,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    mask_to_response: bool = True,
) -> PreparedExample | None:
    """Templated instruction example, loss on response tokens only.

    ``mask_to_response=False`` gives the ``*-alltok`` regime control: identical
    text and template, all-token loss.

    Returns ``None`` when the example fails the length filter, the response is
    entirely truncated away, or fewer than :data:`MIN_RESPONSE_TARGETS` scored
    targets survive.
    """
    prompt = ALPACA_TEMPLATE.format(instruction=instruction)
    input_ids = _encode(tokenizer, prompt + response, max_seq_len)
    seq_len = input_ids.shape[1]
    if seq_len < min_tokens:
        return None
    # Position p scores target token p+1; response targets are token indices
    # >= prompt_len, i.e. positions p >= prompt_len - 1.
    prompt_len = _encode(tokenizer, prompt, max_seq_len).shape[1]
    valid = valid_position_mask(seq_len, skip_first=skip_first)
    if mask_to_response:
        loss_positions = valid.clone()
        loss_positions[: prompt_len - 1] = False
        if int(loss_positions.sum()) < MIN_RESPONSE_TARGETS:
            return None
    else:
        loss_positions = valid
    return PreparedExample(
        input_ids=input_ids,
        loss_positions=loss_positions,
        meta={
            "seq_len": seq_len,
            "n_targets": int(loss_positions.sum()),
            "prompt_len": prompt_len,
        },
    )


def _record_to_example(
    condition: Condition, record: dict, tokenizer: Any, **prepare_kwargs: Any
) -> PreparedExample | None:
    if condition.kind == "ift":
        return prepare_ift(
            tokenizer,
            record[condition.fields[0]],
            record[condition.fields[1]],
            mask_to_response=True,
            **prepare_kwargs,
        )
    if condition.kind == "cpt-on-ift":
        return prepare_ift(
            tokenizer,
            record[condition.fields[0]],
            record[condition.fields[1]],
            mask_to_response=False,
            **prepare_kwargs,
        )
    text = record[condition.fields[0]]
    if not text or len(text.strip()) < 200:  # cheap pre-filter before tokenizing
        return None
    return prepare_cpt(tokenizer, text, **prepare_kwargs)


def iter_condition(
    name: str,
    tokenizer: Any,
    *,
    n_examples: int,
    max_seq_len: int = 256,
    min_tokens: int = MIN_TOKENS,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> Iterator[PreparedExample]:
    """Stream ``n_examples`` prepared examples for a named condition.

    Requires the ``datasets`` package; streams from the Hub (no full download).
    """
    from datasets import load_dataset

    condition = CONDITIONS[name]
    dataset = load_dataset(
        condition.dataset,
        condition.config,
        data_dir=condition.data_dir,
        split=condition.split,
        streaming=True,
    )
    n_yielded = 0
    n_seen = 0
    for record in dataset:
        n_seen += 1
        example = _record_to_example(
            condition,
            record,
            tokenizer,
            max_seq_len=max_seq_len,
            min_tokens=min_tokens,
            skip_first=skip_first,
        )
        if example is None:
            continue
        example.meta["condition"] = name
        yield example
        n_yielded += 1
        if n_yielded >= n_examples:
            break
    logger.info(
        "condition %s: yielded %d examples (scanned %d records)",
        name,
        n_yielded,
        n_seen,
    )
    if n_yielded < n_examples:
        logger.warning(
            "condition %s: only %d/%d examples passed the filters",
            name,
            n_yielded,
            n_examples,
        )
