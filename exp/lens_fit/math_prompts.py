"""On-distribution prompt loader for the lens-refit follow-up.

The original lens fit averages ``J_l`` over WikiText — a general corpus, not
the fine-tuning distribution. The decompose null (LoRA-TMLR adapter writes at
random-B floor mid-band) is consistent with either "the workspace theory is
wrong" or "the WikiText-fitted lens is pointed at the wrong subspace to see
it." Refitting on MetaMathQA text (formatted exactly as ``exp.delta``'s
``math-ift`` condition does) isolates which: same ``jlens.fit`` machinery,
different corpus.
"""

from __future__ import annotations

from exp.delta.data import ALPACA_TEMPLATE, CONDITIONS


def load_metamath_prompts(n_prompts: int, *, min_chars: int = 200) -> list[str]:
    """Return the first ``n_prompts`` MetaMathQA records (Alpaca-formatted
    instruction + response, matching ``exp.delta``'s ``math-ift`` condition)
    of at least ``min_chars`` characters, streamed from the HuggingFace Hub."""
    if n_prompts <= 0:
        return []
    from datasets import load_dataset

    condition = CONDITIONS["math-ift"]
    instruction_field, response_field = condition.fields
    dataset = load_dataset(condition.dataset, split=condition.split, streaming=True)
    prompts: list[str] = []
    for record in dataset:
        text = (
            ALPACA_TEMPLATE.format(instruction=record[instruction_field])
            + record[response_field]
        )
        if len(text.strip()) >= min_chars:
            prompts.append(text)
            if len(prompts) == n_prompts:
                break
    return prompts
