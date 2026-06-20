from __future__ import annotations

import math

from touster.dataset.schema import Dataset, Sample


# ---------------------------------------------------------------------------
# Chat template application
# ---------------------------------------------------------------------------

def _apply_chatml(sample: Sample) -> str:
    """Apply ChatML template: <|im_start|>role\ncontent<|im_end|>\n"""
    parts: list[str] = []
    for msg in sample.messages:
        parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n")
    return "".join(parts)


def _apply_llama3(sample: Sample) -> str:
    """Apply Llama 3 template."""
    parts: list[str] = ["<|begin_of_text|>"]
    for msg in sample.messages:
        parts.append(
            f"<|start_header_id|>{msg.role}<|end_header_id|>\n{msg.content}<|eot_id|>"
        )
    return "".join(parts)


def _apply_alpaca(sample: Sample) -> str:
    """
    Apply Alpaca template (single-turn: ### Instruction: / ### Response:).
    Uses the first user message as Instruction and first assistant message as Response.
    """
    instruction = ""
    response = ""
    for msg in sample.messages:
        if msg.role == "system":
            instruction = msg.content + "\n\n" + instruction
        elif msg.role == "user" and not instruction:
            instruction = msg.content
        elif msg.role == "assistant" and not response:
            response = msg.content

    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


_TEMPLATE_FNS = {
    "chatml": _apply_chatml,
    "llama3": _apply_llama3,
    "alpaca": _apply_alpaca,
}


def apply_chat_template(sample: Sample, template: str = "chatml") -> str:
    """
    Apply a named chat template to a sample.
    Supported: 'chatml', 'llama3', 'alpaca'.
    Raises ValueError on unknown template name.
    """
    fn = _TEMPLATE_FNS.get(template)
    if fn is None:
        raise ValueError(
            f"Unknown template {template!r}. Supported: {sorted(_TEMPLATE_FNS)}"
        )
    return fn(sample)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_token_count(text: str, chars_per_token: float = 3.8) -> int:
    """Quick approximation: int(len(text) / chars_per_token)."""
    return int(len(text) / chars_per_token)


def count_tokens_dataset(ds: Dataset, template: str = "chatml") -> dict:
    """
    Return token count statistics across all samples in the dataset.
    Keys: total, mean, max, min, p95.
    """
    if not ds.samples:
        return {"total": 0, "mean": 0.0, "max": 0, "min": 0, "p95": 0}

    counts = [
        estimate_token_count(apply_chat_template(s, template))
        for s in ds.samples
    ]
    total = sum(counts)
    mean = total / len(counts)
    maximum = max(counts)
    minimum = min(counts)

    # p95: index at 95th percentile
    sorted_counts = sorted(counts)
    p95_idx = min(int(math.ceil(0.95 * len(sorted_counts))) - 1, len(sorted_counts) - 1)
    p95 = sorted_counts[max(p95_idx, 0)]

    return {
        "total": total,
        "mean": mean,
        "max": maximum,
        "min": minimum,
        "p95": p95,
    }
