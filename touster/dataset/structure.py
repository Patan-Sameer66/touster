from __future__ import annotations

import json

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from touster.console import console
from touster.dataset.schema import Dataset, from_list


_CHUNK_CHARS = 3000  # approximate chars per chunk sent to LLM

_SYSTEM_PROMPT = (
    "You are a fine-tuning data extractor. "
    "Given raw text, extract question-and-answer pairs suitable for LLM fine-tuning. "
    "Return ONLY a valid JSON array. No markdown, no explanation outside the JSON."
)

_USER_TEMPLATE = (
    "Extract up to {n} question-and-answer pairs from the following text. "
    "Output a JSON array where each element has a 'messages' key with a list of "
    "{{'role', 'content'}} objects. Use role 'user' for questions and 'assistant' for answers.\n\n"
    "TEXT:\n{chunk}"
)


def _parse_llm_json(text: str) -> list[dict]:
    """Extract and parse a JSON array from LLM text, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()
    return json.loads(text)


def _chunk_text(text: str, chunk_chars: int = _CHUNK_CHARS) -> list[str]:
    """Split text into roughly equal character-length chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > chunk_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text[:chunk_chars]]


def structure_dataset(
    client,
    raw_text: str,
    num_samples: int = 50,
    model: str = "",
) -> Dataset:
    """
    Send chunks of raw_text to the LLM, asking it to extract Q&A pairs as ChatML JSON.
    Returns Dataset.
    """
    chunks = _chunk_text(raw_text)
    per_chunk = max(1, num_samples // max(len(chunks), 1))
    all_samples: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[touster.brand]Structuring dataset[/touster.brand]"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Processing chunks...", total=len(chunks))

        for chunk in chunks:
            if len(all_samples) >= num_samples:
                break
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(n=per_chunk, chunk=chunk),
                },
            ]
            try:
                reply = client.chat(messages, model=model, temperature=0.3, max_tokens=4096)
                batch = _parse_llm_json(reply)
                all_samples.extend(batch)
            except (json.JSONDecodeError, ValueError, RuntimeError):
                # Skip chunk on failure rather than crashing the whole run
                pass
            progress.update(task, advance=1)

    return from_list(all_samples[:num_samples])
