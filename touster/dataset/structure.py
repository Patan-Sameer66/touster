from __future__ import annotations

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from touster.console import console
from touster.dataset.schema import Dataset, from_list
# Reuse the hardened mode-0 parser so mode-1 gets the same repairs:
# trailing-comma stripping, fence removal, invalid-escape fix, flat-wrap, salvage.
from touster.dataset.generate import _parse_llm_json


_CHUNK_CHARS = 3000  # approximate chars per chunk sent to LLM

_SYSTEM_PROMPT = (
    "You are a fine-tuning data extractor. "
    "Given raw text, extract question-and-answer pairs suitable for LLM fine-tuning. "
    "Return ONLY a valid JSON array. No markdown, no explanation outside the JSON."
)

_USER_TEMPLATE = (
    "Extract up to {n} question-and-answer pairs from the following text.\n\n"
    "Output a JSON array where each element has a 'messages' key with a list of message objects. "
    "Each message object must have 'role' (string) and 'content' (string) fields. "
    'Use role "user" for questions and "assistant" for answers.\n\n'
    "Example format:\n"
    '[\n'
    '  {{"messages": [{{"role": "user", "content": "What is X?"}}, '
    '{{"role": "assistant", "content": "X is ..."}}]}}\n'
    ']\n\n'
    "TEXT:\n{chunk}"
)


def _chunk_text(text: str, chunk_chars: int = _CHUNK_CHARS) -> list[str]:
    """Split text into roughly equal character-length chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        # Force-split paragraphs that individually exceed chunk_chars
        if len(para) > chunk_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            for i in range(0, len(para), chunk_chars):
                chunks.append(para[i:i + chunk_chars])
            continue

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
    if not raw_text or not raw_text.strip():
        raise ValueError("structure_dataset: raw_text is empty.")

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

        for chunk_idx, chunk in enumerate(chunks, start=1):
            progress.update(task, advance=1)
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
                # Scale token budget with how many pairs we ask for; constrained
                # JSON output (format="json") prevents most malformed responses.
                max_tokens = max(2048, per_chunk * 300)
                reply = client.chat(
                    messages, model=model, temperature=0.3,
                    max_tokens=max_tokens, format="json",
                )
                batch = _parse_llm_json(reply)
                all_samples.extend(batch)
            except Exception as exc:
                console.print(
                    f"[touster.warning]Skipping chunk {chunk_idx}/{len(chunks)}"
                    f" ({type(exc).__name__}: {exc})[/touster.warning]"
                )

    if not all_samples:
        raise RuntimeError(
            "structure_dataset: all chunks failed to produce samples. "
            "Check LLM configuration and input text quality."
        )

    try:
        return from_list(all_samples[:num_samples])
    except ValueError as exc:
        raise RuntimeError(
            f"structure_dataset: LLM output did not match expected schema. "
            f"Got {len(all_samples)} raw items. Schema error: {exc}"
        ) from exc
