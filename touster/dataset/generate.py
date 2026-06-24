from __future__ import annotations

import json
import re

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from touster.console import console
from touster.dataset.schema import Dataset, from_list


_SYSTEM_PROMPT = (
    "You are a fine-tuning dataset generator. "
    "Output ONLY a valid JSON array — nothing else. "
    "No markdown fences, no comments, no trailing commas, no text before or after the JSON."
)

_USER_TEMPLATE = """\
Generate {batch_size} question-and-answer training examples about: {topic}

Return a JSON array. Each element MUST follow this EXACT structure:
[
  {{"messages": [{{"role": "user", "content": "question"}}, {{"role": "assistant", "content": "answer"}}]}}
]

STRICT RULES — violating any rule will break parsing:
1. Start your response with [ and end with ] — nothing outside
2. No trailing commas after the last item in any array or object
3. No markdown code fences (no ```)
4. No comments (// or /* */)
5. All strings use double quotes, never single quotes
6. Escape double quotes inside strings as \\"

Output {batch_size} elements in the array. Only output the JSON.\
"""

_RETRY_ADDENDUM = """\
Your previous response was not valid JSON. The most common causes:
- TRAILING COMMAS: `"content": "answer",` followed by `}}` — remove the comma
- Markdown fences: remove ``` lines entirely
- Text before/after the array: output ONLY [ ... ]

Correct minimal format (copy exactly):
[
  {{"messages": [{{"role": "user", "content": "What is X?"}}, {{"role": "assistant", "content": "X is ..."}}]}}
]

Try again. Output ONLY the JSON array, no other text.\
"""

# Default max_tokens — 2048 is enough for Q&A pairs; raise per-call if needed
_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_BATCH_SIZE = 5


def _salvage_objects(text: str) -> list[dict]:
    """Scan text for individually-parseable top-level sample objects, skipping broken ones.

    Only returns dicts that have the required 'messages' key so that inner
    message dicts from a truncated response are never mistaken for samples.
    """
    decoder = json.JSONDecoder(strict=False)
    objects: list[dict] = []
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
            if isinstance(obj, dict) and "messages" in obj:
                objects.append(obj)
            pos = end
        except json.JSONDecodeError:
            pos = start + 1
    return objects


def _repair_json(text: str) -> str:
    """Heuristic repairs for the most common LLM JSON mistakes."""
    # Remove JS-style // comments (outside strings — good enough for LLM output)
    text = re.sub(r'//[^\n]*', '', text)
    # Remove JS-style /* */ block comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Strip trailing commas before ] or } — the #1 LLM JSON mistake
    # Loop until stable (nested trailing commas after repair)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r',(\s*[}\]])', r'\1', text)
    return text


def _parse_llm_json(text: str) -> list[dict]:
    """Extract and parse a JSON array from LLM text."""
    text = text.strip()
    # Strip all markdown code fences
    text = re.sub(r'^```[^\n]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    # Find start of JSON array (skip any leading prose)
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found in LLM response")
    text = text[start:]
    # Fix invalid JSON escape sequences LLMs emit (e.g. \p, \s, \U).
    # Negative lookbehind prevents corrupting valid \\ pairs.
    text = re.sub(r'(?<!\\)\\([^"\\/bfnrtu])', r'\\\\\1', text)
    # Apply heuristic repairs (trailing commas, comments)
    text = _repair_json(text)
    # strict=False: allow raw control characters inside strings
    try:
        obj, _ = json.JSONDecoder(strict=False).raw_decode(text)
        if not isinstance(obj, list):
            raise ValueError(f"Expected JSON array, got {type(obj).__name__}")
        if not all(isinstance(item, dict) for item in obj):
            bad = [type(i).__name__ for i in obj if not isinstance(i, dict)]
            raise ValueError(f"JSON array contains non-object items: {bad[:3]}")
        return obj
    except json.JSONDecodeError:
        # Fallback: salvage individually-parseable sample objects
        objects = _salvage_objects(text)
        if not objects:
            raise ValueError("Could not extract any valid objects from LLM response")
        return objects


def _generate_batch(
    client,
    topic: str,
    batch_size: int,
    model: str,
    system_prompt: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> list[dict]:
    """Generate one batch of samples, retrying once on parse failure."""
    messages = [
        {"role": "system", "content": system_prompt or _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(batch_size=batch_size, topic=topic)},
    ]
    try:
        reply = client.chat(messages, model=model, temperature=0.8, max_tokens=max_tokens)
    except Exception as exc:
        raise RuntimeError(
            f"LLM request failed for topic {topic!r} (batch_size={batch_size}): {exc}"
        ) from exc

    try:
        batch = _parse_llm_json(reply)
    except (json.JSONDecodeError, ValueError):
        # Retry once with a clearer instruction
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": _RETRY_ADDENDUM})
        try:
            reply2 = client.chat(messages, model=model, temperature=0.5, max_tokens=max_tokens)
        except Exception as exc:
            raise RuntimeError(
                f"LLM retry request failed for topic {topic!r}: {exc}"
            ) from exc
        try:
            batch = _parse_llm_json(reply2)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"LLM returned invalid JSON after retry. Last response: {reply2[:300]!r}"
            ) from exc

    # Validate top-level shape before returning — inner message dicts or
    # wrong-shaped objects must not silently propagate into all_samples.
    if not all(isinstance(item, dict) and "messages" in item for item in batch):
        bad_shapes = [
            list(item.keys()) if isinstance(item, dict) else type(item).__name__
            for item in batch[:3]
        ]
        raise RuntimeError(
            f"LLM returned objects missing required 'messages' key. "
            f"Got shapes: {bad_shapes}"
        )
    return batch


def generate_dataset(
    client,
    prompt: str,
    num_samples: int,
    model: str = "",
    system_prompt: str = "",
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> Dataset:
    """
    Ask the LLM to generate num_samples fine-tuning examples about prompt.
    Generates in batches. Shows Rich progress bar.
    Returns Dataset.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}")

    all_samples: list[dict] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[touster.brand]Generating dataset[/touster.brand]"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Generating...", total=num_samples)

        while len(all_samples) < num_samples:
            remaining = num_samples - len(all_samples)
            current_batch = min(batch_size, remaining)
            batch = _generate_batch(client, prompt, current_batch, model, system_prompt)
            if not batch:
                raise RuntimeError(
                    f"LLM returned an empty batch for topic {prompt!r}. "
                    "Check your LLM configuration and prompt."
                )
            all_samples.extend(batch[:current_batch])
            progress.update(task, advance=len(batch[:current_batch]))

    try:
        return from_list(all_samples[:num_samples])
    except ValueError as exc:
        raise RuntimeError(
            f"LLM-generated samples failed schema validation for topic {prompt!r}. "
            f"This usually means the LLM returned malformed objects "
            f"(missing 'messages' key, wrong nesting, etc.). Detail: {exc}"
        ) from exc
