from __future__ import annotations

import json
import re

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from touster.console import console, print_warning
from touster.dataset.schema import Dataset, Sample, try_parse_sample


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


def _wrap_flat_messages(items: list[dict]) -> list[dict]:
    """Recover when the LLM returns [{role, content}, ...] instead of [{messages:[...]}, ...].

    Groups consecutive user+assistant pairs into {messages: [user, assistant]} objects.
    Returns items unchanged if they already have the messages key.
    """
    if not items or "messages" in items[0]:
        return items
    # All items look like flat message dicts
    if not all(isinstance(i, dict) and "role" in i and "content" in i for i in items):
        return items
    wrapped: list[dict] = []
    i = 0
    while i < len(items):
        pair: list[dict] = []
        if items[i].get("role") == "user":
            pair.append(items[i])
            i += 1
        if i < len(items) and items[i].get("role") == "assistant":
            pair.append(items[i])
            i += 1
        if pair:
            wrapped.append({"messages": pair})
        else:
            i += 1  # skip unrecognised item
    return wrapped


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
        # Auto-wrap flat [{role,content},...] the model emits when format="json"
        obj = _wrap_flat_messages(obj)
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
        reply = client.chat(messages, model=model, temperature=0.8, max_tokens=max_tokens, format="json")
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
            reply2 = client.chat(messages, model=model, temperature=0.5, max_tokens=max_tokens, format="json")
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

    # A batch producing zero golden-format samples this many times in a row
    # means the LLM can't be salvaged for this run — stop instead of spinning.
    _MAX_CONSECUTIVE_EMPTY = 5

    good_samples: list[Sample] = []
    consecutive_empty = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[touster.brand]Generating dataset[/touster.brand]"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Generating...", total=num_samples)

        while len(good_samples) < num_samples:
            remaining = num_samples - len(good_samples)
            current_batch = min(batch_size, remaining)
            try:
                raw_batch = _generate_batch(client, prompt, current_batch, model, system_prompt)
            except RuntimeError as exc:
                print_warning(f"Batch generation failed, skipping: {exc}")
                raw_batch = []

            # Drop malformed items instead of letting one bad sample take
            # down the whole run — this is the strict golden-format gate.
            parsed = [try_parse_sample(item) for item in raw_batch]
            valid = [s for s in parsed if s is not None]
            dropped = len(raw_batch) - len(valid)
            if dropped:
                print_warning(f"Dropped {dropped} malformed sample(s) from batch (kept {len(valid)}).")

            if not valid:
                consecutive_empty += 1
                if consecutive_empty >= _MAX_CONSECUTIVE_EMPTY:
                    raise RuntimeError(
                        f"LLM produced no valid golden-format samples after "
                        f"{_MAX_CONSECUTIVE_EMPTY} consecutive failed batches for "
                        f"topic {prompt!r}. Check your LLM configuration and prompt."
                    )
                continue

            consecutive_empty = 0
            good_samples.extend(valid[:remaining])
            progress.update(task, advance=len(valid[:remaining]))

    return Dataset(samples=tuple(good_samples[:num_samples]))
