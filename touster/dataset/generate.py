from __future__ import annotations

import json
import re

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from touster.console import console
from touster.dataset.schema import Dataset, from_list


_SYSTEM_PROMPT = (
    "You are a fine-tuning data generator. "
    "Generate high-quality question-and-answer training examples. "
    "Return ONLY a valid JSON array. No markdown, no explanation outside the JSON."
)

_USER_TEMPLATE = (
    "Generate {batch_size} fine-tuning examples about: {topic}\n\n"
    "Output a JSON array of objects. Each object must have a 'messages' key containing "
    "a list of message objects with 'role' and 'content' fields. "
    "Use role 'user' for questions and 'assistant' for answers.\n\n"
    "Example format:\n"
    '[\n'
    '  {{"messages": [{{"role": "user", "content": "What is X?"}},'
    ' {{"role": "assistant", "content": "X is ..."}}]}}\n'
    ']'
)

_RETRY_ADDENDUM = (
    "\n\nIMPORTANT: Your previous response was not valid JSON. "
    "Return ONLY the raw JSON array, no markdown fences, no leading text."
)


def _parse_llm_json(text: str) -> list[dict]:
    """Extract and parse a JSON array from LLM text, stripping markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()
    # Find start of JSON array (skip any leading prose)
    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found in LLM response")
    text = text[start:]
    # Fix invalid JSON escape sequences LLMs emit (e.g. \p, \U, \s, \()
    # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX — everything else is illegal
    text = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', text)
    # raw_decode stops at end of first valid JSON value, ignoring trailing text
    obj, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(obj, list):
        raise ValueError(f"Expected JSON array, got {type(obj).__name__}")
    return obj


def _generate_batch(
    client,
    topic: str,
    batch_size: int,
    model: str,
    system_prompt: str,
) -> list[dict]:
    """Generate one batch of samples, retrying once on parse failure."""
    messages = [
        {"role": "system", "content": system_prompt or _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(batch_size=batch_size, topic=topic)},
    ]
    reply = client.chat(messages, model=model, temperature=0.8, max_tokens=4096)
    try:
        return _parse_llm_json(reply)
    except (json.JSONDecodeError, ValueError):
        # Retry once with a clearer instruction
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": _RETRY_ADDENDUM})
        reply2 = client.chat(messages, model=model, temperature=0.5, max_tokens=4096)
        try:
            return _parse_llm_json(reply2)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"LLM returned invalid JSON after retry. Last response: {reply2[:300]!r}"
            ) from exc


def generate_dataset(
    client,
    prompt: str,
    num_samples: int,
    model: str = "",
    system_prompt: str = "",
) -> Dataset:
    """
    Ask the LLM to generate num_samples fine-tuning examples about prompt.
    Generates in batches of 5. Shows Rich progress bar.
    Returns Dataset.
    """
    batch_size = 5
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

    return from_list(all_samples[:num_samples])
