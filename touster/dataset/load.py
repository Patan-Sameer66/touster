from __future__ import annotations

import json
from pathlib import Path

from touster.dataset.schema import Dataset, Message, Sample, from_list, load_jsonl


_ALPACA_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def _normalize_alpaca_sample(item: dict) -> dict:
    """Convert HF-style {conversations: [{from, value}]} to {messages: [{role, content}]}."""
    conversations = item.get("conversations", [])
    messages = []
    for turn in conversations:
        from_role = turn.get("from", "")
        value = turn.get("value", "")
        role = _ALPACA_ROLE_MAP.get(from_role.lower(), from_role)
        messages.append({"role": role, "content": value})
    return {"messages": messages}


def _is_alpaca_format(data: list[dict]) -> bool:
    """Check whether the dataset looks like HF alpaca/vicuna format."""
    if not data:
        return False
    sample = data[0]
    return "conversations" in sample and "messages" not in sample


def load_dataset(path: Path) -> Dataset:
    """
    Load dataset from .jsonl or .json file.

    Accepts:
    - jsonl: {messages: [{role, content},...]} per line
    - json array: same objects as jsonl but in a list
    - HF-style alpaca/vicuna: {conversations: [{from, value},...]} — normalized to messages

    Raises ValueError with a clear message on unknown format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        # Try standard first; also support alpaca-style jsonl
        with path.open("r", encoding="utf-8") as fh:
            raw_lines: list[dict] = []
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_lines.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

        if _is_alpaca_format(raw_lines):
            normalized = [_normalize_alpaca_sample(item) for item in raw_lines]
            return from_list(normalized)
        return from_list(raw_lines)

    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in file {path}: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array at the top level of {path}, got {type(data).__name__}"
            )

        if _is_alpaca_format(data):
            normalized = [_normalize_alpaca_sample(item) for item in data]
            return from_list(normalized)
        return from_list(data)

    else:
        raise ValueError(
            f"Unknown file format '{suffix}' for {path}. "
            "Supported formats: .jsonl, .json"
        )
