from __future__ import annotations

import json
from pathlib import Path

from touster.dataset.schema import Dataset, from_list


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
    for i, turn in enumerate(conversations):
        from_role = turn.get("from")
        value = turn.get("value")
        if from_role is None or value is None:
            raise ValueError(
                f"Alpaca turn {i} missing required keys 'from' or 'value': {turn!r}"
            )
        role = _ALPACA_ROLE_MAP.get(from_role.lower(), from_role)
        messages.append({"role": role, "content": value})
    return {"messages": messages}


def _is_alpaca_format(data: list[dict]) -> bool:
    """Check whether ALL records use HF alpaca/vicuna format.

    Raises ValueError if the format is inconsistent across records.
    """
    if not data:
        return False
    has_conversations = [
        "conversations" in s and "messages" not in s
        for s in data
    ]
    if any(has_conversations) and not all(has_conversations):
        raise ValueError(
            "Mixed dataset format: some records use 'conversations', others use 'messages'. "
            "All records must use the same format."
        )
    return all(has_conversations)


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
        with path.open("r", encoding="utf-8-sig") as fh:
            raw_lines: list[dict] = []
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_lines.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_no}: Invalid JSON: {exc.msg} (col {exc.colno})"
                    ) from exc

        if _is_alpaca_format(raw_lines):
            normalized = [_normalize_alpaca_sample(item) for item in raw_lines]
            try:
                result = from_list(normalized)
            except ValueError as exc:
                raise ValueError(f"Failed to parse dataset {path}: {exc}") from exc
        else:
            try:
                result = from_list(raw_lines)
            except ValueError as exc:
                raise ValueError(f"Failed to parse dataset {path}: {exc}") from exc

        if len(result) == 0:
            raise ValueError(f"Dataset is empty: {path}")
        return result

    elif suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as fh:
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
            try:
                result = from_list(normalized)
            except ValueError as exc:
                raise ValueError(f"Failed to parse dataset {path}: {exc}") from exc
        else:
            try:
                result = from_list(data)
            except ValueError as exc:
                raise ValueError(f"Failed to parse dataset {path}: {exc}") from exc

        if len(result) == 0:
            raise ValueError(f"Dataset is empty: {path}")
        return result

    else:
        raise ValueError(
            f"Unknown file format '{suffix}' for {path}. "
            "Supported formats: .jsonl, .json"
        )
