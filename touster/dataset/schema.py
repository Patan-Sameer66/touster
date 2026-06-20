from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Message:
    role: str    # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Sample:
    messages: tuple[Message, ...]


@dataclass(frozen=True)
class Dataset:
    samples: tuple[Sample, ...]

    def __len__(self) -> int:
        return len(self.samples)

    def to_list(self) -> list[dict]:
        """Serialize to list of {messages: [{role, content}, ...]} dicts."""
        return [
            {
                "messages": [
                    {"role": m.role, "content": m.content}
                    for m in sample.messages
                ]
            }
            for sample in self.samples
        ]


def _parse_message(raw: object) -> Message:
    """Parse a single message dict, raising ValueError on bad shape."""
    if not isinstance(raw, dict):
        raise ValueError(f"Expected message dict, got {type(raw).__name__}: {raw!r}")
    role = raw.get("role")
    content = raw.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        raise ValueError(
            f"Message must have string 'role' and 'content', got: {raw!r}"
        )
    return Message(role=role, content=content)


def _parse_sample(raw: object) -> Sample:
    """Parse a single sample dict, raising ValueError on bad shape."""
    if not isinstance(raw, dict):
        raise ValueError(f"Expected sample dict, got {type(raw).__name__}: {raw!r}")
    msgs_raw = raw.get("messages")
    if not isinstance(msgs_raw, list):
        raise ValueError(
            f"Sample must have a 'messages' list, got: {raw!r}"
        )
    messages = tuple(_parse_message(m) for m in msgs_raw)
    return Sample(messages=messages)


def from_list(data: list[dict]) -> Dataset:
    """
    Parse list of {messages: [{role, content}]} dicts into Dataset.
    Raises ValueError on bad shape.
    """
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of samples, got {type(data).__name__}")
    samples = tuple(_parse_sample(item) for item in data)
    return Dataset(samples=samples)


def load_jsonl(path: Path) -> Dataset:
    """Load a .jsonl file where each line is a {messages: [...]} JSON object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            samples.append(_parse_sample(obj))
    return Dataset(samples=tuple(samples))


def save_jsonl(ds: Dataset, path: Path) -> None:
    """Save a Dataset as .jsonl, one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in ds.samples:
            obj = {
                "messages": [
                    {"role": m.role, "content": m.content}
                    for m in sample.messages
                ]
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
