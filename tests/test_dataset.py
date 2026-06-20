"""Tests for touster.dataset.*"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from touster.dataset.dedup import dedup_and_filter
from touster.dataset.formats import (
    apply_chat_template,
    count_tokens_dataset,
    estimate_token_count,
)
from touster.dataset.load import load_dataset
from touster.dataset.schema import (
    Dataset,
    Message,
    Sample,
    from_list,
    load_jsonl,
    save_jsonl,
)
from touster.dataset.validate import validate_and_repair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sample(user: str, assistant: str, system: str = "") -> Sample:
    messages: list[Message] = []
    if system:
        messages.append(Message(role="system", content=system))
    messages.append(Message(role="user", content=user))
    messages.append(Message(role="assistant", content=assistant))
    return Sample(messages=tuple(messages))


def make_dataset(*samples: Sample) -> Dataset:
    return Dataset(samples=tuple(samples))


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_schema_roundtrip(tmp_path: Path) -> None:
    """save_jsonl then load_jsonl gives back the same data."""
    ds = make_dataset(
        make_sample("What is 2+2?", "The answer is 4."),
        make_sample("Name a color.", "Red.", system="Be concise."),
    )
    path = tmp_path / "test.jsonl"
    save_jsonl(ds, path)
    loaded = load_jsonl(path)
    assert len(loaded) == len(ds)
    for orig, back in zip(ds.samples, loaded.samples):
        assert orig.messages == back.messages


def test_from_list_valid() -> None:
    """Parse a well-formed list of dicts into a Dataset."""
    data = [
        {"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]},
    ]
    ds = from_list(data)
    assert len(ds) == 1
    assert ds.samples[0].messages[0].role == "user"
    assert ds.samples[0].messages[1].content == "Hi there!"


def test_from_list_missing_messages_raises() -> None:
    """from_list should raise ValueError when 'messages' key is absent."""
    data = [{"text": "oops, wrong format"}]
    with pytest.raises(ValueError):
        from_list(data)


# ---------------------------------------------------------------------------
# Validate tests
# ---------------------------------------------------------------------------

def test_validate_repair_normalizes_roles() -> None:
    """'human' should become 'user', 'gpt' should become 'assistant'."""
    sample = Sample(messages=(
        Message(role="human", content="What is Python?"),
        Message(role="gpt", content="A programming language."),
    ))
    ds = Dataset(samples=(sample,))
    repaired, warnings = validate_and_repair(ds)
    roles = [m.role for m in repaired.samples[0].messages]
    assert "user" in roles
    assert "assistant" in roles
    assert "human" not in roles
    assert "gpt" not in roles
    # Warnings should mention the normalization
    assert any("user" in w or "assistant" in w for w in warnings)


def test_validate_repair_removes_empty_content() -> None:
    """Messages with empty content should be stripped."""
    sample = Sample(messages=(
        Message(role="user", content="Tell me something."),
        Message(role="assistant", content="   "),  # whitespace only
        Message(role="assistant", content="Something useful."),
    ))
    ds = Dataset(samples=(sample,))
    repaired, warnings = validate_and_repair(ds)
    # The empty assistant message should be gone
    contents = [m.content.strip() for m in repaired.samples[0].messages]
    assert "" not in contents
    assert any("empty" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Dedup tests
# ---------------------------------------------------------------------------

def test_dedup_removes_duplicate() -> None:
    """Two identical samples should collapse to one."""
    # Assistant response is long enough to pass the min_assistant_chars filter (default 50)
    sample = make_sample(
        "What is AI?",
        "Artificial Intelligence is a broad field of computer science focused on "
        "building systems that can perform tasks requiring human-like intelligence.",
    )
    ds = make_dataset(sample, sample)
    result = dedup_and_filter(ds)
    assert len(result) == 1


def test_dedup_short_filter() -> None:
    """Sample with a very short assistant turn should be filtered out."""
    short_sample = make_sample("Q?", "Hi.")  # 2-char assistant response
    long_sample = make_sample(
        "What is the capital of France?",
        "The capital of France is Paris, a major European city with rich history.",
    )
    ds = make_dataset(short_sample, long_sample)
    result = dedup_and_filter(ds, min_assistant_chars=50)
    # Short sample should be removed
    assert len(result) == 1
    assert result.samples[0].messages[-1].content == long_sample.messages[-1].content


# ---------------------------------------------------------------------------
# Format / template tests
# ---------------------------------------------------------------------------

def test_apply_chat_template_chatml() -> None:
    """ChatML output must start with <|im_start|> and contain role."""
    sample = make_sample("Hello", "Hi!")
    result = apply_chat_template(sample, template="chatml")
    assert result.startswith("<|im_start|>")
    assert "user" in result
    assert "assistant" in result


def test_apply_chat_template_llama3() -> None:
    """Llama3 output must start with <|begin_of_text|>."""
    sample = make_sample("Hello", "Hi!")
    result = apply_chat_template(sample, template="llama3")
    assert result.startswith("<|begin_of_text|>")
    assert "user" in result
    assert "assistant" in result


def test_token_count_estimate() -> None:
    """estimate_token_count returns a positive int for non-empty text."""
    count = estimate_token_count("The quick brown fox jumps over the lazy dog.")
    assert isinstance(count, int)
    assert count > 0


def test_count_tokens_dataset() -> None:
    """count_tokens_dataset returns a dict with the expected keys."""
    ds = make_dataset(
        make_sample("Question one?", "Answer one is here, fairly long."),
        make_sample("Question two?", "Answer two is also here with content."),
    )
    stats = count_tokens_dataset(ds, template="chatml")
    assert set(stats.keys()) == {"total", "mean", "max", "min", "p95"}
    assert isinstance(stats["total"], int)
    assert stats["total"] > 0
    assert isinstance(stats["mean"], float)
    assert isinstance(stats["max"], int)
    assert isinstance(stats["min"], int)
    assert isinstance(stats["p95"], int)
    assert stats["max"] >= stats["min"]


# ---------------------------------------------------------------------------
# Load tests (format normalization)
# ---------------------------------------------------------------------------

def test_load_dataset_alpaca_normalization(tmp_path: Path) -> None:
    """HF-style conversations with 'from'/'value' should normalize to role/content."""
    alpaca_data = [
        {
            "conversations": [
                {"from": "human", "value": "What is 1+1?"},
                {"from": "gpt", "value": "It is 2."},
            ]
        }
    ]
    path = tmp_path / "alpaca.json"
    path.write_text(json.dumps(alpaca_data), encoding="utf-8")

    ds = load_dataset(path)
    assert len(ds) == 1
    messages = ds.samples[0].messages
    roles = [m.role for m in messages]
    assert "user" in roles
    assert "assistant" in roles
    # Should not contain raw "from"/"value" keys as roles
    assert "human" not in roles
    assert "gpt" not in roles
