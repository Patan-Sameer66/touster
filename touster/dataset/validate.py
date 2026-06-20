from __future__ import annotations

from touster.dataset.schema import Dataset, Message, Sample


_ROLE_NORMALIZATION: dict[str, str] = {
    "human": "user",
    "gpt": "assistant",
    "bot": "assistant",
}

_VALID_ROLES = frozenset({"system", "user", "assistant"})


def _normalize_role(role: str) -> str:
    """Lowercase and apply known aliases."""
    normalized = role.lower().strip()
    return _ROLE_NORMALIZATION.get(normalized, normalized)


def _repair_sample(sample: Sample) -> tuple[Sample | None, list[str]]:
    """
    Repair a single sample:
    - Strip messages with empty content.
    - Normalize role strings.
    - Return None if sample lacks a user turn or assistant turn after repair.
    """
    warnings: list[str] = []
    repaired_messages: list[Message] = []

    for msg in sample.messages:
        content = msg.content.strip()
        if not content:
            warnings.append(
                f"Removed message with empty content (role={msg.role!r})"
            )
            continue
        normalized_role = _normalize_role(msg.role)
        if normalized_role != msg.role:
            warnings.append(
                f"Normalized role {msg.role!r} -> {normalized_role!r}"
            )
        repaired_messages.append(Message(role=normalized_role, content=content))

    if not repaired_messages:
        warnings.append("Dropped sample: no messages remaining after cleanup.")
        return None, warnings

    roles = {m.role for m in repaired_messages}
    if "user" not in roles:
        warnings.append("Dropped sample: no 'user' turn after repair.")
        return None, warnings
    if "assistant" not in roles:
        warnings.append("Dropped sample: no 'assistant' turn after repair.")
        return None, warnings

    return Sample(messages=tuple(repaired_messages)), warnings


def validate_and_repair(ds: Dataset) -> tuple[Dataset, list[str]]:
    """
    Check and repair dataset:
    - Ensure every sample has at least 1 user turn and at least 1 assistant turn.
    - Strip empty content.
    - Normalize role strings.
    Returns (repaired_dataset, list_of_warnings).
    """
    all_warnings: list[str] = []
    good_samples: list[Sample] = []

    for i, sample in enumerate(ds.samples):
        repaired, warnings = _repair_sample(sample)
        for w in warnings:
            all_warnings.append(f"Sample {i}: {w}")
        if repaired is not None:
            good_samples.append(repaired)

    return Dataset(samples=tuple(good_samples)), all_warnings


def validate_strict(ds: Dataset) -> list[str]:
    """
    Run strict validation without mutation.
    Returns list of error strings. Empty list = valid.
    """
    errors: list[str] = []

    for i, sample in enumerate(ds.samples):
        roles = [m.role for m in sample.messages]
        has_user = "user" in roles
        has_assistant = "assistant" in roles

        if not has_user:
            errors.append(f"Sample {i}: missing 'user' turn.")
        if not has_assistant:
            errors.append(f"Sample {i}: missing 'assistant' turn.")

        for j, msg in enumerate(sample.messages):
            if not msg.content.strip():
                errors.append(f"Sample {i}, message {j}: empty content.")
            if msg.role not in _VALID_ROLES:
                errors.append(
                    f"Sample {i}, message {j}: unknown role {msg.role!r}."
                )

    return errors
