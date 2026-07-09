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
    - Strip messages with empty content (warn when content is modified).
    - Normalize role strings.
    - Enforce turn ordering: first non-system turn must be 'user',
      no consecutive same-role turns.
    - Return None if sample lacks a user turn or assistant turn after repair.
    """
    warnings: list[str] = []
    repaired_messages: list[Message] = []

    for msg in sample.messages:
        stripped = msg.content.strip()
        if not stripped:
            warnings.append(
                f"Removed message with empty content (role={msg.role!r})"
            )
            continue
        if stripped != msg.content:
            warnings.append(
                f"Stripped whitespace from message content (role={msg.role!r})"
            )
        normalized_role = _normalize_role(msg.role)
        if normalized_role not in _VALID_ROLES:
            warnings.append(
                f"Removed message with unknown role {msg.role!r} (normalized: {normalized_role!r})"
            )
            continue
        if normalized_role != msg.role:
            warnings.append(
                f"Normalized role {msg.role!r} -> {normalized_role!r}"
            )
        repaired_messages.append(Message(role=normalized_role, content=stripped))

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

    # Enforce ordering: first non-system turn must be 'user';
    # no consecutive same-role turns allowed.
    non_system = [m for m in repaired_messages if m.role != "system"]
    if non_system and non_system[0].role != "user":
        warnings.append(
            f"Dropped sample: first non-system turn is {non_system[0].role!r}, expected 'user'."
        )
        return None, warnings
    for idx in range(len(non_system) - 1):
        if non_system[idx].role == non_system[idx + 1].role:
            warnings.append(
                f"Dropped sample: consecutive {non_system[idx].role!r} turns at positions {idx},{idx+1}."
            )
            return None, warnings

    return Sample(messages=tuple(repaired_messages)), warnings


def validate_and_repair(ds: Dataset) -> tuple[Dataset, list[str]]:
    """
    Check and repair dataset:
    - Ensure every sample has at least 1 user turn and at least 1 assistant turn.
    - Strip empty content.
    - Normalize role strings.
    - Enforce turn ordering.
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
        if not sample.messages:
            errors.append(f"Sample {i}: has no messages.")
            continue

        # Normalize roles before checking so aliased roles don't produce false positives
        normalized_roles = [_normalize_role(m.role) for m in sample.messages]
        has_user = "user" in normalized_roles
        has_assistant = "assistant" in normalized_roles

        if not has_user:
            errors.append(f"Sample {i}: missing 'user' turn.")
        if not has_assistant:
            errors.append(f"Sample {i}: missing 'assistant' turn.")

        for j, msg in enumerate(sample.messages):
            if not msg.content.strip():
                errors.append(f"Sample {i}, message {j}: empty content.")
            if _normalize_role(msg.role) not in _VALID_ROLES:
                errors.append(
                    f"Sample {i}, message {j}: unknown role {msg.role!r}."
                )

    return errors
