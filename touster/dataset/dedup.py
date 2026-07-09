from __future__ import annotations

import logging

from touster.dataset.schema import Dataset, Sample


logger = logging.getLogger(__name__)


def _sample_text(sample: Sample) -> str:
    """Canonical string encoding role+content boundaries for dedup hashing.

    Uses null-byte and SOH as separators — characters that cannot appear in
    parsed JSON string content — so different role/content splits never collide.
    """
    return "\x00".join(f"{m.role}\x01{m.content}" for m in sample.messages)


def _assistant_chars(sample: Sample) -> int:
    """Return total character count of assistant turns in sample."""
    return sum(len(m.content) for m in sample.messages if m.role == "assistant")


def _try_minhash_dedup(
    samples: list[Sample],
    threshold: float,
    num_perm: int = 128,
) -> list[Sample]:
    """
    Near-duplicate removal using datasketch MinHashLSH.
    Returns deduplicated sample list.
    """
    from datasketch import MinHash, MinHashLSH  # type: ignore[import]

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[Sample] = []

    for idx, sample in enumerate(samples):
        text = _sample_text(sample)
        tokens = set(text.lower().split())
        m = MinHash(num_perm=num_perm)
        if not tokens:
            # Empty token set: every empty sample produces an identical MinHash.
            # Treat as a unique key derived from index so they are all kept.
            m.update(f"__empty__{idx}".encode("utf-8"))
        else:
            for token in tokens:
                m.update(token.encode("utf-8"))

        key = str(idx)
        result = lsh.query(m)
        if not result:
            lsh.insert(key, m)
            kept.append(sample)

    return kept


def _exact_dedup(samples: list[Sample]) -> list[Sample]:
    """Exact deduplication using the canonical sample text as key."""
    seen: set[str] = set()
    result = []
    for sample in samples:
        key = _sample_text(sample)
        if key not in seen:
            seen.add(key)
            result.append(sample)
    return result


def dedup_and_filter(
    ds: Dataset,
    similarity_threshold: float = 0.85,
    min_assistant_chars: int = 50,
) -> Dataset:
    """
    1. Remove samples where the assistant turn has fewer than min_assistant_chars.
    2. Near-duplicate removal using MinHashLSH (falls back to exact dedup if unavailable).
    3. Returns a new Dataset (immutable — no mutation of ds).
    """
    if not (0.0 < similarity_threshold <= 1.0):
        raise ValueError(
            f"similarity_threshold must be in (0.0, 1.0], got {similarity_threshold!r}"
        )

    # Step 1: quality filter
    quality_filtered = [
        s for s in ds.samples
        if _assistant_chars(s) >= min_assistant_chars
    ]
    logger.info(
        "Quality filter: %d → %d samples (removed %d)",
        len(ds.samples), len(quality_filtered), len(ds.samples) - len(quality_filtered),
    )

    # Step 2: near-duplicate removal
    used_minhash = False
    try:
        deduped = _try_minhash_dedup(quality_filtered, similarity_threshold)
        used_minhash = True
    except ImportError:
        deduped = _exact_dedup(quality_filtered)
    except MemoryError:
        raise
    except Exception as exc:
        logger.warning(
            "datasketch runtime error (%s: %s); falling back to exact dedup",
            type(exc).__name__, exc,
        )
        deduped = _exact_dedup(quality_filtered)

    logger.info(
        "Dedup (%s): %d → %d samples (removed %d)",
        "minhash" if used_minhash else "exact",
        len(quality_filtered), len(deduped), len(quality_filtered) - len(deduped),
    )

    return Dataset(samples=tuple(deduped))
