from __future__ import annotations

from touster.dataset.schema import Dataset, Sample


def _sample_text(sample: Sample) -> str:
    """Concatenate all message contents into a single string for hashing."""
    return " ".join(m.content for m in sample.messages)


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
        for token in tokens:
            m.update(token.encode("utf-8"))

        key = str(idx)
        result = lsh.query(m)
        if not result:
            lsh.insert(key, m)
            kept.append(sample)
        # else: near-duplicate found — skip

    return kept


def dedup_and_filter(
    ds: Dataset,
    similarity_threshold: float = 0.85,
    min_assistant_chars: int = 50,
) -> Dataset:
    """
    1. Remove samples where the assistant turn has fewer than min_assistant_chars.
    2. Near-duplicate removal using MinHashLSH (falls back to length filter if datasketch unavailable).
    3. Returns a new Dataset (immutable — no mutation of ds).
    """
    # Step 1: quality filter
    quality_filtered = [
        s for s in ds.samples
        if _assistant_chars(s) >= min_assistant_chars
    ]

    # Step 2: near-duplicate removal
    try:
        deduped = _try_minhash_dedup(quality_filtered, similarity_threshold)
    except ImportError:
        # datasketch not available — fall back to exact-duplicate removal
        seen: set[str] = set()
        deduped = []
        for sample in quality_filtered:
            key = _sample_text(sample)
            if key not in seen:
                seen.add(key)
                deduped.append(sample)

    return Dataset(samples=tuple(deduped))
