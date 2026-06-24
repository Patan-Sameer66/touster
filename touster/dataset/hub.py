from __future__ import annotations

import re
import urllib.request
from pathlib import Path


def detect_source_type(source: str) -> str:
    """Return 'url', 'hf', or 'local'."""
    if source.startswith("http://") or source.startswith("https://"):
        return "url"
    # HF dataset ID: author/name — no OS path separators, no file extension
    if re.match(r'^[A-Za-z0-9_\-\.]+/[A-Za-z0-9_\-\.]+$', source):
        return "hf"
    return "local"


def download_url(url: str, cache_dir: Path) -> Path:
    """Download a file from url into cache_dir, return local path."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Derive filename from URL
    filename = url.split("?")[0].rstrip("/").split("/")[-1] or "dataset.jsonl"
    local_path = cache_dir / filename
    if local_path.exists():
        return local_path
    print(f"Downloading {url} → {local_path}")
    urllib.request.urlretrieve(url, local_path)
    return local_path


def _convert_hf_sample(sample: dict) -> dict | None:
    """Convert a single HF dataset sample to {messages: [{role, content}]} format.

    Supports:
    - messages format (OpenAI / toaster native)
    - conversations format (ShareGPT)
    - instruction + output format (Alpaca)
    - prompt + response format
    - prompt + completion format
    """
    # Already in messages format
    if "messages" in sample and isinstance(sample["messages"], list):
        return sample

    # ShareGPT / conversations format
    if "conversations" in sample:
        return {"conversations": sample["conversations"]}

    # Alpaca: instruction + output (+ optional input)
    if "instruction" in sample and "output" in sample:
        instruction = sample["instruction"]
        if sample.get("input", "").strip():
            instruction = instruction + "\n\n" + sample["input"]
        return {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": sample["output"]},
            ]
        }

    # prompt + response
    if "prompt" in sample and "response" in sample:
        return {
            "messages": [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["response"]},
            ]
        }

    # prompt + completion
    if "prompt" in sample and "completion" in sample:
        return {
            "messages": [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["completion"]},
            ]
        }

    # question + answer
    if "question" in sample and "answer" in sample:
        return {
            "messages": [
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": sample["answer"]},
            ]
        }

    return None


def load_hf_dataset(
    repo_id: str,
    split: str = "train",
    cache_dir: Path | None = None,
    max_samples: int | None = None,
) -> list[dict]:
    """Load a HuggingFace Hub dataset and convert to messages format.

    Returns list of raw dicts ready for from_list().
    """
    try:
        from datasets import load_dataset as hf_load  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load HuggingFace datasets. "
            "Install it with: pip install datasets"
        ) from exc

    print(f"Loading HuggingFace dataset: {repo_id} (split={split})")
    ds = hf_load(repo_id, split=split, cache_dir=str(cache_dir) if cache_dir else None)

    converted: list[dict] = []
    skipped = 0
    items = ds if max_samples is None else ds.select(range(min(max_samples, len(ds))))
    for sample in items:
        result = _convert_hf_sample(dict(sample))
        if result is not None:
            converted.append(result)
        else:
            skipped += 1

    if skipped:
        print(f"  Skipped {skipped} samples with unrecognized format.")
    print(f"  Converted {len(converted)} samples from {repo_id}.")
    return converted
