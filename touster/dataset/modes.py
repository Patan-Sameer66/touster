from __future__ import annotations

from pathlib import Path

from touster.config import DatasetConfig
from touster.console import console, print_step, print_success, print_warning
from touster.dataset.dedup import dedup_and_filter
from touster.dataset.generate import generate_dataset
from touster.dataset.hub import detect_source_type, download_url, load_hf_dataset
from touster.dataset.load import load_dataset
from touster.dataset.schema import Dataset, save_jsonl
from touster.dataset.structure import structure_dataset
from touster.dataset.validate import validate_and_repair


def run_dataset_mode(
    cfg: DatasetConfig,
    run_dir: Path,
    client=None,
) -> Path:
    """
    Run the correct dataset mode (0=generate, 1=structure, 2=bring-your-own).
    Applies dedup+filter for modes 0 and 1.
    Validates and repairs all modes.
    Saves final dataset to run_dir/dataset.jsonl.
    Returns the path to the validated dataset file.
    """
    run_dir = Path(run_dir)
    output_path = run_dir / "dataset.jsonl"
    total_steps = 3

    # ------------------------------------------------------------------
    # Step 1: Acquire dataset
    # ------------------------------------------------------------------
    print_step(1, total_steps, "Acquiring dataset")

    if cfg.mode == 0:
        if client is None:
            raise ValueError("Mode 0 (generate) requires an LLM client.")
        console.print(
            f"[touster.dim]Generating {cfg.num_samples} samples for prompt: "
            f"{cfg.prompt[:80]!r}[/touster.dim]"
        )
        ds: Dataset = generate_dataset(
            client=client,
            prompt=cfg.prompt,
            num_samples=cfg.num_samples,
            model=cfg.model,
            batch_size=cfg.gen_batch_size,
        )

    elif cfg.mode == 1:
        if client is None:
            raise ValueError("Mode 1 (structure) requires an LLM client.")
        if cfg.raw_data_path is None:
            raise ValueError("Mode 1 (structure) requires raw_data_path.")
        raw_path = Path(cfg.raw_data_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
        console.print(f"[touster.dim]Reading raw text from {raw_path}[/touster.dim]")
        raw_text = raw_path.read_text(encoding="utf-8")
        ds = structure_dataset(
            client=client,
            raw_text=raw_text,
            num_samples=cfg.num_samples,
            model=cfg.model,
        )

    elif cfg.mode == 2:
        if cfg.dataset_path is None:
            raise ValueError("Mode 2 (bring-your-own) requires dataset_path.")
        ds_path = _resolve_dataset_source(str(cfg.dataset_path), run_dir)
        console.print(f"[touster.dim]Loading dataset from {ds_path}[/touster.dim]")
        ds = load_dataset(ds_path)

    else:
        raise ValueError(f"Unknown dataset mode: {cfg.mode}")

    print_success(f"Acquired {len(ds)} samples.")

    # ------------------------------------------------------------------
    # Step 2: Dedup + filter (modes 0 and 1 only)
    # ------------------------------------------------------------------
    print_step(2, total_steps, "Dedup & quality filter")

    if cfg.mode in (0, 1):
        before = len(ds)
        ds = dedup_and_filter(ds)
        after = len(ds)
        removed = before - after
        if removed:
            print_warning(f"Removed {removed} duplicates/low-quality samples.")
        if cfg.mode in (0, 1) and after < cfg.num_samples:
            print_warning(
                f"Only {after}/{cfg.num_samples} samples remain after dedup. "
                "Consider generating more raw samples or lowering min_assistant_chars."
            )
        print_success(f"{after} samples after dedup.")
    else:
        console.print("[touster.dim]Skipping dedup for mode 2 (bring-your-own).[/touster.dim]")

    # ------------------------------------------------------------------
    # Step 3: Validate & repair
    # ------------------------------------------------------------------
    print_step(3, total_steps, "Validate & repair")

    ds, warnings = validate_and_repair(ds)
    for w in warnings:
        print_warning(w)

    if len(ds) == 0:
        raise RuntimeError(
            "Dataset is empty after dedup and validation. "
            "Check input quality, LLM configuration, or reduce min_assistant_chars threshold."
        )

    # Create run_dir only when we are sure we have data to write
    run_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(ds, output_path)
    print_success(f"Dataset saved to {output_path} ({len(ds)} samples).")

    return output_path


# ── helpers ──────────────────────────────────────────────────────────────────

def _resolve_dataset_source(source: str, run_dir: Path) -> Path:
    """Resolve a dataset source to a local file path.

    Handles:
    - Local file paths (returned as-is)
    - Direct URLs (downloaded to run_dir/cache/)
    - HuggingFace dataset IDs like "author/dataset-name[/split]"
    """
    src_type = detect_source_type(source)

    if src_type == "local":
        return Path(source)

    cache_dir = run_dir / "cache"

    if src_type == "url":
        return download_url(source, cache_dir)

    # HuggingFace dataset ID — may include a split suffix "author/name/split"
    parts = source.split("/")
    if len(parts) == 3:
        repo_id = "/".join(parts[:2])
        split = parts[2]
    else:
        repo_id = source
        split = "train"

    raw_records = load_hf_dataset(repo_id, split=split, cache_dir=cache_dir)
    if not raw_records:
        raise RuntimeError(f"HuggingFace dataset '{source}' returned 0 convertible samples.")

    # Save to a local JSONL so the normal load_dataset path handles it
    import json
    local_path = cache_dir / f"{source.replace('/', '__')}_{split}.jsonl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with local_path.open("w", encoding="utf-8", newline="") as f:
        for rec in raw_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return local_path
