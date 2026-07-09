from __future__ import annotations

from pathlib import Path

from touster import display
from touster.config import DatasetConfig
from touster.dataset.dedup import dedup_and_filter
from touster.dataset.generate import generate_dataset
from touster.dataset.hub import detect_source_type, download_url, load_hf_dataset
from touster.dataset.load import load_dataset
from touster.dataset.quality import filter_by_quality
from touster.dataset.schema import Dataset, save_jsonl
from touster.dataset.structure import structure_dataset
from touster.dataset.validate import validate_and_repair
from touster.llm.client import LLMClient
from touster.llm.factory import build_client_for_dataset


def validate_dataset_config(cfg: DatasetConfig) -> Path | None:
    """Validate cfg for its mode before any compute/API spend.

    Returns a pre-resolved local Path for a mode-2 local file (so the
    data-source stage can skip re-resolving it), else None. Raises
    ValueError/FileNotFoundError on misconfiguration.
    """
    if cfg.mode == 2:
        if not cfg.dataset_path:
            raise ValueError("Mode 2: set dataset_path to a local file, URL, or HF dataset ID.")
        source = str(cfg.dataset_path)
        stype = detect_source_type(source)
        if stype == "local":
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
            print(f"Mode 2 — local file: {path} (data-source stage will skip re-resolving it)")
            return path
        print(f"Mode 2 — {stype}: {source} (will resolve in the data-source stage)")
        return None

    if cfg.mode == 0 and not cfg.prompt:
        raise ValueError("Mode 0: set prompt to a non-empty topic string.")
    if cfg.mode == 1 and not cfg.raw_data_path:
        raise ValueError("Mode 1: set raw_data_path to your raw .txt/.md file.")
    return None


def estimate_datagen_cost(cfg: DatasetConfig, has_llm: bool) -> str:
    """Rough dataset-generation cost/time estimate, printed before any spend starts.

    Dataset-gen only — full pipeline time/cost (including the tuning stage)
    lands once that stage exists; see overview.md.
    """
    if cfg.mode == 2 or not has_llm:
        return "Dataset: no LLM calls (mode 2 or heuristic-only)."
    calls = -(-cfg.num_samples // max(cfg.gen_batch_size, 1))  # ceil division
    judge_calls = cfg.num_samples if cfg.mode in (0, 1) else 0
    return f"Dataset: ~{calls} generation call(s) + ~{judge_calls} quality-judge call(s). Cost is $0 on Ollama, API pricing otherwise."


def prepare_dataset_stage(
    base_model: str,
    mode: int,
    topic: str,
    dataset_path: str | None,
    num_samples: int,
    gen_batch_size: int,
    min_quality_score: float,
    quality_scale: int,
    api_key: str,
    api_base: str,
    api_model: str,
    ollama_port: int,
    ollama_model: str,
) -> tuple[DatasetConfig, Path | None, LLMClient | None]:
    """Build + validate the dataset config, build the LLM client, and print
    the config summary + cost estimate — everything the Config cell needs,
    in one call instead of inlining it in the notebook.

    Returns (dataset_cfg, validated_path, llm_client).
    """
    cfg = DatasetConfig(
        mode=mode, prompt=topic,
        raw_data_path=Path(dataset_path) if mode == 1 and dataset_path else None,
        dataset_path=Path(dataset_path) if dataset_path else None,
        num_samples=num_samples, gen_batch_size=gen_batch_size,
        model=(ollama_model if not api_key else api_model),
        min_quality_score=min_quality_score, quality_scale=quality_scale,
    )
    validated_path = validate_dataset_config(cfg)
    llm_client, llm_label = build_client_for_dataset(mode, api_key, api_base, api_model, ollama_port, ollama_model)

    print(f"Model    : {base_model}")
    print(f"Dataset  : mode {mode}  n={num_samples}  quality>={min_quality_score}/{quality_scale}")
    print(f"LLM      : {llm_label}")
    print(estimate_datagen_cost(cfg, llm_client is not None))

    return cfg, validated_path, llm_client


def run_dataset_mode(
    cfg: DatasetConfig,
    run_dir: Path,
    client=None,
) -> Path:
    """
    Run the correct dataset mode (0=generate, 1=structure, 2=bring-your-own).
    Modes 0/1 only: dedup+filter, then an LLM-judge quality gate.
    Validates and repairs all modes.
    Saves final dataset to run_dir/dataset.jsonl.
    Returns the path to the validated dataset file.
    """
    run_dir = Path(run_dir)
    output_path = run_dir / "dataset.jsonl"
    total_steps = 4
    raw_text = ""  # captured for mode 1's groundedness quality check

    # ------------------------------------------------------------------
    # Step 1: Acquire dataset
    # ------------------------------------------------------------------
    display.step(1, total_steps, "Acquiring dataset")

    if cfg.mode == 0:
        if client is None:
            raise ValueError("Mode 0 (generate) requires an LLM client.")
        print(f"Generating {cfg.num_samples} samples for prompt: {cfg.prompt[:80]!r}")
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
        print(f"Reading raw text from {raw_path}")
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
        print(f"Loading dataset from {ds_path}")
        ds = load_dataset(ds_path)

    else:
        raise ValueError(f"Unknown dataset mode: {cfg.mode}")

    display.success(f"Acquired {len(ds)} samples.")

    # ------------------------------------------------------------------
    # Step 2: Dedup + filter (modes 0 and 1 only) — cheap, do it before
    # the expensive per-sample quality judge below.
    # ------------------------------------------------------------------
    display.step(2, total_steps, "Dedup & near-duplicate filter")

    if cfg.mode in (0, 1):
        before = len(ds)
        ds = dedup_and_filter(ds)
        after = len(ds)
        removed = before - after
        if removed:
            display.warning(f"Removed {removed} duplicates/low-quality samples.")
        if after < cfg.num_samples:
            display.warning(
                f"Only {after}/{cfg.num_samples} samples remain after dedup. "
                "Consider generating more raw samples or lowering min_assistant_chars."
            )
        display.success(f"{after} samples after dedup.")
    else:
        print("Skipping dedup for mode 2 (bring-your-own).")

    # ------------------------------------------------------------------
    # Step 3: LLM-judge quality gate (modes 0 and 1 only) — golden format
    # only checks shape; this checks whether the answer is actually good.
    # ------------------------------------------------------------------
    display.step(3, total_steps, "Quality gate")

    if cfg.mode in (0, 1):
        before = len(ds)
        ds, quality_warnings = filter_by_quality(
            ds, client, cfg.model, cfg.mode,
            min_score=cfg.min_quality_score, scale=cfg.quality_scale,
            source_text=raw_text,
        )
        for w in quality_warnings:
            display.warning(w)
        display.success(f"{len(ds)}/{before} samples passed the quality gate.")
    else:
        print("Skipping quality gate for mode 2 (bring-your-own).")

    # ------------------------------------------------------------------
    # Step 4: Validate & repair
    # ------------------------------------------------------------------
    display.step(4, total_steps, "Validate & repair")

    ds, warnings = validate_and_repair(ds)
    for w in warnings:
        display.warning(w)

    if len(ds) == 0:
        raise RuntimeError(
            "Dataset is empty after dedup, quality gate, and validation. "
            "Check input quality, LLM configuration, or lower min_quality_score."
        )

    # Create run_dir only when we are sure we have data to write
    run_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(ds, output_path)
    display.success(f"Dataset saved to {output_path} ({len(ds)} samples).")

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
