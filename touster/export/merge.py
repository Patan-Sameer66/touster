from __future__ import annotations

"""Merge LoRA adapter into base model weights to produce a standalone 16-bit model."""

import json
from pathlib import Path

from touster.console import console, print_success, print_warning


def export_merged(adapter_path: Path, run_dir: Path, dtype: str = "float16") -> Path:
    """
    Merge LoRA adapter into base model and save as merged 16-bit weights.

    Steps:
    1. Load base model id from adapter_path/adapter_config.json
       (field: "base_model_name_or_path")
    2. Load base model with AutoModelForCausalLM
    3. Load PEFT model via PeftModel.from_pretrained(base, adapter_path)
    4. Call peft_model.merge_and_unload() to get merged model
    5. Save to run_dir/merged_weights/ with save_pretrained + tokenizer

    Returns path to merged_weights dir.
    Raises RuntimeError with clear message if adapter_config.json not found.
    """
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise RuntimeError(
            f"adapter_config.json not found at {adapter_config_path}. "
            "Ensure adapter_path points to a valid PEFT adapter directory."
        )

    adapter_config = json.loads(adapter_config_path.read_text())
    base_model_id = adapter_config.get("base_model_name_or_path", "")
    if not base_model_id:
        raise RuntimeError(
            "adapter_config.json does not contain 'base_model_name_or_path'. "
            "The adapter directory may be corrupt."
        )

    console.print(
        f"[touster.dim]Merging adapter into base model:[/touster.dim] "
        f"[touster.model]{base_model_id}[/touster.model]"
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch_dtype = getattr(torch, dtype, torch.float16)

    with console.status("[touster.step]Loading base model...[/touster.step]"):
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

    with console.status("[touster.step]Loading PEFT adapter...[/touster.step]"):
        peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    with console.status("[touster.step]Merging and unloading LoRA weights...[/touster.step]"):
        merged_model = peft_model.merge_and_unload()

    merged_dir = run_dir / "merged_weights"
    merged_dir.mkdir(parents=True, exist_ok=True)

    with console.status(f"[touster.step]Saving merged model to {merged_dir}[/touster.step]"):
        merged_model.save_pretrained(str(merged_dir))

        # Save tokenizer if present in adapter dir or base model
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            except Exception:
                tokenizer = None
                print_warning("Could not load tokenizer — skipping tokenizer save.")

        if tokenizer is not None:
            tokenizer.save_pretrained(str(merged_dir))

    print_success(f"Merged model saved to: {merged_dir}")
    return merged_dir
