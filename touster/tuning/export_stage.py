"""Config-driven export — runs at the end of the tuning stage, not a
separate pipeline stage. See overview.md stage 4: "Final run trains the
winning (or fallback) recipe to completion, then executes stage 2's export
toggles: local save, merged weights, GGUF, model card, optional HF Hub push."
"""

from __future__ import annotations

import shutil
from pathlib import Path

from touster import display
from touster.config import ExportConfig, RecipeConfig
from touster.export.gguf import export_gguf
from touster.export.merge import export_merged
from touster.export.modelcard import write_model_card


def run_export_stage(
    recipe: RecipeConfig,
    adapter_path: Path,
    run_dir: Path,
    export_cfg: ExportConfig,
) -> dict[str, Path | None]:
    """Run every export the config asks for. Never raises — one export
    failing (e.g. GGUF needs llama-cpp-python, which is optional) must not
    take down the ones that already succeeded or the ones after it.

    Returns {"adapter": Path, "merged": Path | None, "gguf": Path | None,
    "model_card": Path | None} for the dashboard/summary stage.
    """
    run_dir = Path(run_dir)
    adapter_path = Path(adapter_path)
    local_dir = Path(export_cfg.local_save_dir) if export_cfg.save_local else None
    if local_dir:
        local_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Path | None] = {"adapter": adapter_path, "merged": None, "gguf": None, "model_card": None}

    if export_cfg.save_local:
        try:
            dest = local_dir / "adapter"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(adapter_path, dest)
            display.success(f"Adapter (local): {dest}")
        except Exception as e:
            display.warning(f"Could not copy adapter to local save dir: {e}")

    if export_cfg.export_merged:
        try:
            merged_path = export_merged(adapter_path, run_dir)
            results["merged"] = merged_path
            if export_cfg.save_local:
                dest = local_dir / "merged_weights"
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(merged_path, dest)
                print(f"Merged (local): {dest}")
        except Exception as e:
            display.warning(f"Merged-weights export failed: {e}")

    if export_cfg.export_gguf:
        try:
            gguf_path = export_gguf(adapter_path, run_dir, quantization=export_cfg.gguf_quantize)
            results["gguf"] = gguf_path
            if export_cfg.save_local:
                dest = local_dir / gguf_path.name
                shutil.copy2(gguf_path, dest)
                print(f"GGUF (local): {dest}")
        except Exception as e:
            display.warning(f"GGUF export failed: {e}")

    push_to_hub = bool(export_cfg.hf_token and export_cfg.hf_repo_id)
    if push_to_hub:
        try:
            import huggingface_hub
            huggingface_hub.login(token=export_cfg.hf_token, add_to_git_credential=False)
        except Exception as e:
            display.warning(f"HF Hub login failed ({e}) — model card will save locally only.")
            push_to_hub = False

    try:
        card_path = write_model_card(
            recipe=recipe,
            run_dir=run_dir,
            push_to_hub=push_to_hub,
            repo_id=export_cfg.hf_repo_id,
        )
        results["model_card"] = card_path
    except Exception as e:
        display.warning(f"Model card generation failed: {e}")

    return results
