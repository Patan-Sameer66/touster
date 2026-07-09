"""Build the tuning-stage configs + print the cost estimate — everything the
Config cell needs for stage 4, in one call instead of inlining it in the
notebook. Mirrors touster.dataset.modes.prepare_dataset_stage."""

from __future__ import annotations

from pathlib import Path

from touster.config import ExportConfig, LoopConfig, RecipeConfig
from touster.hardware.detect import detect_hardware
from touster.hardware.estimate import estimate_tokens_per_second
from touster.tuning.estimate_cost import estimate_and_print


def prepare_tuning_stage(
    base_model: str,
    max_trials: int,
    trial_steps: int,
    final_steps: int,
    use_llm_prior: bool,
    judge_top_k: int,
    judge_prompts: int,
    ds_cfg,
    save_local: bool,
    local_save_dir: str,
    export_merged: bool,
    export_gguf: bool,
    gguf_quantize: str,
    hf_token: str,
    hf_repo_id: str,
) -> tuple[RecipeConfig, LoopConfig, ExportConfig]:
    """Build RecipeConfig/LoopConfig/ExportConfig from raw config-cell values
    and print the upfront cost/time estimate before any GPU compute starts.
    """
    recipe = RecipeConfig(base_model=base_model, max_steps=final_steps)
    loop_cfg = LoopConfig(
        trial_max_steps=trial_steps, max_trials=max_trials,
        judge_top_k=judge_top_k, judge_prompts=judge_prompts,
        use_llm_prior=use_llm_prior,
    )
    export_cfg = ExportConfig(
        save_local=save_local, local_save_dir=Path(local_save_dir),
        export_merged=export_merged, export_gguf=export_gguf, gguf_quantize=gguf_quantize,
        hf_token=hf_token, hf_repo_id=hf_repo_id,
    )

    hw = detect_hardware()
    tps = estimate_tokens_per_second(1.0, hw.gpu_bandwidth_gbps)  # rough 1B-param baseline
    estimate_and_print(hw, ds_cfg, recipe, loop_cfg, tps)

    return recipe, loop_cfg, export_cfg
