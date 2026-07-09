"""Upfront cost + time estimate for the tuning stage, shown before committing
any GPU compute. Dataset-generation cost is a separate estimate
(touster.dataset.modes.estimate_datagen_cost) — this one covers the
search loop + final training run only. See overview.md stage 2."""

from __future__ import annotations

from touster import display
from touster.config import DatasetConfig, HardwareConfig, LoopConfig, RecipeConfig

_API_COST_PER_1K_TOKENS = 0.0001  # gpt-4o-mini ballpark
_CHARS_PER_TOKEN = 3.8
_AVG_SAMPLE_CHARS = 500


def estimate_and_print(
    hw: HardwareConfig,
    ds_cfg: DatasetConfig,
    recipe: RecipeConfig,
    loop_cfg: LoopConfig,
    estimated_tps: float,
) -> None:
    """Print the upfront tuning-stage cost/time estimate before any GPU compute."""
    est = _compute(hw, ds_cfg, recipe, loop_cfg, estimated_tps)
    cost_str = "$0 (local)" if est["api_cost"] == 0 else f"~${est['api_cost']:.2f}"

    if est["time_reliable"]:
        rows = [
            ["Loop trials", str(loop_cfg.max_trials)],
            ["Steps / trial", str(loop_cfg.trial_max_steps)],
            ["Final train steps", str(recipe.max_steps)],
            ["Est. loop time", _fmt_time(est["loop_secs"])],
            ["Est. final train time", _fmt_time(est["final_secs"])],
            ["Est. total time", _fmt_time(est["total_secs"])],
            ["Est. judge API cost", cost_str],
        ]
    else:
        # No GPU bandwidth figure to estimate from (CPU/unknown platform) —
        # a formula built for GPU throughput would print a nonsense ETA
        # (hours where it should be minutes). Say so plainly instead.
        rows = [
            ["Loop trials", str(loop_cfg.max_trials)],
            ["Steps / trial", str(loop_cfg.trial_max_steps)],
            ["Final train steps", str(recipe.max_steps)],
            ["Est. time", "not reliable on CPU — budget 30-60 min for a small model / short trial budget"],
            ["Est. judge API cost", cost_str],
        ]
    display.table(["", ""], rows, title="Tuning stage — upfront estimate")


def _compute(
    hw: HardwareConfig,
    ds_cfg: DatasetConfig,
    recipe: RecipeConfig,
    loop_cfg: LoopConfig,
    estimated_tps: float,
) -> dict:
    # API cost — judge calls on top-k survivors only (dataset-gen cost is separate)
    api_tokens = loop_cfg.judge_top_k * loop_cfg.judge_prompts * 600 / _CHARS_PER_TOKEN
    api_cost = (api_tokens / 1000) * _API_COST_PER_1K_TOKENS if hw.platform != "cpu" else 0.0

    if estimated_tps <= 0:
        # Formula below assumes GPU-bandwidth-derived throughput; with no
        # reliable tps figure (CPU/unknown platform) it produces a wildly
        # wrong ETA (hours where it should be minutes) — don't print one.
        return {"time_reliable": False, "loop_secs": 0.0, "final_secs": 0.0, "total_secs": 0.0, "api_cost": api_cost}

    tokens_per_step = recipe.batch_size * 512  # approx max_length

    # Training time estimate: steps * tokens / (tps * batch_overhead)
    trial_secs = (loop_cfg.trial_max_steps * tokens_per_step) / (estimated_tps * recipe.batch_size)
    loop_secs = trial_secs * loop_cfg.max_trials
    final_secs = (recipe.max_steps * tokens_per_step) / (estimated_tps * recipe.batch_size)

    return {
        "time_reliable": True,
        "loop_secs": loop_secs,
        "final_secs": final_secs,
        "total_secs": loop_secs + final_secs,
        "api_cost": api_cost,
    }


def _fmt_time(secs: float) -> str:
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    return f"{secs // 3600}h {(secs % 3600) // 60}m"
