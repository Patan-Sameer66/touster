from __future__ import annotations

"""Upfront cost + time estimate shown before committing any compute."""

from touster.config import DatasetConfig, HardwareConfig, LoopConfig, RecipeConfig
from touster.console import console
from rich.panel import Panel
from rich.table import Table


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
    """Print the upfront cost/time estimate panel before any GPU compute."""
    est = _compute(hw, ds_cfg, recipe, loop_cfg, estimated_tps)

    table = Table.grid(padding=(0, 2))
    table.add_column(style="touster.dim")
    table.add_column()

    table.add_row("Loop trials", str(loop_cfg.max_trials))
    table.add_row("Steps / trial", str(loop_cfg.trial_max_steps))
    table.add_row("Final train steps", str(recipe.max_steps))
    table.add_row("Dataset samples", str(ds_cfg.num_samples))
    table.add_row("Est. loop time", _fmt_time(est["loop_secs"]))
    table.add_row("Est. final train time", _fmt_time(est["final_secs"]))
    table.add_row("Est. total time", f"[bold]{_fmt_time(est['total_secs'])}[/bold]")
    cost_str = "$0 (local)" if est["api_cost"] == 0 else f"~${est['api_cost']:.2f}"
    table.add_row("Est. API cost", f"[bold]{cost_str}[/bold]")

    console.print(
        Panel(
            table,
            title="[touster.brand]Upfront estimate[/touster.brand]",
            border_style="touster.dim",
        )
    )


def _compute(
    hw: HardwareConfig,
    ds_cfg: DatasetConfig,
    recipe: RecipeConfig,
    loop_cfg: LoopConfig,
    estimated_tps: float,
) -> dict:
    tps = max(estimated_tps, 1.0)
    tokens_per_step = recipe.batch_size * 512  # approx max_length

    # Training time estimate: steps * tokens / (tps * batch_overhead)
    trial_secs = (loop_cfg.trial_max_steps * tokens_per_step) / (tps * recipe.batch_size) if tps else 300
    loop_secs = trial_secs * loop_cfg.max_trials
    final_secs = (recipe.max_steps * tokens_per_step) / (tps * recipe.batch_size) if tps else 600

    # API cost (dataset generation only — training is local)
    api_tokens = 0
    if ds_cfg.mode in (0, 1):
        api_tokens += ds_cfg.num_samples * _AVG_SAMPLE_CHARS / _CHARS_PER_TOKEN
    # Judge calls
    api_tokens += loop_cfg.judge_top_k * loop_cfg.judge_prompts * 600 / _CHARS_PER_TOKEN
    api_cost = (api_tokens / 1000) * _API_COST_PER_1K_TOKENS if hw.platform != "cpu" else 0.0

    return {
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
