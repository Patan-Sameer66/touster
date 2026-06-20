from __future__ import annotations

"""Run one fixed-budget trial and return its eval score."""

import time
from pathlib import Path

from touster.config import LoopConfig, RecipeConfig
from touster.console import console
from touster.state import ExperimentRecord, append_experiment
from touster.tuning.checkpoint import checkpoint_path, save_checkpoint, LoopCheckpoint


def run_trial(
    trial_id: int,
    recipe: RecipeConfig,
    backend,
    dataset_path: Path,
    loop_cfg: LoopConfig,
    run_dir: Path,
    current_best_bpb: float,
) -> tuple[float, Path | None]:
    """
    Run one short trial with fixed budget.
    Returns (eval_bpb, adapter_path_if_kept).
    Saves ExperimentRecord to experiments.jsonl.
    """
    from touster.tuning.eval import eval_bpb

    console.print(
        f"  [touster.dim]Trial [bold]{trial_id}[/bold] "
        f"lr={recipe.learning_rate:.2e} rank={recipe.lora_rank} "
        f"steps={loop_cfg.trial_max_steps}[/touster.dim]"
    )
    start = time.time()

    try:
        train_result = backend.train_steps(
            dataset_path=dataset_path,
            max_steps=loop_cfg.trial_max_steps,
            batch_size=recipe.batch_size,
            gradient_accumulation_steps=recipe.gradient_accumulation_steps,
            learning_rate=recipe.learning_rate,
            warmup_steps=recipe.warmup_steps,
            scheduler=recipe.scheduler,
            wall_clock_limit_secs=loop_cfg.trial_wall_clock_secs,
        )
    except Exception as e:
        console.print(f"  [touster.error]Trial {trial_id} training failed: {e}[/touster.error]")
        return float("inf"), None

    elapsed = time.time() - start

    try:
        bpb = eval_bpb(backend, dataset_path)
    except Exception as e:
        console.print(f"  [touster.warning]Eval failed: {e}[/touster.warning]")
        bpb = float("inf")

    kept = bpb < current_best_bpb
    adapter_path: Path | None = None

    if kept:
        adapter_path = checkpoint_path(run_dir, trial_id)
        try:
            backend.save_adapter(adapter_path)
        except Exception as e:
            console.print(f"  [touster.warning]Could not save adapter: {e}[/touster.warning]")

    record = ExperimentRecord(
        trial_id=trial_id,
        recipe_diff={},
        eval_bpb=bpb,
        judge_score=None,
        kept=kept,
        wall_clock_secs=elapsed,
        steps=train_result.get("steps", 0),
    )
    append_experiment(run_dir, record)

    _status = "[touster.success]kept ✓[/touster.success]" if kept else "[touster.dim]discarded[/touster.dim]"
    console.print(f"    bpb={bpb:.4f} {_status} ({elapsed:.0f}s)")

    return bpb, adapter_path
