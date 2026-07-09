"""Run one fixed-budget trial and return its eval score."""

from __future__ import annotations

import time
from pathlib import Path

from touster import display
from touster.config import LoopConfig, RecipeConfig
from touster.state import ExperimentRecord, append_experiment
from touster.tuning.backends.base import TrainerBackend
from touster.tuning.checkpoint import checkpoint_path


def run_trial(
    trial_id: int,
    recipe: RecipeConfig,
    backend: TrainerBackend,
    dataset_path: Path,
    loop_cfg: LoopConfig,
    run_dir: Path,
    current_best_bpb: float,
    recipe_diff: dict | None = None,
) -> tuple[float, Path | None]:
    """
    Run one short trial with fixed budget.
    Returns (eval_bpb, adapter_path_if_kept).
    Saves ExperimentRecord to experiments.jsonl.
    """
    from touster.tuning.eval import eval_bpb

    print(
        f"  Trial {trial_id} lr={recipe.learning_rate:.2e} rank={recipe.lora_rank} "
        f"steps={loop_cfg.trial_max_steps}"
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
        import traceback
        display.error(f"Trial {trial_id} training failed: {e}")
        print(traceback.format_exc())
        return float("inf"), None

    elapsed = time.time() - start

    try:
        bpb = eval_bpb(backend, dataset_path)
    except Exception as e:
        display.warning(f"Eval failed: {e}")
        bpb = float("inf")

    kept = bpb < current_best_bpb
    adapter_path: Path | None = None

    if kept:
        adapter_path = checkpoint_path(run_dir, trial_id)
        try:
            backend.save_adapter(adapter_path)
        except Exception as e:
            display.warning(f"Could not save adapter: {e}")

    record = ExperimentRecord(
        trial_id=trial_id,
        recipe_diff=recipe_diff or {},
        eval_bpb=bpb,
        judge_score=None,
        kept=kept,
        wall_clock_secs=elapsed,
        steps=train_result.get("steps", 0),
    )
    append_experiment(run_dir, record)

    status = "kept" if kept else "discarded"
    print(f"    bpb={bpb:.4f} {status} ({elapsed:.0f}s)")

    return bpb, adapter_path
