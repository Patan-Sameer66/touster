from __future__ import annotations

"""Autoresearch-style self-improvement loop."""

from pathlib import Path

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from touster.config import LoopConfig, RecipeConfig
from touster.console import console, print_success, print_warning
from touster.state import load_experiments
from touster.tuning.checkpoint import LoopCheckpoint, load_checkpoint, save_checkpoint
from touster.tuning.trial import run_trial


_PROGRAM_MD_TEMPLATE = """\
# Touster fine-tuning program

## Goal
Minimize eval bpb (bits-per-byte) on the held-out split.

## Strategy
- Start conservative: low learning rate (2e-4), moderate rank (16).
- If bpb improves: try increasing rank or learning rate slightly.
- If bpb diverges (> 10% worse than best): halve learning rate immediately.
- Prefer cosine scheduler for smooth decay.
- alpha should equal rank (standard LoRA practice).
- Do not change base_model, dataset, or eval harness.

## Constraints
Only modify: learning_rate, lora_rank, lora_alpha, target_modules,
warmup_steps, num_epochs, max_steps, batch_size,
gradient_accumulation_steps, scheduler.
"""


def run_loop(
    recipe: RecipeConfig,
    loop_cfg: LoopConfig,
    dataset_path: Path,
    run_dir: Path,
    client=None,
) -> tuple[RecipeConfig, Path]:
    """
    Self-improvement loop.
    Returns (winning_recipe, final_adapter_path).
    """
    from touster.hardware.detect import detect_hardware
    from touster.tuning.backends.factory import get_backend
    from touster.tuning.agent import propose_heuristic, propose_llm
    from touster.tuning.final import run_final

    # Write program.md if not present
    program_path = run_dir / "program.md"
    if not program_path.exists():
        program_path.write_text(_PROGRAM_MD_TEMPLATE)

    # Detect backend
    hw = detect_hardware()
    backend = get_backend(hw)
    backend.load_model(
        model_id=recipe.base_model,
        lora_rank=recipe.lora_rank,
        lora_alpha=recipe.lora_alpha,
        target_modules=recipe.target_modules,
    )

    # Resume from checkpoint if present
    ckpt = load_checkpoint(run_dir)
    current_recipe = recipe
    best_bpb = ckpt.best_bpb if ckpt else float("inf")
    best_trial_id = ckpt.best_trial_id if ckpt else -1
    start_trial = ckpt.current_trial if ckpt else 0

    if ckpt:
        print_warning(f"Resuming from trial {start_trial}, best bpb={best_bpb:.4f}")

    program_md = program_path.read_text()
    last_bpb = best_bpb

    console.rule(
        f"[touster.brand]Self-improvement loop[/touster.brand] — "
        f"[touster.dim]{loop_cfg.max_trials} trials max[/touster.dim]",
        style="touster.dim",
    )

    with Progress(
        TextColumn("[touster.step]Trial {task.completed}/{task.total}[/touster.step]"),
        BarColumn(bar_width=30, style="touster.dim", complete_style="touster.brand"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("loop", total=loop_cfg.max_trials)
        progress.advance(task, start_trial)

        for trial_id in range(start_trial, loop_cfg.max_trials):
            # Propose recipe change
            diff: dict = {}
            if trial_id > 0:
                if client and loop_cfg.use_llm_proposer:
                    diff = propose_llm(client, current_recipe, program_md, last_bpb, best_bpb, trial_id)
                else:
                    diff = propose_heuristic(current_recipe, trial_id, last_bpb, best_bpb)
                try:
                    current_recipe = current_recipe.apply_diff(diff)
                except ValueError as e:
                    print_warning(f"Invalid proposal {diff}: {e}. Skipping.")
                    diff = {}

            # Reload model with potentially new LoRA config
            if diff:
                backend.unload()
                backend.load_model(
                    model_id=recipe.base_model,
                    lora_rank=current_recipe.lora_rank,
                    lora_alpha=current_recipe.lora_alpha,
                    target_modules=current_recipe.target_modules,
                )

            bpb, adapter_path = run_trial(
                trial_id=trial_id,
                recipe=current_recipe,
                backend=backend,
                dataset_path=dataset_path,
                loop_cfg=loop_cfg,
                run_dir=run_dir,
                current_best_bpb=best_bpb,
                recipe_diff=diff,
            )

            last_bpb = bpb
            if bpb < best_bpb:
                best_bpb = bpb
                best_trial_id = trial_id

            # Checkpoint loop state
            save_checkpoint(
                run_dir,
                LoopCheckpoint(
                    current_trial=trial_id + 1,
                    best_trial_id=best_trial_id,
                    best_bpb=best_bpb,
                    best_recipe_diff=diff,
                    total_trials_run=trial_id + 1,
                ),
            )
            progress.advance(task)

    backend.unload()

    # LLM-judge top-k finalists
    experiments = load_experiments(run_dir)
    if client and experiments:
        _run_judge_top_k(experiments, recipe, backend, dataset_path, client, loop_cfg, run_dir)

    # Load winning recipe for final run
    print_success(f"Best trial: {best_trial_id} (bpb={best_bpb:.4f})")
    console.print("\n[touster.brand]Running final full training with winning recipe…[/touster.brand]")

    # Reload backend for final run
    backend = get_backend(hw)
    backend.load_model(
        model_id=recipe.base_model,
        lora_rank=current_recipe.lora_rank,
        lora_alpha=current_recipe.lora_alpha,
        target_modules=current_recipe.target_modules,
    )
    final_adapter = run_final(current_recipe, backend, dataset_path, run_dir)
    backend.unload()

    return current_recipe, final_adapter


def _run_judge_top_k(experiments, recipe, backend, dataset_path, client, loop_cfg, run_dir):
    """Run LLM-as-judge on top-k trials, loading each trial's saved adapter."""
    from touster.tuning.eval import eval_llm_judge
    from touster.hardware.detect import detect_hardware
    from touster.tuning.backends.factory import get_backend
    from touster.tuning.checkpoint import checkpoint_path

    kept = sorted([e for e in experiments if e.kept], key=lambda e: e.eval_bpb)
    top_k = kept[: loop_cfg.judge_top_k]
    if not top_k:
        return

    console.print(f"\n[touster.dim]Running LLM-judge on top-{len(top_k)} trials…[/touster.dim]")
    hw = detect_hardware()

    for exp in top_k:
        trial_adapter = checkpoint_path(run_dir, exp.trial_id)
        be = get_backend(hw)
        be.load_model(
            recipe.base_model,
            recipe.lora_rank,
            recipe.lora_alpha,
            list(recipe.target_modules),
        )
        # Load the per-trial saved adapter if it exists
        if trial_adapter.exists():
            try:
                from peft import PeftModel
                from transformers import AutoModelForCausalLM
                base = AutoModelForCausalLM.from_pretrained(recipe.base_model)
                peft_m = PeftModel.from_pretrained(base, str(trial_adapter))
                be._model = peft_m
            except Exception as e:
                console.print(f"  [touster.warning]Could not load adapter for trial {exp.trial_id}: {e}[/touster.warning]")

        score = eval_llm_judge(be, client, dataset_path, n_prompts=loop_cfg.judge_prompts)
        console.print(f"  Trial {exp.trial_id}: bpb={exp.eval_bpb:.4f} judge={score:.1f}/10")
        be.unload()
