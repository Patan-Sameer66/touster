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
    best_recipe = recipe  # the recipe that produced best_bpb (used for final run)
    best_bpb = ckpt.best_bpb if ckpt else float("inf")
    best_trial_id = ckpt.best_trial_id if ckpt else -1
    start_trial = ckpt.current_trial if ckpt else 0

    if ckpt:
        print_warning(f"Resuming from trial {start_trial}, best bpb={best_bpb:.4f}")

    program_md = program_path.read_text()
    last_bpb = best_bpb
    # Track which LoRA structure is currently loaded so we reload only when it
    # actually changes — and never run a recipe against a mismatched adapter.
    loaded_structure = _structure(current_recipe)

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
            # Propose a change BRANCHING FROM THE BEST recipe so far — not the
            # last (possibly worse) one. This prevents a single bad trial from
            # dragging the whole search downward (lr-halving death spiral).
            diff: dict = {}
            if trial_id > 0:
                if client and loop_cfg.use_llm_proposer:
                    diff = propose_llm(client, best_recipe, program_md, last_bpb, best_bpb, trial_id)
                else:
                    diff = propose_heuristic(best_recipe, trial_id, last_bpb, best_bpb)
                try:
                    current_recipe = best_recipe.apply_diff(diff)
                except ValueError as e:
                    print_warning(f"Invalid proposal {diff}: {e}. Skipping.")
                    diff = {}
                    current_recipe = best_recipe
                # Skip no-op proposals that would waste a full trial
                if diff and all(getattr(best_recipe, k) == getattr(current_recipe, k)
                                for k in diff):
                    print_warning(f"Trial {trial_id}: proposal {diff} is a no-op. Skipping.")
                    progress.advance(task)
                    continue

            # Reload only when the LoRA structure actually differs from what is
            # loaded — correct regardless of which recipe we branched from.
            if _structure(current_recipe) != loaded_structure:
                reload_ok = _reload_backend(backend, recipe.base_model, current_recipe)
                if reload_ok:
                    loaded_structure = _structure(current_recipe)
                else:
                    # Structural reload is unrecoverable for this trial (e.g. an
                    # OOM at a bigger rank) — restore the last-known-good
                    # structure so subsequent trials aren't dragged down too,
                    # and count this trial as failed rather than crashing.
                    print_warning(
                        f"Trial {trial_id}: could not load structure {_structure(current_recipe)}, "
                        "reverting to last-known-good and marking trial failed."
                    )
                    if not _reload_backend(backend, recipe.base_model, best_recipe):
                        print_warning(
                            "Backend is unrecoverable — stopping the loop early "
                            "and falling back to the default recipe."
                        )
                        break
                    loaded_structure = _structure(best_recipe)
                    last_bpb = float("inf")
                    progress.advance(task)
                    continue

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
                best_recipe = current_recipe  # snapshot the winning recipe

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

    try:
        backend.unload()
    except Exception:
        pass

    if best_trial_id == -1:
        # Every trial failed (training crash, eval crash, or unrecoverable
        # reload) and there's no known-solvable fix left to try here — fall
        # back to the caller-supplied default recipe rather than crashing
        # the notebook. The final run below still trains a real adapter.
        print_warning(
            "All trials failed to produce a valid adapter — falling back to "
            "the default recipe for the final training run."
        )
        best_recipe = recipe

    # LLM-judge top-k finalists
    experiments = load_experiments(run_dir)
    if client and experiments:
        _run_judge_top_k(experiments, recipe, backend, dataset_path, client, loop_cfg, run_dir)

    # Load winning recipe for final run
    if best_trial_id == -1:
        print_success("Using fallback default recipe (no trial improved on it).")
    else:
        print_success(f"Best trial: {best_trial_id} (bpb={best_bpb:.4f})")
    console.print(
        f"\n[touster.brand]Running final full training with winning recipe[/touster.brand] "
        f"[touster.dim]lr={best_recipe.learning_rate:.2e} rank={best_recipe.lora_rank} "
        f"scheduler={best_recipe.scheduler}[/touster.dim]"
    )

    # Reload backend for final run — always the BEST recipe, never the last trial's
    backend = get_backend(hw)
    backend.load_model(
        model_id=recipe.base_model,
        lora_rank=best_recipe.lora_rank,
        lora_alpha=best_recipe.lora_alpha,
        target_modules=best_recipe.target_modules,
    )
    final_adapter = run_final(best_recipe, backend, dataset_path, run_dir)
    backend.unload()

    return best_recipe, final_adapter


def _structure(r) -> tuple:
    """The LoRA structural identity that requires a model reload when changed."""
    return (r.lora_rank, r.lora_alpha, tuple(r.target_modules))


def _reload_backend(backend, base_model: str, recipe: RecipeConfig) -> bool:
    """Unload and reload backend with recipe's LoRA structure. Returns False on failure."""
    try:
        backend.unload()
        backend.load_model(
            model_id=base_model,
            lora_rank=recipe.lora_rank,
            lora_alpha=recipe.lora_alpha,
            target_modules=recipe.target_modules,
        )
        return True
    except Exception as e:
        print_warning(f"Backend reload failed: {e}")
        return False


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
        if not trial_adapter.exists():
            console.print(
                f"  [touster.warning]Trial {exp.trial_id}: adapter missing, skipping judge[/touster.warning]"
            )
            continue
        be = get_backend(hw)
        try:
            # PeftModel.from_pretrained reads the adapter's own rank/alpha from
            # adapter_config.json, so the structural args here are placeholders.
            be.load_model(
                recipe.base_model,
                recipe.lora_rank,
                recipe.lora_alpha,
                list(recipe.target_modules),
            )
            try:
                from peft import PeftModel
                from transformers import AutoModelForCausalLM
                base = AutoModelForCausalLM.from_pretrained(recipe.base_model)
                be._model = PeftModel.from_pretrained(base, str(trial_adapter))
            except Exception as e:
                # Never score the bare base model as if it were the adapter
                console.print(
                    f"  [touster.warning]Trial {exp.trial_id}: adapter load failed ({e}), skipping[/touster.warning]"
                )
                continue
            score = eval_llm_judge(be, client, dataset_path, n_prompts=loop_cfg.judge_prompts)
            console.print(f"  Trial {exp.trial_id}: bpb={exp.eval_bpb:.4f} judge={score:.1f}/10")
        finally:
            be.unload()
