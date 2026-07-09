"""Tuning loop — TPE search (Optuna) + fixed-budget trials.

Trial 0 is always the default recipe (the branch point every later trial
proposes a change from). Every trial after that is proposed by Optuna's TPE
sampler via touster/tuning/optuna_search.py — see that module's docstring
and research.md section 3 for why the old freeform-LLM proposer is gone.

Resilience carried forward from the pre-rewrite crash fixes: a structural
LoRA reload failure reverts to the last-known-good structure instead of
crashing the loop, and if every trial fails, this falls back to the
caller's default recipe and still trains a real final adapter — it never
raises and kills the notebook.
"""

from __future__ import annotations

import math
from pathlib import Path

import optuna

from touster import display
from touster.config import LoopConfig, RecipeConfig
from touster.llm.client import LLMClient
from touster.state import ExperimentRecord, RunState, append_experiment, load_experiments, load_state, save_state
from touster.tuning.backends.base import TrainerBackend
from touster.tuning.checkpoint import LoopCheckpoint, load_checkpoint, save_checkpoint
from touster.tuning.optuna_search import close_study, create_study, narrow_search_space_with_llm, suggest_recipe_diff
from touster.tuning.trial import run_trial


def run_loop(
    recipe: RecipeConfig,
    loop_cfg: LoopConfig,
    dataset_path: Path,
    run_dir: Path,
    client: LLMClient | None = None,
) -> tuple[RecipeConfig, Path]:
    """
    Self-improvement loop. Returns (winning_recipe, final_adapter_path).
    """
    from touster.hardware.detect import detect_hardware
    from touster.tuning.backends.factory import get_backend
    from touster.tuning.final import run_final

    run_dir = Path(run_dir)

    hw = detect_hardware()
    backend = get_backend(hw)
    backend.load_model(
        model_id=recipe.base_model,
        lora_rank=recipe.lora_rank,
        lora_alpha=recipe.lora_alpha,
        target_modules=recipe.target_modules,
    )

    run_state = load_state(run_dir) or RunState(
        run_dir=run_dir, base_model=recipe.base_model, dataset_path=str(dataset_path),
    )
    run_state.phase = "loop"
    save_state(run_state)

    # Resume from checkpoint if present
    ckpt = load_checkpoint(run_dir)
    best_bpb = ckpt.best_bpb if ckpt else float("inf")
    best_trial_id = ckpt.best_trial_id if ckpt else -1
    start_trial = ckpt.current_trial if ckpt else 0

    # best_recipe_diff is the diff belonging to best_trial_id specifically —
    # NOT necessarily the last trial's diff (that trial might have been
    # discarded). Reconstruct best_recipe from it so a resumed run trains
    # the actual winning recipe, not silently fall back to the default.
    best_recipe_diff: dict = ckpt.best_recipe_diff if ckpt else {}
    best_recipe = recipe
    if best_trial_id != -1 and best_recipe_diff:
        try:
            best_recipe = recipe.apply_diff(best_recipe_diff)
        except ValueError as e:
            display.warning(f"Checkpoint's best_recipe_diff {best_recipe_diff} failed to apply ({e}) — using default recipe.")
            best_recipe = recipe

    if ckpt:
        display.warning(f"Resuming from trial {start_trial}, best bpb={best_bpb:.4f}")

    if hw.platform == "mlx":
        # mlx_lm doesn't expose per-token log-probs (see mlx_backend.eval_loss),
        # so bpb is always NaN on this platform and no trial can ever beat the
        # default — the search loop always falls back. Said here explicitly
        # rather than leaving it as a silent -1 best_trial_id at the end.
        display.warning(
            "MLX backend can't compute eval bpb — the search loop can't rank "
            "trials on this platform and will always fall back to the default recipe."
        )

    # Optuna's SQLite storage makes the study itself resumable across
    # notebook restarts — no need to manually replay past trials into it.
    study = create_study(run_dir, seed=0)
    search_space = narrow_search_space_with_llm(client if loop_cfg.use_llm_prior else None, recipe)

    # Track which LoRA structure is currently loaded so we reload only when it
    # actually changes — and never run a recipe against a mismatched adapter.
    loaded_structure = _structure(recipe)

    print(f"Tuning loop — {loop_cfg.max_trials} trials max")

    for trial_id in range(start_trial, loop_cfg.max_trials):
        optuna_trial: optuna.trial.Trial | None = None
        if trial_id == 0:
            # Trial 0 is always the unmodified default recipe — the baseline
            # every later trial branches from.
            diff: dict = {}
            current_recipe = best_recipe
        else:
            optuna_trial = study.ask()
            diff = suggest_recipe_diff(optuna_trial, search_space)
            try:
                # Branch from the fixed trial-0 default, not the current
                # best-so-far. suggest_recipe_diff always returns every
                # tunable knob (an absolute config, not a partial delta), so
                # this is provably identical to branching from best_recipe
                # today — but doing it explicitly means it stays correct if
                # a future proposer ever returns a partial diff instead.
                current_recipe = recipe.apply_diff(diff)
            except ValueError as e:
                display.warning(f"Invalid proposal {diff}: {e}. Skipping.")
                study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                _log_failed_trial(run_dir, trial_id, diff, best_trial_id, best_bpb, best_recipe_diff)
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
                display.warning(
                    f"Trial {trial_id}: could not load structure {_structure(current_recipe)}, "
                    "reverting to last-known-good and marking trial failed."
                )
                if optuna_trial is not None:
                    study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                _log_failed_trial(run_dir, trial_id, diff, best_trial_id, best_bpb, best_recipe_diff)
                if not _reload_backend(backend, recipe.base_model, best_recipe):
                    display.warning(
                        "Backend is unrecoverable — stopping the loop early "
                        "and falling back to the default recipe."
                    )
                    break
                loaded_structure = _structure(best_recipe)
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

        if optuna_trial is not None:
            if math.isfinite(bpb):
                study.tell(optuna_trial, bpb)
            else:
                # A failed trial teaches TPE nothing useful about the loss
                # surface — mark it FAIL instead of feeding it inf as a value.
                study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)

        if bpb < best_bpb:
            best_bpb = bpb
            best_trial_id = trial_id
            best_recipe = current_recipe  # snapshot the winning recipe
            best_recipe_diff = diff       # the diff THIS specific trial used — not just "last trial run"

        # Checkpoint loop state — best_recipe_diff, not this trial's diff,
        # so a resume reconstructs the actual best recipe even if the most
        # recent trial before a disconnect was a discarded one.
        save_checkpoint(
            run_dir,
            LoopCheckpoint(
                current_trial=trial_id + 1,
                best_trial_id=best_trial_id,
                best_bpb=best_bpb,
                best_recipe_diff=best_recipe_diff,
                total_trials_run=trial_id + 1,
            ),
        )
        print(f"Trial {trial_id + 1}/{loop_cfg.max_trials} complete")

    close_study(study)  # release the SQLite handle — see optuna_search.close_study

    try:
        backend.unload()
    except Exception as e:
        display.warning(f"Backend unload failed (non-fatal): {e}")

    run_state.phase = "final"
    run_state.best_trial_id = best_trial_id
    run_state.best_bpb = best_bpb
    run_state.total_trials = min(loop_cfg.max_trials, start_trial + 1)
    save_state(run_state)

    if best_trial_id == -1:
        # Every trial failed (training crash, eval crash, or unrecoverable
        # reload) and there's no known-solvable fix left to try here — fall
        # back to the caller-supplied default recipe rather than crashing
        # the notebook. The final run below still trains a real adapter.
        display.warning(
            "All trials failed to produce a valid adapter — falling back to "
            "the default recipe for the final training run."
        )
        best_recipe = recipe

    # LLM-judge top-k finalists — never let a judge-pass failure (OOM, HF
    # Hub error, etc.) take down a run that already found a winner.
    experiments = load_experiments(run_dir)
    if client and experiments:
        try:
            _run_judge_top_k(experiments, recipe, backend, dataset_path, client, loop_cfg, run_dir)
        except Exception as e:
            display.warning(f"LLM-judge pass failed ({e}) — continuing to final training without it.")

    # Load winning recipe for final run
    if best_trial_id == -1:
        display.success("Using fallback default recipe (no trial improved on it).")
    else:
        display.success(f"Best trial: {best_trial_id} (bpb={best_bpb:.4f})")
    print(
        f"Running final full training with winning recipe "
        f"lr={best_recipe.learning_rate:.2e} rank={best_recipe.lora_rank} "
        f"scheduler={best_recipe.scheduler}"
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
    try:
        backend.unload()
    except Exception as e:
        display.warning(f"Backend unload failed (non-fatal): {e}")

    run_state.phase = "done"
    run_state.final_adapter_path = str(final_adapter)
    save_state(run_state)

    return best_recipe, final_adapter


def _structure(r) -> tuple:
    """The LoRA structural identity that requires a model reload when changed."""
    return (r.lora_rank, r.lora_alpha, tuple(r.target_modules))


def _reload_backend(backend: TrainerBackend, base_model: str, recipe: RecipeConfig) -> bool:
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
        display.warning(f"Backend reload failed: {e}")
        return False


def _log_failed_trial(
    run_dir: Path,
    trial_id: int,
    diff: dict,
    best_trial_id: int,
    best_bpb: float,
    best_recipe_diff: dict,
) -> None:
    """Checkpoint + log a trial that never reached run_trial (invalid proposal
    or unrecoverable structural reload). Without this, these trials leave no
    record — a resume can't tell "never ran" from "ran and failed," and the
    checkpoint's current_trial would be one step behind on these paths."""
    save_checkpoint(
        run_dir,
        LoopCheckpoint(
            current_trial=trial_id + 1,
            best_trial_id=best_trial_id,
            best_bpb=best_bpb,
            best_recipe_diff=best_recipe_diff,
            total_trials_run=trial_id + 1,
        ),
    )
    append_experiment(
        run_dir,
        ExperimentRecord(
            trial_id=trial_id, recipe_diff=diff, eval_bpb=float("inf"),
            judge_score=None, kept=False, wall_clock_secs=0.0, steps=0,
        ),
    )


def _run_judge_top_k(
    experiments: list,
    recipe: RecipeConfig,
    backend: TrainerBackend,
    dataset_path: Path,
    client: LLMClient,
    loop_cfg: LoopConfig,
    run_dir: Path,
) -> None:
    """Run LLM-as-judge on top-k trials, loading each trial's saved adapter."""
    from touster.tuning.eval import eval_llm_judge
    from touster.hardware.detect import detect_hardware
    from touster.tuning.backends.factory import get_backend
    from touster.tuning.checkpoint import checkpoint_path

    kept = sorted([e for e in experiments if e.kept], key=lambda e: e.eval_bpb)
    top_k = kept[: loop_cfg.judge_top_k]
    if not top_k:
        return

    hw = detect_hardware()
    if hw.platform == "mlx":
        display.warning("LLM-judge pass needs PEFT/transformers, not supported on the MLX backend — skipping.")
        return

    print(f"Running LLM-judge on top-{len(top_k)} trials...")

    for exp in top_k:
        trial_adapter = checkpoint_path(run_dir, exp.trial_id)
        if not trial_adapter.exists():
            display.warning(f"Trial {exp.trial_id}: adapter missing, skipping judge")
            continue
        be = get_backend(hw)
        try:
            # Load the base model once and attach the trial's own trained
            # adapter directly — no separate be.load_model() call first,
            # which would load a second full base-model copy just to
            # immediately throw it away with a fresh, untrained LoRA attached.
            # PeftModel.from_pretrained reads the adapter's own rank/alpha
            # from adapter_config.json, so recipe's LoRA fields aren't needed.
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            base = AutoModelForCausalLM.from_pretrained(recipe.base_model)
            be._tokenizer = AutoTokenizer.from_pretrained(recipe.base_model)
            if be._tokenizer.pad_token is None:
                be._tokenizer.pad_token = be._tokenizer.eos_token
            be._model = PeftModel.from_pretrained(base, str(trial_adapter))
        except Exception as e:
            # Never score the bare base model as if it were the adapter
            display.warning(f"Trial {exp.trial_id}: adapter load failed ({e}), skipping")
            continue
        try:
            score = eval_llm_judge(be, client, dataset_path, n_prompts=loop_cfg.judge_prompts)
            print(f"  Trial {exp.trial_id}: bpb={exp.eval_bpb:.4f} judge={score:.1f}/10")
        finally:
            be.unload()
