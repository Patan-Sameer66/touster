from __future__ import annotations

"""Touster CLI — orchestrates the 5-step fine-tuning pipeline."""

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from touster.console import console, print_error, print_step, print_success
from touster.config import DatasetConfig, LoopConfig, RecipeConfig, RunConfig
from touster.state import RunState, load_state, save_state

app = typer.Typer(
    name="touster",
    help="All-in-one LoRA fine-tuning pipeline. From zero dataset to exportable model.",
    add_completion=False,
    rich_markup_mode="rich",
)

TOTAL_STEPS = 5


def _abort(msg: str) -> None:
    print_error(msg)
    raise typer.Exit(1)


@app.command()
def main(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", "-d", help="Directory to store run artefacts."),
    ] = Path("runs/latest"),
    resume: Annotated[bool, typer.Option("--resume", help="Resume an interrupted run.")] = False,
    model: Annotated[str, typer.Option("--model", help="Base model id (HF or Ollama).")] = "",
    dataset_mode: Annotated[int, typer.Option("--dataset-mode", min=0, max=2)] = 0,
    dataset_path: Annotated[Optional[Path], typer.Option("--dataset-path")] = None,
    num_samples: Annotated[int, typer.Option("--num-samples")] = 200,
    max_trials: Annotated[int, typer.Option("--max-trials")] = 20,
    trial_steps: Annotated[int, typer.Option("--trial-steps")] = 200,
    ollama_port: Annotated[int, typer.Option("--ollama-port")] = 11434,
    api_key: Annotated[str, typer.Option("--api-key", envvar="OPENAI_API_KEY")] = "",
    api_base: Annotated[str, typer.Option("--api-base", envvar="OPENAI_API_BASE")] = "",
    skip_export: Annotated[bool, typer.Option("--skip-export")] = False,
) -> None:
    """
    Fine-tune an LLM end-to-end.

    [bold amber]Steps:[/bold amber]
      1. Hardware analysis — what can you actually train here?
      2. Dataset — generate, structure, or load.
      3. Validate + dry-run preview.
      4. Self-improvement loop -> best recipe -> final training run.
      5. Dashboard + export (GGUF / merged / model card).
    """
    console.print(
        "\n[touster.brand]🍞  Touster[/touster.brand] — fine-tuning for people who don't want to become fine-tuning experts.\n",
    )

    run_dir = run_dir.resolve()

    # ── Resume: reload existing state ────────────────────────────────────────
    state: RunState | None = None
    if resume:
        state = load_state(run_dir)
        if state is None:
            _abort(f"No run found at {run_dir}. Cannot resume.")
        console.print(f"[touster.warning]Resuming run from phase=[touster.code]{state.phase}[/touster.code][/touster.warning]\n")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Hardware analysis ─────────────────────────────────────────────
    if state is None or state.phase == "init":
        print_step(1, TOTAL_STEPS, "Hardware analysis")
        try:
            from touster.hardware.detect import detect_hardware
            from touster.hardware.report import print_hardware_report
            hw = detect_hardware()
            chosen_model = print_hardware_report(hw, suggested_model=model or None)
            if model:
                chosen_model = model
        except ImportError as e:
            print_warning = lambda m: console.print(f"[touster.warning]⚠  {m}[/touster.warning]")
            print_warning(f"Hardware module not yet installed ({e}). Using CPU defaults.")
            from touster.config import HardwareConfig
            hw = HardwareConfig(platform="cpu")
            chosen_model = model or "sshleifer/tiny-gpt2"

    # ── Step 2: Dataset ───────────────────────────────────────────────────────
    if state is None or state.phase in ("init", "dataset"):
        print_step(2, TOTAL_STEPS, "Dataset")
        try:
            from touster.dataset.modes import run_dataset_mode
            from touster.llm.factory import build_client

            llm_client = None
            if api_key or api_base:
                llm_client = build_client(api_key=api_key, api_base=api_base)
            elif dataset_mode in (0, 1):
                llm_client = build_client(ollama_port=ollama_port)

            ds_cfg = DatasetConfig(
                mode=dataset_mode,  # type: ignore[arg-type]
                num_samples=num_samples,
                dataset_path=dataset_path,
            )
            validated_path = run_dataset_mode(ds_cfg, run_dir, llm_client)
        except ImportError:
            console.print("[touster.dim]  (dataset module stub — building)[/touster.dim]")
            validated_path = dataset_path or run_dir / "dataset.jsonl"

    # ── Step 3: Preview ───────────────────────────────────────────────────────
    if state is None or state.phase in ("init", "dataset", "preview"):
        print_step(3, TOTAL_STEPS, "Dry-run preview")
        try:
            from touster.dataset.preview import print_preview
            recipe = RecipeConfig(base_model=chosen_model)
            print_preview(validated_path, recipe)
        except (ImportError, NameError):
            console.print("[touster.dim]  (preview stub — building)[/touster.dim]")
            recipe = RecipeConfig(base_model=model or "sshleifer/tiny-gpt2")

    # ── Step 4: Self-improvement loop ─────────────────────────────────────────
    if state is None or state.phase in ("init", "dataset", "preview", "loop"):
        print_step(4, TOTAL_STEPS, "Self-improvement loop → fine-tuning")
        try:
            from touster.tuning.loop import run_loop
            loop_cfg = LoopConfig(
                trial_max_steps=trial_steps,
                max_trials=max_trials,
            )
            best_recipe, adapter_path = run_loop(recipe, loop_cfg, validated_path, run_dir)
            print_success(f"Winning recipe trained. Adapter at [touster.code]{adapter_path}[/touster.code]")
        except ImportError:
            console.print("[touster.dim]  (tuning module stub — building)[/touster.dim]")
            adapter_path = run_dir / "adapter"
            best_recipe = recipe

    # ── Step 5: Dashboard + Export ────────────────────────────────────────────
    print_step(5, TOTAL_STEPS, "Dashboard & export")
    try:
        from touster.dashboard.app import launch_dashboard
        launch_dashboard(best_recipe.base_model, str(adapter_path), run_dir)
    except ImportError:
        console.print("[touster.dim]  (dashboard stub — building)[/touster.dim]")

    if not skip_export:
        try:
            from touster.export.gguf import export_gguf
            from touster.export.merge import export_merged
            from touster.export.modelcard import write_model_card
            gguf_path = export_gguf(adapter_path, run_dir)
            merged_path = export_merged(adapter_path, run_dir)
            write_model_card(best_recipe, run_dir)
            print_success(f"GGUF → [touster.code]{gguf_path}[/touster.code]")
            print_success(f"Merged → [touster.code]{merged_path}[/touster.code]")
        except ImportError:
            console.print("[touster.dim]  (export stub — building)[/touster.dim]")

    console.print("\n[touster.brand]🍞  Done. Your model is ready.[/touster.brand]\n")


if __name__ == "__main__":
    app()
