"""Train winning recipe to full completion."""

from __future__ import annotations

from pathlib import Path

from touster import display
from touster.config import RecipeConfig
from touster.tuning.backends.base import TrainerBackend


def run_final(
    recipe: RecipeConfig,
    backend: TrainerBackend,
    dataset_path: Path,
    run_dir: Path,
) -> Path:
    """
    Train the winning recipe for the full duration (recipe.num_epochs / recipe.max_steps).
    Saves adapter to run_dir/final_adapter/.
    Returns path to saved adapter.
    """
    output_dir = run_dir / "final_adapter"
    print(
        f"Final run: lr={recipe.learning_rate:.2e} rank={recipe.lora_rank} "
        f"epochs={recipe.num_epochs} steps={recipe.max_steps}"
    )
    result = backend.train_steps(
        dataset_path=dataset_path,
        max_steps=recipe.max_steps,
        batch_size=recipe.batch_size,
        gradient_accumulation_steps=recipe.gradient_accumulation_steps,
        learning_rate=recipe.learning_rate,
        warmup_steps=recipe.warmup_steps,
        scheduler=recipe.scheduler,
    )
    backend.save_adapter(output_dir)
    display.success(
        f"Final training done — {result['steps']} steps, "
        f"loss={result.get('train_loss', float('nan')):.4f}"
    )
    return output_dir
