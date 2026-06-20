from __future__ import annotations

"""Train winning recipe to full completion."""

from pathlib import Path

from touster.config import RecipeConfig
from touster.console import console, print_success


def run_final(
    recipe: RecipeConfig,
    backend,
    dataset_path: Path,
    run_dir: Path,
) -> Path:
    """
    Train the winning recipe for the full duration (recipe.num_epochs / recipe.max_steps).
    Saves adapter to run_dir/final_adapter/.
    Returns path to saved adapter.
    """
    output_dir = run_dir / "final_adapter"
    console.print(
        f"  [touster.dim]Final run: lr={recipe.learning_rate:.2e} "
        f"rank={recipe.lora_rank} epochs={recipe.num_epochs} "
        f"steps={recipe.max_steps}[/touster.dim]"
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
    print_success(
        f"Final training done — {result['steps']} steps, "
        f"loss={result.get('train_loss', float('nan')):.4f}"
    )
    return output_dir
