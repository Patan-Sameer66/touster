from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class TrainerBackend(Protocol):
    """Structural protocol for all training backends (CPU/NVIDIA/MLX)."""

    def load_model(self, model_id: str, lora_rank: int, lora_alpha: int, target_modules: list[str]) -> None:
        """Load base model + attach LoRA adapters."""
        ...

    def train_steps(
        self,
        dataset_path: Path,
        max_steps: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        warmup_steps: int,
        scheduler: str,
        wall_clock_limit_secs: int = 0,
    ) -> dict:
        """Run training for up to max_steps (or wall_clock_limit_secs). Returns {"steps": int, "train_loss": float}."""
        ...

    def eval_loss(self, dataset_path: Path, eval_fraction: float = 0.1) -> float:
        """Compute eval loss (cross-entropy) on held-out split. Lower = better."""
        ...

    def save_adapter(self, output_dir: Path) -> None:
        """Save LoRA adapter weights to output_dir."""
        ...

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Run inference with the loaded (base+adapter) model."""
        ...

    def unload(self) -> None:
        """Release model from memory."""
        ...
