from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

from touster.console import console


class MLXBackend:
    """Apple Silicon backend via mlx-lm. Requires macOS + Apple Silicon."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_id: str = ""

    def load_model(
        self,
        model_id: str,
        lora_rank: int,
        lora_alpha: int,
        target_modules: list[str],
    ) -> None:
        from mlx_lm import load

        console.print(f"  [touster.dim]Loading [touster.model]{model_id}[/touster.model] via MLX…[/touster.dim]")
        self._model, self._tokenizer = load(model_id)
        self._model_id = model_id
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha

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
        """Delegate to mlx_lm.lora CLI for training."""
        import tempfile, os

        tmp_adapter = Path(tempfile.mkdtemp()) / "adapter"
        cmd = [
            "python", "-m", "mlx_lm.lora",
            "--model", self._model_id,
            "--train",
            "--data", str(dataset_path.parent),
            "--iters", str(max_steps),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--lora-layers", "4",
            "--adapter-path", str(tmp_adapter),
        ]
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=wall_clock_limit_secs or 3600)
        elapsed = time.time() - start

        if result.returncode != 0:
            raise RuntimeError(f"mlx_lm.lora failed: {result.stderr[-500:]}")

        self._adapter_path = tmp_adapter
        return {"steps": max_steps, "train_loss": float("nan")}

    def eval_loss(self, dataset_path: Path, eval_fraction: float = 0.1) -> float:
        """Approximate eval via mlx_lm generate perplexity."""
        from touster.tuning.backends.cpu_backend import _load_samples
        from mlx_lm import generate

        samples = _load_samples(dataset_path)
        n_eval = max(1, int(len(samples) * eval_fraction))
        eval_samples = samples[-n_eval:]

        raise NotImplementedError(
            "MLXBackend.eval_loss: mlx_lm does not expose per-token log-probs in this version. "
            "Use bpb from validation loss logged during training instead."
        )

    def save_adapter(self, output_dir: Path) -> None:
        import shutil
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self, "_adapter_path") and self._adapter_path.exists():
            shutil.copytree(str(self._adapter_path), str(output_dir), dirs_exist_ok=True)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        from mlx_lm import generate
        return generate(self._model, self._tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=False)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
