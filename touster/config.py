from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

# Re-export replace so callers never need to import dataclasses directly
__all__ = [
    "HardwareConfig",
    "DatasetConfig",
    "RecipeConfig",
    "LoopConfig",
    "RunConfig",
    "replace",
    "ALLOWED_RECIPE_KNOBS",
]

# Knobs the agent is allowed to change — everything else is off-limits.
ALLOWED_RECIPE_KNOBS = frozenset(
    {
        "learning_rate",
        "lora_rank",
        "lora_alpha",
        "target_modules",
        "warmup_steps",
        "num_epochs",
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "scheduler",
    }
)

Platform = Literal["cuda", "mlx", "cpu"]
DatasetMode = Literal[0, 1, 2]  # 0=generate, 1=structure, 2=bring-your-own

# Numeric guardrails for agent-proposed recipe knobs.
_RECIPE_BOUNDS = {
    "learning_rate": (1e-6, 1e-1),
    "lora_rank": (1, 512),
    "lora_alpha": (1, 1024),
    "warmup_steps": (0, 100_000),
    "num_epochs": (1, 100),
    "max_steps": (1, 1_000_000),
    "batch_size": (1, 1024),
    "gradient_accumulation_steps": (1, 1024),
}
_VALID_SCHEDULERS = frozenset({"cosine", "linear", "constant"})


@dataclass(frozen=True)
class HardwareConfig:
    platform: Platform = "cpu"
    gpu_name: str = ""
    vram_bytes: int = 0
    ram_bytes: int = 0
    cpu_count: int = 1
    gpu_bandwidth_gbps: float = 0.0


@dataclass(frozen=True)
class DatasetConfig:
    mode: DatasetMode = 0
    prompt: str = ""
    raw_data_path: Path | None = None
    dataset_path: Path | None = None
    num_samples: int = 50
    gen_batch_size: int = 10
    model: str = ""
    eval_fraction: float = 0.1


@dataclass(frozen=True)
class RecipeConfig:
    """Exactly the knobs the agent may tune. Frozen; agent proposes a dict diff."""

    base_model: str = "sshleifer/tiny-gpt2"
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: int = 16
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    warmup_steps: int = 10
    num_epochs: int = 1
    max_steps: int = 200
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    scheduler: str = "cosine"

    def apply_diff(self, diff: dict) -> "RecipeConfig":
        """Return new RecipeConfig with validated diff applied."""
        unknown = set(diff) - ALLOWED_RECIPE_KNOBS
        if unknown:
            raise ValueError(f"Agent proposed disallowed knobs: {unknown}")
        patched = dict(diff)
        if "target_modules" in patched:
            patched["target_modules"] = tuple(patched["target_modules"])
            if not patched["target_modules"]:
                raise ValueError("target_modules must not be empty")
        # Numeric bounds — prevents lr→0 underflow spiral and absurd configs
        for key, (lo, hi) in _RECIPE_BOUNDS.items():
            if key in patched:
                val = patched[key]
                if not isinstance(val, (int, float)) or not (lo <= val <= hi):
                    raise ValueError(f"{key}={val!r} out of bounds [{lo}, {hi}]")
        if "scheduler" in patched and patched["scheduler"] not in _VALID_SCHEDULERS:
            raise ValueError(
                f"scheduler={patched['scheduler']!r} not in {sorted(_VALID_SCHEDULERS)}"
            )
        return replace(self, **patched)


@dataclass(frozen=True)
class LoopConfig:
    """Controls the self-improvement search loop."""

    trial_max_steps: int = 200
    trial_wall_clock_secs: int = 300
    max_trials: int = 20
    judge_top_k: int = 3
    judge_prompts: int = 20
    use_llm_proposer: bool = True


@dataclass(frozen=True)
class RunConfig:
    run_dir: Path = Path("runs/default")
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    recipe: RecipeConfig = field(default_factory=RecipeConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    resume: bool = False
