from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class LoopCheckpoint:
    """Serializable loop state — survives Colab disconnects."""
    current_trial: int
    best_trial_id: int
    best_bpb: float
    best_recipe_diff: dict
    total_trials_run: int


def save_checkpoint(run_dir: Path, ckpt: LoopCheckpoint) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "loop_checkpoint.json").write_text(json.dumps(asdict(ckpt), indent=2))


def load_checkpoint(run_dir: Path) -> LoopCheckpoint | None:
    p = run_dir / "loop_checkpoint.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return LoopCheckpoint(**data)


def checkpoint_path(run_dir: Path, trial_id: int) -> Path:
    """Path for adapter checkpoint from trial trial_id."""
    return run_dir / "checkpoints" / f"trial_{trial_id:03d}"
