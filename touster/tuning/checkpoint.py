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
    d = asdict(ckpt)
    if d.get("best_bpb") == float("inf") or (isinstance(d.get("best_bpb"), float) and d["best_bpb"] != d["best_bpb"]):
        d["best_bpb"] = None
    target = run_dir / "loop_checkpoint.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
    tmp.replace(target)


def load_checkpoint(run_dir: Path) -> LoopCheckpoint | None:
    p = run_dir / "loop_checkpoint.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("best_bpb") is None:
            data["best_bpb"] = float("inf")
        return LoopCheckpoint(**data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def checkpoint_path(run_dir: Path, trial_id: int) -> Path:
    """Path for adapter checkpoint from trial trial_id."""
    return run_dir / "checkpoints" / f"trial_{trial_id:03d}"
