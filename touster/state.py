from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["RunState", "ExperimentRecord", "load_state", "save_state", "append_experiment"]


@dataclass
class ExperimentRecord:
    trial_id: int
    recipe_diff: dict[str, Any]
    eval_bpb: float
    judge_score: float | None
    kept: bool
    wall_clock_secs: float
    steps: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class RunState:
    run_dir: Path
    base_model: str
    dataset_path: str
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    phase: str = "init"       # init | loop | final | dashboard | export | done
    best_trial_id: int = -1
    best_bpb: float = float("inf")
    total_trials: int = 0
    final_adapter_path: str = ""
    gguf_path: str = ""
    merged_path: str = ""


def _run_json(run_dir: Path) -> Path:
    return run_dir / "run.json"


def _experiments_jsonl(run_dir: Path) -> Path:
    return run_dir / "experiments.jsonl"


def load_state(run_dir: Path) -> RunState | None:
    p = _run_json(run_dir)
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    data["run_dir"] = Path(data["run_dir"])
    return RunState(**data)


def save_state(state: RunState) -> None:
    state.run_dir.mkdir(parents=True, exist_ok=True)
    d = asdict(state)
    d["run_dir"] = str(d["run_dir"])
    _run_json(state.run_dir).write_text(json.dumps(d, indent=2))


def append_experiment(run_dir: Path, record: ExperimentRecord) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with _experiments_jsonl(run_dir).open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_experiments(run_dir: Path) -> list[ExperimentRecord]:
    p = _experiments_jsonl(run_dir)
    if not p.exists():
        return []
    records = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(ExperimentRecord(**json.loads(line)))
    return records
