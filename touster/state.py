from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
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
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class RunState:
    run_dir: Path
    base_model: str
    dataset_path: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
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
    data = json.loads(p.read_text(encoding="utf-8"))
    data["run_dir"] = Path(data["run_dir"])
    if data.get("best_bpb") is None:
        data["best_bpb"] = float("inf")
    known = {f.name for f in fields(RunState)}
    return RunState(**{k: v for k, v in data.items() if k in known})


def save_state(state: RunState) -> None:
    state.run_dir.mkdir(parents=True, exist_ok=True)
    d = asdict(state)
    d["run_dir"] = str(d["run_dir"])
    if d.get("best_bpb") == float("inf"):
        d["best_bpb"] = None
    _run_json(state.run_dir).write_text(json.dumps(d, indent=2), encoding="utf-8")


def append_experiment(run_dir: Path, record: ExperimentRecord) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    d = asdict(record)
    if d.get("eval_bpb") == float("inf"):
        d["eval_bpb"] = None
    with _experiments_jsonl(run_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(d) + "\n")


def load_experiments(run_dir: Path) -> list[ExperimentRecord]:
    p = _experiments_jsonl(run_dir)
    if not p.exists():
        return []
    known = {f.name for f in fields(ExperimentRecord)}
    records = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if d.get("eval_bpb") is None:
                d["eval_bpb"] = float("inf")
            records.append(ExperimentRecord(**{k: v for k, v in d.items() if k in known}))
        except (json.JSONDecodeError, TypeError):
            pass
    return records
