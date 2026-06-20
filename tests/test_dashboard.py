"""Lightweight dashboard tests — no model loading, no Textual UI launch."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── Import smoke-tests ────────────────────────────────────────────────────────

def test_dashboard_imports_cleanly() -> None:
    """Importing launch_dashboard must not require torch/transformers."""
    from touster.dashboard.app import launch_dashboard  # noqa: F401


def test_compare_imports_cleanly() -> None:
    """Importing ModelPair must not require torch/transformers."""
    from touster.dashboard.compare import ModelPair  # noqa: F401


# ── ModelPair — no adapter path ───────────────────────────────────────────────

def test_compare_no_adapter_returns_empty_finetuned(tmp_path: Path) -> None:
    """ModelPair with a non-existent adapter path must not crash.

    generate_finetuned must return either '' or the same text as generate_base
    without touching the filesystem or loading any model.
    """
    from touster.dashboard.compare import ModelPair

    # Point at a path that definitely does not exist
    bogus_adapter = tmp_path / "nonexistent_adapter"
    assert not bogus_adapter.exists()

    pair = ModelPair(base_model_id="gpt2", adapter_path=bogus_adapter)

    # Models are NOT loaded — pair._base_backend is None
    # generate_* should return "" gracefully (guarded by `if self._base_backend is None`)
    base_out = pair.generate_base("hello")
    ft_out = pair.generate_finetuned("hello")

    assert isinstance(base_out, str)
    assert isinstance(ft_out, str)
    # With no model loaded: both must be empty strings
    assert base_out == ""
    assert ft_out == ""
    # has_adapter must be False because the path does not exist
    assert not pair.has_adapter
    assert not pair.is_loaded


def test_compare_no_adapter_none_path() -> None:
    """ModelPair with adapter_path=None also degrades gracefully."""
    from touster.dashboard.compare import ModelPair

    pair = ModelPair(base_model_id="gpt2", adapter_path=None)

    assert not pair.has_adapter
    assert not pair.is_loaded
    assert pair.generate_base("test") == ""
    assert pair.generate_finetuned("test") == ""


# ── State loading ─────────────────────────────────────────────────────────────

def test_run_summary_loads_from_state(tmp_path: Path) -> None:
    """Create minimal run.json + experiments.jsonl and verify state loads."""
    from touster.state import (
        ExperimentRecord,
        RunState,
        append_experiment,
        load_experiments,
        load_state,
        save_state,
    )

    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    state = RunState(
        run_dir=run_dir,
        base_model="gpt2",
        dataset_path="data/train.jsonl",
        phase="loop",
        best_bpb=2.34,
        total_trials=3,
    )
    save_state(state)

    for trial_id in range(3):
        rec = ExperimentRecord(
            trial_id=trial_id,
            recipe_diff={"learning_rate": 1e-4},
            eval_bpb=2.5 - trial_id * 0.1,
            judge_score=None,
            kept=trial_id == 2,
            wall_clock_secs=10.0,
            steps=5,
        )
        append_experiment(run_dir, rec)

    # Round-trip load
    loaded_state = load_state(run_dir)
    assert loaded_state is not None
    assert loaded_state.base_model == "gpt2"
    assert loaded_state.total_trials == 3
    assert loaded_state.best_bpb == pytest.approx(2.34)

    experiments = load_experiments(run_dir)
    assert len(experiments) == 3
    assert experiments[2].kept is True
    assert experiments[0].kept is False


def test_load_state_missing_dir_returns_none(tmp_path: Path) -> None:
    """load_state on a non-existent run directory must return None."""
    from touster.state import load_state

    result = load_state(tmp_path / "no_such_run")
    assert result is None


def test_load_experiments_missing_file_returns_empty(tmp_path: Path) -> None:
    """load_experiments on a run dir with no experiments.jsonl returns []."""
    from touster.state import load_experiments

    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    experiments = load_experiments(run_dir)
    assert experiments == []


# ── _load_run_summary helper ──────────────────────────────────────────────────

def test_load_run_summary_missing_dir(tmp_path: Path) -> None:
    """_load_run_summary on a missing run dir returns placeholder strings, not exceptions."""
    from touster.dashboard.app import _load_run_summary

    summary, exp_lines = _load_run_summary(tmp_path / "no_run")
    assert isinstance(summary, str)
    assert isinstance(exp_lines, list)


def test_load_run_summary_with_valid_run(tmp_path: Path) -> None:
    """_load_run_summary returns human-readable summary for a valid run."""
    from touster.state import RunState, save_state
    from touster.dashboard.app import _load_run_summary

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = RunState(
        run_dir=run_dir,
        base_model="meta-llama/Llama-3.2-1B",
        dataset_path="data/my_data.jsonl",
        phase="done",
        best_bpb=1.85,
        total_trials=5,
    )
    save_state(state)

    summary, exp_lines = _load_run_summary(run_dir)
    assert "meta-llama/Llama-3.2-1B" in summary
    assert isinstance(exp_lines, list)
