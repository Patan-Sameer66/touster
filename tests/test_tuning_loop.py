"""Regression tests for touster/tuning/loop.py's Optuna-based tuning loop.

These specifically cover bugs found and fixed during the phase-2 build (see
research.md section 6): resume silently reconstructing the wrong recipe,
checkpoint gaps on early-exit paths, and the fallback-to-default contract.
Uses a mock backend — no torch/transformers/peft required.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from touster.config import LoopConfig, RecipeConfig


class MockBackend:
    """bpb is a deterministic convex function of learning_rate, minimized at TARGET_LR."""

    TARGET_LR = 3e-4

    def __init__(self) -> None:
        self._last_lr = 2e-4
        self.trials: list[tuple[float, float]] = []

    def load_model(self, model_id, lora_rank, lora_alpha, target_modules) -> None:
        pass

    def train_steps(self, dataset_path, max_steps, batch_size, gradient_accumulation_steps,
                     learning_rate, warmup_steps, scheduler, wall_clock_limit_secs=0) -> dict:
        self._last_lr = learning_rate
        return {"steps": max_steps, "train_loss": 1.0}

    def eval_loss(self, dataset_path, eval_fraction=0.1) -> float:
        bpb = abs(math.log10(self._last_lr) - math.log10(self.TARGET_LR)) + 0.1
        self.trials.append((self._last_lr, bpb))
        return bpb

    def save_adapter(self, output_dir) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "adapter_config.json").write_text("{}")

    def generate(self, prompt, max_new_tokens=256) -> str:
        return "mock output"

    def unload(self) -> None:
        pass


class NeverImprovesBackend:
    """Every trial's eval crashes/returns inf — simulates a fully broken backend."""

    def load_model(self, model_id, lora_rank, lora_alpha, target_modules) -> None:
        pass

    def train_steps(self, *a, **k) -> dict:
        return {"steps": 5, "train_loss": 1.0}

    def eval_loss(self, *a, **k) -> float:
        return float("inf")

    def save_adapter(self, output_dir) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def generate(self, prompt, max_new_tokens=256) -> str:
        return "x"

    def unload(self) -> None:
        pass


class FlakyReloadBackend:
    """Fails to reload whenever lora_rank != 16 — simulates OOM at a bigger rank."""

    def load_model(self, model_id, lora_rank, lora_alpha, target_modules) -> None:
        if lora_rank != 16:
            raise RuntimeError("simulated OOM at this rank")

    def train_steps(self, *a, **k) -> dict:
        return {"steps": 5, "train_loss": 1.0}

    def eval_loss(self, *a, **k) -> float:
        return 0.5

    def save_adapter(self, output_dir) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def generate(self, prompt, max_new_tokens=256) -> str:
        return "x"

    def unload(self) -> None:
        pass


def _write_dataset(path: Path, n: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            obj = {"messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ]}
            f.write(json.dumps(obj) + "\n")


@pytest.fixture
def run_dir(tmp_path) -> Path:
    return tmp_path / "run"


@pytest.fixture
def dataset_path(run_dir) -> Path:
    p = run_dir / "dataset.jsonl"
    _write_dataset(p)
    return p


def test_tpe_search_converges_to_best_trial(monkeypatch, run_dir, dataset_path):
    """The TPE sampler should identify and return the actual best trial's recipe."""
    import touster.tuning.backends.factory as factory_mod
    from touster.tuning.loop import run_loop

    backend = MockBackend()
    monkeypatch.setattr(factory_mod, "get_backend", lambda hw: backend)

    recipe = RecipeConfig(base_model="mock-model", learning_rate=2e-4)
    loop_cfg = LoopConfig(max_trials=10, trial_max_steps=5, use_llm_prior=False, judge_top_k=0, judge_prompts=0)

    best_recipe, adapter_path = run_loop(recipe, loop_cfg, dataset_path, run_dir, client=None)

    best_lr, _ = min(backend.trials, key=lambda t: t[1])
    assert abs(best_recipe.learning_rate - best_lr) < 1e-9
    assert Path(adapter_path).exists()


def test_fallback_to_default_when_all_trials_fail(monkeypatch, run_dir, dataset_path):
    """If every trial fails, the loop must fall back to the default recipe and
    still produce a real final adapter — never raise."""
    import touster.tuning.backends.factory as factory_mod
    from touster.tuning.loop import run_loop

    monkeypatch.setattr(factory_mod, "get_backend", lambda hw: NeverImprovesBackend())

    recipe = RecipeConfig(base_model="mock-model", learning_rate=2e-4)
    loop_cfg = LoopConfig(max_trials=4, trial_max_steps=5, use_llm_prior=False, judge_top_k=0, judge_prompts=0)

    best_recipe, adapter_path = run_loop(recipe, loop_cfg, dataset_path, run_dir, client=None)

    assert best_recipe == recipe
    assert Path(adapter_path).exists()


def test_resume_reconstructs_best_recipe_not_last_trial(monkeypatch, run_dir, dataset_path):
    """Regression test for the bug where resume reconstructed whichever trial
    ran last before a disconnect, instead of the actual best trial."""
    import touster.tuning.backends.factory as factory_mod
    import touster.tuning.loop as loop_mod
    from touster.tuning.checkpoint import load_checkpoint

    backend1 = MockBackend()
    monkeypatch.setattr(factory_mod, "get_backend", lambda hw: backend1)
    recipe = RecipeConfig(base_model="mock-model", learning_rate=2e-4)

    # 5 trials — with seed=0 this deterministically makes trial 3 the best and
    # trial 4 (the last one before "disconnect") worse than it.
    loop_cfg = LoopConfig(max_trials=5, trial_max_steps=5, use_llm_prior=False, judge_top_k=0, judge_prompts=0)
    loop_mod.run_loop(recipe, loop_cfg, dataset_path, run_dir, client=None)

    ckpt = load_checkpoint(run_dir)
    best_lr_run1, _ = min(backend1.trials, key=lambda t: t[1])
    last_lr_run1, _ = backend1.trials[-1]
    assert abs(last_lr_run1 - best_lr_run1) > 1e-9, "test setup didn't stress the adversarial case"

    backend2 = MockBackend()
    monkeypatch.setattr(factory_mod, "get_backend", lambda hw: backend2)
    loop_cfg2 = LoopConfig(max_trials=ckpt.current_trial, trial_max_steps=5,
                            use_llm_prior=False, judge_top_k=0, judge_prompts=0)
    best_recipe_2, adapter_path_2 = loop_mod.run_loop(recipe, loop_cfg2, dataset_path, run_dir, client=None)

    assert abs(best_recipe_2.learning_rate - best_lr_run1) < 1e-9
    assert Path(adapter_path_2).exists()


def test_structural_reload_failure_still_checkpointed(monkeypatch, run_dir, dataset_path):
    """Regression test: trials that fail before reaching run_trial (invalid
    proposal, unrecoverable reload) must still be checkpointed/logged, not
    silently skipped — otherwise a resume repeats or loses that trial slot."""
    import touster.tuning.backends.factory as factory_mod
    from touster.tuning.loop import run_loop
    from touster.state import load_experiments
    from touster.tuning.checkpoint import load_checkpoint

    monkeypatch.setattr(factory_mod, "get_backend", lambda hw: FlakyReloadBackend())

    recipe = RecipeConfig(base_model="mock-model", learning_rate=2e-4, lora_rank=16)
    loop_cfg = LoopConfig(max_trials=5, trial_max_steps=5, use_llm_prior=False, judge_top_k=0, judge_prompts=0)

    run_loop(recipe, loop_cfg, dataset_path, run_dir, client=None)

    experiments = load_experiments(run_dir)
    ckpt = load_checkpoint(run_dir)
    logged_ids = sorted(e.trial_id for e in experiments)

    assert logged_ids == list(range(5))
    assert ckpt.current_trial == 5


def test_optuna_storage_file_releases_cleanly(monkeypatch, run_dir, dataset_path):
    """Regression test for the Windows file-handle leak: the study's SQLite
    file must be closeable (and thus the run_dir removable) after run_loop
    returns, without a PermissionError."""
    import touster.tuning.backends.factory as factory_mod
    from touster.tuning.loop import run_loop
    import shutil

    monkeypatch.setattr(factory_mod, "get_backend", lambda hw: MockBackend())
    recipe = RecipeConfig(base_model="mock-model", learning_rate=2e-4)
    loop_cfg = LoopConfig(max_trials=3, trial_max_steps=5, use_llm_prior=False, judge_top_k=0, judge_prompts=0)

    run_loop(recipe, loop_cfg, dataset_path, run_dir, client=None)

    shutil.rmtree(run_dir)  # must not raise PermissionError on Windows
    assert not run_dir.exists()
