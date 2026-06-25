#!/usr/bin/env python
"""Standalone self-improvement-loop test harness.

Run (fast, no model):   python tests/test_loop.py
Real tiny-model on CPU:  python tests/test_loop.py --real

The fast path uses a MockBackend so we test the LOOP LOGIC (best-recipe
tracking, bounds validation, checkpointing) in seconds with no downloads.
The --real path runs ONE actual LoRA trial on sshleifer/tiny-gpt2 on CPU to
smoke-test the training/eval/generate code path.

Paste the full output back so fixes can target the actual failures.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles default to cp1252 and crash on non-ASCII output
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from touster.config import LoopConfig, RecipeConfig

_PASS = 0
_FAIL = 0
_RESULTS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        _RESULTS.append(f"pass  {name}")
    else:
        _FAIL += 1
        _RESULTS.append(f"FAIL  {name}: {detail}")


def check_raises(name: str, fn) -> None:
    global _PASS, _FAIL
    try:
        fn()
        _FAIL += 1
        _RESULTS.append(f"FAIL  {name}: expected exception, none raised")
    except Exception:  # noqa: BLE001
        _PASS += 1
        _RESULTS.append(f"pass  {name}")


# ── mock backend: bpb is a deterministic function of learning_rate ─────────
class MockBackend:
    """Records every trial's (lr, bpb). bpb minimized at TARGET_LR."""

    TARGET_LR = 5e-4

    def __init__(self) -> None:
        self._last_lr = 2e-4
        self.trials: list[tuple[float, float]] = []  # (lr, bpb)

    def load_model(self, model_id, lora_rank, lora_alpha, target_modules) -> None:
        pass

    def train_steps(self, dataset_path, max_steps, batch_size,
                    gradient_accumulation_steps, learning_rate, warmup_steps,
                    scheduler, wall_clock_limit_secs=0) -> dict:
        self._last_lr = learning_rate
        return {"steps": max_steps, "train_loss": 1.0}

    def eval_loss(self, dataset_path, eval_fraction=0.1) -> float:
        # convex bowl in log-lr space, minimum at TARGET_LR
        bpb = abs(math.log10(self._last_lr) - math.log10(self.TARGET_LR)) + 0.1
        self.trials.append((self._last_lr, bpb))
        return bpb

    def save_adapter(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        return "mock output"

    def unload(self) -> None:
        pass


def _write_dummy_dataset(path: Path, n: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            obj = {"messages": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i}"},
            ]}
            f.write(json.dumps(obj) + "\n")


def run_loop_logic() -> None:
    print("\n== self-improvement loop logic (MockBackend) ==")
    import touster.tuning.backends.factory as factory_mod

    backend = MockBackend()
    factory_mod.get_backend = lambda hw: backend  # inject mock

    from touster.tuning.loop import run_loop

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        ds_path = run_dir / "dataset.jsonl"
        _write_dummy_dataset(ds_path)

        recipe = RecipeConfig(base_model="mock-model", learning_rate=2e-4)
        loop_cfg = LoopConfig(max_trials=8, trial_max_steps=5,
                              use_llm_proposer=False, judge_top_k=0, judge_prompts=0)

        try:
            best_recipe, adapter_path = run_loop(
                recipe=recipe, loop_cfg=loop_cfg,
                dataset_path=ds_path, run_dir=run_dir, client=None,
            )
        except Exception as exc:  # noqa: BLE001
            check("run_loop completes without crashing", False, f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            return

        check("run_loop completes without crashing", True)

        # which lr actually produced the lowest bpb across all trials
        best_lr, best_bpb = min(backend.trials, key=lambda t: t[1])
        print(f"  trials (lr->bpb): {[(f'{lr:.0e}', round(b,3)) for lr,b in backend.trials]}")
        print(f"  observed best: lr={best_lr:.0e} bpb={best_bpb:.3f}")
        print(f"  run_loop returned: lr={best_recipe.learning_rate:.0e}")

        # CRITICAL: the recipe returned for the final run must be the BEST
        # trial's recipe, not the last trial's.
        check(
            "best_recipe.learning_rate == lr of lowest-bpb trial",
            abs(best_recipe.learning_rate - best_lr) < 1e-12,
            f"returned lr={best_recipe.learning_rate:.2e} but best trial used lr={best_lr:.2e}",
        )

        check("final adapter path exists", Path(adapter_path).exists(),
              f"missing: {adapter_path}")


def run_apply_diff_bounds() -> None:
    print("\n== RecipeConfig.apply_diff validation ==")
    r = RecipeConfig()

    # valid change should work
    check("valid lr change accepted",
          r.apply_diff({"learning_rate": 1e-4}).learning_rate == 1e-4)

    # unknown knob already rejected
    check_raises("unknown knob rejected", lambda: r.apply_diff({"nope": 1}))

    # these SHOULD raise (numeric/empty bounds) — will FAIL until bounds added
    check_raises("learning_rate=0 rejected", lambda: r.apply_diff({"learning_rate": 0.0}))
    check_raises("negative lora_rank rejected", lambda: r.apply_diff({"lora_rank": -4}))
    check_raises("empty target_modules rejected", lambda: r.apply_diff({"target_modules": []}))


def run_checkpoint_roundtrip() -> None:
    print("\n== checkpoint save/load round-trip ==")
    from touster.tuning.checkpoint import LoopCheckpoint, load_checkpoint, save_checkpoint

    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        ck = LoopCheckpoint(current_trial=3, best_trial_id=1, best_bpb=float("inf"),
                            best_recipe_diff={"learning_rate": 1e-4}, total_trials_run=3)
        save_checkpoint(run_dir, ck)
        loaded = load_checkpoint(run_dir)
        check("checkpoint round-trips", loaded is not None and loaded.current_trial == 3,
              f"loaded={loaded}")
        check("inf best_bpb survives round-trip",
              loaded is not None and loaded.best_bpb == float("inf"),
              f"best_bpb={getattr(loaded, 'best_bpb', None)}")


# ── optional real tiny-model CPU smoke test ────────────────────────────────
def run_real() -> None:
    print("\n== REAL CPU smoke test (sshleifer/tiny-gpt2) ==")
    try:
        from touster.tuning.backends.cpu_backend import CPUBackend

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            ds_path = run_dir / "dataset.jsonl"
            _write_dummy_dataset(ds_path, n=8)

            be = CPUBackend()
            be.load_model("sshleifer/tiny-gpt2", lora_rank=8, lora_alpha=8,
                          target_modules=["c_attn"])
            res = be.train_steps(ds_path, max_steps=3, batch_size=2,
                                 gradient_accumulation_steps=1, learning_rate=2e-4,
                                 warmup_steps=0, scheduler="cosine")
            print(f"  train: {res}")
            check("train_loss finite", math.isfinite(res.get("train_loss", float('inf'))),
                  f"loss={res.get('train_loss')}")

            bpb = be.eval_loss(ds_path, eval_fraction=0.25)
            print(f"  eval bpb: {bpb}")
            check("eval bpb finite", math.isfinite(bpb), f"bpb={bpb}")

            out = be.generate("Hello", max_new_tokens=10)
            print(f"  generate: {out!r}")
            check("generate returns text", isinstance(out, str))
            be.unload()
    except Exception as exc:  # noqa: BLE001
        check("real CPU smoke test", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def main() -> int:
    run_apply_diff_bounds()
    run_checkpoint_roundtrip()
    run_loop_logic()
    if "--real" in sys.argv:
        run_real()

    print("\n" + "=" * 60)
    for line in _RESULTS:
        print(line)
    print("=" * 60)
    print(f"TOTAL: {_PASS} passed, {_FAIL} failed")
    print("\nExpected: all pass on the mock path. Run with --real on Colab/GPU")
    print("(where peft+torch are installed) to smoke-test the real training path.")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
