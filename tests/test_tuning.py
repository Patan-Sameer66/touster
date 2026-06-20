from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from touster.config import ALLOWED_RECIPE_KNOBS, LoopConfig, RecipeConfig


# ── RecipeConfig guardrails ───────────────────────────────────────────────────

class TestRecipeGuardrails:
    def test_allowed_knob_applies(self):
        r = RecipeConfig()
        r2 = r.apply_diff({"learning_rate": 1e-3})
        assert r2.learning_rate == pytest.approx(1e-3)
        assert r.learning_rate != r2.learning_rate  # immutable

    def test_disallowed_knob_raises(self):
        r = RecipeConfig()
        with pytest.raises(ValueError, match="disallowed"):
            r.apply_diff({"base_model": "evil/model"})

    def test_disallowed_dataset_key_raises(self):
        r = RecipeConfig()
        with pytest.raises(ValueError, match="disallowed"):
            r.apply_diff({"dataset_path": "/evil"})

    def test_multiple_allowed_knobs(self):
        r = RecipeConfig()
        r2 = r.apply_diff({"learning_rate": 5e-5, "lora_rank": 32})
        assert r2.lora_rank == 32

    def test_allowed_knobs_set_complete(self):
        assert "learning_rate" in ALLOWED_RECIPE_KNOBS
        assert "lora_rank" in ALLOWED_RECIPE_KNOBS
        assert "base_model" not in ALLOWED_RECIPE_KNOBS


# ── Agent proposer ────────────────────────────────────────────────────────────

class TestHeuristicProposer:
    def test_returns_dict(self):
        from touster.tuning.agent import propose_heuristic
        diff = propose_heuristic(RecipeConfig(), trial_id=1, last_bpb=2.0, best_bpb=2.0)
        assert isinstance(diff, dict)

    def test_only_allowed_keys(self):
        from touster.tuning.agent import propose_heuristic
        for trial_id in range(20):
            diff = propose_heuristic(RecipeConfig(), trial_id=trial_id, last_bpb=2.0, best_bpb=2.0)
            for key in diff:
                assert key in ALLOWED_RECIPE_KNOBS, f"Disallowed key: {key}"

    def test_divergence_halves_lr(self):
        from touster.tuning.agent import propose_heuristic
        recipe = RecipeConfig(learning_rate=2e-4)
        diff = propose_heuristic(recipe, trial_id=5, last_bpb=3.0, best_bpb=2.0)
        assert "learning_rate" in diff
        assert diff["learning_rate"] == pytest.approx(1e-4)


# ── Eval bpb ─────────────────────────────────────────────────────────────────

def _create_local_tiny_model(tmp_path: Path) -> Path:
    """Create a tiny GPT2 causal LM locally (no network) and return its path."""
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    # Build a simple word-level tokenizer with vocab that covers test text.
    # IDs start at 0 — pad=0, unk=1, then words.
    VOCAB_SIZE = 512
    vocab: dict[str, int] = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
    # Add every printable ASCII char as a word so any test text tokenizes safely
    for i in range(32, 127):
        word = chr(i)
        if word not in vocab:
            vocab[word] = len(vocab)
    # Fill up to VOCAB_SIZE with dummy tokens
    while len(vocab) < VOCAB_SIZE:
        vocab[f"<tok{len(vocab)}>"] = len(vocab)

    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    model_dir = tmp_path / "tiny_model"
    model_dir.mkdir()
    cfg = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=512,   # must match max_length in _encode_samples
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(str(model_dir))
    tok.save_pretrained(str(model_dir))
    return model_dir


class TestEvalBpb:
    def _make_dataset(self, tmp_path: Path, n: int = 10) -> Path:
        p = tmp_path / "dataset.jsonl"
        samples = [
            {"messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4."},
            ]}
            for _ in range(n)
        ]
        p.write_text("\n".join(json.dumps(s) for s in samples))
        return p

    def test_bpb_finite_after_loading(self, tmp_path):
        """Tiny local model bpb should be finite (not inf) after loading."""
        from touster.tuning.backends.cpu_backend import CPUBackend
        from touster.tuning.eval import eval_bpb

        model_dir = _create_local_tiny_model(tmp_path)
        backend = CPUBackend()
        backend.load_model(str(model_dir), lora_rank=4, lora_alpha=4, target_modules=["c_attn"])
        dataset_path = self._make_dataset(tmp_path)
        bpb = eval_bpb(backend, dataset_path, eval_fraction=0.5)
        backend.unload()
        assert bpb < float("inf")
        assert bpb > 0

    def test_bpb_empty_dataset_returns_inf(self, tmp_path):
        from touster.tuning.backends.cpu_backend import CPUBackend
        from touster.tuning.eval import eval_bpb

        model_dir = _create_local_tiny_model(tmp_path)
        backend = CPUBackend()
        backend.load_model(str(model_dir), lora_rank=4, lora_alpha=4, target_modules=["c_attn"])
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        bpb = eval_bpb(backend, empty, eval_fraction=0.5)
        backend.unload()
        assert bpb == float("inf")


# ── Checkpoint ───────────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        from touster.tuning.checkpoint import LoopCheckpoint, load_checkpoint, save_checkpoint

        ckpt = LoopCheckpoint(
            current_trial=3,
            best_trial_id=2,
            best_bpb=1.23,
            best_recipe_diff={"learning_rate": 1e-4},
            total_trials_run=3,
        )
        save_checkpoint(tmp_path, ckpt)
        loaded = load_checkpoint(tmp_path)
        assert loaded is not None
        assert loaded.best_bpb == pytest.approx(1.23)
        assert loaded.current_trial == 3
        assert loaded.best_recipe_diff == {"learning_rate": 1e-4}

    def test_load_nonexistent_returns_none(self, tmp_path):
        from touster.tuning.checkpoint import load_checkpoint
        assert load_checkpoint(tmp_path / "nonexistent") is None


# ── Cost estimate ─────────────────────────────────────────────────────────────

class TestEstimateCost:
    def test_local_api_cost_zero(self):
        from touster.config import DatasetConfig, HardwareConfig
        from touster.tuning.estimate_cost import _compute

        hw = HardwareConfig(platform="cpu")
        ds_cfg = DatasetConfig(mode=0, num_samples=100)
        result = _compute(hw, ds_cfg, RecipeConfig(), LoopConfig(), estimated_tps=5.0)
        assert result["api_cost"] == 0.0

    def test_total_time_positive(self):
        from touster.config import DatasetConfig, HardwareConfig
        from touster.tuning.estimate_cost import _compute

        hw = HardwareConfig(platform="cuda", gpu_bandwidth_gbps=1008.0)
        ds_cfg = DatasetConfig(mode=0, num_samples=200)
        result = _compute(hw, ds_cfg, RecipeConfig(), LoopConfig(), estimated_tps=100.0)
        assert result["total_secs"] > 0
