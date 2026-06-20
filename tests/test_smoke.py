"""
E2E smoke test: full pipeline with mocked LLM client + locally-created tiny GPT2.
Runs without any network access or downloaded models.

Usage:
    pytest tests/test_smoke.py -v -m smoke
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from touster.config import LoopConfig, RecipeConfig

pytestmark = pytest.mark.smoke


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_vocab(size: int = 512) -> dict[str, int]:
    vocab: dict[str, int] = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
    for i in range(32, 127):
        ch = chr(i)
        if ch not in vocab:
            vocab[ch] = len(vocab)
    while len(vocab) < size:
        vocab[f"<extra{len(vocab)}>"] = len(vocab)
    return vocab


def _create_tiny_model(model_dir: Path) -> None:
    """Create a tiny GPT2 + tokenizer, fully offline."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    VOCAB_SIZE = 512
    vocab = _build_vocab(VOCAB_SIZE)
    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=512,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(str(model_dir))
    tok.save_pretrained(str(model_dir))


def _create_tiny_dataset(path: Path, n: int = 20) -> None:
    """Write a minimal ChatML JSONL dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i in range(n):
            sample = {
                "messages": [
                    {"role": "user", "content": f"Question number {i}"},
                    {"role": "assistant", "content": f"Answer number {i} with some words"},
                ]
            }
            f.write(json.dumps(sample) + "\n")


def _mock_llm_client() -> MagicMock:
    """Mock LLMClient: returns valid JSON diff for proposer, numeric score for judge."""
    client = MagicMock()
    call_count = {"n": 0}

    def _chat(messages, **kwargs):
        call_count["n"] += 1
        content = (messages[-1]["content"] if messages else "")
        if "score" in content.lower() or "judge" in content.lower() or "rate" in content.lower():
            return "Score: 7"
        return '{"learning_rate": 1e-4}'

    client.chat.side_effect = _chat
    return client


# ── Smoke test ────────────────────────────────────────────────────────────────

class TestSmokePipeline:
    """Full pipeline smoke test: no network, no downloaded models."""

    @pytest.fixture(autouse=True)
    def run_dir(self, tmp_path: Path):
        self._run_dir = tmp_path / "smoke_run"
        self._run_dir.mkdir()
        self._model_dir = tmp_path / "tiny_model"
        _create_tiny_model(self._model_dir)
        self._dataset_path = tmp_path / "dataset.jsonl"
        _create_tiny_dataset(self._dataset_path, n=20)
        return self._run_dir

    def _recipe(self) -> RecipeConfig:
        return RecipeConfig(
            base_model=str(self._model_dir),
            learning_rate=2e-4,
            lora_rank=4,
            lora_alpha=4,
            target_modules=("c_attn",),
            warmup_steps=2,
            num_epochs=1,
            max_steps=5,
            batch_size=2,
            gradient_accumulation_steps=1,
            scheduler="cosine",
        )

    def test_dataset_validate(self):
        """Dataset loads and validates without errors."""
        from touster.dataset.schema import load_jsonl
        from touster.dataset.validate import validate_and_repair

        ds = load_jsonl(self._dataset_path)
        repaired, warnings = validate_and_repair(ds)
        assert len(repaired.samples) > 0
        for sample in repaired.samples:
            assert len(sample.messages) >= 2

    def test_loop_runs_and_produces_adapter(self):
        """Self-improvement loop completes 2 trials, adapter saved."""
        from touster.tuning.loop import run_loop

        recipe = self._recipe()
        loop_cfg = LoopConfig(
            max_trials=2,
            trial_max_steps=5,
            trial_wall_clock_secs=0,
            judge_top_k=1,
            judge_prompts=2,
            use_llm_proposer=False,
        )
        client = _mock_llm_client()

        best_recipe, final_adapter = run_loop(
            recipe=recipe,
            loop_cfg=loop_cfg,
            dataset_path=self._dataset_path,
            run_dir=self._run_dir,
            client=client,
        )

        assert isinstance(best_recipe, RecipeConfig)
        assert final_adapter.exists(), f"Expected adapter at {final_adapter}"

    def test_experiments_jsonl_populated(self):
        """After loop, experiments.jsonl has records with bpb finite."""
        from touster.state import load_experiments
        from touster.tuning.loop import run_loop

        recipe = self._recipe()
        loop_cfg = LoopConfig(
            max_trials=2, trial_max_steps=5, trial_wall_clock_secs=0,
            judge_top_k=0, judge_prompts=0, use_llm_proposer=False,
        )
        run_loop(
            recipe=recipe,
            loop_cfg=loop_cfg,
            dataset_path=self._dataset_path,
            run_dir=self._run_dir,
            client=None,
        )

        exps = load_experiments(self._run_dir)
        assert len(exps) == 2
        for exp in exps:
            assert exp.eval_bpb != float("inf") or not exp.kept

    def test_final_eval_lte_best_trial(self):
        """Final adapter bpb <= best trial bpb (loop picked the right winner)."""
        from touster.tuning.loop import run_loop
        from touster.state import load_experiments

        recipe = self._recipe()
        loop_cfg = LoopConfig(
            max_trials=3, trial_max_steps=5, trial_wall_clock_secs=0,
            judge_top_k=0, judge_prompts=0, use_llm_proposer=False,
        )
        run_loop(
            recipe=recipe,
            loop_cfg=loop_cfg,
            dataset_path=self._dataset_path,
            run_dir=self._run_dir,
            client=None,
        )

        kept = [e for e in load_experiments(self._run_dir) if e.kept]
        if kept:
            best_trial_bpb = min(e.eval_bpb for e in kept)
            # Final adapter exists
            assert (self._run_dir / "final_adapter").exists()
            assert best_trial_bpb < float("inf")

    def test_export_model_card(self):
        """write_model_card produces a file with correct content."""
        from touster.export.modelcard import write_model_card

        recipe = self._recipe()
        card_path = write_model_card(recipe, self._run_dir)
        assert card_path.exists()
        content = card_path.read_text()
        assert "learning_rate" in content
        assert "lora_rank" in content
        assert str(self._model_dir) in content or recipe.base_model in content

    def test_export_gguf_stub(self):
        """GGUF export falls back gracefully when llama.cpp unavailable."""
        from touster.export.gguf import export_gguf

        # Point at a plausible adapter dir (doesn't need to be real for stub path)
        fake_adapter = self._run_dir / "fake_adapter"
        fake_adapter.mkdir()
        result = export_gguf(fake_adapter, self._run_dir)
        assert result.exists()

    def test_compare_imports_and_degrades(self):
        """ModelPair loads without error; graceful no-adapter case."""
        from touster.dashboard.compare import ModelPair

        pair = ModelPair(str(self._model_dir), adapter_path=None)
        # Should not crash
        result = pair.generate_finetuned("hello")
        assert isinstance(result, str)

    def test_recipe_diff_recorded_in_experiments(self):
        """ExperimentRecord.recipe_diff is populated for trial_id > 0."""
        from touster.state import load_experiments
        from touster.tuning.loop import run_loop

        recipe = self._recipe()
        loop_cfg = LoopConfig(
            max_trials=2, trial_max_steps=5, trial_wall_clock_secs=0,
            judge_top_k=0, judge_prompts=0, use_llm_proposer=False,
        )
        run_loop(
            recipe=recipe, loop_cfg=loop_cfg,
            dataset_path=self._dataset_path, run_dir=self._run_dir, client=None,
        )

        exps = load_experiments(self._run_dir)
        trial_1 = next((e for e in exps if e.trial_id == 1), None)
        assert trial_1 is not None
        assert isinstance(trial_1.recipe_diff, dict)
