from __future__ import annotations

"""Tests for touster/export/ — all tests run offline (no network calls)."""

import json
from pathlib import Path

import pytest

from touster.config import RecipeConfig


# ---------------------------------------------------------------------------
# Helper: create a tiny local GPT2 + LoRA adapter (no network)
# ---------------------------------------------------------------------------

def _create_local_tiny_adapter(tmp_path: Path) -> tuple[Path, Path]:
    """Create a tiny GPT2 + LoRA adapter locally for testing merge."""
    import json as _json

    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from peft import LoraConfig, get_peft_model

    VOCAB_SIZE = 512
    vocab: dict[str, int] = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
    for i in range(32, 127):
        w = chr(i)
        if w not in vocab:
            vocab[w] = len(vocab)
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

    model_dir = tmp_path / "base_model"
    model_dir.mkdir()
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
    base = GPT2LMHeadModel(cfg)
    base.save_pretrained(str(model_dir))
    tok.save_pretrained(str(model_dir))

    # Create LoRA adapter
    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=4,
        target_modules=["c_attn"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, lora_cfg)

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    peft_model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))

    # PEFT leaves base_model_name_or_path="" for in-memory models.
    # Patch it so export_merged can locate the base model locally.
    cfg_path = adapter_dir / "adapter_config.json"
    cfg_data = json.loads(cfg_path.read_text())
    cfg_data["base_model_name_or_path"] = str(model_dir)
    cfg_path.write_text(json.dumps(cfg_data, indent=2))

    return model_dir, adapter_dir


# ---------------------------------------------------------------------------
# merge.py tests
# ---------------------------------------------------------------------------

class TestExportMerged:
    def test_export_merged_no_adapter_raises(self, tmp_path: Path) -> None:
        """Calling export_merged with a non-existent adapter dir raises RuntimeError."""
        from touster.export.merge import export_merged

        with pytest.raises(RuntimeError, match="adapter_config.json"):
            export_merged(Path("nonexistent_adapter_xyz"), tmp_path)

    def test_export_merged_missing_config_raises(self, tmp_path: Path) -> None:
        """Calling export_merged on a dir without adapter_config.json raises RuntimeError."""
        from touster.export.merge import export_merged

        empty_adapter = tmp_path / "empty_adapter"
        empty_adapter.mkdir()

        with pytest.raises(RuntimeError, match="adapter_config.json"):
            export_merged(empty_adapter, tmp_path / "run")

    def test_export_merged_with_mock_adapter(self, tmp_path: Path) -> None:
        """Merge a tiny local GPT2 + LoRA adapter — no network required."""
        from touster.export.merge import export_merged

        _model_dir, adapter_dir = _create_local_tiny_adapter(tmp_path)
        run_dir = tmp_path / "run"

        merged_path = export_merged(adapter_dir, run_dir)

        assert merged_path.exists(), f"merged_weights dir not found: {merged_path}"
        assert merged_path.is_dir()
        # Expect at least a config.json inside
        assert (merged_path / "config.json").exists(), \
            "config.json missing from merged_weights"

    def test_export_merged_returns_merged_weights_subdir(self, tmp_path: Path) -> None:
        """Merged weights should land in run_dir/merged_weights/."""
        from touster.export.merge import export_merged

        _model_dir, adapter_dir = _create_local_tiny_adapter(tmp_path)
        run_dir = tmp_path / "myrun"

        merged_path = export_merged(adapter_dir, run_dir)

        assert merged_path == run_dir / "merged_weights"


# ---------------------------------------------------------------------------
# gguf.py tests
# ---------------------------------------------------------------------------

class TestExportGguf:
    def test_export_gguf_stub_fallback(self, tmp_path: Path) -> None:
        """When llama.cpp is not available, export_gguf returns a path without crashing."""
        import sys
        import unittest.mock as mock

        # Ensure unsloth and llama_cpp appear unavailable
        with mock.patch.dict(sys.modules, {"unsloth": None, "llama_cpp": None}):
            # Also ensure import inside functions fails cleanly
            from touster.export import gguf as gguf_mod

            # Patch _try_unsloth_gguf and _try_llama_cpp_gguf to return None
            with mock.patch.object(gguf_mod, "_try_unsloth_gguf", return_value=None), \
                 mock.patch.object(gguf_mod, "_try_llama_cpp_gguf", return_value=None):

                # Create a minimal adapter_config.json so merge step has something
                _model_dir, adapter_dir = _create_local_tiny_adapter(tmp_path)
                run_dir = tmp_path / "run"

                result = gguf_mod.export_gguf(adapter_dir, run_dir, quantization="q4_k_m")

                assert result is not None, "export_gguf should return a Path"
                assert isinstance(result, Path), f"Expected Path, got {type(result)}"

    def test_export_gguf_stub_writes_json_content(self, tmp_path: Path) -> None:
        """The stub file should contain the expected JSON keys."""
        import unittest.mock as mock
        from touster.export import gguf as gguf_mod

        with mock.patch.object(gguf_mod, "_try_unsloth_gguf", return_value=None), \
             mock.patch.object(gguf_mod, "_try_llama_cpp_gguf", return_value=None):

            _model_dir, adapter_dir = _create_local_tiny_adapter(tmp_path)
            run_dir = tmp_path / "run"

            result = gguf_mod.export_gguf(adapter_dir, run_dir, quantization="q4_k_m")

            # Should be the stub file
            assert result.name == "model.gguf.stub", \
                f"Expected stub file, got: {result.name}"
            content = json.loads(result.read_text())
            assert content["status"] == "gguf_export_requires_llama_cpp"
            assert "merged_path" in content
            assert content["quantization"] == "q4_k_m"

    def test_export_gguf_never_crashes_silently(self, tmp_path: Path) -> None:
        """export_gguf on a missing adapter falls back to stub instead of raising."""
        import unittest.mock as mock
        from touster.export import gguf as gguf_mod

        with mock.patch.object(gguf_mod, "_try_unsloth_gguf", return_value=None), \
             mock.patch.object(gguf_mod, "_try_llama_cpp_gguf", return_value=None):

            # No adapter_config.json — merge will fail, should still return stub
            bad_adapter = tmp_path / "bad_adapter"
            bad_adapter.mkdir()
            run_dir = tmp_path / "run"

            result = gguf_mod.export_gguf(bad_adapter, run_dir, quantization="q4_k_m")
            assert result is not None
            assert result.exists()


# ---------------------------------------------------------------------------
# modelcard.py tests
# ---------------------------------------------------------------------------

class TestWriteModelCard:
    def test_write_model_card_creates_file(self, tmp_path: Path) -> None:
        """write_model_card should create model_card.md in run_dir."""
        from touster.export.modelcard import write_model_card

        recipe = RecipeConfig()
        result = write_model_card(recipe, tmp_path)

        assert result == tmp_path / "model_card.md"
        assert result.exists(), "model_card.md was not created"

    def test_write_model_card_contains_base_model(self, tmp_path: Path) -> None:
        """model_card.md should mention the base_model name."""
        from touster.export.modelcard import write_model_card

        recipe = RecipeConfig(base_model="sshleifer/tiny-gpt2")
        write_model_card(recipe, tmp_path)

        content = (tmp_path / "model_card.md").read_text(encoding="utf-8")
        assert "sshleifer/tiny-gpt2" in content, \
            "Base model name not found in model card"

    def test_write_model_card_contains_training_details(self, tmp_path: Path) -> None:
        """model_card.md should mention learning_rate and lora_rank."""
        from touster.export.modelcard import write_model_card

        recipe = RecipeConfig(learning_rate=3e-4, lora_rank=32)
        write_model_card(recipe, tmp_path)

        content = (tmp_path / "model_card.md").read_text(encoding="utf-8")
        assert "learning_rate" in content, "learning_rate not in model card"
        assert "lora_rank" in content, "lora_rank not in model card"

    def test_write_model_card_includes_scheduler(self, tmp_path: Path) -> None:
        """model_card.md should mention the scheduler."""
        from touster.export.modelcard import write_model_card

        recipe = RecipeConfig(scheduler="linear")
        write_model_card(recipe, tmp_path)

        content = (tmp_path / "model_card.md").read_text(encoding="utf-8")
        assert "linear" in content, "Scheduler not found in model card"

    def test_write_model_card_includes_bpb_when_run_json_present(
        self, tmp_path: Path
    ) -> None:
        """When run.json exists with best_bpb, card should include it."""
        from touster.export.modelcard import write_model_card

        run_json = tmp_path / "run.json"
        run_json.write_text(json.dumps({
            "best_bpb": 1.2345,
            "run_dir": str(tmp_path),
            "base_model": "sshleifer/tiny-gpt2",
            "dataset_path": "",
            "started_at": "2024-01-01T00:00:00",
            "phase": "done",
            "best_trial_id": 1,
            "total_trials": 5,
            "final_adapter_path": "",
            "gguf_path": "",
            "merged_path": "",
        }))

        recipe = RecipeConfig()
        write_model_card(recipe, tmp_path)

        content = (tmp_path / "model_card.md").read_text(encoding="utf-8")
        assert "1.2345" in content, "best_bpb not found in model card"

    def test_write_model_card_returns_path(self, tmp_path: Path) -> None:
        """write_model_card returns the Path to the generated file."""
        from touster.export.modelcard import write_model_card

        result = write_model_card(RecipeConfig(), tmp_path)
        assert isinstance(result, Path)

    def test_write_model_card_includes_touster_footer(self, tmp_path: Path) -> None:
        """model_card.md should include the Touster footer."""
        from touster.export.modelcard import write_model_card

        write_model_card(RecipeConfig(), tmp_path)

        content = (tmp_path / "model_card.md").read_text(encoding="utf-8")
        assert "Touster" in content, "Touster footer not found in model card"

    def test_write_model_card_no_hub_push_when_flag_false(
        self, tmp_path: Path
    ) -> None:
        """When push_to_hub=False, no Hub interaction should occur."""
        import unittest.mock as mock
        from touster.export import modelcard as mc_mod

        with mock.patch.object(mc_mod, "_push_to_hub") as mock_push:
            write_result = mc_mod.write_model_card(
                RecipeConfig(), tmp_path, push_to_hub=False, repo_id="org/model"
            )
            mock_push.assert_not_called()
            assert write_result.exists()
