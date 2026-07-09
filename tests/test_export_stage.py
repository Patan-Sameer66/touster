"""Regression tests for touster/tuning/export_stage.py's export orchestration."""
from __future__ import annotations

from pathlib import Path

from touster.config import ExportConfig, RecipeConfig


def test_export_stage_runs_all_exports_and_local_copies(tmp_path, monkeypatch):
    import touster.tuning.export_stage as export_stage

    def fake_merge(adapter_path, run_dir, dtype="float16"):
        d = run_dir / "merged_weights"
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.bin").write_text("fake")
        return d

    def fake_gguf(adapter_path, run_dir, quantization="q4_k_m"):
        d = run_dir / "gguf"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "model.gguf"
        p.write_text("fake")
        return p

    monkeypatch.setattr(export_stage, "export_merged", fake_merge)
    monkeypatch.setattr(export_stage, "export_gguf", fake_gguf)

    run_dir = tmp_path / "run"
    adapter = run_dir / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")

    local_dir = tmp_path / "out"
    export_cfg = ExportConfig(
        save_local=True, local_save_dir=local_dir,
        export_merged=True, export_gguf=True, gguf_quantize="q4_k_m",
    )
    recipe = RecipeConfig(base_model="test-model")

    results = export_stage.run_export_stage(recipe, adapter, run_dir, export_cfg)

    assert results["adapter"] == adapter
    assert results["merged"] is not None and results["merged"].exists()
    assert results["gguf"] is not None and results["gguf"].exists()
    assert results["model_card"] is not None and results["model_card"].exists()
    assert (local_dir / "adapter" / "adapter_config.json").exists()
    assert (local_dir / "merged_weights" / "model.bin").exists()
    assert (local_dir / "model.gguf").exists()


def test_export_stage_one_failure_does_not_block_others(tmp_path, monkeypatch):
    """One export failing (e.g. GGUF needs optional llama-cpp-python) must not
    take down exports that already succeeded or the ones after it."""
    import touster.tuning.export_stage as export_stage

    def fake_merge(adapter_path, run_dir, dtype="float16"):
        d = run_dir / "merged_weights"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def failing_gguf(adapter_path, run_dir, quantization="q4_k_m"):
        raise RuntimeError("llama-cpp-python not installed")

    monkeypatch.setattr(export_stage, "export_merged", fake_merge)
    monkeypatch.setattr(export_stage, "export_gguf", failing_gguf)

    run_dir = tmp_path / "run"
    adapter = run_dir / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")

    export_cfg = ExportConfig(save_local=False, export_merged=True, export_gguf=True)
    recipe = RecipeConfig(base_model="test-model")

    results = export_stage.run_export_stage(recipe, adapter, run_dir, export_cfg)

    assert results["merged"] is not None  # succeeded despite gguf failing after it
    assert results["gguf"] is None        # failed gracefully
    assert results["model_card"] is not None  # ran despite gguf failing before it
