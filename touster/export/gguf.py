"""Export merged model weights to GGUF format for Ollama/llama.cpp."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from touster import display
from touster.export.merge import export_merged


def _try_unsloth_gguf(
    adapter_path: Path,
    run_dir: Path,
    quantization: str,
) -> Path | None:
    """Attempt GGUF export via Unsloth. Returns output path or None."""
    try:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
    except ImportError:
        return None

    try:
        import json as _json

        adapter_config_path = adapter_path / "adapter_config.json"
        if not adapter_config_path.exists():
            return None
        cfg = _json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model_id = cfg.get("base_model_name_or_path", "")

        print("Loading model via Unsloth...")
        # Load base model + adapter together; Unsloth auto-detects adapter dirs
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            load_in_4bit=False,
        )

        gguf_dir = run_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)

        print("Saving GGUF via Unsloth...")
        model.save_pretrained_gguf(
            str(gguf_dir / "model"),
            tokenizer,
            quantization_method=quantization,
        )

        # Unsloth writes model-<quant>.gguf
        candidates = list(gguf_dir.glob("*.gguf"))
        if candidates:
            gguf_path = candidates[0]
            display.success(f"GGUF saved via Unsloth: {gguf_path}")
            return gguf_path

    except Exception as exc:  # noqa: BLE001
        display.warning(f"Unsloth GGUF export failed: {exc}")

    return None


def _try_llama_cpp_gguf(
    merged_dir: Path,
    run_dir: Path,
    quantization: str,
) -> Path | None:
    """Attempt GGUF export via llama-cpp-python convert script. Returns path or None."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return None

    # Locate the convert script shipped with llama-cpp-python
    llama_cpp_pkg = Path(llama_cpp.__file__).parent
    convert_candidates = [
        llama_cpp_pkg / "convert_hf_to_gguf.py",
        llama_cpp_pkg / "llama" / "convert_hf_to_gguf.py",
        # Older naming
        llama_cpp_pkg / "convert.py",
    ]
    convert_script: Path | None = None
    for candidate in convert_candidates:
        if candidate.exists():
            convert_script = candidate
            break

    if convert_script is None:
        display.warning("llama-cpp-python found but convert script not located.")
        return None

    gguf_dir = run_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    output_file = gguf_dir / "model.gguf"

    cmd = [
        sys.executable,
        str(convert_script),
        str(merged_dir),
        "--outfile",
        str(output_file),
        "--outtype",
        quantization,
    ]

    print("Running llama.cpp convert script...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            display.warning(f"llama.cpp convert script failed:\n{result.stderr}")
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        display.warning(f"llama.cpp convert script error: {exc}")
        return None

    if output_file.exists():
        display.success(f"GGUF saved via llama.cpp: {output_file}")
        return output_file

    return None


def _write_stub(merged_dir: Path, run_dir: Path, quantization: str) -> Path:
    """Write a stub placeholder file and warn the user."""
    gguf_dir = run_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    stub_path = gguf_dir / "model.gguf.stub"
    stub_data = {
        "status": "gguf_export_requires_llama_cpp",
        "merged_path": str(merged_dir),
        "quantization": quantization,
    }
    stub_path.write_text(json.dumps(stub_data, indent=2), encoding="utf-8")

    display.warning(
        "GGUF export requires llama-cpp-python. Run: pip install llama-cpp-python"
    )
    print(f"Stub written to: {stub_path}")
    return stub_path


def export_gguf(
    adapter_path: Path,
    run_dir: Path,
    quantization: str = "q4_k_m",
) -> Path:
    """
    Convert merged model to GGUF.

    Strategy (in order of preference):
    1. If Unsloth is installed: use model.save_pretrained_gguf(quantization=quantization).
    2. Else if llama.cpp convert script is available (llama_cpp): use it.
    3. Else: merge weights first, then attempt llama.cpp via subprocess.
    4. Fallback: save a stub GGUF placeholder file and warn the user.

    Always returns a Path (even if it's the stub). Never crashes silently.
    """
    print(f"GGUF export quantization={quantization}")

    # Strategy 1: Unsloth (handles merge + GGUF in one shot)
    result = _try_unsloth_gguf(adapter_path, run_dir, quantization)
    if result is not None:
        return result

    # Merge weights first (needed for strategies 2 & 3)
    try:
        merged_dir = export_merged(adapter_path, run_dir)
    except RuntimeError as exc:
        display.warning(f"Merge step failed: {exc}. Falling back to stub.")
        stub_dir = run_dir / "gguf"
        stub_dir.mkdir(parents=True, exist_ok=True)
        stub_path = stub_dir / "model.gguf.stub"
        stub_data = {
            "status": "gguf_export_requires_llama_cpp",
            "merged_path": str(adapter_path),
            "quantization": quantization,
        }
        stub_path.write_text(json.dumps(stub_data, indent=2), encoding="utf-8")
        display.warning(
            "GGUF export requires llama-cpp-python. Run: pip install llama-cpp-python"
        )
        return stub_path

    # Strategy 2 & 3: llama-cpp-python convert script
    result = _try_llama_cpp_gguf(merged_dir, run_dir, quantization)
    if result is not None:
        return result

    # Strategy 4: Fallback stub
    return _write_stub(merged_dir, run_dir, quantization)
