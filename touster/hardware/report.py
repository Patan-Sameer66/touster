"""Hardware report — HTML via touster.display, not a terminal library."""
from __future__ import annotations

import sys

from touster import display
from touster.config import HardwareConfig
from touster.hardware.catalog import ModelEntry, get_catalog, get_trainable
from touster.hardware.estimate import estimate_tokens_per_second, estimate_vram_needed

_MARGINAL_THRESHOLD = 0.10  # 10% margin = "marginal"


def _platform_label(hw: HardwareConfig) -> str:
    return {"cuda": "CUDA / NVIDIA", "mlx": "Apple Silicon (MLX)"}.get(hw.platform, "CPU only")


def _fits_symbol(vram_bytes: int, entry: ModelEntry) -> str:
    needed_gb = estimate_vram_needed(entry.param_billions, entry.default_quant_bits)
    available_gb = vram_bytes / 1e9

    if available_gb <= 0:
        return "OK" if needed_gb < 2.0 else "XX"

    ratio = available_gb / needed_gb if needed_gb > 0 else float("inf")
    if ratio >= (1 + _MARGINAL_THRESHOLD):
        return "YES"
    if ratio >= (1 - _MARGINAL_THRESHOLD):
        return "~"
    return "NO"


def print_hardware_report(
    hw: HardwareConfig,
    suggested_model: str | None = None,
) -> str:
    """Print the hardware specs + ranked model table; return the chosen model id."""
    vram_str = f"{hw.vram_bytes / 1e9:.1f} GB" if hw.vram_bytes > 0 else "None"
    display.table(
        ["", ""],
        [
            ["Platform", _platform_label(hw)],
            ["GPU", f"{hw.gpu_name or 'None'} ({vram_str})"],
            ["System RAM", f"{hw.ram_bytes / 1e9:.1f} GB"],
            ["CPU cores", str(hw.cpu_count)],
        ],
        title="System",
    )

    catalog = get_catalog()
    trainable = get_trainable(hw, catalog)
    if not trainable:
        display.warning("No models fit on this hardware. Defaulting to tiny-gpt2 for CPU validation.")
        return "sshleifer/tiny-gpt2"

    rows = []
    for rank, entry in enumerate(trainable, start=1):
        tps = estimate_tokens_per_second(entry.param_billions, hw.gpu_bandwidth_gbps, entry.default_quant_bits)
        rows.append([
            rank, entry.id, f"{entry.param_billions:.1f}B",
            f"{estimate_vram_needed(entry.param_billions, entry.default_quant_bits):.1f}",
            f"{tps:.0f}" if tps > 0 else "-",
            f"{entry.quality_score:.0f}",
            _fits_symbol(hw.vram_bytes, entry),
        ])
    display.table(["#", "Model", "Params", "VRAM(GB)", "t/s", "Quality", "Fits"], rows, title="Model ranking")

    top_model = trainable[0]
    if suggested_model:
        match = next((e for e in trainable if suggested_model in (e.id, e.hf_id)), None)
        default_choice = match.id if match else top_model.id
    else:
        default_choice = top_model.id

    print(f"Top suggestion: {default_choice} (enter a model id above, or press Enter to accept)")
    interactive = sys.stdin is not None and hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    chosen = default_choice
    if interactive:
        try:
            typed = input(f"Model [{default_choice}]: ").strip()
            chosen = typed or default_choice
        except (EOFError, KeyboardInterrupt):
            chosen = default_choice

    resolved = next((e.hf_id for e in trainable if chosen in (e.id, e.hf_id)), chosen)
    display.success(f"Selected: {resolved}")
    return resolved
