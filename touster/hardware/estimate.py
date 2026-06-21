"""Pure math utilities for VRAM estimation and performance prediction.

No I/O, no imports of hardware detection — all functions are stateless and testable.
"""
from __future__ import annotations


def estimate_vram_needed(
    param_billions: float,
    bits_per_param: int,
    kv_cache_fraction: float = 0.15,
    activation_fraction: float = 0.10,
    framework_overhead_gb: float = 0.5,
) -> float:
    """Return estimated VRAM in GB for loading and running inference/training."""
    param_bytes = param_billions * 1e9 * bits_per_param / 8
    param_gb = param_bytes / 1e9

    total_gb = (
        param_gb
        + param_gb * kv_cache_fraction
        + param_gb * activation_fraction
        + framework_overhead_gb
    )
    return total_gb


def estimate_tokens_per_second(
    param_billions: float,
    bandwidth_gbps: float,
    bits_per_param: int = 4,
) -> float:
    """Estimate inference tokens/second from memory bandwidth.

    Formula: t/s ≈ bandwidth_GB/s / (param_billions * 1e9 * bits / 8 / 1e9)
    """
    if bandwidth_gbps <= 0 or param_billions <= 0 or bits_per_param <= 0:
        return 0.0

    model_gb = param_billions * 1e9 * bits_per_param / 8 / 1e9
    return bandwidth_gbps / model_gb


def model_fits(
    vram_bytes: int,
    param_billions: float,
    bits_per_param: int = 4,
) -> bool:
    """Return True if the model fits within *vram_bytes* of VRAM."""
    vram_gb = vram_bytes / 1e9
    needed_gb = estimate_vram_needed(param_billions, bits_per_param)
    return vram_gb >= needed_gb
