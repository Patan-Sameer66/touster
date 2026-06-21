from __future__ import annotations

from touster.config import HardwareConfig


def get_backend(hw: HardwareConfig):
    """Return the correct TrainerBackend for the detected hardware."""
    if hw.platform == "cuda":
        try:
            from touster.tuning.backends.unsloth_backend import UnslothBackend
            return UnslothBackend()
        except (ImportError, RuntimeError, SystemExit):
            pass
        # Fall through to HF/PEFT CUDA or CPU if unsloth not installed
    if hw.platform == "mlx":
        try:
            from touster.tuning.backends.mlx_backend import MLXBackend
            return MLXBackend()
        except ImportError:
            pass
    from touster.tuning.backends.cpu_backend import CPUBackend
    return CPUBackend()
