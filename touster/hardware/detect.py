"""Hardware detection — discovers platform, GPU, RAM, and CPU count."""
from __future__ import annotations

import platform
import sys

import psutil

from touster.config import HardwareConfig

# Approximate memory bandwidth in GB/s per GPU model.
BANDWIDTH_TABLE: dict[str, float] = {
    "RTX 4090": 1008,
    "RTX 4080": 717,
    "RTX 4070 Ti": 672,
    "RTX 4070": 504,
    "RTX 4060 Ti": 288,
    "RTX 4060": 272,
    "RTX 3090": 936,
    "RTX 3080": 760,
    "RTX 3070": 448,
    "RTX 3060": 336,
    "A100": 2000,
    "H100": 3350,
    "RTX 4080 SUPER": 736,
    "RTX 4070 SUPER": 504,
    "RTX 4070 Ti SUPER": 672,
    # Colab/cloud GPUs
    "Tesla T4": 300,
    "T4": 300,
    "A10G": 600,
    "V100": 900,
    "L4": 300,
    # Apple Silicon
    "Apple M1": 68,
    "Apple M2": 100,
    "Apple M3": 150,
    "Apple M4": 273,
    "Apple M1 Pro": 200,
    "Apple M2 Pro": 200,
    "Apple M3 Pro": 150,
    "Apple M1 Max": 400,
    "Apple M2 Max": 400,
    "Apple M3 Max": 300,
}


def _lookup_bandwidth(gpu_name: str) -> float:
    """Return bandwidth (GB/s) for *gpu_name* using longest-key-match in the table."""
    name_lower = gpu_name.lower()
    matches = [(key, bw) for key, bw in BANDWIDTH_TABLE.items() if key.lower() in name_lower]
    if not matches:
        return 0.0
    return max(matches, key=lambda x: len(x[0]))[1]


def _try_cuda() -> HardwareConfig | None:
    """Attempt NVIDIA GPU detection via pynvml; returns None if unavailable."""
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            pynvml.nvmlShutdown()
            return None

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name: str = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8", errors="replace")

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_bytes: int = int(mem_info.total)
        bandwidth_gbps = _lookup_bandwidth(gpu_name)

        pynvml.nvmlShutdown()

        ram_bytes = psutil.virtual_memory().total
        cpu_count = psutil.cpu_count(logical=False) or 1

        return HardwareConfig(
            platform="cuda",
            gpu_name=gpu_name,
            vram_bytes=vram_bytes,
            ram_bytes=ram_bytes,
            cpu_count=cpu_count,
            gpu_bandwidth_gbps=bandwidth_gbps,
        )
    except Exception:  # pynvml can raise various NVMLErrors
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return None


def _try_mlx() -> HardwareConfig | None:
    """Attempt Apple Silicon detection via mlx; returns None if unavailable."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    try:
        import mlx  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return None

    ram_bytes = psutil.virtual_memory().total
    cpu_count = psutil.cpu_count(logical=False) or 1

    # Detect Apple chip name for bandwidth lookup.
    # machdep.cpu.brand_string works on both Intel and Apple Silicon on modern macOS.
    chip_raw = ""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        chip_raw = result.stdout.strip()
    except Exception:
        pass

    if not chip_raw:
        # Fallback: infer from hw.model (e.g. "MacBookPro18,3")
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.model"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            chip_raw = result.stdout.strip()
        except Exception:
            pass

    gpu_name = chip_raw if chip_raw else "Apple Silicon"
    bandwidth_gbps = _lookup_bandwidth(gpu_name)

    return HardwareConfig(
        platform="mlx",
        gpu_name=gpu_name,
        # Unified memory: VRAM == system RAM
        vram_bytes=ram_bytes,
        ram_bytes=ram_bytes,
        cpu_count=cpu_count,
        gpu_bandwidth_gbps=bandwidth_gbps,
    )


def detect_hardware() -> HardwareConfig:
    """Detect the current system hardware and return a HardwareConfig."""
    ram_bytes = psutil.virtual_memory().total
    cpu_count = psutil.cpu_count(logical=False) or 1

    # Priority 1 — NVIDIA CUDA GPU
    cuda_cfg = _try_cuda()
    if cuda_cfg is not None:
        return cuda_cfg

    # Priority 2 — Apple Silicon (MLX)
    mlx_cfg = _try_mlx()
    if mlx_cfg is not None:
        return mlx_cfg

    # Fallback — CPU only
    return HardwareConfig(
        platform="cpu",
        gpu_name="",
        vram_bytes=0,
        ram_bytes=ram_bytes,
        cpu_count=cpu_count,
        gpu_bandwidth_gbps=0.0,
    )
