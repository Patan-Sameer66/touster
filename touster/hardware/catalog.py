"""Model catalog — registry of LoRA-trainable models with metadata."""
from __future__ import annotations

from dataclasses import dataclass

from touster.config import HardwareConfig
from touster.hardware.estimate import estimate_tokens_per_second, estimate_vram_needed, model_fits


@dataclass(frozen=True)
class ModelEntry:
    """Immutable descriptor for a single trainable model."""

    id: str                       # Display id, e.g. "qwen2.5-7b"
    hf_id: str                    # HuggingFace model id
    ollama_id: str                # Ollama pull id (empty string if unsupported)
    param_billions: float
    arch: str                     # Architecture family
    quality_score: float          # 0-100 composite benchmark score (hand-curated)
    default_quant_bits: int       # 4 or 8
    default_lora_targets: tuple[str, ...]


_CATALOG: list[ModelEntry] = [
    # ── Qwen 2.5 ──────────────────────────────────────────────────────────────
    ModelEntry(
        id="qwen2.5-7b",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        ollama_id="qwen2.5:7b",
        param_billions=7.6,
        arch="qwen2",
        quality_score=82.0,
        default_quant_bits=4,
        default_lora_targets=("q_proj", "v_proj"),
    ),
    ModelEntry(
        id="qwen2.5-3b",
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        ollama_id="qwen2.5:3b",
        param_billions=3.1,
        arch="qwen2",
        quality_score=72.0,
        default_quant_bits=4,
        default_lora_targets=("q_proj", "v_proj"),
    ),
    ModelEntry(
        id="qwen2.5-0.5b",
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        ollama_id="qwen2.5:0.5b",
        param_billions=0.5,
        arch="qwen2",
        quality_score=55.0,
        default_quant_bits=4,
        default_lora_targets=("q_proj", "v_proj"),
    ),
    # ── Llama 3.x ─────────────────────────────────────────────────────────────
    ModelEntry(
        id="llama3.2-3b",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
        ollama_id="llama3.2:3b",
        param_billions=3.2,
        arch="llama3",
        quality_score=74.0,
        default_quant_bits=4,
        default_lora_targets=("q_proj", "v_proj"),
    ),
    ModelEntry(
        id="llama3.1-8b",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        ollama_id="llama3.1:8b",
        param_billions=8.0,
        arch="llama3",
        quality_score=80.0,
        default_quant_bits=4,
        default_lora_targets=("q_proj", "v_proj"),
    ),
    ModelEntry(
        id="llama3.2-1b",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        ollama_id="llama3.2:1b",
        param_billions=1.24,
        arch="llama3",
        quality_score=60.0,
        default_quant_bits=4,
        default_lora_targets=("q_proj", "v_proj"),
    ),
    # ── Mistral ───────────────────────────────────────────────────────────────
    ModelEntry(
        id="mistral-7b",
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        ollama_id="mistral:7b",
        param_billions=7.24,
        arch="mistral",
        quality_score=78.0,
        default_quant_bits=4,
        default_lora_targets=("q_proj", "v_proj"),
    ),
    # ── Phi-3 / Phi-3.5 ───────────────────────────────────────────────────────
    ModelEntry(
        id="phi3.5-mini",
        hf_id="microsoft/Phi-3.5-mini-instruct",
        ollama_id="phi3.5:mini",
        param_billions=3.82,
        arch="phi3",
        quality_score=76.0,
        default_quant_bits=4,
        default_lora_targets=("q_proj", "v_proj"),
    ),
    ModelEntry(
        id="phi3-mini-4k",
        hf_id="microsoft/Phi-3-mini-4k-instruct",
        ollama_id="phi3:mini",
        param_billions=3.82,
        arch="phi3",
        quality_score=73.0,
        default_quant_bits=4,
        default_lora_targets=("q_proj", "v_proj"),
    ),
    # ── Gemma 2 ───────────────────────────────────────────────────────────────
    ModelEntry(
        id="gemma2-2b",
        hf_id="google/gemma-2-2b-it",
        ollama_id="gemma2:2b",
        param_billions=2.61,
        arch="gemma",
        quality_score=70.0,
        default_quant_bits=4,
        default_lora_targets=("q_proj", "v_proj"),
    ),
    ModelEntry(
        id="gemma2-9b",
        hf_id="google/gemma-2-9b-it",
        ollama_id="gemma2:9b",
        param_billions=9.24,
        arch="gemma",
        quality_score=83.0,
        default_quant_bits=4,
        default_lora_targets=("q_proj", "v_proj"),
    ),
    # ── SmolLM2 ───────────────────────────────────────────────────────────────
    ModelEntry(
        id="smollm2-1.7b",
        hf_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        ollama_id="",
        param_billions=1.71,
        arch="llama",
        quality_score=65.0,
        default_quant_bits=4,
        default_lora_targets=("q_proj", "v_proj"),
    ),
    # ── CPU validation / tiny model ───────────────────────────────────────────
    ModelEntry(
        id="tiny-gpt2",
        hf_id="sshleifer/tiny-gpt2",
        ollama_id="",
        param_billions=0.117,
        arch="gpt2",
        quality_score=5.0,
        default_quant_bits=8,
        default_lora_targets=("c_attn",),
    ),
]


def get_catalog() -> list[ModelEntry]:
    """Return the full list of known trainable models."""
    return list(_CATALOG)


def get_trainable(
    hw: HardwareConfig,
    catalog: list[ModelEntry] | None = None,
) -> list[ModelEntry]:
    """Return models that fit on *hw*, sorted by combined score descending.

    Combined score = (t/s estimate * quality_score / 100). CPU-only systems
    return only models that have a near-zero VRAM footprint (fits in 0 VRAM).
    """
    entries = catalog if catalog is not None else get_catalog()

    bandwidth = hw.gpu_bandwidth_gbps

    # CPU path: vram_bytes == 0, so model_fits is always False.
    # We still allow tiny models that need negligible memory to run on CPU.
    # Uses the SAME estimate_vram_needed() the "fits" column renders with —
    # otherwise a model can rank as trainable here yet show "doesn't fit"
    # in the table (weights-only formula here vs. full-footprint there).
    if hw.platform == "cpu" or hw.vram_bytes == 0:
        # Allow models that need < 2 GB so tiny-gpt2 always appears
        cpu_trainable = [
            e for e in entries
            if estimate_vram_needed(e.param_billions, e.default_quant_bits) < 2.0
        ]
        # Sort by quality descending
        return sorted(cpu_trainable, key=lambda e: e.quality_score, reverse=True)

    def score(entry: ModelEntry) -> float:
        tps = estimate_tokens_per_second(
            entry.param_billions, bandwidth, entry.default_quant_bits
        )
        return tps * entry.quality_score / 100.0

    trainable = [e for e in entries if model_fits(hw.vram_bytes, e.param_billions, e.default_quant_bits)]
    return sorted(trainable, key=score, reverse=True)
