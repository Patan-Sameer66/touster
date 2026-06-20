"""Tests for the touster.hardware module."""
from __future__ import annotations

import pytest

from touster.config import HardwareConfig
from touster.hardware.catalog import get_catalog, get_trainable
from touster.hardware.estimate import (
    estimate_tokens_per_second,
    estimate_vram_needed,
    model_fits,
)


class TestEstimateVramNeeded:
    def test_7b_4bit_in_expected_range(self) -> None:
        """7B model @ 4-bit should need between 3.0 and 6.0 GB."""
        gb = estimate_vram_needed(param_billions=7.0, bits_per_param=4)
        assert 3.0 <= gb <= 6.0, f"Expected 3.0–6.0 GB, got {gb:.2f}"

    def test_larger_model_needs_more_vram(self) -> None:
        """70B model should require more VRAM than 7B model."""
        small = estimate_vram_needed(7.0, 4)
        large = estimate_vram_needed(70.0, 4)
        assert large > small

    def test_fp16_needs_more_than_4bit(self) -> None:
        """Same model in fp16 should need more VRAM than 4-bit."""
        q4 = estimate_vram_needed(7.0, 4)
        fp16 = estimate_vram_needed(7.0, 16)
        assert fp16 > q4

    def test_overhead_included(self) -> None:
        """Framework overhead should always be included in the result."""
        gb = estimate_vram_needed(0.0, 4, framework_overhead_gb=1.0)
        assert gb >= 1.0


class TestEstimateTokensPerSecond:
    def test_rtx4090_7b_over_50_tps(self) -> None:
        """7B model on RTX 4090 (1008 GBps) @ 4-bit should exceed 50 t/s."""
        tps = estimate_tokens_per_second(
            param_billions=7.0, bandwidth_gbps=1008.0, bits_per_param=4
        )
        assert tps > 50, f"Expected > 50 t/s, got {tps:.1f}"

    def test_zero_bandwidth_returns_zero(self) -> None:
        """Zero bandwidth should return 0 t/s."""
        tps = estimate_tokens_per_second(7.0, bandwidth_gbps=0.0)
        assert tps == 0.0

    def test_larger_model_is_slower(self) -> None:
        """70B model should be slower than 7B at same bandwidth."""
        tps_7b = estimate_tokens_per_second(7.0, 1008.0)
        tps_70b = estimate_tokens_per_second(70.0, 1008.0)
        assert tps_7b > tps_70b


class TestModelFits:
    def test_7b_fits_in_16gb(self) -> None:
        """7B model @ 4-bit should fit in 16 GB VRAM."""
        vram_16gb = 16 * 1024 ** 3
        assert model_fits(vram_16gb, param_billions=7.0, bits_per_param=4)

    def test_70b_does_not_fit_in_16gb(self) -> None:
        """70B model @ 4-bit should NOT fit in 16 GB VRAM."""
        vram_16gb = 16 * 1024 ** 3
        assert not model_fits(vram_16gb, param_billions=70.0, bits_per_param=4)

    def test_zero_vram_nothing_fits(self) -> None:
        """Zero VRAM means no model fits."""
        assert not model_fits(0, param_billions=0.5, bits_per_param=4)


class TestGetCatalog:
    def test_nonempty(self) -> None:
        """Catalog should contain at least 10 models."""
        catalog = get_catalog()
        assert len(catalog) >= 10, f"Expected >= 10 models, got {len(catalog)}"

    def test_all_have_hf_id(self) -> None:
        """Every catalog entry must have a non-empty HuggingFace id."""
        for entry in get_catalog():
            assert entry.hf_id, f"Entry {entry.id!r} has no hf_id"

    def test_tiny_gpt2_present(self) -> None:
        """sshleifer/tiny-gpt2 must be in the catalog."""
        hf_ids = {e.hf_id for e in get_catalog()}
        assert "sshleifer/tiny-gpt2" in hf_ids

    def test_all_frozen(self) -> None:
        """ModelEntry instances must be immutable (frozen=True)."""
        entry = get_catalog()[0]
        with pytest.raises((AttributeError, TypeError)):
            entry.id = "mutated"  # type: ignore[misc]


class TestGetTrainable:
    def _cpu_hw(self) -> HardwareConfig:
        return HardwareConfig(
            platform="cpu",
            gpu_name="",
            vram_bytes=0,
            ram_bytes=16 * 1024 ** 3,
            cpu_count=4,
            gpu_bandwidth_gbps=0.0,
        )

    def test_cpu_returns_at_least_tiny_gpt2(self) -> None:
        """CPU hardware with 0 VRAM must include tiny-gpt2."""
        hw = self._cpu_hw()
        trainable = get_trainable(hw)
        ids = {e.hf_id for e in trainable}
        assert "sshleifer/tiny-gpt2" in ids, f"tiny-gpt2 missing; got {ids}"

    def test_16gb_vram_returns_multiple_models(self) -> None:
        """16 GB VRAM should accommodate several models."""
        hw = HardwareConfig(
            platform="cuda",
            gpu_name="RTX 4090",
            vram_bytes=16 * 1024 ** 3,
            ram_bytes=32 * 1024 ** 3,
            cpu_count=8,
            gpu_bandwidth_gbps=1008.0,
        )
        trainable = get_trainable(hw)
        assert len(trainable) >= 3, f"Expected >= 3 models, got {len(trainable)}"

    def test_sorted_by_combined_score_descending(self) -> None:
        """Models should be returned highest combined-score (t/s * quality) first."""
        from touster.hardware.estimate import estimate_tokens_per_second

        hw = HardwareConfig(
            platform="cuda",
            gpu_name="RTX 4090",
            vram_bytes=24 * 1024 ** 3,
            ram_bytes=64 * 1024 ** 3,
            cpu_count=8,
            gpu_bandwidth_gbps=1008.0,
        )
        trainable = get_trainable(hw)

        def combined_score(entry) -> float:
            tps = estimate_tokens_per_second(
                entry.param_billions, hw.gpu_bandwidth_gbps, entry.default_quant_bits
            )
            return tps * entry.quality_score / 100.0

        if len(trainable) >= 2:
            scores = [combined_score(e) for e in trainable]
            # Verify list is non-increasing (sorted descending)
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"Score at rank {i+1} ({scores[i]:.1f}) < rank {i+2} ({scores[i+1]:.1f})"
                )
