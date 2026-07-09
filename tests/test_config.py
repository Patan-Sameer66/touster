"""Regression tests for touster/config.py's RecipeConfig.apply_diff bounds checks."""
from __future__ import annotations

import pytest

from touster.config import RecipeConfig


def test_apply_diff_accepts_valid_change():
    r = RecipeConfig()
    updated = r.apply_diff({"learning_rate": 1e-4})
    assert updated.learning_rate == 1e-4


def test_apply_diff_rejects_unknown_knob():
    r = RecipeConfig()
    with pytest.raises(ValueError):
        r.apply_diff({"nope": 1})


def test_apply_diff_rejects_out_of_bounds():
    r = RecipeConfig()
    with pytest.raises(ValueError):
        r.apply_diff({"learning_rate": 0.0})
    with pytest.raises(ValueError):
        r.apply_diff({"lora_rank": -4})


def test_apply_diff_rejects_empty_target_modules():
    r = RecipeConfig()
    with pytest.raises(ValueError):
        r.apply_diff({"target_modules": []})


def test_apply_diff_rejects_bool_for_numeric_knob():
    """Regression test: bool is an int subclass in Python, so isinstance(True, int)
    is True — without an explicit bool check, {"lora_rank": True} would silently
    coerce to lora_rank=1 instead of being rejected as a type error."""
    r = RecipeConfig()
    with pytest.raises(ValueError):
        r.apply_diff({"lora_rank": True})
