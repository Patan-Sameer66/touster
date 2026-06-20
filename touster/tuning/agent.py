from __future__ import annotations

"""Recipe proposer: LLM-agent with deterministic heuristic fallback."""

import json
import random
import re
from dataclasses import asdict

from touster.config import ALLOWED_RECIPE_KNOBS, RecipeConfig
from touster.console import console


# ── LLM-based proposer ───────────────────────────────────────────────────────

_SYSTEM = """You are a LoRA fine-tuning hyperparameter optimizer.
You receive the current recipe (as JSON) and the program instructions.
Propose ONE change to improve the model. Output ONLY a JSON object with
the single field you want to change and its new value. Example: {"learning_rate": 1e-4}
Only use these keys: """ + ", ".join(sorted(ALLOWED_RECIPE_KNOBS))


def propose_llm(
    client,
    recipe: RecipeConfig,
    program_md: str,
    last_bpb: float,
    best_bpb: float,
    trial_id: int,
) -> dict:
    """Ask LLM to propose a recipe diff. Returns validated diff dict."""
    recipe_json = json.dumps(_recipe_to_dict(recipe), indent=2)
    user_msg = (
        f"Program instructions:\n{program_md}\n\n"
        f"Current recipe:\n{recipe_json}\n\n"
        f"Trial {trial_id}: last bpb={last_bpb:.4f}, best bpb={best_bpb:.4f}. "
        f"Propose ONE hyperparameter change to improve eval bpb (lower is better)."
    )
    try:
        reply = client.chat(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=128,
        )
        diff = _parse_diff(reply)
        if diff:
            _validate_diff(diff)
            return diff
    except Exception as e:
        console.print(f"  [touster.warning]LLM proposer failed ({e}), using heuristic[/touster.warning]")

    return propose_heuristic(recipe, trial_id, last_bpb, best_bpb)


# ── Heuristic fallback ────────────────────────────────────────────────────────

# Bounded search grids
_LR_GRID = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
_RANK_GRID = [8, 16, 32, 64]
_BATCH_GRID = [1, 2, 4, 8]
_WARMUP_GRID = [0, 5, 10, 20]
_SCHEDULER_GRID = ["cosine", "linear", "constant"]


def propose_heuristic(
    recipe: RecipeConfig,
    trial_id: int,
    last_bpb: float,
    best_bpb: float,
) -> dict:
    """Simple coordinate-descent heuristic with halve-on-divergence."""
    rng = random.Random(trial_id)

    # Divergence detection: loss got worse by >10%
    if last_bpb != float("inf") and last_bpb > best_bpb * 1.10:
        return {"learning_rate": recipe.learning_rate / 2}

    def _alt(grid, current):
        opts = [x for x in grid if x != current]
        return opts if opts else grid

    strategies = [
        lambda: {"learning_rate": rng.choice(_alt(_LR_GRID, recipe.learning_rate))},
        lambda: {"lora_rank": rng.choice(_alt(_RANK_GRID, recipe.lora_rank))},
        lambda: {"lora_alpha": recipe.lora_rank},  # alpha = rank is a common heuristic
        lambda: {"warmup_steps": rng.choice(_alt(_WARMUP_GRID, recipe.warmup_steps))},
        lambda: {"scheduler": rng.choice(_alt(_SCHEDULER_GRID, recipe.scheduler))},
    ]
    return rng.choice(strategies)()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _recipe_to_dict(recipe: RecipeConfig) -> dict:
    d = asdict(recipe)
    d.pop("base_model", None)
    return d


def _parse_diff(reply: str) -> dict | None:
    """Extract the first JSON object from LLM reply (handles nested brackets like lists)."""
    # Scan for balanced braces so {"target_modules": ["q_proj"]} parses correctly
    start = reply.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(reply[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(reply[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _validate_diff(diff: dict) -> None:
    unknown = set(diff) - ALLOWED_RECIPE_KNOBS
    if unknown:
        raise ValueError(f"Disallowed knobs in proposal: {unknown}")
