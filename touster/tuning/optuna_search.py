"""TPE-based hyperparameter search — replaces the old freeform-LLM-diff
proposer (touster/tuning/agent.py in the pre-rewrite codebase).

Root cause this fixes (see research.md section 3): the old proposer had no
surrogate model — it either picked a random grid value or asked an LLM to
freehand-guess one new number, with no memory of *why* past guesses did or
didn't work beyond "better than best so far, or not." In a 3-20 trial
budget that's statistically unlikely to beat the default recipe.

Optuna's TPE sampler *is* the surrogate model: after every trial it splits
history into "good" and "bad" groups and samples the next point from where
those groups' distributions diverge most — the actual "smart darts"
mechanism. The LLM, if given, only narrows the search space bounds once up
front (a prior) — it never proposes trial values directly. That split
(LLM narrows, TPE decides) is what the 2026 arXiv paper (2602.11171) found
actually works, versus either alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import optuna

from touster.config import RecipeConfig
from touster.llm.client import LLMClient

# Keep Optuna's own logging out of the notebook — we print our own trial lines.
optuna.logging.set_verbosity(optuna.logging.WARNING)

_DEFAULT_SEARCH_SPACE = {
    "lr_bounds": [5e-5, 1e-3],
    "rank_choices": [8, 16, 32, 64],
    "warmup_choices": [0, 5, 10, 20],
    "scheduler_choices": ["cosine", "linear", "constant"],
}


def create_study(run_dir: Path, seed: int | None = None) -> optuna.Study:
    """Create (or resume) a TPE-sampler study, persisted to run_dir/optuna.db.

    SQLite storage is what makes this resumable across notebook restarts —
    Optuna reloads the full trial history from disk instead of us having to
    manually replay it into the sampler.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(run_dir / 'optuna.db').as_posix()}"
    study = optuna.create_study(
        study_name="touster",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=3),
    )
    _fail_stale_running_trials(study)
    return study


def _fail_stale_running_trials(study: optuna.Study) -> None:
    """Mark any trial left in RUNNING state as FAILED.

    A trial in RUNNING means a previous process died between `study.ask()`
    (which persists the trial row immediately) and `study.tell()` — a hard
    kill, OOM, or unclean Colab disconnect. Optuna doesn't auto-resolve
    these, and TPE would otherwise silently model an unfinished trial as
    part of the search history on every future resume.
    """
    stale = study.get_trials(states=(optuna.trial.TrialState.RUNNING,))
    for t in stale:
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)


def narrow_search_space_with_llm(client: LLMClient | None, recipe: RecipeConfig, model: str = "") -> dict:
    """Ask the LLM to narrow the search space bounds — a prior, not a decision.

    Never raises — falls back to the module defaults on any failure (bad
    reply, no client, network error). Narrowing is an optimization; a wrong
    or missing answer must never block the search from running.
    """
    if client is None:
        return dict(_DEFAULT_SEARCH_SPACE)

    prompt = (
        "You are narrowing a hyperparameter search space for LoRA fine-tuning "
        f"of {recipe.base_model}. Suggest a narrower learning-rate range and a "
        "shortlist of LoRA ranks worth trying for a model this size. "
        'Reply with ONLY a JSON object: {"lr_min": float, "lr_max": float, "ranks": [int, ...]}'
    )
    try:
        reply = client.chat([{"role": "user", "content": prompt}], model=model, temperature=0.2, max_tokens=128)
        start, end = reply.find("{"), reply.rfind("}")
        if start == -1 or end == -1:
            return dict(_DEFAULT_SEARCH_SPACE)
        parsed = json.loads(reply[start:end + 1])
        lr_min, lr_max = float(parsed["lr_min"]), float(parsed["lr_max"])
        ranks = sorted({int(r) for r in parsed["ranks"]})
        if not (0 < lr_min < lr_max) or not ranks:
            return dict(_DEFAULT_SEARCH_SPACE)
        return {
            "lr_bounds": [lr_min, lr_max],
            "rank_choices": ranks,
            "warmup_choices": list(_DEFAULT_SEARCH_SPACE["warmup_choices"]),
            "scheduler_choices": list(_DEFAULT_SEARCH_SPACE["scheduler_choices"]),
        }
    except Exception:
        return dict(_DEFAULT_SEARCH_SPACE)


def close_study(study: optuna.Study) -> None:
    """Release the SQLite file handle Optuna's storage holds open.

    Without this, the .db file stays locked for the life of the process —
    harmless on Linux/macOS, but it breaks temp-dir cleanup and re-running
    the tuning cell in the same notebook session on Windows (file still
    locked by the previous run). Best-effort: reaches into Optuna's private
    storage internals since there's no public "close" API for this.
    """
    try:
        backend = study._storage._backend
        backend.remove_session()
        backend.engine.dispose()
    except AttributeError:
        pass  # in-memory or a different storage backend — nothing to release


def suggest_recipe_diff(trial: optuna.trial.Trial, search_space: dict) -> dict:
    """The actual decision-maker for this trial's recipe — no LLM call happens here."""
    lr_min, lr_max = search_space["lr_bounds"]
    rank = trial.suggest_categorical("lora_rank", search_space["rank_choices"])
    return {
        "learning_rate": trial.suggest_float("learning_rate", lr_min, lr_max, log=True),
        "lora_rank": rank,
        "lora_alpha": rank,  # alpha = rank is standard LoRA practice
        "warmup_steps": trial.suggest_categorical("warmup_steps", search_space["warmup_choices"]),
        "scheduler": trial.suggest_categorical("scheduler", search_space["scheduler_choices"]),
    }
