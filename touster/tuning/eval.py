from __future__ import annotations

import json
import re
from pathlib import Path

from touster.llm.client import LLMClient
from touster.tuning.backends.base import TrainerBackend


def eval_bpb(backend: TrainerBackend, dataset_path: Path, eval_fraction: float = 0.1) -> float:
    """Compute bits-per-byte on held-out eval split. Lower is better."""
    return backend.eval_loss(dataset_path, eval_fraction)


def eval_llm_judge(
    backend: TrainerBackend,
    client: LLMClient,
    dataset_path: Path,
    n_prompts: int = 20,
    judge_model: str = "",
) -> float:
    """
    LLM-as-judge: score base+adapter outputs on 1-10 scale.
    Returns mean score (higher = better).
    Uses the same LLM client as dataset generation.
    """
    samples = _load_samples(dataset_path)
    eval_samples = samples[-n_prompts:]

    scores = []
    for sample in eval_samples:
        msgs = sample.get("messages", [])
        user_turn = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        expected = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")

        if not user_turn:
            continue

        try:
            generated = backend.generate(user_turn, max_new_tokens=256)
        except Exception:
            generated = ""

        judge_prompt = _judge_prompt(user_turn, expected, generated)
        try:
            reply = client.chat(
                [{"role": "user", "content": judge_prompt}],
                model=judge_model,
                temperature=0.0,
                max_tokens=64,
            )
            score = _parse_score(reply)
            if score is not None:
                scores.append(score)
        except Exception:
            continue

    if not scores:
        return 5.0
    return sum(scores) / len(scores)


def _judge_prompt(question: str, reference: str, generated: str) -> str:
    return (
        f"Rate the quality of this AI response on a scale of 1-10.\n\n"
        f"Question: {question[:300]}\n\n"
        f"Reference answer: {reference[:300]}\n\n"
        f"Model response: {generated[:300]}\n\n"
        f"Score (1=terrible, 10=perfect). Reply with just the number."
    )


def _parse_score(reply: str) -> float | None:
    """Extract numeric score from judge reply."""
    match = re.search(r"\b(\d+(?:\.\d+)?)\b", reply.strip())
    if match:
        score = float(match.group(1))
        return max(1.0, min(10.0, score))
    return None


def _load_samples(path: Path) -> list[dict]:
    if not path.exists():
        return []
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    return samples
