"""LLM-judge quality gate for generated/structured datasets.

Golden format (schema.py) checks *shape* — has messages, has user+assistant,
non-empty content. It says nothing about whether the answer is actually
correct or good. This module adds that check, using an LLM judge:

- Mode 0 (generate): no external ground truth exists — the LLM wrote both
  the question and the answer — so judge the pair on its own merits
  (coherent, on-topic, complete).
- Mode 1 (structure): the source raw text IS the ground truth — judge
  whether the extracted Q&A pair is faithful to it (no invented facts).
- Mode 2 (bring-your-own): already-trusted data, never filtered here.

Never crashes the pipeline on a bad/odd judge reply — same resilience
contract as the rest of dataset generation: parse defensively, fall back to
a neutral score on any failure, keep going.
"""
from __future__ import annotations

import re

from touster import display
from touster.dataset.schema import Dataset, Sample

_SELF_JUDGE_TEMPLATE = """\
Rate this question-and-answer pair's quality on a scale of 1-{scale}: is the \
answer correct, complete, and well-formed for the question?
Question: {q}
Answer: {a}
Reply with ONLY a number."""

_GROUNDED_JUDGE_TEMPLATE = """\
Rate 1-{scale}: is this question-and-answer pair faithful to the source text \
below (no invented facts, no contradictions)?
Source: {source}
Question: {q}
Answer: {a}
Reply with ONLY a number."""

_MAX_JUDGE_TOKENS = 8  # judge only needs to reply with a number — keep it cheap


def _sample_qa(sample: Sample) -> tuple[str, str]:
    """Pull the first user question and first assistant answer out of a sample."""
    q = next((m.content for m in sample.messages if m.role == "user"), "")
    a = next((m.content for m in sample.messages if m.role == "assistant"), "")
    return q, a


def _parse_score(reply: str, scale: int) -> float | None:
    """Extract a numeric score from a judge reply. Never raises."""
    match = re.search(r"\d+(?:\.\d+)?", reply.strip())
    if not match:
        return None
    return max(1.0, min(float(scale), float(match.group())))


def _judge_one(client, prompt: str, model: str, scale: int) -> float:
    """Score one prompt. Falls back to the scale midpoint on any failure —
    a network hiccup shouldn't unfairly drop (or unfairly keep) a sample."""
    fallback = (scale + 1) / 2
    try:
        reply = client.chat([{"role": "user", "content": prompt}], model=model, temperature=0.0, max_tokens=_MAX_JUDGE_TOKENS)
    except Exception as exc:
        display.warning(f"Quality judge call failed ({exc}), using neutral score {fallback}")
        return fallback
    score = _parse_score(reply, scale)
    if score is None:
        display.warning(f"Quality judge gave an unparseable reply ({reply[:40]!r}), using neutral score {fallback}")
        return fallback
    return score


def filter_by_quality(
    ds: Dataset,
    client,
    model: str,
    mode: int,
    min_score: float,
    scale: int,
    source_text: str = "",
) -> tuple[Dataset, list[str]]:
    """Drop samples scoring below min_score. Modes 0/1 only; mode 2 is a no-op passthrough."""
    if mode not in (0, 1) or client is None:
        return ds, []

    kept: list[Sample] = []
    warnings: list[str] = []
    for i, sample in enumerate(ds.samples):
        q, a = _sample_qa(sample)
        if mode == 0:
            prompt = _SELF_JUDGE_TEMPLATE.format(scale=scale, q=q, a=a)
        else:
            prompt = _GROUNDED_JUDGE_TEMPLATE.format(scale=scale, source=source_text[:3000], q=q, a=a)

        score = _judge_one(client, prompt, model, scale)
        if score < min_score:
            warnings.append(f"Sample {i}: dropped, quality score {score:.1f}/{scale} < {min_score}")
            continue
        kept.append(sample)

    return Dataset(samples=tuple(kept)), warnings
