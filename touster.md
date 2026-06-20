# Touster 🍞

> Toasting a loaf — fine-tuning an LLM.

An all-in-one fine-tuning pipeline that takes you from **"I have no dataset"** to **"here's my tuned, tested, exportable model"** without 40 tabs open, manual config hell, or guessing hyperparameters.

---

## Intro

Fine-tuning today is a scavenger hunt. You find a dataset somewhere, format it by hand, write a config file guessing learning rate and LoRA rank, kick off a run, wonder if it worked, and then manually convert weights to something usable. Each step is a separate tool and a separate failure point.

Touster collapses that into one guided pipeline. The dataset can be **generated for you**. The hyperparameters are **searched, not guessed** — using an autoresearch-style self-improvement loop. Evaluation is **built in, not an afterthought**. And the whole thing runs in a visually rich terminal with a real before/after dashboard.

The pitch in one line: **fine-tuning for people who don't want to become fine-tuning experts.**

---

## Goal

Build the best all-in-one fine-tuning pipeline — faster, more effective, and less painful than the ordinary path (manual Unsloth configs, hand-formatting datasets, guessing parameters). Touster should:

- Generate, structure, or accept datasets — three input modes.
- Tell you what you can actually train on your hardware before you commit.
- **Search** for the best training recipe instead of making you guess (autoresearch-style loop).
- Show rich, interactive terminal output throughout.
- End with a real dashboard comparing old vs new, with live input/output testing.
- Export to something usable (GGUF, merged weights, model card).

**Stretch target:** 5k–10k GitHub stars. The terminal UI and a 2-click Colab are the viral surface — invest there disproportionately.

---

## What makes Touster different

Ordinary fine-tuning:

```
find dataset → format manually → write config (guess LR/rank/epochs)
→ run → wonder if it worked → manually convert to GGUF
```

Touster:

```
analyze hardware → generate/structure/load dataset → auto-validate
→ search best recipe (self-improvement loop) → train
→ before/after dashboard → export
```

The differentiator isn't raw speed (Unsloth already won speed). It's **removing decisions**. The dataset can be generated, the hyperparameters are searched, and eval is built in.

---

## Compatibility

Be specific, don't overpromise — bitsandbytes/Triton break on Windows, and Mac has no CUDA (MLX is a separate stack).

| Platform | Path |
|---|---|
| Linux + NVIDIA | Full GPU training |
| Google Colab | Full GPU training (2-click notebook) |
| Windows + NVIDIA | Via WSL |
| macOS (Apple Silicon) | MLX path |
| CPU / any | Tiny-model fallback |

---

## Dataset modes

Three sources of training data:

- **Mode 0 — Generate.** User describes what they want; Touster generates the dataset via API call or local Ollama call. Default sample count, user-adjustable.
- **Mode 1 — Structure.** User has raw, unstructured data; Touster uses an API or Ollama call to structure it into a fine-tuning-ready format.
- **Mode 2 — Bring your own.** User already has a dataset. No API/Ollama call needed. Skips the validation-rebuild step.

A **dedup + quality-filter pass** sits between generation and validation for Modes 0 and 1 — AI-generated data is full of near-duplicates and format drift.

---

## Flow (fixed)

### 1. Hardware analysis

A `whichllm`-style scan of the current system. Output:

- **System specs** in one box.
- A **table** of models (by id) that can be pulled from Ollama or HF and tuned on this machine: model name, params, estimated t/s, quality score.
- **Top 3** (by t/s + quality) rendered brighter.

This is the strongest hook — the "what can I actually train on this machine?" question is the most screenshot-able moment. Make it gorgeous.

### 2. User input

- API config (if using an API), or Ollama port (if local).
- If multiple local models are present → choosable option.
- Prompt for the AI to generate the dataset as required.
- Number of samples — sensible default, user can change.

### 3. Dataset validation

*(Skipped for Mode 2.)*

Verify dataset structure and make it conformant for fine-tuning. Catches malformed rows, wrong fields, format drift before any compute is spent.

A **dry-run preview** caps this step: show a few formatted samples, the chosen chat template, token counts, and the auto-selected starting hyperparameters — *before* committing compute.

### 4. Fine-tuning with self-improvement loop

The autoresearch-adapted core. Interactive training UI in the terminal.

**Three-file structure** (mirrors `karpathy/autoresearch`):

- **`recipe.py`** — the single file the agent edits. Holds the LoRA config + loop: learning rate, rank/alpha, target modules, batch size, epochs, warmup, scheduler. The agent tunes *only* these knobs — not architecture.
- **`program.md`** — plain-English instructions that drive the agent ("start conservative on LR, prioritize eval score, halve LR and retry on divergence"). The **human** edits this. This is the headline feature: *programming the tuner in English.*
- **`prepare.py`** — frozen scaffolding. Loads the base model, formats the dataset, holds the eval set + scoring function. The agent never touches it.

**The loop:**

```
1. Agent reads program.md + current recipe.py
2. Agent proposes a change (e.g. "LR 2e-4, rank 32")
3. Run a SHORT trial — fixed budget (e.g. 200 steps OR 5 min wall-clock)
4. Measure eval score on a held-out set
5. Compare to best-so-far → keep or discard
6. Log the experiment (change, score, decision)
7. Repeat until budget exhausted (N trials / X hours)
8. Final run: take the winning recipe, train to completion
```

**Why fixed trial budget:** every trial runs the same number of steps, so the only variable is the agent's change — not run length. This makes keep/discard decisions trustworthy. (Inherited directly from autoresearch.)

**Why a short trial predicts the full run:** a bad LR diverges or plateaus within ~100–200 steps; a good one shows clean descent. The loop is *early-signal pruning* — cheaply eliminating the ~80% of configs that are clearly wrong, which is exactly what manual tuners waste days on. It finds a strong recipe for *your* hardware + dataset, not a global optimum.

**Eval metric:**

- Inside the loop → cheap **eval loss / bpb** on a held-out split (deterministic, run dozens of times). bpb is vocab-independent, so it's fairly comparable.
- For finalists only → **LLM-as-judge**, reusing the API/Ollama connection from dataset generation. Score ~20–30 held-out prompts old-vs-new on a 1–10 scale.
- **Hybrid:** rank all trials by cheap eval loss, then run the expensive judge only on the top 3 survivors. Fast loop, quality-aligned final pick.

**Agent guardrails:** the agent may tune LR, rank/alpha, target modules, warmup, epochs/steps. It may **not** touch the base model (user picks in step 1), the dataset, or the eval harness. Unlike autoresearch (pretrain research, everything is fair game), Touster wants a *reliable tool*, not a science experiment.

### 5. LLM dashboard

- All run info + examples.
- **Old vs new** comparison.
- A terminal window for **testing both** — feed inputs, see outputs from both models side by side.

---

## What's borrowed from autoresearch (and what isn't)

`karpathy/autoresearch` is pretrain-from-scratch, single-NVIDIA-GPU, deliberately minimal — an agent rewrites the whole model in `train.py` against `val_bpb` on a fixed 5-minute budget.

**Borrowed:**

- The **`program.md` pattern** — human programs the agent in plain English.
- The **keep/discard loop** with a **fixed trial budget** for comparability.
- **bpb** as a fair, vocab-independent eval metric.
- The "find the best model *for your platform* in a time budget" framing.

**Deliberately different:**

- Touster **fine-tunes existing models** (LoRA), it does not pretrain from scratch.
- The agent tunes a **recipe**, not architecture — with guardrails.
- **Cross-platform** by design (autoresearch is NVIDIA-only; cross-platform lives in its forks).
- Includes **dataset generation, validation, dashboard, and export** — out of autoresearch's scope.

**Honest framing for the eventual README:**

> Touster uses an autoresearch-style self-improvement loop: instead of guessing hyperparameters, an agent runs short fine-tuning trials against a held-out eval, keeps what improves the score, and discards what doesn't — converging on the best recipe for your hardware and dataset. Inspired by Karpathy's autoresearch, adapted from pretrain-from-scratch to LoRA fine-tuning.

Don't claim "powered by autoresearch" if it's really an internal loop — that repo is famous and people will check. Credit as inspiration; that's accurate and rides the wave without overclaiming.

---

## Must-not-miss (gaps to close)

- **Evaluation is the actual core** — without a real metric the self-improvement loop has nothing to optimize. bpb + LLM-judge, as above.
- **Export / deployment** — GGUF (for Ollama/llama.cpp), merged weights, pushed HF model card. Stopping at "adapter on disk" loses half the users.
- **Resumability & checkpointing** — Colab disconnects, spot instances die. A run that can't resume won't be trusted.
- **Dataset dedup / cleaning** — between generation and validation for Modes 0/1.
- **Cost / time estimate upfront** — "≈45 min, ≈$0 local or ≈$1.20 API gen calls" before a run.

---

## Viral levers (for the star goal)

- A stunning **demo GIF** of the terminal UI in the README — this is the asset.
- A **2-click Colab** notebook.
- One well-timed post (HN / r/LocalLLaMA / X).
- **Interop bonus:** autoresearch links cross-platform forks in its README. A clean fine-tune layer that interoperates with that ecosystem could earn a link from the parent repo — worth more than any single post.
