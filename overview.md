# Touster — pipeline overview (rewrite, 2026-07-06)

Clean-slate rewrite. Five stages, one notebook, package does the heavy lifting.

## Stack decision: display
Old build used `rich` for all terminal output (tables, panels, progress
bars). Swapped to **`IPython.display` (HTML)** — the notebook/Colab cell is
the real surface, not a terminal, so render actual HTML (`display(HTML(...))`)
for the hardware table, config summary, trial log, and dashboard instead of
rich's console-width-guessing terminal styling. Applies to every stage below
that prints a table or status — not called out per-stage.

## 1. Hardware check
Detect GPU/CPU/RAM, print a ranked table of models that fit, with rough
speed estimates. Read-only — nothing to configure here.

**Install** isn't its own stage — light deps (psutil, pynvml) install here;
heavy deps (torch/peft/transformers, GPU/CPU/MLX variant) install right
after config, once hardware + chosen model are both known.

## 2. Config
Single editable block. Everything downstream reads from here:
- base model, dataset mode + its inputs
- loop size (trials, steps)
- LLM client (optional — API or Ollama)
- **save/export toggles** (local save, merged weights, GGUF, HF Hub push) —
  configured here, *executed* at the end of stage 4, not a separate stage

**Cost/time estimate** prints here, before any compute or API spend starts:
`≈45 min, ≈$0 (local Ollama)` or `≈$1.20 (API dataset gen + judge calls)`.
Computed from stage 1's t/s estimate × `max_trials × trial_steps` for time,
and `num_samples × tokens/sample × API price` (or $0 if Ollama) for cost.
One print line, not a stage of its own.

## 3. Data source (3 modes)
- **Mode 0 — generate**: topic string → LLM writes Q&A pairs
- **Mode 1 — structure**: raw `.txt`/`.md` → LLM turns it into Q&A pairs
- **Mode 2 — bring-your-own**: local file / URL / HF dataset id → load directly

All three end in the same contract: dedup + quality-filter (modes 0/1 only,
skipped for mode 2) → validate + repair → **golden-format dataset on disk**.
Dataset gen must never crash the notebook on one bad LLM sample — drop and
continue, only hard-fail if nothing usable comes out after retries.
A short preview (sample rows, token counts, starting hyperparameters) caps
this stage as a sanity check before any compute is spent.

## 4. Tuning
Search + train + export, one stage:
1. Fixed-budget trials (same step count each trial, so only the recipe
   changes) against a held-out eval split (bpb).
2. **Search algorithm — the actual rewrite target.** Current heuristic +
   freeform-LLM-diff proposer has no surrogate model and can't reliably beat
   the default recipe in a 3–20 trial budget. Replace with an Optuna TPE
   sampler (or equivalent surrogate-based search); demote the LLM to a
   language-guided prior that narrows the search space, not the sole
   decision-maker.
3. LLM-as-judge on the top-k survivors only (cheap eval loss first, expensive
   judge second).
4. **Resumable** — checkpoint every trial, so a Colab disconnect or spot-
   instance death doesn't lose the run. Cross-cutting property of this
   stage, not a stage of its own.
5. **If every trial fails or nothing beats the default** (backend crash,
   OOM, bad reload): fall back to the default recipe and still run final
   training. Never raise and kill the notebook.
6. Final run trains the winning (or fallback) recipe to completion, then
   executes stage 2's export toggles: local save, merged weights, GGUF,
   model card, optional HF Hub push.

## 5. Dashboard
One consolidated view, not five separate cells:
- trial log (every trial's eval score + recipe diff, kept/discarded)
- base-vs-fine-tuned side-by-side generation on fixed + your own prompts
- run summary (best score, winning recipe, every artifact + its size)
