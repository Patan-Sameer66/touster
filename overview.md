# Touster — pipeline overview (rewrite, 2026-07-06)

Clean-slate rewrite. Five stages, one notebook, package does the heavy lifting.

## 1. Hardware check
Detect GPU/CPU/RAM, print a ranked table of models that fit, with rough
speed estimates. Read-only — nothing to configure here.

## 2. Config
Single editable block. Everything downstream reads from here:
- base model, dataset mode + its inputs
- loop size (trials, steps)
- LLM client (optional — API or Ollama)
- **save/export toggles** (local save, merged weights, GGUF, HF Hub push) —
  configured here, *executed* at the end of stage 4, not a separate stage

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
4. Checkpoint every trial — resumable if the run dies partway.
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

---

## Open question for review
Is 5 stages the right cut, or is something missing? Candidates I'd flag
(delete this section once resolved):
- **Install** isn't a numbered stage above — folds into stage 1 (light deps)
  and stage 2→3 transition (heavy deps, once hardware is known). Worth a
  one-line callout in the notebook even if it's not a "stage."
- **Resumability** (checkpoint/resume across Colab disconnects) is a
  cross-cutting property of stage 4 — called out above so it doesn't get
  lost in the rewrite, not a stage of its own.
- Nothing else looks missing for a fixed pipeline. If you want a 6th stage
  it'd be an explicit **cost/time estimate** upfront ("≈45 min, ≈$0 local")
  before committing to a run — currently absent everywhere.
