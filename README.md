# Touster

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Patan-Sameer66/touster/blob/main/touster.ipynb)

Fine-tuning an LLM is nine hours of reading blog posts about learning rates, one hour of copying somebody's `TrainingArguments`, and a dawning realization that you set `lora_alpha` to a number you chose because it looked nice. Touster is the pipeline that does all of that so you don't have to have opinions.

Point it at a topic (or a file, or a Hugging Face dataset). It builds the data, scans your hardware, runs a search loop that keeps trying recipes until one stops embarrassing itself, shows you before-and-after, and hands you weights you can actually run. One notebook. No decisions required, though you're welcome to have some.

## What it actually does

Six phases, top to bottom, no wiring:

1. **Hardware scan** — looks at your GPU/CPU/RAM and tells you which models will fit instead of letting you find out via OOM at step 40.
2. **Dataset** — three ways to get one (below). Dedups near-duplicates and validates the shapes so garbage doesn't reach the trainer.
3. **Self-improvement loop** — the interesting part. Short trials, each proposing one recipe change, keeping the best, throwing away the rest.
4. **Dashboard** — base model vs. fine-tuned, same prompts, side by side, so you can confirm something happened.
5. **Export** — merged fp16 weights for transformers, GGUF for Ollama/llama.cpp, a model card, optional push to the Hub.

## Three ways to get a dataset

Set `DATASET_MODE` in the config cell:

| Mode | You bring | Touster does |
|------|-----------|--------------|
| `0` generate | a topic string | an LLM writes Q&A pairs |
| `1` structure | raw `.txt` / `.md` | an LLM carves it into Q&A pairs |
| `2` bring-your-own | a `.jsonl`, a URL, or a HF dataset id | loads and converts it, no LLM |

Mode 2 speaks fluent messages-format, Alpaca, and ShareGPT, and will download from a URL or straight off the Hub (`tatsu-lab/alpaca`, or `org/name/split` if you're picky about which split).

## The LLM is optional and it wants you to know that

Modes 0 and 1 need a text model to write the data — local **Ollama** or any OpenAI-compatible API. The optional LLM-judge and the smart recipe-proposer use it too. Leave every LLM field blank and nothing breaks: the loop falls back to a deterministic heuristic search that costs zero dollars and zero tokens and still finds you a decent recipe. It's less clever. It's also free. You decide what you're worth.

## The loop, because it's the whole point

Naive hyperparameter search wanders. It finds a good recipe at trial 3, then spends trials 4 through 19 mutating that recipe further and further downhill, and finally trains the production model on whatever sad configuration it drifted into by the end. Touster's loop branches every new proposal from the **best trial so far**, not the last one, so a bad guess costs you one trial instead of the whole run. The final model is trained on the recipe that actually won. This sounds obvious. It was not obvious to the first version.

Guardrails included: learning rate can't underflow to zero, `lora_rank` can't go negative, `target_modules` can't be empty, and no-op proposals get skipped instead of burning a trial on the illusion of progress.

## Quickstart

**Colab (recommended):** click the badge. Runtime -> change runtime type -> T4 GPU (free). Runtime -> Run all. Edit the config cell if you have preferences; otherwise watch it work.

**Local:**

```bash
# GPU (Linux/WSL + CUDA; Unsloth where supported, HF+PEFT otherwise)
pip install "touster[gpu] @ git+https://github.com/Patan-Sameer66/touster.git"

# CPU-only (slow, but it runs and it validates the whole pipeline)
pip install "touster[cpu] @ git+https://github.com/Patan-Sameer66/touster.git"

# Apple Silicon
pip install "touster[mlx] @ git+https://github.com/Patan-Sameer66/touster.git"
```

Then open `touster.ipynb` and run top to bottom. Every code cell has a one-line note above it explaining what it does, so you're never guessing which cell is the one that matters. (It's cell 2. Cell 2 is the one that matters.)

## Configuration

You edit one cell. It holds the base model, the dataset mode, the loop size (`MAX_TRIALS`, `TRIAL_STEPS`), optional LLM credentials, and export toggles (`EXPORT_MERGED`, `EXPORT_GGUF`, `GGUF_QUANTIZE`). Everything downstream reads from it. If you find yourself editing a different cell, stop and ask yourself why.

## Tests

Two standalone harnesses, no ceremony, run them directly:

```bash
python tests/test_datagen.py          # parser battery vs. every malformed LLM output we've seen
python tests/test_datagen.py --live   # optional: real generation through a running Ollama
python tests/test_loop.py             # loop logic on a mock backend, seconds, no downloads
python tests/test_loop.py --real      # optional: one real LoRA trial on tiny-gpt2
```

The mode-0 JSON parser survives trailing commas, markdown fences, unescaped backslashes, flat-vs-nested message shapes, comments, and truncation, because small models produce all of those, usually in the same response. The full `pytest` suite lives in `tests/` and covers dataset, tuning, export, and an end-to-end smoke run.

## Honest limitations

- CPU works and is genuinely useful for validating the pipeline, but a 7B model on CPU is a way to learn patience, not a way to fine-tune quickly.
- Unsloth is Linux/CUDA; on Windows or plain CUDA it quietly uses HF + PEFT instead.
- The heuristic proposer is fine. The LLM proposer is better. Neither is a hyperparameter oracle, and anyone who tells you they have one is selling something.

## License

See repository.
