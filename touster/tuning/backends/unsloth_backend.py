from __future__ import annotations

import json
import math
import time
from pathlib import Path

from touster import display


class UnslothBackend:
    """NVIDIA GPU backend via Unsloth (2x speed over vanilla HF). Requires Linux/WSL + CUDA."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load_model(
        self,
        model_id: str,
        lora_rank: int,
        lora_alpha: int,
        target_modules: list[str],
    ) -> None:
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        from unsloth import FastLanguageModel

        print(f"Loading {model_id} via Unsloth...")

        # Suppress Unsloth's own verbose startup banner — unrelated to our
        # output stack, just noisy stdout/stderr from the library itself.
        _sink = io.StringIO()
        with redirect_stdout(_sink), redirect_stderr(_sink):
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                load_in_4bit=True,
            )

        effective_targets = _find_lora_targets(self._model, target_modules)
        if set(effective_targets) != set(target_modules):
            display.warning(
                f"target_modules {list(target_modules)} not in model -- "
                f"auto-detected: {effective_targets}"
            )

        self._model = FastLanguageModel.get_peft_model(
            self._model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=effective_targets,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    def train_steps(
        self,
        dataset_path: Path,
        max_steps: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        warmup_steps: int,
        scheduler: str,
        wall_clock_limit_secs: int = 0,
    ) -> dict:
        import tempfile
        from datasets import load_dataset as hf_load
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import is_bfloat16_supported

        assert self._model is not None and self._tokenizer is not None

        raw_ds = hf_load("json", data_files=str(dataset_path), split="train")

        def _format(example):
            msgs = example.get("messages", [])
            text = ""
            for m in msgs:
                text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            return {"text": text}

        formatted = raw_ds.map(_format, remove_columns=raw_ds.column_names)

        use_bf16 = is_bfloat16_supported()
        args = TrainingArguments(
            output_dir=str(Path(tempfile.gettempdir()) / "touster_unsloth"),
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            lr_scheduler_type=scheduler,
            fp16=not use_bf16,
            bf16=use_bf16,
            optim="adamw_8bit",
            logging_steps=max(1, max_steps // 5),
            save_steps=max_steps + 1,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=formatted,
            dataset_text_field="text",
            max_seq_length=512,
            args=args,
        )
        start = time.time()
        result = trainer.train()
        return {"steps": result.global_step, "train_loss": result.training_loss}

    def eval_loss(self, dataset_path: Path, eval_fraction: float = 0.1) -> float:
        import torch

        assert self._model is not None and self._tokenizer is not None
        from touster.tuning.backends.cpu_backend import _load_samples, _encode_samples

        samples = _load_samples(dataset_path)
        n_eval = max(1, int(len(samples) * eval_fraction))
        eval_samples = samples[-n_eval:]
        encodings = _encode_samples(eval_samples, self._tokenizer)
        if not encodings:
            return float("inf")

        self._model.eval()
        total_loss = 0.0
        total_tokens = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.no_grad():
            for i in range(len(eval_samples)):
                input_ids = encodings["input_ids"][i].unsqueeze(0).to(device)
                attention_mask = encodings["attention_mask"][i].unsqueeze(0).to(device)
                labels = encodings["labels"][i].unsqueeze(0).to(device)
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                n_tokens = (labels != -100).sum().item()
                total_loss += outputs.loss.item() * n_tokens
                total_tokens += n_tokens

        nats_per_token = total_loss / max(total_tokens, 1)
        return (nats_per_token / math.log(2)) / 3.8

    def save_adapter(self, output_dir: Path) -> None:
        assert self._model is not None and self._tokenizer is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(output_dir))
        self._tokenizer.save_pretrained(str(output_dir))

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        import torch
        from unsloth import FastLanguageModel

        assert self._model is not None and self._tokenizer is not None
        FastLanguageModel.for_inference(self._model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def unload(self) -> None:
        import torch
        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()


# -- helpers ------------------------------------------------------------------

def _find_lora_targets(model, requested: list[str]) -> list[str]:
    """Return LoRA target module names that actually exist in model.

    Mirrors cpu_backend._find_lora_targets -- kept in sync so both backends
    auto-detect the correct layers for any model architecture.
    """
    import torch.nn as nn

    present = {name.split(".")[-1] for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    matched = [t for t in requested if t in present]
    if matched:
        return matched

    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "c_attn", "c_proj",
        "query_key_value", "dense",
        "dense_h_to_4h", "dense_4h_to_h",
    ]
    detected = [c for c in candidates if c in present]
    return detected if detected else list(present)
