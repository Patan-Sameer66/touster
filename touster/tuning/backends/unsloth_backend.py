from __future__ import annotations

import json
import math
import time
from pathlib import Path

from touster.console import console


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
        from unsloth import FastLanguageModel

        console.print(f"  [touster.dim]Loading [touster.model]{model_id}[/touster.model] via Unsloth…[/touster.dim]")
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            load_in_4bit=True,
        )
        self._model = FastLanguageModel.get_peft_model(
            self._model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
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
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset as hf_load

        assert self._model is not None and self._tokenizer is not None

        raw_ds = hf_load("json", data_files=str(dataset_path), split="train")

        def _format(example):
            msgs = example.get("messages", [])
            text = ""
            for m in msgs:
                text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            return {"text": text}

        formatted = raw_ds.map(_format)

        args = TrainingArguments(
            output_dir="/tmp/touster_unsloth",
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            lr_scheduler_type=scheduler,
            fp16=True,
            logging_steps=10,
            save_steps=max_steps + 1,
        )
        trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=formatted,
            dataset_text_field="text",
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
        with torch.no_grad():
            for i in range(len(eval_samples)):
                input_ids = encodings["input_ids"][i].unsqueeze(0).cuda()
                attention_mask = encodings["attention_mask"][i].unsqueeze(0).cuda()
                labels = encodings["labels"][i].unsqueeze(0).cuda()
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
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def unload(self) -> None:
        import torch
        self._model = None
        self._tokenizer = None
        torch.cuda.empty_cache()
