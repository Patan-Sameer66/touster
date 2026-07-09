from __future__ import annotations

import json
import time
from pathlib import Path

from touster import display


class CPUBackend:
    """HF transformers + PEFT backend. Runs on CPU; validates the full pipeline locally."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_id: str = ""

    def load_model(
        self,
        model_id: str,
        lora_rank: int,
        lora_alpha: int,
        target_modules: list[str],
    ) -> None:
        """Load base model + attach LoRA adapters in-memory."""
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {model_id} on CPU...")
        self._model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(model_id)

        effective_targets = _find_lora_targets(base, tuple(target_modules))
        if set(effective_targets) != set(target_modules):
            display.warning(
                f"target_modules {list(target_modules)} not in model -- "
                f"auto-detected: {effective_targets}"
            )

        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=effective_targets,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(base, lora_cfg)
        display.success(f"Model ready -- {self._model.num_parameters():,} params total")

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
        """Train for up to max_steps. Returns {"steps": int, "train_loss": float}."""
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
        from transformers import DataCollatorForLanguageModeling

        assert self._model is not None and self._tokenizer is not None

        samples = _load_samples(dataset_path)
        train_samples = samples[: int(len(samples) * 0.9)]
        encodings = _encode_samples(train_samples, self._tokenizer)
        if not encodings:
            return {"steps": 0, "train_loss": float("inf")}

        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], encodings["labels"])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._model.train()
        optimizer = AdamW(self._model.parameters(), lr=learning_rate)
        if scheduler == "cosine":
            sched = CosineAnnealingLR(optimizer, T_max=max_steps)
        else:
            sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

        step = 0
        total_loss = 0.0
        start = time.time()
        accum_loss = torch.tensor(0.0)

        for batch in loader:
            if step >= max_steps:
                break
            if wall_clock_limit_secs and (time.time() - start) > wall_clock_limit_secs:
                break

            input_ids, attention_mask, labels = batch
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.detach()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                sched.step()
                optimizer.zero_grad()
                total_loss += accum_loss.item()
                accum_loss = torch.tensor(0.0)

            step += 1

        avg_loss = total_loss / max(step // gradient_accumulation_steps, 1)
        return {"steps": step, "train_loss": avg_loss}
    def eval_loss(self, dataset_path: Path, eval_fraction: float = 0.1) -> float:
        """Cross-entropy eval loss on held-out split (bits-per-byte compatible)."""
        import torch

        assert self._model is not None and self._tokenizer is not None

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
                input_ids = encodings["input_ids"][i].unsqueeze(0)
                attention_mask = encodings["attention_mask"][i].unsqueeze(0)
                labels = encodings["labels"][i].unsqueeze(0)
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                n_tokens = (labels != -100).sum().item()
                total_loss += outputs.loss.item() * n_tokens
                total_tokens += n_tokens

        if total_tokens == 0:
            return float("inf")
        # convert cross-entropy (nats) -> bits-per-byte approximation
        # avg chars/token ~= 3.8; bits = nats / ln(2)
        import math
        nats_per_token = total_loss / total_tokens
        bpb = (nats_per_token / math.log(2)) / 3.8
        return bpb

    def save_adapter(self, output_dir: Path) -> None:
        """Save LoRA adapter to disk."""
        assert self._model is not None and self._tokenizer is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(output_dir))
        self._tokenizer.save_pretrained(str(output_dir))

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Run inference."""
        import torch

        assert self._model is not None and self._tokenizer is not None
        self._model.eval()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def unload(self) -> None:
        """Free model from memory."""
        self._model = None
        self._tokenizer = None


# -- helpers ------------------------------------------------------------------

def _find_lora_targets(model, requested: tuple[str, ...]) -> list[str]:
    """Return LoRA target module names that actually exist in model.

    If none of the requested names match, auto-detects attention/MLP linear
    layers so any model architecture works without manual config.
    """
    import torch.nn as nn

    present = {name.split(".")[-1] for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    matched = [t for t in requested if t in present]
    if matched:
        return matched

    # Priority-ordered candidates covering LLaMA/Qwen/Mistral/Phi/GPT-2/Falcon
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",      # LLaMA - Qwen - Mistral - Phi
        "gate_proj", "up_proj", "down_proj",           # LLaMA - Qwen MLP
        "c_attn", "c_proj",                            # GPT-2
        "query_key_value", "dense",                    # Falcon - MPT
        "dense_h_to_4h", "dense_4h_to_h",             # Falcon MLP
    ]
    detected = [c for c in candidates if c in present]
    return detected if detected else list(present)


def _load_samples(path: Path) -> list[dict]:
    """Load JSONL dataset. Returns list of {messages: [...]} dicts."""
    if not path.exists():
        return []
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    return samples


def _encode_samples(samples: list[dict], tokenizer, max_length: int = 256) -> dict | None:
    """Encode samples to tensors using simple text concatenation."""
    import torch

    if not samples:
        return None

    texts = []
    for s in samples:
        msgs = s.get("messages", [])
        text = ""
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        texts.append(text)

    tokenizer.padding_side = "right"
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    encoded["labels"] = labels
    return encoded
