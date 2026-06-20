from __future__ import annotations

from pathlib import Path


class ModelPair:
    """Holds base model and base+adapter for side-by-side comparison."""

    def __init__(self, base_model_id: str, adapter_path: Path | None = None) -> None:
        self._base_model_id = base_model_id
        self._adapter_path = adapter_path
        self._base_backend: object | None = None
        self._ft_backend: object | None = None
        self._loaded = False

    # ── public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load base model on CPU.  If adapter_path exists, also load fine-tuned version."""
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        # Base backend ─ plain CPUBackend without any LoRA adapter
        self._base_backend = _PlainInferenceBackend()
        self._base_backend.load(self._base_model_id)

        # Fine-tuned backend ─ only when adapter directory actually exists
        if self._adapter_path is not None and self._adapter_path.exists():
            self._ft_backend = _AdapterInferenceBackend()
            self._ft_backend.load(self._base_model_id, self._adapter_path)
        else:
            self._ft_backend = None

        self._loaded = True

    def generate_base(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate from base model only."""
        if self._base_backend is None:
            return ""
        return self._base_backend.generate(prompt, max_new_tokens=max_new_tokens)

    def generate_finetuned(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate from base + adapter.  Falls back to base if no adapter loaded."""
        if self._ft_backend is None:
            if self._base_backend is None:
                return ""
            return self._base_backend.generate(prompt, max_new_tokens=max_new_tokens)
        return self._ft_backend.generate(prompt, max_new_tokens=max_new_tokens)

    def unload(self) -> None:
        """Free models from memory."""
        if self._base_backend is not None:
            self._base_backend.unload()
            self._base_backend = None
        if self._ft_backend is not None:
            self._ft_backend.unload()
            self._ft_backend = None
        self._loaded = False

    @property
    def has_adapter(self) -> bool:
        return self._ft_backend is not None

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ── internal helpers ──────────────────────────────────────────────────────────


class _PlainInferenceBackend:
    """Bare-bones HF inference without LoRA (base model only)."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load(self, model_id: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(model_id)
        self._model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        import torch  # lazy

        if self._model is None or self._tokenizer is None:
            return ""
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
        self._model = None
        self._tokenizer = None


class _AdapterInferenceBackend:
    """HF inference with a PEFT LoRA adapter loaded on top of the base model."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load(self, model_id: str, adapter_path: Path) -> None:
        from peft import PeftModel  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_id)
        self._model = PeftModel.from_pretrained(base, str(adapter_path))
        self._model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        import torch  # lazy

        if self._model is None or self._tokenizer is None:
            return ""
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
        self._model = None
        self._tokenizer = None
