from __future__ import annotations

from touster.llm.client import LLMClient
from touster.llm.ollama_client import OllamaClient
from touster.llm.openai_client import OpenAIClient


def build_client(
    api_key: str = "",
    api_base: str = "",
    ollama_port: int = 11434,
    model: str = "",
) -> LLMClient:
    """
    Build the appropriate LLM client:
    - If api_key is provided: OpenAIClient (OpenAI-compatible REST)
    - Otherwise: OllamaClient (local Ollama server)
    """
    if api_key:
        if api_base and not api_base.startswith(("http://", "https://")):
            api_base = "https://" + api_base
        base_url = api_base or "https://api.openai.com/v1"
        return OpenAIClient(api_key=api_key, base_url=base_url, model=model)
    return OllamaClient(port=ollama_port, model=model)


def build_client_for_dataset(
    mode: int,
    api_key: str,
    api_base: str,
    api_model: str,
    ollama_port: int,
    ollama_model: str,
) -> tuple[LLMClient | None, str]:
    """Build (client, label) for dataset modes 0/1. Mode 2 needs neither.

    Never raises — an unreachable Ollama server just means "no LLM" and a
    heuristic-only run, not a crashed notebook cell.
    """
    if mode not in (0, 1):
        return None, "none (mode 2 — bring your own)"
    if api_key or api_base:
        client = build_client(api_key=api_key, api_base=api_base, model=api_model)
        return client, f"API — {api_base or 'api.openai.com'} / {api_model or 'default'}"
    try:
        client = build_client(ollama_port=ollama_port, model=ollama_model)
        available = client.list_models()
        picked = ollama_model or (available[0] if available else "")
        return client, f"Ollama — {picked or '?'} (available={available})"
    except Exception as exc:
        return None, f"none — Ollama unavailable ({exc}), heuristic-only"
