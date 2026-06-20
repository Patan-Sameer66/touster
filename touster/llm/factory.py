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
        base_url = api_base or "https://api.openai.com/v1"
        return OpenAIClient(api_key=api_key, base_url=base_url, model=model)
    return OllamaClient(port=ollama_port, model=model)
