from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Structural protocol for all LLM backends."""

    def chat(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        format: str = "",
    ) -> str:
        """Send messages, return assistant reply string."""
        ...

    def list_models(self) -> list[str]:
        """Return available model ids."""
        ...
