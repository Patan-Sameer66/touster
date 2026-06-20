from __future__ import annotations

import httpx


class OllamaClient:
    """Ollama HTTP client — talks to a local Ollama server."""

    def __init__(self, port: int = 11434, model: str = "") -> None:
        self.base_url = f"http://localhost:{port}"
        self.model = model

    def chat(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """POST /api/chat and return assistant reply string."""
        chosen_model = model or self.model
        payload = {
            "model": chosen_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300.0,
            )
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Ollama request failed: {exc}. Is Ollama running?"
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama error {response.status_code}: {response.text}"
            )

        data = response.json()
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected Ollama response shape: {data}"
            ) from exc

    def list_models(self) -> list[str]:
        """GET /api/tags -> list of model names. Falls back to [] on error."""
        try:
            response = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=10.0,
            )
            if response.status_code != 200:
                return []
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
