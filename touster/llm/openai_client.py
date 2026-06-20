from __future__ import annotations

import httpx


class OpenAIClient:
    """OpenAI-compatible REST client (no openai SDK dependency)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Send messages to /chat/completions and return assistant reply."""
        chosen_model = model or self.model
        payload = {
            "model": chosen_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=120.0,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"OpenAI request failed (network error): {exc}"
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI API error {response.status_code}: {response.text}"
            )

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected OpenAI response shape: {data}"
            ) from exc

    def list_models(self) -> list[str]:
        """Return available model ids from GET /models. Falls back to [] on error."""
        try:
            response = httpx.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=30.0,
            )
            if response.status_code != 200:
                return []
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []
