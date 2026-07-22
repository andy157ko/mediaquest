"""LLM access layer.

Deliberately thin and provider-agnostic: the rest of the code calls
`chat(system, user)` and never touches HTTP. To add Gemini/Groq later,
implement another Provider and swap it in `get_provider()` — nothing
else in the codebase changes.
"""

from __future__ import annotations

import json
from typing import Protocol

import requests

from .config import config


class LLMError(RuntimeError):
    pass


class Provider(Protocol):
    def chat(self, system: str, user: str) -> str: ...


class OllamaProvider:
    """Talks to a local Ollama server. Free, offline, no API key."""

    def __init__(self, host: str, model: str, timeout: int):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Low temperature: we want faithful synthesis, not creativity.
            "options": {"temperature": 0.2},
        }
        try:
            resp = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise LLMError(
                f"Could not reach Ollama at {self.host}. "
                "Is it running? Start it with `ollama serve` and pull a model "
                f"with `ollama pull {self.model}`."
            ) from e
        except requests.exceptions.Timeout as e:
            raise LLMError(
                f"Ollama timed out after {self.timeout}s. Try a smaller model "
                "or reduce MQ_MAX_RESULTS / MQ_PER_SOURCE_CHARS."
            ) from e

        if resp.status_code == 404:
            raise LLMError(
                f"Model '{self.model}' not found in Ollama. "
                f"Pull it with `ollama pull {self.model}`."
            )
        if not resp.ok:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        return data.get("message", {}).get("content", "").strip()


def get_provider() -> Provider:
    """Single place to choose the backend. Swap here to change providers."""
    return OllamaProvider(
        host=config.ollama_host,
        model=config.model,
        timeout=config.request_timeout,
    )


def chat(system: str, user: str) -> str:
    return get_provider().chat(system, user)


def chat_json(system: str, user: str) -> dict:
    """Chat, expecting a JSON object back. Tolerates code fences / stray prose."""
    raw = chat(system, user)
    return _extract_json(raw)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fences
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip("` \n")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise LLMError(f"Model did not return valid JSON. Got: {text[:300]}")
