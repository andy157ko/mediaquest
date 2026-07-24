"""LLM access layer.

Deliberately thin and provider-agnostic: the rest of the code calls
`chat(system, user)` and never touches HTTP. Providers implement a single
`chat()` method; `get_provider()` picks one from config. Adding another
(e.g. Gemini) is one class here — nothing else in the codebase changes.
"""

from __future__ import annotations

import json
import random
import time
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


class GroqProvider:
    """Groq's OpenAI-compatible chat API. Free tier, very fast, needs a key.

    Free tiers rate-limit (HTTP 429), so we retry with exponential backoff,
    honoring the server's Retry-After when present.
    """

    URL = "https://api.groq.com/openai/v1/chat/completions"
    MAX_RETRIES = 4

    def __init__(self, api_key: str, model: str, timeout: int):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, system: str, user: str) -> str:
        if not self.api_key:
            raise LLMError(
                "Provider is 'groq' but no API key is set. Get a free key at "
                "https://console.groq.com/keys and put it in a .env file as "
                "GROQ_API_KEY=... (or export GROQ_API_KEY)."
            )
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.post(
                    self.URL, headers=headers, json=payload, timeout=self.timeout
                )
            except requests.exceptions.RequestException as e:
                raise LLMError(f"Could not reach Groq: {e}") from e

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == self.MAX_RETRIES - 1:
                    raise LLMError(
                        f"Groq rate-limited/unavailable ({resp.status_code}) after "
                        f"{self.MAX_RETRIES} tries. Wait a moment or lower "
                        "MQ_CONCURRENCY."
                    )
                # Honor Retry-After if given, else exponential backoff + jitter.
                wait = resp.headers.get("retry-after")
                delay = float(wait) if wait else (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)
                continue

            if resp.status_code == 401:
                raise LLMError("Groq rejected the API key (401). Check GROQ_API_KEY.")
            if resp.status_code == 404:
                raise LLMError(
                    f"Groq model '{self.model}' not found. Set MQ_GROQ_MODEL to a "
                    "current model (see https://console.groq.com/docs/models)."
                )
            if not resp.ok:
                raise LLMError(f"Groq error {resp.status_code}: {resp.text[:300]}")

            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

        raise LLMError("Groq: exhausted retries.")  # unreachable, keeps type-checkers happy


def get_provider() -> Provider:
    """Single place to choose the backend. Controlled by config.provider."""
    if config.provider == "groq":
        return GroqProvider(
            api_key=config.groq_api_key,
            model=config.groq_model,
            timeout=config.groq_timeout,
        )
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
