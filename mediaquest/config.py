"""Configuration — everything tunable lives here, overridable via env vars.

Nothing here requires a paid key. Defaults target a local Ollama install.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # --- LLM (Ollama, local & free) ---
    ollama_host: str = os.environ.get("MQ_OLLAMA_HOST", "http://localhost:11434")
    # A small, capable default. Swap via `export MQ_MODEL=...`.
    model: str = os.environ.get("MQ_MODEL", "llama3.1:8b")
    # How many transcript characters we feed the model per source in the map step.
    # Local models have small context windows, so we truncate defensively.
    per_source_chars: int = _env_int("MQ_PER_SOURCE_CHARS", 12000)
    request_timeout: int = _env_int("MQ_TIMEOUT", 300)

    # --- Platforms ---
    # TikTok is opt-in: it has no free keyword search (we discover via web
    # search) and no captions (every clip is Whisper-transcribed), so it's
    # slower and lower-signal than YouTube. Enable per-request in the UI/CLI.
    tiktok_results: int = _env_int("MQ_TIKTOK_RESULTS", 6)
    # Drop TikToks whose transcript is too short to carry real information
    # (silent/meme clips). Measured in characters.
    tiktok_min_chars: int = _env_int("MQ_TIKTOK_MIN_CHARS", 100)

    # --- Search ---
    max_results: int = _env_int("MQ_MAX_RESULTS", 10)
    # Cap videos from any single channel so "more results" means broader
    # coverage, not five clips from the same creator. 0 disables the cap.
    max_per_channel: int = _env_int("MQ_MAX_PER_CHANNEL", 2)
    # Skip anything shorter/longer than these (seconds). Filters out
    # low-signal shorts and multi-hour streams.
    min_duration: int = _env_int("MQ_MIN_DURATION", 60)
    max_duration: int = _env_int("MQ_MAX_DURATION", 3600)

    # --- Concurrency ---
    # LLM "read" calls to run at once. Local Ollama is largely serial
    # (hardware-bound), so keep modest; bump high with a free cloud model.
    concurrency: int = _env_int("MQ_CONCURRENCY", 4)
    # Caption fetching is deliberately gentle: YouTube rate-limits / IP-blocks
    # the caption endpoint under bursty load, so we fetch few-at-a-time with
    # backoff rather than hammering it.
    transcript_workers: int = _env_int("MQ_TRANSCRIPT_WORKERS", 2)
    transcript_retries: int = _env_int("MQ_TRANSCRIPT_RETRIES", 2)

    # --- Whisper fallback (opt-in) ---
    # When captions are missing OR YouTube blocks the caption endpoint, download
    # the audio and transcribe it locally with Whisper. Slower, but bypasses the
    # block and works on caption-less videos. Requires: pip install faster-whisper.
    whisper_fallback: bool = os.environ.get("MQ_WHISPER_FALLBACK", "") not in ("", "0", "false")
    # tiny (~75MB, fast) | base (~150MB) | small (~500MB, best quality/speed bal.)
    whisper_model: str = os.environ.get("MQ_WHISPER_MODEL", "tiny")

    # --- Fact-check ---
    # A claim needs at least this many independent sources to be "corroborated".
    corroboration_threshold: int = _env_int("MQ_CORROBORATION", 2)

    # Languages to accept for transcripts, in preference order.
    transcript_languages: tuple = ("en", "en-US", "en-GB")


config = Config()
