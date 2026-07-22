"""Whisper fallback — transcribe a video's audio locally when captions fail.

This exists for two situations:
  1. The video genuinely has no captions.
  2. YouTube is rate-limiting / IP-blocking the caption endpoint (`timedtext`).

Both are solved the same way: pull the audio stream (a different endpoint —
Google's video CDN, not `timedtext`) and transcribe it offline with Whisper.
The same code path will serve TikTok later.

Everything here is free and local. `faster-whisper` is an optional dependency
imported lazily, so the base tool runs without it.
"""

from __future__ import annotations

import glob
import os
import tempfile
from typing import Optional

from .config import config

# The decode/transcribe model is expensive to construct, so cache one per
# model name across calls within a process.
_MODEL_CACHE: dict = {}


class WhisperUnavailable(RuntimeError):
    """faster-whisper isn't installed."""


def _get_model():
    name = config.whisper_model
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise WhisperUnavailable(
            "Whisper fallback needs faster-whisper. Install it with:\n"
            "    .venv/bin/python -m pip install faster-whisper"
        ) from e
    # int8 on CPU: small memory, good speed, negligible quality loss for this use.
    model = WhisperModel(name, device="cpu", compute_type="int8")
    _MODEL_CACHE[name] = model
    return model


def _download_audio(video_url: str, workdir: str) -> Optional[str]:
    """Download a single pre-encoded audio stream (no ffmpeg/muxing needed)."""
    from yt_dlp import YoutubeDL

    out = os.path.join(workdir, "audio.%(ext)s")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,   # keep the download bar out of our progress stream
        "noplaylist": True,
        # Prefer a single already-encoded stream so we never invoke ffmpeg.
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio",
        "outtmpl": out,
    }
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([video_url])
    except Exception:
        return None
    files = glob.glob(os.path.join(workdir, "audio.*"))
    return files[0] if files else None


def transcribe(video_url: str) -> str:
    """Return transcript text for a video by transcribing its audio, or ""."""
    model = _get_model()  # raises WhisperUnavailable if not installed
    with tempfile.TemporaryDirectory(prefix="mq_whisper_") as workdir:
        audio_path = _download_audio(video_url, workdir)
        if not audio_path:
            return ""
        try:
            segments, _ = model.transcribe(audio_path, language="en")
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception:
            return ""
