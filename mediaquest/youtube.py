"""YouTube access — search and transcripts, both free and key-less.

Search uses yt-dlp's `ytsearch` (no Data API quota, no key).
Transcripts use youtube-transcript-api (captions only, no video download).
Both are best-effort: a video with no captions is simply skipped upstream.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from .config import config
from .models import Source


class TranscriptBlocked(RuntimeError):
    """YouTube rate-limited / IP-blocked the caption endpoint (temporary)."""


_BLOCK_MARKERS = ("ipblocked", "too many requests", "429", "blocking requests from your ip")


def _looks_blocked(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "ipblocked" in name or "toomanyrequests" in name or any(
        m in msg for m in _BLOCK_MARKERS
    )


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
def search(query: str, max_results: Optional[int] = None) -> List[Source]:
    """Return candidate videos for a query, newest-relevance first.

    Uses yt-dlp flat extraction so we get metadata without hitting each page.
    Filters by duration to drop shorts and multi-hour streams.
    """
    from yt_dlp import YoutubeDL

    n = max_results or config.max_results
    # Over-fetch a wide candidate pool (search is free & fast): many entries get
    # dropped by duration, channel-diversity, or missing transcripts downstream.
    fetch = max(n * 4, 30)
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "default_search": "ytsearch",
    }

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{fetch}:{query}", download=False)

    entries = (info or {}).get("entries", []) or []
    per_channel: dict = {}
    sources: List[Source] = []
    for e in entries:
        if not e or not e.get("id"):
            continue
        duration = e.get("duration")
        if duration is not None and not (
            config.min_duration <= duration <= config.max_duration
        ):
            continue
        channel = e.get("channel") or e.get("uploader") or "Unknown channel"
        # Diversity cap: don't let one creator dominate the pool.
        if config.max_per_channel:
            seen = per_channel.get(channel, 0)
            if seen >= config.max_per_channel:
                continue
            per_channel[channel] = seen + 1
        sources.append(
            Source(
                index=len(sources) + 1,
                platform="youtube",
                video_id=e["id"],
                title=e.get("title") or "(untitled)",
                channel=channel,
                url=e.get("url") or f"https://www.youtube.com/watch?v={e['id']}",
                duration=duration,
                views=e.get("view_count"),
            )
        )
        if len(sources) >= n:
            break

    return sources


# --------------------------------------------------------------------------
# Transcripts
# --------------------------------------------------------------------------
def _fetch_captions_once(video_id: str) -> str:
    """One caption attempt. Returns "" if the video has none; raises
    TranscriptBlocked if YouTube is blocking us so callers can react."""
    langs = list(config.transcript_languages)
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return ""

    segments = None
    last_exc: Optional[Exception] = None

    # New instance-based API (youtube-transcript-api >= 1.0)
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=langs)
        segments = [{"text": getattr(s, "text", "")} for s in fetched]
    except Exception as e:
        last_exc = e
        segments = None

    # Old classmethod API (<= 0.6.x)
    if segments is None and hasattr(YouTubeTranscriptApi, "get_transcript"):
        try:
            segments = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
        except Exception as e:
            last_exc = e
            segments = None

    if segments is None:
        if last_exc is not None and _looks_blocked(last_exc):
            raise TranscriptBlocked(str(last_exc))
        return ""  # genuinely no captions in our languages

    text = " ".join(
        (seg.get("text", "") if isinstance(seg, dict) else str(seg)).strip()
        for seg in segments
    )
    return text.replace("[Music]", " ").replace("[Applause]", " ").strip()


def fetch_captions(video_id: str) -> str:
    """Caption text with polite retry/backoff. Raises TranscriptBlocked if
    still blocked after retries; returns "" if the video has no captions."""
    attempts = max(1, config.transcript_retries + 1)
    for i in range(attempts):
        try:
            return _fetch_captions_once(video_id)
        except TranscriptBlocked:
            if i == attempts - 1:
                raise
            # Exponential backoff with jitter to let a soft rate-limit clear.
            time.sleep((2 ** i) + random.uniform(0, 0.5))
    return ""


def attach_transcripts(sources: List[Source], progress=None) -> List[Source]:
    """Fill in `.transcript`/`.transcript_source`; keep sources that have text.

    Strategy: fetch captions gently (low concurrency + backoff). For any source
    with no captions — because the video lacks them OR YouTube blocked us — fall
    back to local Whisper transcription if enabled (MQ_WHISPER_FALLBACK=1).
    """
    def emit(msg: str) -> None:
        if progress:
            progress(msg)

    workers = max(1, min(config.transcript_workers, len(sources)))
    blocked = False

    def _cap(s: Source):
        nonlocal blocked
        try:
            s.transcript = fetch_captions(s.video_id)
            if s.transcript:
                s.transcript_source = "captions"
        except TranscriptBlocked:
            blocked = True  # caption endpoint is rate-limiting us

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_cap, sources))

    need_fallback = [s for s in sources if not s.transcript]
    if blocked:
        emit(f"  ⚠ YouTube is rate-limiting caption requests "
             f"({len(need_fallback)} affected).")

    if need_fallback and config.whisper_fallback:
        from . import whisper_stt
        emit(f"  ↳ Whisper fallback: transcribing {len(need_fallback)} "
             f"video(s) from audio (model={config.whisper_model})…")
        for s in need_fallback:
            try:
                s.transcript = whisper_stt.transcribe(s.url)
                if s.transcript:
                    s.transcript_source = "whisper"
                    emit(f"    · [{s.video_id}] {len(s.transcript)} chars via Whisper")
            except whisper_stt.WhisperUnavailable as e:
                emit(f"    ✗ {e}")
                break
    elif need_fallback and blocked:
        emit("  ↳ Tip: enable Whisper to bypass the block: "
             "export MQ_WHISPER_FALLBACK=1  (needs: pip install faster-whisper)")

    kept = [s for s in sources if s.transcript]
    # Re-number so citations stay contiguous ([1]..[n]) after drops.
    for i, s in enumerate(kept, start=1):
        s.index = i
    return kept
