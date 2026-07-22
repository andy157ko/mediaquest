"""TikTok access — discovery + transcription, free and key-less.

TikTok has no free keyword-search API (yt-dlp has no search extractor, and its
hashtag extractor is broken), and no caption API. So:

  - Discovery: query a web search engine for `site:tiktok.com <query>` and keep
    the real `/@user/video/<id>` links. This sidesteps TikTok's bot-protected
    search entirely — we ask the search engine, not TikTok.
  - Transcription: reuse the Whisper audio path (yt-dlp downloads the combined
    clip, faster-whisper transcribes it). Every TikTok goes through this.

Short, silent, or meme clips carry little information, so we drop transcripts
below a minimum length.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .config import config
from .models import Source

# Matches a canonical TikTok video URL and captures the handle + numeric id.
_VIDEO_RE = re.compile(r"tiktok\.com/@([^/?#]+)/video/(\d+)")


def _clean_title(raw: str, handle: str) -> str:
    """Turn a search-result title into something readable."""
    t = (raw or "").strip()
    # Search titles often look like "handle on TikTok" or end with "| TikTok".
    t = re.sub(r"\s*\|\s*TikTok\s*$", "", t, flags=re.I)
    t = re.sub(r"\s+on TikTok$", "", t, flags=re.I)
    t = " ".join(t.split())  # collapse whitespace/newlines from captions
    if len(t) > 90:
        t = t[:87].rstrip() + "…"
    return t or f"TikTok by @{handle}"


def search(query: str, max_results: Optional[int] = None) -> List[Source]:
    """Discover TikTok videos for a query via web search. Returns Sources
    (without transcripts); de-duplicated by video id, order preserved."""
    try:
        from ddgs import DDGS
    except ImportError:
        return []

    n = max_results or config.tiktok_results
    sources: List[Source] = []
    seen: set = set()

    try:
        with DDGS() as ddgs:
            # Over-fetch: many results are profile/tag pages, not videos.
            for r in ddgs.text(f"site:tiktok.com {query}", max_results=n * 4):
                url = r.get("href", "") or ""
                m = _VIDEO_RE.search(url)
                if not m:
                    continue
                handle, vid = m.group(1), m.group(2)
                if vid in seen:
                    continue
                seen.add(vid)
                sources.append(
                    Source(
                        index=len(sources) + 1,
                        platform="tiktok",
                        video_id=vid,
                        title=_clean_title(r.get("title", ""), handle),
                        channel=f"@{handle}",
                        url=f"https://www.tiktok.com/@{handle}/video/{vid}",
                    )
                )
                if len(sources) >= n:
                    break
    except Exception:
        return sources  # partial results are fine; upstream handles empties

    return sources


def attach_transcripts(sources: List[Source], progress=None) -> List[Source]:
    """Transcribe each TikTok via Whisper; keep those with enough content.

    Serial on purpose: transcription is CPU-bound on one shared Whisper model,
    and TikToks are short so this stays fast.
    """
    def emit(msg: str) -> None:
        if progress:
            progress(msg)

    from . import whisper_stt

    kept: List[Source] = []
    for i, s in enumerate(sources, start=1):
        try:
            text = whisper_stt.transcribe(s.url)
        except whisper_stt.WhisperUnavailable as e:
            emit(f"  ✗ {e}")
            break
        if text and len(text) >= config.tiktok_min_chars:
            s.transcript = text
            s.transcript_source = "whisper"
            kept.append(s)
            emit(f"  · ({i}/{len(sources)}) {s.channel} — {len(text)} chars")
        else:
            emit(f"  · ({i}/{len(sources)}) {s.channel} — too short, skipped")
    return kept
