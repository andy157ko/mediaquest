"""Plain data structures passed between pipeline stages.

Kept dependency-free so both the CLI and a future web/JSON layer can use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Source:
    """A single piece of media (currently a YouTube video)."""

    index: int                      # 1-based citation number, e.g. [1]
    platform: str                   # "youtube", later "tiktok"
    video_id: str
    title: str
    channel: str
    url: str
    duration: Optional[int] = None  # seconds
    views: Optional[int] = None
    transcript: str = ""            # full text; empty if none available
    transcript_source: str = ""     # "captions" | "whisper" | ""
    key_points: List[str] = field(default_factory=list)  # filled in map step

    def citation(self) -> str:
        return f"[{self.index}] {self.title} — {self.channel}\n    {self.url}"


@dataclass
class Claim:
    """A factual statement extracted from the answer, plus its support."""

    text: str
    supporting_sources: List[int] = field(default_factory=list)  # source indexes
    status: str = "unverified"      # "corroborated" | "single-source" | "unverified"

    @property
    def support_count(self) -> int:
        return len(self.supporting_sources)


@dataclass
class Answer:
    """The final result handed back to the caller (CLI or web)."""

    query: str
    summary: str                    # the synthesized, cited prose answer
    sources: List[Source] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
