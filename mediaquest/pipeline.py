"""The orchestration: query in, cited + fact-checked Answer out.

Stages:
  1. search      — find candidate videos (yt-dlp)
  2. transcripts — pull captions, drop videos without them
  3. map         — per-source: extract key points relevant to the query
  4. reduce      — synthesize one cited answer across all sources
  5. factcheck   — extract claims, score by independent-source agreement

Every stage reports through an optional `progress` callback so the CLI
(and later a web UI) can show what's happening without this module knowing
anything about the UI.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

from . import youtube, llm
from .config import config
from .models import Answer, Claim, Source

Progress = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# --------------------------------------------------------------------------
# Stage 3: map — distill each transcript to query-relevant points
# --------------------------------------------------------------------------
_MAP_SYSTEM = (
    "You extract only the information from a video transcript that helps answer "
    "a specific question. You never invent facts. If the transcript is off-topic, "
    "you say so."
)


def _map_source(source: Source, query: str) -> List[str]:
    transcript = source.transcript[: config.per_source_chars]
    user = (
        f"QUESTION: {query}\n\n"
        f"VIDEO TITLE: {source.title}\n"
        f"CHANNEL: {source.channel}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        "List the specific, concrete points from THIS transcript that help answer "
        "the question. Use short bullet lines starting with '- '. Include numbers, "
        "steps, and named techniques when present. If the transcript does not "
        "address the question, reply with exactly: NONE"
    )
    out = llm.chat(_MAP_SYSTEM, user).strip()
    if out.upper().startswith("NONE"):
        return []
    points = [
        line.lstrip("-•* ").strip()
        for line in out.splitlines()
        if line.strip() and line.strip() not in {"-", "•", "*"}
    ]
    return [p for p in points if p]


# --------------------------------------------------------------------------
# Stage 4: reduce — synthesize a single cited answer
# --------------------------------------------------------------------------
_SYNTH_SYSTEM = (
    "You are a careful research assistant. You answer the user's question using "
    "ONLY the provided source notes. You cite every claim inline with [n] markers "
    "matching the source numbers. You never use outside knowledge. When sources "
    "disagree, you note the disagreement. You write clear, useful prose."
)


def _source_notes(sources: List[Source]) -> str:
    """Render sources' distilled key points as a numbered, citeable block."""
    blocks = []
    for s in sources:
        if not s.key_points:
            continue
        pts = "\n".join(f"  - {p}" for p in s.key_points)
        blocks.append(f"SOURCE [{s.index}] — {s.title} ({s.channel}):\n{pts}")
    return "\n\n".join(blocks)


def _synthesize(query: str, sources: List[Source]) -> str:
    user = (
        f"QUESTION: {query}\n\n"
        f"SOURCE NOTES:\n{_source_notes(sources)}\n\n"
        "Write a well-organized answer to the question using only these notes. "
        "Cite each point with the matching [n]. Prefer advice that multiple "
        "sources agree on, and say when something comes from only one source. "
        "End with a short 'Bottom line' takeaway."
    )
    return llm.chat(_SYNTH_SYSTEM, user).strip()


# --------------------------------------------------------------------------
# Stage 5: factcheck — claims + cross-source corroboration
# --------------------------------------------------------------------------
_CLAIM_SYSTEM = (
    "You verify which sources support each factual claim. You are strict: a source "
    "supports a claim only if its notes actually state it. Output valid JSON only."
)


def _factcheck(query: str, answer_text: str, sources: List[Source]) -> List[Claim]:
    catalog = []
    for s in sources:
        pts = "; ".join(s.key_points) if s.key_points else "(no relevant points)"
        catalog.append(f"[{s.index}] {s.title}: {pts}")
    catalog_text = "\n".join(catalog)

    user = (
        f"QUESTION: {query}\n\n"
        f"ANSWER TO CHECK:\n{answer_text}\n\n"
        f"SOURCE NOTES:\n{catalog_text}\n\n"
        "Extract the key factual claims from the answer. For each, list the source "
        "numbers whose notes genuinely support it. Return JSON of the exact form:\n"
        '{"claims": [{"text": "...", "supporting_sources": [1,3]}]}'
    )
    try:
        data = llm.chat_json(_CLAIM_SYSTEM, user)
    except llm.LLMError:
        return []

    claims: List[Claim] = []
    valid_ids = {s.index for s in sources}
    for item in data.get("claims", []):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        support = [
            int(i) for i in item.get("supporting_sources", [])
            if isinstance(i, (int, float)) and int(i) in valid_ids
        ]
        support = sorted(set(support))
        if len(support) >= config.corroboration_threshold:
            status = "corroborated"
        elif len(support) == 1:
            status = "single-source"
        else:
            status = "unverified"
        claims.append(Claim(text=text, supporting_sources=support, status=status))
    return claims


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def research(query: str, progress: Optional[Progress] = None) -> Answer:
    p = progress or _noop

    p(f"Searching YouTube for “{query}”…")
    candidates = youtube.search(query)
    if not candidates:
        return Answer(query=query, summary="No videos found for that query.")
    p(f"Found {len(candidates)} candidate videos.")

    p("Fetching transcripts…")
    sources = youtube.attach_transcripts(candidates, progress=p)
    if not sources:
        if config.whisper_fallback:
            msg = ("Found videos, but couldn't get any transcript — even the "
                   "Whisper audio fallback failed (network, or faster-whisper "
                   "not installed). Check the messages above.")
        else:
            msg = ("Found videos, but none had usable captions — either they "
                   "lack captions or YouTube is temporarily rate-limiting caption "
                   "requests from your IP. Wait a few minutes, or enable the "
                   "Whisper fallback to transcribe audio directly:\n"
                   "    export MQ_WHISPER_FALLBACK=1   (needs: pip install faster-whisper)")
        return Answer(query=query, summary=msg)
    by_cap = sum(1 for s in sources if s.transcript_source == "captions")
    by_whisper = sum(1 for s in sources if s.transcript_source == "whisper")
    detail = f"{by_cap} via captions"
    if by_whisper:
        detail += f", {by_whisper} via Whisper audio"
    p(f"{len(sources)} videos have transcripts ({detail}).")

    p(f"Reading {len(sources)} videos (extracting key points)…")
    workers = max(1, min(config.concurrency, len(sources)))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_map_source, s, query): s for s in sources}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                s.key_points = fut.result()
            except llm.LLMError:
                s.key_points = []
            done += 1
            p(f"  · ({done}/{len(sources)}) [{s.index}] {s.title[:55]} — "
              f"{len(s.key_points)} points")

    # Preserve original relevance order for citation numbering.
    used = [s for s in sources if s.key_points]

    if not used:
        return Answer(
            query=query,
            summary="Transcripts were found but none addressed your question.",
            sources=sources,
        )

    p("Synthesizing a cited answer…")
    summary = _synthesize(query, used)

    p("Fact-checking claims across sources…")
    claims = _factcheck(query, summary, used)

    return Answer(query=query, summary=summary, sources=used, claims=claims)


# --------------------------------------------------------------------------
# Follow-ups — answer further questions from the ALREADY-gathered sources.
# No new search or transcription: reuses the distilled key points, so it's
# fast, free, cited to the same videos, and unaffected by YouTube throttling.
# --------------------------------------------------------------------------
_FOLLOWUP_SYSTEM = (
    "You are a careful research assistant continuing a conversation about a set of "
    "videos. Answer the user's follow-up using ONLY the provided source notes and "
    "the conversation so far. Cite every claim inline with [n] markers matching the "
    "source numbers. Never use outside knowledge. If the notes don't cover the "
    "follow-up, say so plainly and suggest running a fresh search for it."
)


def follow_up(
    sources: List[Source],
    original_query: str,
    original_summary: str,
    history: List[dict],
    question: str,
    progress: Optional[Progress] = None,
) -> Answer:
    """Answer `question` from existing `sources`. `history` is a list of
    {"question", "answer"} dicts from earlier follow-ups (oldest first)."""
    p = progress or _noop

    convo = [f"ORIGINAL QUESTION: {original_query}", f"ANSWER: {original_summary}"]
    for turn in history:
        convo.append(f"FOLLOW-UP: {turn.get('question','')}")
        convo.append(f"ANSWER: {turn.get('answer','')}")
    convo_text = "\n\n".join(convo)

    p("Answering from the videos already gathered…")
    user = (
        f"SOURCE NOTES:\n{_source_notes(sources)}\n\n"
        f"CONVERSATION SO FAR:\n{convo_text}\n\n"
        f"NEW FOLLOW-UP QUESTION: {question}\n\n"
        "Answer the new follow-up using only the source notes above. Cite with [n]. "
        "Keep it focused on what was actually asked."
    )
    summary = llm.chat(_FOLLOWUP_SYSTEM, user).strip()

    p("Fact-checking…")
    claims = _factcheck(question, summary, sources)

    return Answer(query=question, summary=summary, sources=sources, claims=claims)
