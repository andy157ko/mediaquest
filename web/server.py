"""FastAPI backend for MediaQuest.

Deliberately thin: it owns no research logic — it streams the same
`pipeline.research()` used by the CLI, forwarding the pipeline's progress
callback to the browser over Server-Sent Events (SSE) so the user watches
each stage happen live.

Run it:
    .venv/bin/python -m web.server
    # then open http://localhost:8000
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import sys
import tempfile
import threading
from pathlib import Path

# Make `import mediaquest` work regardless of how we're launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from mediaquest import pipeline
from mediaquest.config import config
from mediaquest.models import Answer, Claim, Source

app = FastAPI(title="MediaQuest")

_STATIC = Path(__file__).resolve().parent / "static"

# Ollama is effectively serial and we tweak the shared `config` per request,
# so run one research job at a time. Plenty for a local, single-user tool.
_lock = threading.Lock()

# Research sessions let follow-ups reuse already-gathered sources without
# re-searching. Kept in memory AND mirrored to disk so they survive a server
# restart (otherwise every restart invalidates open browser sessions).
# { sid: {"answer": Answer, "turns": [ {"question", "answer"} ] } }
_sessions: dict = {}
_MAX_SESSIONS = 50
_SESSIONS_DIR = Path(tempfile.gettempdir()) / "mediaquest_sessions"
_SESSIONS_DIR.mkdir(exist_ok=True)


def _persist(sid: str, session: dict) -> None:
    """Mirror a session to disk as JSON (best-effort; never breaks a request)."""
    try:
        data = {"answer": session["answer"].to_dict(), "turns": session["turns"]}
        (_SESSIONS_DIR / f"{sid}.json").write_text(
            json.dumps(data, ensure_ascii=False)
        )
    except Exception:
        pass


def _answer_from_dict(d: dict) -> Answer:
    return Answer(
        query=d["query"],
        summary=d["summary"],
        sources=[Source(**s) for s in d.get("sources", [])],
        claims=[Claim(**c) for c in d.get("claims", [])],
    )


def _get_session(sid: str):
    """Look up a session in memory, falling back to the on-disk copy."""
    if sid in _sessions:
        return _sessions[sid]
    path = _SESSIONS_DIR / f"{sid}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            session = {"answer": _answer_from_dict(data["answer"]),
                       "turns": data.get("turns", [])}
            _sessions[sid] = session
            return session
        except Exception:
            return None
    return None


def _remember(answer) -> str:
    sid = secrets.token_hex(8)
    # Bound in-memory size; the disk copy remains for later reload.
    if len(_sessions) >= _MAX_SESSIONS:
        _sessions.pop(next(iter(_sessions)), None)
    session = {"answer": answer, "turns": []}
    _sessions[sid] = session
    _persist(sid, session)
    return sid


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _apply_provider(provider: str) -> None:
    """Switch the active LLM backend for this request, if a valid one is asked."""
    if provider in ("groq", "ollama"):
        config.provider = provider


def _research_events(query: str, results: int, whisper: bool, tiktok: bool,
                     provider: str = ""):
    """Generator yielding SSE frames: progress* → (result | error) → done."""
    q: "queue.Queue" = queue.Queue()

    def progress(msg: str) -> None:
        q.put({"type": "progress", "msg": msg})

    def run() -> None:
        with _lock:
            try:
                _apply_provider(provider)
                if results:
                    config.max_results = results
                # TikTok always needs Whisper (no captions), so enable it too.
                config.whisper_fallback = bool(whisper) or bool(tiktok)
                platforms = ["youtube"] + (["tiktok"] if tiktok else [])
                answer = pipeline.research(
                    query, progress=progress, platforms=platforms
                )
                payload = answer.to_dict()
                # Only keep a session worth following up on if we got sources.
                if answer.sources:
                    payload["session_id"] = _remember(answer)
                q.put({"type": "result", "answer": payload})
            except Exception as e:  # surface, don't crash the stream
                q.put({"type": "error", "msg": f"{type(e).__name__}: {e}"})
            finally:
                q.put({"type": "done"})

    threading.Thread(target=run, daemon=True).start()

    while True:
        event = q.get()
        yield _sse(event)
        if event["type"] == "done":
            break


def _followup_events(session_id: str, question: str, provider: str = ""):
    """SSE frames for a follow-up answered from a stored session's sources."""
    q: "queue.Queue" = queue.Queue()
    session = _get_session(session_id)

    if session is None:
        def only_error():
            yield _sse({"type": "error",
                        "msg": "Session expired — please run the search again."})
            yield _sse({"type": "done"})
        return only_error()

    def progress(msg: str) -> None:
        q.put({"type": "progress", "msg": msg})

    def run() -> None:
        with _lock:
            try:
                _apply_provider(provider)
                base = session["answer"]
                answer = pipeline.follow_up(
                    sources=base.sources,
                    original_query=base.query,
                    original_summary=base.summary,
                    history=session["turns"],
                    question=question,
                    progress=progress,
                )
                session["turns"].append(
                    {"question": question, "answer": answer.summary}
                )
                _persist(session_id, session)  # keep the growing thread on disk
                payload = answer.to_dict()
                payload["session_id"] = session_id
                q.put({"type": "result", "answer": payload})
            except Exception as e:
                q.put({"type": "error", "msg": f"{type(e).__name__}: {e}"})
            finally:
                q.put({"type": "done"})

    threading.Thread(target=run, daemon=True).start()

    def gen():
        while True:
            event = q.get()
            yield _sse(event)
            if event["type"] == "done":
                break
    return gen()


@app.get("/api/research")
def research(
    q: str = Query(..., min_length=2, description="The question to research"),
    results: int = Query(0, ge=0, le=30),
    whisper: int = Query(0, ge=0, le=1),
    tiktok: int = Query(0, ge=0, le=1),
    provider: str = Query("", pattern="^(groq|ollama)?$"),
):
    return StreamingResponse(
        _research_events(q.strip(), results, bool(whisper), bool(tiktok), provider),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/followup")
def followup(
    session_id: str = Query(..., min_length=4),
    q: str = Query(..., min_length=2, description="The follow-up question"),
    provider: str = Query("", pattern="^(groq|ollama)?$"),
):
    return StreamingResponse(
        _followup_events(session_id, q.strip(), provider),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _ollama_up() -> bool:
    try:
        return requests.get(f"{config.ollama_host}/api/tags", timeout=0.6).ok
    except Exception:
        return False


@app.get("/api/info")
def info():
    """Active backend + which providers the user can switch to."""
    provider = config.provider
    model = config.groq_model if provider == "groq" else config.model
    return {
        "provider": provider,
        "model": model,
        "groq_model": config.groq_model,
        "ollama_model": config.model,
        "available": {
            "groq": bool(config.groq_api_key),
            "ollama": _ollama_up(),
        },
    }


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn

    # PORT (set by the harness/preview when autoPort is on) wins, then
    # MQ_WEB_PORT, then a sensible default for manual `bash run_web.sh` runs.
    port = int(os.environ.get("PORT") or os.environ.get("MQ_WEB_PORT") or "8000")
    print(f"MediaQuest web UI → http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
