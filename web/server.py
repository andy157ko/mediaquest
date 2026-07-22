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
import threading
from pathlib import Path

# Make `import mediaquest` work regardless of how we're launched.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from mediaquest import pipeline
from mediaquest.config import config

app = FastAPI(title="MediaQuest")

_STATIC = Path(__file__).resolve().parent / "static"

# Ollama is effectively serial and we tweak the shared `config` per request,
# so run one research job at a time. Plenty for a local, single-user tool.
_lock = threading.Lock()

# In-memory research sessions, so follow-ups can reuse already-gathered sources
# without re-searching. Keyed by an opaque session id. Fine for a local tool;
# a persistent store would be the upgrade for multi-user / restart survival.
# { sid: {"answer": Answer, "turns": [ {"question", "answer"} ] } }
_sessions: dict = {}
_MAX_SESSIONS = 50


def _remember(answer) -> str:
    sid = secrets.token_hex(8)
    # Bound memory: drop the oldest session if we're over the cap.
    if len(_sessions) >= _MAX_SESSIONS:
        _sessions.pop(next(iter(_sessions)), None)
    _sessions[sid] = {"answer": answer, "turns": []}
    return sid


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _research_events(query: str, results: int, whisper: bool):
    """Generator yielding SSE frames: progress* → (result | error) → done."""
    q: "queue.Queue" = queue.Queue()

    def progress(msg: str) -> None:
        q.put({"type": "progress", "msg": msg})

    def run() -> None:
        with _lock:
            try:
                if results:
                    config.max_results = results
                config.whisper_fallback = bool(whisper)
                answer = pipeline.research(query, progress=progress)
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


def _followup_events(session_id: str, question: str):
    """SSE frames for a follow-up answered from a stored session's sources."""
    q: "queue.Queue" = queue.Queue()
    session = _sessions.get(session_id)

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
):
    return StreamingResponse(
        _research_events(q.strip(), results, bool(whisper)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/followup")
def followup(
    session_id: str = Query(..., min_length=4),
    q: str = Query(..., min_length=2, description="The follow-up question"),
):
    return StreamingResponse(
        _followup_events(session_id, q.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("MQ_WEB_PORT", "8000"))
    print(f"MediaQuest web UI → http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
