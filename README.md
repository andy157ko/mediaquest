# MediaQuest

Ask a question, get a **cited, fact-checked answer built from online videos** —
so you don't have to watch 10 videos to learn one thing.

It searches YouTube, reads the transcripts (no video download), and uses a
**local, free LLM (Ollama)** to synthesize an answer with inline `[n]` citations
and a cross-source fact-check. No paid API keys anywhere.

```
.venv/bin/python cli.py "best gym routine for beginners"
```

## How it works

```
query ─▶ search (yt-dlp, keyless) ─▶ transcripts (captions, no download)
      ─▶ map: per-video key points ─▶ reduce: cited synthesis
      ─▶ fact-check: claims scored by how many independent videos agree
```

- **Free & offline.** Search and transcripts need no API key; the LLM runs
  locally via Ollama.
- **Cited.** Every point is tagged `[n]`; sources are listed at the end.
- **Fact-checked.** Each claim is marked `✓ corroborated` (multiple videos agree),
  `△ single source`, or `✗ unverified`.

## Videos without a transcript

Two things can leave a video with no captions, and MediaQuest handles both:

1. **The video genuinely has none** (music, small creators). It's normally just
   dropped — that's why we search a wide pool and keep the best that have text.
2. **YouTube rate-limits / IP-blocks the caption endpoint.** Under bursty load
   YouTube returns `IpBlocked` / HTTP 429 for caption requests (temporary, clears
   in minutes–hours). We fetch captions *gently* (low concurrency + backoff) to
   avoid this, and we report a block honestly instead of pretending "no captions."

**The fallback that beats both — Whisper.** Enable it to download the audio and
transcribe locally (a different endpoint, so it sidesteps the block; also works
on caption-less videos):

```bash
pip install faster-whisper          # into your .venv
export MQ_WHISPER_FALLBACK=1
.venv/bin/python cli.py "best gym routine for beginners"
```

It's slower than captions (~20s per video with the `tiny` model on CPU) but
reliable. `MQ_WHISPER_MODEL` = `tiny` (default) | `base` | `small`.

## Setup (one time)

```bash
cd Agent-Media
bash setup.sh
```

This installs the Python deps, installs Ollama if needed, and pulls the model
(`llama3.1:8b` by default — a few GB, downloaded once).

## Web UI

```bash
bash run_web.sh          # starts Ollama if needed, serves on http://localhost:8000
```

A single-page app: type a question, watch each stage stream live (search →
transcripts → reading → synthesis → fact-check), then read the cited answer with
clickable `[n]` chips, colour-coded fact-check badges, and source cards
(thumbnail, channel, and whether the transcript came from captions or Whisper).
The backend (`web/server.py`) is a thin FastAPI wrapper that streams the very
same `pipeline.research()` the CLI uses, over Server-Sent Events.

**Follow-up questions.** After an answer you can keep asking — follow-ups are
answered *from the videos already gathered* (via `pipeline.follow_up()`), so
they're instant, free, cited to the same sources, and don't re-hit YouTube. If
the videos don't cover the follow-up, the answer says so rather than guessing.
The research session is held in memory server-side and keyed by an id the page
tracks; each follow-up carries the running conversation for context.

## Usage (CLI)

```bash
.venv/bin/python cli.py "how to fix a leaky faucet"
.venv/bin/python cli.py "is creatine safe" --results 8
.venv/bin/python cli.py "best gym routine" --json       # machine-readable output
.venv/bin/python cli.py "learn to solder" --model qwen2.5:7b
```

## Configuration

All optional, via environment variables:

| Var                 | Default              | Meaning                             |
|---------------------|----------------------|-------------------------------------|
| `MQ_MODEL`          | `llama3.1:8b`        | Ollama model                        |
| `MQ_MAX_RESULTS`    | `10`                 | Videos used per query               |
| `MQ_MAX_PER_CHANNEL`| `2`                  | Cap per creator (0 = no cap)        |
| `MQ_CONCURRENCY`    | `4`                  | Parallel LLM-read calls             |
| `MQ_TRANSCRIPT_WORKERS` | `2`              | Parallel caption fetches (kept low) |
| `MQ_WHISPER_FALLBACK` | *(off)*            | `1` = transcribe audio when captions fail |
| `MQ_WHISPER_MODEL`  | `tiny`               | `tiny` \| `base` \| `small`          |
| `MQ_MIN_DURATION`   | `60`                 | Skip videos shorter than N seconds  |
| `MQ_MAX_DURATION`   | `3600`               | Skip videos longer than N seconds   |
| `MQ_CORROBORATION`  | `2`                  | Sources needed to mark a claim solid|
| `MQ_OLLAMA_HOST`    | `http://localhost:11434` | Ollama endpoint                 |

## Project layout

```
mediaquest/
  config.py     settings (env-overridable)
  models.py     Source / Claim / Answer data types
  llm.py        Ollama client — swap here to add Gemini/Groq later
  youtube.py    keyless search + caption transcripts
  pipeline.py   search → map → reduce → fact-check
  whisper_stt.py audio-download + local Whisper fallback
cli.py          terminal entry point
web/
  server.py     FastAPI backend — streams pipeline.research() over SSE
  static/       single-file frontend (no build step, no CDN)
```

The core (`mediaquest/`) is UI-free on purpose: both the CLI and the web app
just import `pipeline.research(query)`.

## Roadmap

- **TikTok**: download audio with yt-dlp → transcribe with Whisper → same pipeline.
- **Web fact-check**: corroborate claims against text sources, not just other videos.
- **Swap in a free cloud LLM** (Gemini/Groq) for much faster synthesis at scale —
  a one-file change in `mediaquest/llm.py`.
```

## Notes on use & license

This is a personal/educational project. It relies on `yt-dlp` and
`youtube-transcript-api` to fetch publicly available video transcripts and
audio; please use it responsibly, for personal research, and respect the Terms
of Service and rate limits of the platforms you query. Cited answers are only as
reliable as their source videos — verify anything important.

Licensed under the [MIT License](LICENSE).
