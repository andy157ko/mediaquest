#!/usr/bin/env bash
# One-time setup for MediaQuest. Everything here is free.
set -e

echo "==> Creating an isolated virtualenv (.venv)…"
# yt-dlp has dropped Python 3.9; prefer a modern interpreter if present.
PYBIN=$(command -v python3.14 || command -v python3.13 || command -v python3.12 || command -v python3)
echo "    using $PYBIN ($($PYBIN --version 2>&1))"
"$PYBIN" -m venv .venv

echo "==> Installing Python dependencies…"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "==> Checking for Ollama (the free, local LLM engine)…"
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama not found. Installing via Homebrew…"
  if command -v brew >/dev/null 2>&1; then
    brew install ollama
  else
    echo "Homebrew not found. Install Ollama manually from https://ollama.com/download"
    exit 1
  fi
else
  echo "Ollama already installed."
fi

# Start the Ollama server in the background if it isn't running.
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "==> Starting Ollama server…"
  ollama serve >/tmp/ollama.log 2>&1 &
  sleep 3
fi

MODEL="${MQ_MODEL:-llama3.1:8b}"
echo
echo "==> Pulling model '$MODEL' (a few GB, one-time download)…"
ollama pull "$MODEL"

echo
echo "==> Done. Try it:"
echo "    .venv/bin/python cli.py \"best gym routine for beginners\""
