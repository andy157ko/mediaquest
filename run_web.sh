#!/usr/bin/env bash
# Launch the MediaQuest web UI. Assumes `bash setup.sh` has been run.
set -e
cd "$(dirname "$0")"

# Make sure Ollama is up (the web UI needs it for synthesis).
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama…"
  ollama serve >/tmp/ollama.log 2>&1 &
  until curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do sleep 1; done
fi

exec .venv/bin/python -m web.server
