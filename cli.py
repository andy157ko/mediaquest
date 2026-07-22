#!/usr/bin/env python3
"""MediaQuest CLI — ask a question, get a cited answer built from videos.

Usage:
    python3 cli.py "best gym routine for beginners"
    python3 cli.py "how to fix a leaky faucet" --results 8
    python3 cli.py "is creatine safe" --json

Requires a running local Ollama (see setup.sh / README).
"""

from __future__ import annotations

import argparse
import json
import sys

from mediaquest import pipeline
from mediaquest.config import config
from mediaquest.llm import LLMError
from mediaquest.models import Answer

# Optional pretty output; degrade gracefully if `rich` isn't installed.
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.rule import Rule

    _console = Console()

    def info(msg: str) -> None:
        _console.print(f"[dim]{msg}[/dim]")

    def render(answer: Answer) -> None:
        _console.print(Rule(f"Answer: {answer.query}"))
        _console.print(Markdown(answer.summary))
        _render_claims(answer)
        _render_sources(answer)

    def _rule(label: str) -> None:
        _console.print(Rule(label))

    def _plain(s: str) -> None:
        _console.print(s)

except ImportError:  # plain-text fallback
    def info(msg: str) -> None:
        print(msg, file=sys.stderr)

    def _rule(label: str) -> None:
        print("\n" + "=" * 8 + f" {label} " + "=" * 8)

    def _plain(s: str) -> None:
        print(s)

    def render(answer: Answer) -> None:
        _rule(f"Answer: {answer.query}")
        print(answer.summary)
        _render_claims(answer)
        _render_sources(answer)


_STATUS_MARK = {
    "corroborated": "✓ corroborated",
    "single-source": "△ single source",
    "unverified": "✗ unverified",
}


def _render_claims(answer: Answer) -> None:
    if not answer.claims:
        return
    _rule("Fact-check")
    for c in answer.claims:
        mark = _STATUS_MARK.get(c.status, c.status)
        srcs = ", ".join(f"[{i}]" for i in c.supporting_sources) or "none"
        _plain(f"  {mark}  ({srcs})  {c.text}")


def _render_sources(answer: Answer) -> None:
    _rule("Sources")
    for s in answer.sources:
        _plain(s.citation())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a cited, fact-checked answer from online videos."
    )
    parser.add_argument("query", help="What you want to learn about.")
    parser.add_argument(
        "--results", type=int, default=None,
        help=f"Number of videos to use (default {config.max_results}).",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Ollama model to use (default {config.model}).",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the raw result as JSON."
    )
    args = parser.parse_args()

    if args.results:
        config.max_results = args.results
    if args.model:
        config.model = args.model

    try:
        answer = pipeline.research(args.query, progress=None if args.json else info)
    except LLMError as e:
        print(f"\nLLM error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    if args.json:
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
    else:
        render(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
