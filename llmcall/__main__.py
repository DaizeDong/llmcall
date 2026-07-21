"""CLI: the language-agnostic entry point for PowerShell/bash callers. Prompt on stdin, answer on
stdout, exit 0 if a provider answered else 1.

  echo "..." | python -m llmcall [--chain codex,cc,claude] [--schema f.json] [--timeout 120] \
                                 [--model M] [--effort E] [--notify STREAM] [--web-search]

--chain also accepts `gemini` (the search-grounded Gemini CLI), e.g. --chain gemini.
--web-search opts into the network search tool (off by default; relaxes read-only).
With --schema, stdout is the validated JSON object; otherwise stdout is the raw text.
"""
from __future__ import annotations

import argparse
import json
import sys

from .core import DEFAULT_CHAIN, call


def main() -> int:
    ap = argparse.ArgumentParser(prog="llmcall", description="headless codex -> cc -> claude judgment")
    ap.add_argument("--chain", default=",".join(DEFAULT_CHAIN))
    ap.add_argument("--schema", default=None, help="path to a JSON-Schema file for validated output")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--notify", default=None, help="the relay project stream to alert on total failure")
    ap.add_argument("--mode", choices=["judge", "research", "agent"], default="judge",
                    help="capability tier: judge (default, read-only) | research (web) | agent (full)")
    ap.add_argument("--web-search", dest="web_search", action="store_const", const=True, default=None,
                    help="force the network search tool on (else follows --mode)")
    a = ap.parse_args()

    schema = None
    if a.schema:
        with open(a.schema, encoding="utf-8") as f:
            schema = json.load(f)
    prompt = sys.stdin.read()
    r = call(prompt, chain=[c.strip() for c in a.chain.split(",") if c.strip()],
             schema=schema, mode=a.mode, timeout=a.timeout, model=a.model, effort=a.effort,
             notify=a.notify, web_search=a.web_search)
    if not r:
        sys.stderr.write((r.error or "chain failed") + "\n")
        return 1
    sys.stdout.write(json.dumps(r.data, ensure_ascii=False) if schema else r.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
