"""CLI: the language-agnostic entry point for PowerShell/bash callers. Prompt on stdin, answer on
stdout, exit 0 if a provider answered else 1.

  echo "..." | python -m llmcall [--chain codex,cc,claude] [--schema f.json] [--timeout 120] \
                                 [--model M] [--effort E] [--notify STREAM]

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
    a = ap.parse_args()

    schema = None
    if a.schema:
        with open(a.schema, encoding="utf-8") as f:
            schema = json.load(f)
    prompt = sys.stdin.read()
    r = call(prompt, chain=[c.strip() for c in a.chain.split(",") if c.strip()],
             schema=schema, timeout=a.timeout, model=a.model, effort=a.effort, notify=a.notify)
    if not r:
        sys.stderr.write((r.error or "chain failed") + "\n")
        return 1
    sys.stdout.write(json.dumps(r.data, ensure_ascii=False) if schema else r.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
