"""A small JSON-Schema-subset validator (no third-party dep). Covers only what the fleet's callers
actually use: type (object/array/string/number/integer/boolean), required, enum, nested properties
and items. Returns (ok, error_message)."""
from __future__ import annotations

import json
from typing import Any, Optional, Tuple


def _iter_top_level(text: str):
    """Yield each top-level brace/bracket-balanced {...} or [...] substring, in order. String-aware:
    a { } [ ] inside a JSON double-quoted string (honoring \\ escapes) does not move the depth, so
    prose or a log line that merely mentions a brace cannot corrupt the scan."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    yield text[start:i + 1]
                    start = -1


def extract_json(text: str) -> Any:
    """Pull a JSON object/array out of a model's text (which may wrap it in prose or log chrome).

    Returns the FIRST top-level balanced {...}/[...] span that actually json.loads. This is strictly
    more robust than a greedy `\\{.*\\}` regex (which over-captures from the first brace to the last and
    is defeated by any stray brace in surrounding prose), and it is what every schema= consumer relies
    on. For the rarer "pick the object that HAS key X" need, pass call(..., extract=<callable>)."""
    if not text:
        return None
    for cand in _iter_top_level(text):
        try:
            return json.loads(cand)
        except (ValueError, TypeError):
            continue
    return None


def validate(obj: Any, schema: dict) -> Tuple[bool, Optional[str]]:
    t = schema.get("type")
    if t == "object":
        if not isinstance(obj, dict):
            return False, "expected object"
        for req in schema.get("required", []):
            if req not in obj:
                return False, f"missing required '{req}'"
        for k, sub in (schema.get("properties") or {}).items():
            if k in obj:
                ok, e = validate(obj[k], sub)
                if not ok:
                    return False, f"{k}: {e}"
    elif t == "array":
        if not isinstance(obj, list):
            return False, "expected array"
        item = schema.get("items")
        if item:
            for i, el in enumerate(obj):
                ok, e = validate(el, item)
                if not ok:
                    return False, f"[{i}]: {e}"
    elif t == "string":
        if not isinstance(obj, str):
            return False, "expected string"
    elif t == "number":
        if isinstance(obj, bool) or not isinstance(obj, (int, float)):
            return False, "expected number"
    elif t == "integer":
        if isinstance(obj, bool) or not isinstance(obj, int):
            return False, "expected integer"
    elif t == "boolean":
        if not isinstance(obj, bool):
            return False, "expected boolean"
    enum = schema.get("enum")
    if enum is not None and obj not in enum:
        return False, f"not in enum {enum}"
    return True, None
