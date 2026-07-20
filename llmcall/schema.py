"""A small JSON-Schema-subset validator (no third-party dep). Covers only what the fleet's callers
actually use: type (object/array/string/number/integer/boolean), required, enum, nested properties
and items. Returns (ok, error_message)."""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Tuple


def extract_json(text: str) -> Any:
    """Pull the first JSON object or array out of a model's text (which may wrap it in prose)."""
    if not text:
        return None
    m = re.search(r"\[.*\]|\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
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
