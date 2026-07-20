"""llmcall core: the cost/health chain codex -> cc -> claude for read-only, one-shot TEXT JUDGMENT.

One place solves every headless footgun the fleet re-solved 3-4 times: no-window creationflags,
`cmd /c` for .cmd launchers, absolute-path fallback under a scheduled task's minimal PATH, codex
`-o` outfile (strips the reasoning preamble), utf-8 to dodge GBK mojibake, and the MANDATORY _NO_MCP
(else ~26 MCP servers load and hang the one-shot). Read-only by construction: codex runs `-s
read-only` and cc/claude run with MCP disabled, so a judgment call can never be handed a tool.

`call()` NEVER raises: any provider failure is caught and the chain moves on; total failure returns a
falsy Result. Pure stdlib.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .schema import extract_json, validate

_NOWINDOW = {"creationflags": 0x08000000} if sys.platform == "win32" else {}
DEFAULT_CHAIN: Tuple[str, ...] = ("codex", "cc", "claude")

# Binary resolution: PATH first, then these absolute fallbacks (a scheduled task's minimal PATH would
# otherwise silently slide the chain to a pricier provider or drop codex entirely).
_CODEX_PATHS = [os.path.expanduser(r"~/AppData/Roaming/npm/codex.cmd"),
                os.path.expanduser(r"~/AppData/Roaming/npm/codex")]
_CC_PATHS = [os.path.expanduser(r"~/.local/bin/cc.cmd"), os.path.expanduser(r"~/.local/bin/cc")]
_CLAUDE_PATHS = [os.path.expanduser(r"~/.local/bin/claude.exe"),
                 os.path.expanduser(r"~/.local/bin/claude")]

# Disabling MCP is mandatory for a headless one-shot: both Claude Code CLIs otherwise load every
# configured MCP server and hang after the work is done, running out the time limit -> empty answer.
_NO_MCP = ("--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}')

_CODEX_FALLBACK_MODEL = "gpt-5.6-sol"
_CODEX_FALLBACK_EFFORT = "max"
_CLAUDE_FALLBACK_MODEL = "claude-opus-4-8"


@dataclass
class Attempt:
    provider: str
    ok: bool
    ms: int
    error: Optional[str] = None


@dataclass
class Result:
    """One canonical return shape (closes the str|None vs {available,raw} vs str divergence).
    Truthy + str-coercible so `str(r) if r else None` reproduces the old `str | None` contract."""
    text: str = ""
    provider: Optional[str] = None
    data: Any = None                     # validated object when schema= was given
    error: Optional[str] = None
    attempts: List[Attempt] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.provider is not None

    def __str__(self) -> str:
        return self.text


def _find(name: str, candidates: List[str]) -> Optional[str]:
    p = shutil.which(name)
    if p:
        return p
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _argv(binp: str, *args: str) -> List[str]:
    """A .cmd/.bat launcher must be run via `cmd /c` on Windows; run other binaries directly."""
    if sys.platform == "win32" and binp.lower().endswith((".cmd", ".bat")):
        return ["cmd", "/c", binp, *args]
    return [binp, *args]


def _run(cmd: List[str], prompt: str, timeout: float) -> Tuple[Optional[str], Optional[str]]:
    try:
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, **_NOWINDOW)
    except (subprocess.TimeoutExpired, OSError) as e:
        return None, str(e)[:200]
    if p.returncode != 0:
        return None, ((p.stderr or "").strip()[:200] or f"exit {p.returncode}")
    return (p.stdout or ""), None


# ---- single source of truth for model + effort (kwarg -> ~/.codex/config.toml -> fallback) --------
def _codex_config() -> dict:
    path = os.path.expanduser("~/.codex/config.toml")
    try:
        import tomllib  # stdlib on 3.11+
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _resolve_model(kind: str, model: Optional[str], effort: Optional[str]) -> Tuple[str, Optional[str]]:
    if kind == "codex":
        cfg = _codex_config()
        return (model or cfg.get("model") or _CODEX_FALLBACK_MODEL,
                effort or cfg.get("model_reasoning_effort") or _CODEX_FALLBACK_EFFORT)
    return model or _CLAUDE_FALLBACK_MODEL, None


def _unwrap_envelope(stdout: str) -> str:
    """Claude Code `--output-format json` wraps the text in {result: "..."}; unwrap it."""
    if not stdout:
        return ""
    try:
        env = json.loads(stdout)
        if isinstance(env, dict) and "result" in env:
            return env.get("result") or ""
    except Exception:
        pass
    return stdout


# ---- providers: each returns (text|None, error|None) ---------------------------------------------
def _codex(prompt, timeout, model, effort):
    binp = _find("codex", _CODEX_PATHS)
    if not binp:
        return None, "codex not found"
    m, eff = _resolve_model("codex", model, effort)
    fd, outpath = tempfile.mkstemp(prefix="llmcall_codex_", suffix=".txt")
    os.close(fd)
    try:
        cmd = _argv(binp, "exec", "-m", m, "-c", f"model_reasoning_effort={eff}",
                    "-s", "read-only", "--skip-git-repo-check", "--ephemeral",
                    "-c", "mcp_servers={}", "--color", "never", "-o", outpath, "-")
        stdout, err = _run(cmd, prompt, timeout)
        if stdout is None:
            return None, err
        try:
            with open(outpath, "r", encoding="utf-8") as f:
                return f.read(), None
        except OSError as e:
            return None, str(e)[:120]
    finally:
        try:
            os.remove(outpath)
        except OSError:
            pass


def _claude_family(name, paths, prompt, timeout, model):
    binp = _find(name, paths)
    if not binp:
        return None, f"{name} not found"
    m, _ = _resolve_model("claude", model, None)
    stdout, err = _run(_argv(binp, "-p", "--model", m, "--output-format", "json", *_NO_MCP),
                       prompt, timeout)
    if stdout is None:
        return None, err
    return _unwrap_envelope(stdout), None


def _invoke(name, prompt, timeout, model, effort):
    if name == "codex":
        return _codex(prompt, timeout, model, effort)
    if name == "cc":
        return _claude_family("cc", _CC_PATHS, prompt, timeout, model)
    if name == "claude":
        return _claude_family("claude", _CLAUDE_PATHS, prompt, timeout, model)
    return None, f"unknown provider {name}"


# ---- optional layers -----------------------------------------------------------------------------
def _validate_or_retry(name, prompt, text, schema, timeout, model, effort):
    """Return (data, error). Extract+validate the JSON; on failure retry the SAME provider once with a
    nudge (a per-provider self-correction) before the caller falls through to the next provider."""
    obj = extract_json(text)
    if obj is not None:
        ok, _ = validate(obj, schema)
        if ok:
            return obj, None
    nudge = prompt + "\n\nReturn ONLY valid JSON matching the required shape. No prose, no markdown."
    raw, err = _invoke(name, nudge, timeout, model, effort)
    obj = extract_json((raw or "").strip())
    if obj is not None:
        ok, e = validate(obj, schema)
        if ok:
            return obj, None
        return None, f"schema invalid after retry: {e}"
    return None, "no valid JSON after retry"


def _notify(stream: str, msg: str) -> None:
    relay = os.path.expanduser("the relay")
    if not os.path.isfile(relay):
        return
    try:
        subprocess.run([sys.executable, relay, "send", "--stream", stream, "--text", msg],
                       capture_output=True, text=True, encoding="utf-8", timeout=30, **_NOWINDOW)
    except Exception:
        pass


# ---- the chain -----------------------------------------------------------------------------------
def call(prompt: str, *, chain=DEFAULT_CHAIN, schema=None, timeout: float = 120.0,
         model: Optional[str] = None, effort: Optional[str] = None,
         notify: Optional[str] = None) -> Result:
    """Try providers in `chain` order; the first non-empty (and, with schema=, schema-valid) answer
    wins. Never raises. Returns a Result (falsy if the whole chain failed)."""
    r = Result()
    for name in chain:
        t0 = time.time()
        data = None
        try:
            raw, err = _invoke(name, prompt, timeout, model, effort)
            text = (raw or "").strip()
            if text and schema is not None:
                data, verr = _validate_or_retry(name, prompt, text, schema, timeout, model, effort)
                if data is None:
                    text, err = "", verr  # schema-invalid counts as a provider miss
        except Exception as e:  # the never-raises guarantee: a provider bug cannot escape the chain
            text, err = "", str(e)[:200]
        ms = int((time.time() - t0) * 1000)
        if not text:
            r.attempts.append(Attempt(name, False, ms, err))
            continue
        r.text, r.provider, r.data = text, name, data
        r.attempts.append(Attempt(name, True, ms))
        return r
    r.error = r.attempts[-1].error if r.attempts else "no provider available"
    if notify:
        _notify(notify, f"llmcall chain failed ({','.join(chain)}): {r.error}")
    return r
