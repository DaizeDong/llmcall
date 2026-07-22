"""llmcall core: the cost/health chain codex -> cc -> claude for read-only, one-shot TEXT JUDGMENT.

One place solves every headless footgun the fleet re-solved 3-4 times: no-window creationflags,
`cmd /c` for .cmd launchers, absolute-path fallback under a scheduled task's minimal PATH, codex
`-o` outfile (strips the reasoning preamble), utf-8 to dodge GBK mojibake, and the MANDATORY _NO_MCP
(else ~26 MCP servers load and hang the one-shot). Read-only by construction: codex runs `-s
read-only` and cc/claude run with MCP disabled, so a judgment call can never be handed a tool.

Two opt-in extensions widen this without changing the default:
- `chain=["gemini"]` routes to the Gemini CLI (search-grounded, for discovery diversity); gemini is
  never in the default chain.
- `web_search=True` is an explicit relaxation of the read-only default: it grants the network search
  tool (codex web_search; cc/claude built-in WebSearch/WebFetch) while codex stays FS-read-only. It is
  a research call, off by default, never implied.

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
_GEMINI_PATHS = [os.path.expanduser(r"~/AppData/Roaming/npm/gemini.cmd"),
                 os.path.expanduser(r"~/AppData/Roaming/npm/gemini")]

# Disabling MCP is mandatory for a headless one-shot: both Claude Code CLIs otherwise load every
# configured MCP server and hang after the work is done, running out the time limit -> empty answer.
_NO_MCP = ("--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}')

_CODEX_FALLBACK_MODEL = "gpt-5.6-sol"
_CODEX_FALLBACK_EFFORT = "max"
_CLAUDE_FALLBACK_MODEL = "claude-opus-4-8"
_GEMINI_FALLBACK_MODEL = "gemini-3-pro-preview"

# Opt-in web tools. web_search=True relaxes the read-only default: codex gains the web_search tool
# (still FS-read-only), cc/claude gain the built-in WebSearch/WebFetch (which work with MCP disabled).
# gemini is search-grounded by design, so it needs no extra flag. Never on by default.
_CC_WEB_TOOLS = ("--allowedTools", "WebSearch", "WebFetch")

# Capability tiers. mode="judge" (default) = read-only judgment (all footguns above). mode="research" =
# judge + the network search tool (== web_search=True). mode="agent" = full agency, provider-split:
# codex flips its sandbox to workspace-write in-process; cc/claude DELEGATE to the agent runner (the resilient
# full-session skill runner) rather than growing an agentic runner inside the judgment primitive.
_MODES = ("judge", "research", "agent")
_RUN_CLAUDE_AGENT = os.path.expanduser(r"the agent runner")


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
    depth: int = 0                       # refinement passes taken (0 for a plain call)

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
    if kind == "gemini":
        return model or _GEMINI_FALLBACK_MODEL, None
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
def _codex(prompt, timeout, model, effort, web_search=False, agentic=False):
    binp = _find("codex", _CODEX_PATHS)
    if not binp:
        return None, "codex not found"
    m, eff = _resolve_model("codex", model, effort)
    fd, outpath = tempfile.mkstemp(prefix="llmcall_codex_", suffix=".txt")
    os.close(fd)
    try:
        # mode="agent" flips the sandbox to workspace-write (FS write + full shell); judge/research stay
        # read-only. web_search is ORTHOGONAL (adds the network tool at either sandbox level), so an
        # FS-write agent gets the network only if the caller also asked for it.
        sandbox = "workspace-write" if agentic else "read-only"
        extra = ("-c", "tools.web_search=true") if web_search else ()
        cmd = _argv(binp, "exec", "-m", m, "-c", f"model_reasoning_effort={eff}",
                    "-s", sandbox, "--skip-git-repo-check", "--ephemeral",
                    "-c", "mcp_servers={}", *extra, "--color", "never", "-o", outpath, "-")
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


def _claude_family(name, paths, prompt, timeout, model, web_search=False, agentic=False):
    if agentic:
        return _claude_agent(prompt, timeout)
    binp = _find(name, paths)
    if not binp:
        return None, f"{name} not found"
    m, _ = _resolve_model("claude", model, None)
    # WebSearch/WebFetch are built-in Claude Code tools; they work with MCP disabled.
    tools = _CC_WEB_TOOLS if web_search else ()
    stdout, err = _run(_argv(binp, "-p", "--model", m, "--output-format", "json", *_NO_MCP, *tools),
                       prompt, timeout)
    if stdout is None:
        return None, err
    return _unwrap_envelope(stdout), None


def _claude_agent(prompt, timeout):
    """mode="agent" for cc/claude DELEGATES to the agent runner (agent-runner -Capture): it reuses that
    runner's resilient cc -> claude-direct transport (gateway-unset fallback) and full-session tool
    access, and returns the final answer on stdout. Never used on untrusted input; never the default;
    codex has NO the agent runner home (it cannot run Claude Code skills) so its agent path stays in-process."""
    if not os.path.isfile(_RUN_CLAUDE_AGENT):
        return None, "agent-runner not found (mode='agent' delegate unavailable)"
    fd, logpath = tempfile.mkstemp(prefix="llmcall_agent_", suffix=".log")
    os.close(fd)
    try:
        cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", _RUN_CLAUDE_AGENT,
               "-Prompt", prompt, "-Log", logpath, "-Capture"]
        stdout, err = _run(cmd, "", timeout)
        if stdout is None:
            return None, err
        return _unwrap_envelope(stdout.strip()), None
    finally:
        try:
            os.remove(logpath)
        except OSError:
            pass


def _gemini(prompt, timeout, model):
    """Gemini via the gemini-cli one-shot (`gemini -m <model> -p <prompt>`). Search-grounded by design,
    so web_search needs no extra flag. Reached as an EXCLUSIVE provider (chain=["gemini"]) for discovery
    diversity, never in the default codex->cc->claude chain. If gemini-cli is not installed it returns a
    not-found error and the chain moves on, exactly like any other missing provider."""
    binp = _find("gemini", _GEMINI_PATHS)
    if not binp:
        return None, "gemini not found"
    m, _ = _resolve_model("gemini", model, None)
    # gemini-cli takes the prompt as the -p argument (no stdin); output is plain text (no JSON envelope).
    stdout, err = _run(_argv(binp, "-m", m, "-p", prompt), "", timeout)
    if stdout is None:
        return None, err
    return stdout, None


def _invoke(name, prompt, timeout, model, effort, web_search=False, agentic=False):
    if name == "codex":
        return _codex(prompt, timeout, model, effort, web_search, agentic)
    if name == "cc":
        return _claude_family("cc", _CC_PATHS, prompt, timeout, model, web_search, agentic)
    if name == "claude":
        return _claude_family("claude", _CLAUDE_PATHS, prompt, timeout, model, web_search, agentic)
    if name == "gemini":
        return _gemini(prompt, timeout, model)
    return None, f"unknown provider {name}"


# ---- optional layers -----------------------------------------------------------------------------
def _apply(text, schema, extract):
    """Turn provider text into (obj, ok). extract= (a caller callable) WINS over schema=; a throwing
    extractor counts as a miss, never an escape (preserves never-raises). A None result = miss."""
    if extract is not None:
        try:
            obj = extract(text)
        except Exception:
            obj = None
        return obj, (obj is not None)
    obj = extract_json(text)
    if obj is None:
        return None, False
    ok, _ = validate(obj, schema)
    return (obj if ok else None), ok


def _extract_or_retry(name, prompt, text, schema, extract, timeout, model, effort, web_search, agentic):
    """Return (data, error). Apply schema=/extract= to the text; on a miss retry the SAME provider once
    with a nudge (per-provider self-correction) before the caller falls through to the next provider.
    This is the general form of the old schema-only path: schema= is _apply's extract_json+validate."""
    obj, ok = _apply(text, schema, extract)
    if ok:
        return obj, None
    nudge = prompt + "\n\nReturn ONLY valid JSON matching the required shape. No prose, no markdown."
    raw, err = _invoke(name, nudge, timeout, model, effort, web_search, agentic)
    obj, ok = _apply((raw or "").strip(), schema, extract)
    if ok:
        return obj, None
    return None, (err or "no valid result after retry")


def _notify(stream: str, msg: str) -> None:
    relay = os.path.expanduser("the relay")
    if not os.path.isfile(relay):
        return
    try:
        subprocess.run([sys.executable, relay, "send", "--stream", stream, "--text", msg],
                       capture_output=True, text=True, encoding="utf-8", timeout=30, **_NOWINDOW)
    except Exception:
        pass


# ---- provider-mix telemetry (append-only, out-of-repo, never affects the call) -------------------
# One JSONL line per call recording WHICH provider actually served. Makes silent provider degradation
# visible (e.g. the cc provider dying -> a provider silently degrades, whose
# only other signal is the provider bill). Off nowhere by default because it is pure local telemetry
# (no network, no PII -- prompt/reply are recorded only as CHAR COUNTS). LLMCALL_LEDGER=0 disables it;
# LLMCALL_LEDGER=<path> overrides the location. A ledger write can NEVER raise into the caller.
_LEDGER_DEFAULT = os.path.expanduser(r"~/.llmcall/ledger.jsonl")


def _ledger_path() -> Optional[str]:
    val = os.environ.get("LLMCALL_LEDGER")
    if val is not None:
        if val.strip() in ("0", "off", "false", ""):
            return None
        return os.path.expanduser(val)
    return _LEDGER_DEFAULT


def _ledger(rec: dict) -> None:
    path = _ledger_path()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # telemetry must never break a call


# ---- the chain -----------------------------------------------------------------------------------
def _log(log, msg):
    if log:
        try:
            log(msg)
        except Exception:
            pass


def call(prompt: str, *, chain=DEFAULT_CHAIN, schema=None, extract=None, mode: str = "judge",
         timeout: float = 120.0, model: Optional[str] = None, effort: Optional[str] = None,
         notify: Optional[str] = None, log=None, web_search: Optional[bool] = None) -> Result:
    """Try providers in `chain` order; the first non-empty (and, with schema=/extract=, parseable)
    answer wins. Never raises. Returns a Result (falsy if the whole chain failed).

    Structured output: schema=<JSON-Schema> validates + returns Result.data; extract=<callable(text)->obj>
    is the general form (return None to reject -> same-provider retry then fall through). extract= wins
    over schema=; both get one same-provider nudge-retry before the next provider.

    Capability tiers via mode= (default "judge" = read-only, MCP off, deterministic -- for classify /
    extract / draft / parse where the whole input is in the prompt): "research" grants the network
    search tool (== web_search=True); "agent" grants full agency (codex workspace-write in-process;
    cc/claude delegate to the agent runner). web_search overrides mode's network (None follows mode, True/False
    force). mode="agent" is an explicit escape hatch -- never use it on untrusted input.

    log=<callable(str)> is invoked once per attempt ("<provider>: answered / unavailable")."""
    web = web_search if web_search is not None else (mode == "research")
    agentic = (mode == "agent")
    r = Result()
    for name in chain:
        t0 = time.time()
        data = None
        try:
            raw, err = _invoke(name, prompt, timeout, model, effort, web, agentic)
            text = (raw or "").strip()
            if text and (schema is not None or extract is not None):
                data, verr = _extract_or_retry(name, prompt, text, schema, extract,
                                                timeout, model, effort, web, agentic)
                if data is None:
                    text, err = "", verr  # unparseable / schema-invalid counts as a provider miss
        except Exception as e:  # the never-raises guarantee: a provider bug cannot escape the chain
            text, err = "", str(e)[:200]
        ms = int((time.time() - t0) * 1000)
        if not text:
            _log(log, f"{name}: unavailable ({err})")
            r.attempts.append(Attempt(name, False, ms, err))
            continue
        _log(log, f"{name}: answered")
        r.text, r.provider, r.data = text, name, data
        r.attempts.append(Attempt(name, True, ms))
        _ledger({"provider": name, "chain": list(chain), "mode": mode, "web": web,
                 "prompt_chars": len(prompt), "reply_chars": len(text),
                 "ms": ms, "attempts": len(r.attempts), "ok": True})
        return r
    r.error = r.attempts[-1].error if r.attempts else "no provider available"
    _ledger({"provider": None, "chain": list(chain), "mode": mode, "web": web,
             "prompt_chars": len(prompt), "reply_chars": 0,
             "attempts": len(r.attempts), "ok": False, "error": (r.error or "")[:120]})
    if notify:
        _notify(notify, f"llmcall chain failed ({','.join(chain)}): {r.error}")
    return r


# ---- iterative deepening (opt-in) ----------------------------------------------------------------
_JUDGE_SYS = (
    "You are independently reviewing an answer produced for a task. Decide whether it is COMPLETE and "
    "CORRECT, or whether one more pass would make it materially better. Reply EXACTLY 'DONE' if it is "
    "good enough, or 'CONTINUE: <one concrete line on what to improve>'. Be strict about DONE: only "
    "say CONTINUE when another pass would genuinely help, not for cosmetic wording."
)


def _self_judge(orig_prompt, answer, chain, timeout, model, effort):
    """Independent review that decides done/continue. Runs on a ROTATED chain (a different provider
    than most likely generated the answer) so it is not the generator grading itself. Returns the
    verdict text ('DONE' / 'CONTINUE: ...') or None on failure (treated as DONE)."""
    judge_chain = (list(chain[1:]) + list(chain[:1])) if len(chain) > 1 else list(chain)
    jp = (f"{_JUDGE_SYS}\n\nTASK:\n{orig_prompt[:2000]}\n\nANSWER:\n{answer[:4000]}\n\nYour verdict:")
    r = call(jp, chain=judge_chain, timeout=timeout, model=model, effort=effort)
    return r.text.strip() if r else None


def refine(prompt: str, *, max_depth: int = 3, judge=None, chain=DEFAULT_CHAIN, schema=None,
           extract=None, mode: str = "judge", timeout: float = 120.0, model: Optional[str] = None,
           effort: Optional[str] = None, notify: Optional[str] = None, log=None,
           web_search: Optional[bool] = None) -> Result:
    """Iterative deepening: generate an answer, then decide from that answer whether to think harder,
    up to max_depth further passes. Generalizes the "a single headless call cannot be course-corrected"
    pattern into the primitive. Two judge modes:

      judge=None (default self-refine): after each pass an INDEPENDENT model call (rotated chain)
        reviews the answer and replies DONE or CONTINUE:<what to improve>; on CONTINUE the answer is
        regenerated with that critique. Stops at DONE, at convergence (the answer stops changing), or
        at max_depth.
      judge=callable: judge(result, depth) -> str | None. Return a follow-up prompt to go deeper, or
        None to accept the current result. Full control.

    Returns the final Result (Result.depth = passes taken, Result.attempts spans them all). Never
    raises; a total failure at any pass returns the best Result so far."""
    r = call(prompt, chain=chain, schema=schema, extract=extract, mode=mode, timeout=timeout, model=model, effort=effort, web_search=web_search, log=log)
    attempts = list(r.attempts)
    if not r:
        r.attempts = attempts
        if notify:
            _notify(notify, f"llmcall refine: first pass failed: {r.error}")
        return r
    for depth in range(1, max(0, max_depth) + 1):
        try:
            if callable(judge):
                nxt = judge(r, depth)
                if not nxt:
                    break
                r2 = call(nxt, chain=chain, schema=schema, extract=extract, mode=mode, timeout=timeout, model=model, effort=effort, web_search=web_search, log=log)
            else:
                verdict = _self_judge(prompt, r.text, chain, timeout, model, effort)
                if verdict is None or verdict.upper().startswith("DONE"):
                    break
                improve = (f"{prompt}\n\nYour previous answer:\n{r.text}\n\nAn independent reviewer says "
                           f"it can be improved:\n{verdict}\n\nProduce a better, complete answer.")
                r2 = call(improve, chain=chain, schema=schema, extract=extract, mode=mode, timeout=timeout, model=model, effort=effort, web_search=web_search, log=log)
        except Exception:
            break
        if not r2:
            break
        attempts += r2.attempts
        converged = r2.text.strip() == r.text.strip()
        r = r2
        r.depth = depth
        if converged:
            break
    r.attempts = attempts
    return r
